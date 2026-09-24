module LayaMetalExt

using Laya: Laya, LayerNorm, Linear
using Metal: Metal, MetalBackend, MtlArray, MtlMatrix, MtlThreadGroupArray, @metal, threadgroup_barrier,
    simd_shuffle, simd_shuffle_xor, simd_ballot, simd_vote_any, thread_index_in_simdgroup, simdgroup_index_in_threadgroup,
    thread_position_in_threadgroup_1d, threadgroup_position_in_grid_1d, threadgroup_position_in_grid_3d,
    thread_position_in_grid_2d, thread_position_in_grid_3d,
    simdgroup_load, simdgroup_store, simdgroup_multiply_accumulate
using Metal: GPUArrays, MPS
# Metal.jl's cached matmul graphs (internal names, pinned by the `[compat]` bound on Metal)
using Metal.MPSGraphs: MatmulGraphKey, CachedMatmulGraph, _matmul_graph_cache, _matmul_graph_cache_lock,
    MPSGraphTensor, MPSGraphTensorData, default_exec_desc
using Metal.ObjectiveC.Foundation: @autoreleasepool, NSDictionary, nil
using PrecompileTools: @setup_workload, @compile_workload

# `Laya.load(repo; backend=MetalBackend())`. The model code in Laya is generic over array
# types; this extension moves the weights to the GPU and specializes the hot spots for
# `MtlArray`: one fused kernel for attention, simdgroup reductions for LayerNorm, a GeGLU
# kernel, and MPSGraph matrix products for the linear layers. Intermediates come from a
# buffer pool, so a warm forward pass allocates no device memory.

function Laya.load_backend_model(::MetalBackend, dir::AbstractString, ::Type{T}) where {T}
    T in (Float32, Float16) || throw(ArgumentError("MetalBackend supports Float32 and Float16, not $T"))
    Laya.adapt_arrays(MtlArray, first(Laya.load_model(dir; dtype=T)))
end

# ---------------------------------------------------------------------------- buffer pool

# Intermediates come from a pool keyed by (command queue, element type, byte size). Metal.jl
# passes kernel arguments by GPU address and keeps only the Julia objects alive until the
# command buffer completes, so a buffer must not go back to Metal while queued work may
# still touch it: `release!` never frees. Reusing a pooled buffer is safe because work on one
# queue runs in submission order. The element type is part of the key, and a new buffer is
# zeroed, because MPSGraph's matmul reads its output operand even with beta = 0 (NaN garbage
# would leak into the product); a same-typed buffer written by our own kernels is finite.
# Pool buffers return through their DataRef finalizer, so arrays that are dropped instead of
# released come back at the next GC. Buffers of other origins that are released wait in
# `GRAVE`, freed after waiting for the GPU once it holds `GRAVE_LIMIT` bytes; the pool's own
# buffers are freed only when it exceeds a quarter of the recommended working set.
const POOL = Dict{Tuple{UInt,DataType,Int},Vector{Metal.MTLBuffer}}()
const GRAVE = GPUArrays.DataRef[]
const POOL_LOCK = ReentrantLock()
const POOL_BYTES = Ref(0)
const POOL_MISSES = Ref(0)      # allocations the pool could not serve (for tests)
const POOL_PAGE = 16384
const GRAVE_LIMIT = 256 << 20   # bytes of released foreign buffers kept before waiting for the GPU
const GRAVE_BYTES = Ref(0)

pool_limit() = Int(Metal.device().recommendedMaxWorkingSetSize) ÷ 4
queue_id() = objectid(Metal.global_queue(Metal.device()))

"""The finalizer of a pool buffer's reference: the last owner hands the buffer back."""
struct PoolReturn
    key::Tuple{UInt,DataType,Int}
end
function (f::PoolReturn)(buf::Metal.MTLBuffer)
    @lock POOL_LOCK begin
        push!(get!(Vector{Metal.MTLBuffer}, POOL, f.key), buf)
        POOL_BYTES[] += f.key[3]
    end
    nothing
end

"""An uninitialized `MtlArray{T}` of size `dims`, from the pool when possible."""
function pooled(::Type{T}, dims::Dims{N}) where {T,N}
    bytes = cld(max(prod(dims) * sizeof(T), 1), POOL_PAGE) * POOL_PAGE
    key = (queue_id(), T, bytes)
    buf = @lock POOL_LOCK begin
        free = get(POOL, key, nothing)
        if free === nothing || isempty(free)
            nothing
        else
            POOL_BYTES[] -= bytes
            pop!(free)
        end
    end
    fresh = buf === nothing
    if fresh
        POOL_MISSES[] += 1
        POOL_BYTES[] > pool_limit() && evict!()
        buf = Metal.alloc(Metal.device(), bytes; storage=Metal.PrivateStorage)
    end
    GRAVE_BYTES[] > GRAVE_LIMIT && bury!()
    ref = GPUArrays.DataRef(PoolReturn(key), buf)
    x = MtlArray{T,N,Metal.PrivateStorage}(ref, dims; maxsize=bytes)
    GPUArrays.unsafe_free!(ref)     # `x` now holds the only reference
    fresh && fill!(x, zero(T))
    x
end
pooled(::Type{T}, dims::Integer...) where {T} = pooled(T, Int.(dims))

# Free the grave (and, for `evict!`, the pool's buffers) from the task that owns the queue,
# so that waiting for the GPU covers everything that may still use them.
function bury!()
    Metal.synchronize()
    refs = @lock POOL_LOCK begin
        refs = copy(GRAVE)
        empty!(GRAVE)
        GRAVE_BYTES[] = 0
        refs
    end
    foreach(GPUArrays.unsafe_free!, refs)
    nothing
end

function evict!()
    bury!()
    bufs = @lock POOL_LOCK begin
        bufs = reduce(vcat, values(POOL); init=Metal.MTLBuffer[])
        empty!(POOL)
        POOL_BYTES[] = 0
        bufs
    end
    foreach(Metal.free, bufs)
    nothing
end

# Dropping the last reference to a pool buffer returns it (see `PoolReturn`); a buffer of
# another origin that is solely owned waits in the grave; a view only drops its reference.
function Laya.release_one!(x::MtlArray)
    d = x.data
    d.freed && return
    if !(d.rc.finalizer isa PoolReturn) && x.offset == 0 && d.rc.count[] == 1
        @lock POOL_LOCK begin
            push!(GRAVE, copy(d))
            GRAVE_BYTES[] += Int(d[].length)
        end
    end
    Metal.unsafe_free!(x)
end

# ---------------------------------------------------------------------------- host <-> device

# Host arrays (token ids, masks, indices, pooled features) go to the device through shared
# buffers written with memcpy. Metal.jl's own upload synchronizes the whole queue and stages
# through a fresh buffer every time. A shared buffer can only be rewritten by the host once
# the GPU is done with it, so released upload buffers wait in `UPLOAD_PENDING` until a
# download (which waits for the queue) or, past `UPLOAD_LIMIT`, an explicit wait moves them
# back to `UPLOAD_FREE`.
const UPLOAD_FREE = Dict{Tuple{UInt,DataType,Int},Vector{Metal.MTLBuffer}}()
const UPLOAD_PENDING = Tuple{Tuple{UInt,DataType,Int},Metal.MTLBuffer}[]
const UPLOAD_PENDING_BYTES = Ref(0)
const UPLOAD_LIMIT = 64 << 20

struct UploadReturn
    key::Tuple{UInt,DataType,Int}
end
function (f::UploadReturn)(buf::Metal.MTLBuffer)
    @lock POOL_LOCK begin
        push!(UPLOAD_PENDING, (f.key, buf))
        UPLOAD_PENDING_BYTES[] += f.key[3]
    end
    nothing
end

"""Move released upload buffers to the free list; call only once the GPU is idle."""
function recycle_uploads!()
    @lock POOL_LOCK begin
        for (key, buf) in UPLOAD_PENDING
            push!(get!(Vector{Metal.MTLBuffer}, UPLOAD_FREE, key), buf)
        end
        empty!(UPLOAD_PENDING)
        UPLOAD_PENDING_BYTES[] = 0
    end
    nothing
end

function Laya.on_device_of(::MtlArray, x::AbstractArray{T,N}) where {T,N}
    xa = convert(Array{T,N}, x)
    bytes = cld(max(sizeof(xa), 1), POOL_PAGE) * POOL_PAGE
    key = (queue_id(), T, bytes)
    take() = @lock POOL_LOCK begin
        free = get(UPLOAD_FREE, key, nothing)
        free === nothing || isempty(free) ? nothing : pop!(free)
    end
    buf = take()
    if buf === nothing && UPLOAD_PENDING_BYTES[] > UPLOAD_LIMIT
        Metal.synchronize()
        recycle_uploads!()
        buf = take()
    end
    buf === nothing && (buf = Metal.alloc(Metal.device(), bytes; storage=Metal.SharedStorage))
    GC.@preserve xa unsafe_copyto!(convert(Ptr{T}, Metal.MTL.contents(buf)), pointer(xa), length(xa))
    ref = GPUArrays.DataRef(UploadReturn(key), buf)
    y = MtlArray{T,N,Metal.SharedStorage}(ref, size(xa); maxsize=bytes)
    GPUArrays.unsafe_free!(ref)     # `y` now holds the only reference
    y
end

# A download waits for the queue, so every upload released so far can be reused.
function Laya.to_host(x::MtlArray)
    a = Array(x)
    recycle_uploads!()
    a
end

# ---------------------------------------------------------------------------- matrix products

"""
    matmul!(C, A, B, tA, tB, m, n, k; batch=1)

`C = op(A) * op(B)` for column-major `C` `(m, n)` and `op(A)` `(m, k)`, `batch` times over
consecutive matrices, whatever the arrays' own shapes, through Metal.jl's cached MPSGraph
matmul graphs. (MPSMatrixMultiplication encoded by hand was faster to enqueue but produced NaN
about once per hundred forwards, so it is not used.)
"""
function matmul!(C::MtlArray, A::MtlArray, B::MtlArray, tA::Bool, tB::Bool, m::Integer, n::Integer, k::Integer;
                 batch::Integer=1)
    if batch == 1
        c = reshape(C, m, n)
        a = tA ? reshape(A, k, m) : reshape(A, m, k)
        b = tB ? reshape(B, n, k) : reshape(B, k, n)
        graph_matmul_batched!(c, a, b, tA ? 'T' : 'N', tB ? 'T' : 'N')
        # Drop the views' references now, so the parents can go back to the pool (`reshape`
        # returns the array itself when the shape already matches).
        c === C || Metal.unsafe_free!(c)
        a === A || Metal.unsafe_free!(a)
        b === B || Metal.unsafe_free!(b)
    else
        graph_matmul_batched!(C, A, B, tA ? 'T' : 'N', tB ? 'T' : 'N')
    end
    C
end

# `Metal.MPSGraphs.graph_matmul!` (alpha = 1, beta = 0) with the graph encoded into the batched
# command buffer of Metal.jl's queue. `graph_matmul!` itself commits a command buffer of its own
# per call, which also flushes the kernel batch: about 170 command buffers per forward pass,
# each with its submission latency. Encoded in the batch, a single-question forward pass runs
# in a handful of command buffers. The operands stay alive with the batch's roots.
@autoreleasepool function graph_matmul_batched!(c::MtlArray, a::MtlArray{T}, b::MtlArray{T}, tA::Char, tB::Char) where {T}
    key = MatmulGraphKey(a, b, c, true, false, tA, tB)
    cached = @lock _matmul_graph_cache_lock get!(_matmul_graph_cache, key) do
        CachedMatmulGraph(key)
    end
    feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
        cached.place_a => MPSGraphTensorData(a), cached.place_b => MPSGraphTensorData(b),
        cached.place_c => MPSGraphTensorData(c))
    results = Dict{MPSGraphTensor,MPSGraphTensorData}(cached.result => feeds[cached.place_c])
    bq = Metal.global_queue(Metal.device())
    Metal.end_encoder!(bq)
    cmdbuf = MPS.MPSCommandBuffer(Metal.ensure_cmdbuf!(bq))
    MPS.encode!(cmdbuf, cached.graph, NSDictionary(feeds), NSDictionary(results), nil, default_exec_desc())
    Metal.record_operation!(bq, a, b, c, feeds, results, cmdbuf)
    Metal.maybe_autoflush!(bq)
    c
end

function (l::Linear{<:MtlMatrix{T}})(x::MtlArray{T}) where {T}
    k, m = size(l.weight)                          # weight is (in, out)
    n = length(x) ÷ k
    y = pooled(T, (m, size(x)[2:end]...))
    matmul!(y, l.weight, x, true, false, m, n, k)
    l.bias === nothing || launch!(bias_kernel, (m, n), y, l.bias, m, n)
    y
end

"""
    launch!(kernel, dims, args...)

Run `kernel(args...)` with one thread per work item of the 1-, 2- or 3-D grid `dims`; integer
arguments become `Int32`. Kernels read their item with `grid2()` / `grid3()` and check the
first two coordinates, which the threadgroup shape may round up.
"""
function launch!(kernel, dims::Dims{N}, args...) where {N}
    args32 = map(a -> a isa Integer && !(a isa Bool) ? Int32(a) : a, args)
    tx = min(dims[1], 256)
    ty = N >= 2 ? min(dims[2], max(1, 256 ÷ tx)) : 1
    groups = (cld(dims[1], tx), N >= 2 ? cld(dims[2], ty) : 1, N >= 3 ? dims[3] : 1)
    @metal threads=(tx, ty, 1) groups=groups kernel(args32...)
end

# Index math stays in Int32: 64-bit division is slow on Apple GPUs, and the grids are chosen
# so that no kernel needs more than one division per thread.
@inline grid2() = (p = thread_position_in_grid_2d(); (Int32(p.x), Int32(p.y)))
@inline grid3() = (p = thread_position_in_grid_3d(); (Int32(p.x), Int32(p.y), Int32(p.z)))
@inline divmod1(a::Int32, n::Int32) = (q = (a - Int32(1)) ÷ n; (q + Int32(1), a - q * n))   # (cld(a, n), mod1(a, n))

# ---------------------------------------------------------------------------- reductions

# Metal's fast exp (relative error about 1e-6, exp(-Inf) = 0): the softmax kernels spend a
# sixth of their time in the precise one.
@inline fast_exp(x::Float32) = ccall("extern air.fast_exp.f32", llvmcall, Float32, (Float32,), x)

# LayerNorm and softmax reduce over one column per simdgroup (32 lanes, shuffles instead of
# barriers), `REDUCE_COLS` columns per threadgroup. Metal.jl's simdgroup and lane indices are
# 1-based like its grid positions.
const SIMD = 32
const REDUCE_COLS = 8

"""`op`-reduction of `v` over the simdgroup; every lane gets the result."""
@inline function simd_reduce(op, v::Float32)
    v = op(v, simd_shuffle_xor(v, Int16(16)))
    v = op(v, simd_shuffle_xor(v, Int16(8)))
    v = op(v, simd_shuffle_xor(v, Int16(4)))
    v = op(v, simd_shuffle_xor(v, Int16(2)))
    v = op(v, simd_shuffle_xor(v, Int16(1)))
    v
end

"""This thread's lane (1-based) and the column its simdgroup reduces."""
@inline lane_col() = (Int32(thread_index_in_simdgroup()),
    (Int32(threadgroup_position_in_grid_1d()) - Int32(1)) * Int32(REDUCE_COLS) + Int32(simdgroup_index_in_threadgroup()))

launch_reduce!(kernel, ncol::Integer, args...) =
    @metal threads=SIMD*REDUCE_COLS groups=cld(ncol, REDUCE_COLS) kernel(args..., Int32(ncol))

# Statistics in Float32 as MLX's fast.layer_norm (also for Float16). With `r !== nothing` the
# column is first replaced by the residual sum `x + r`, which is also written to `z`. Each lane
# keeps its `NV` elements of the column in registers, so the column is read once.
function layernorm_kernel(y, z, x, r, w, b, d, eps, ::Val{NV}, ncol) where {NV}
    lane, col = lane_col()
    col <= ncol || return
    o = (col - Int32(1)) * d
    v = ntuple(Val(NV)) do k
        i = lane + Int32(SIMD) * Int32(k - 1)
        if i > d
            0.0f0
        elseif r === nothing
            @inbounds Float32(x[o+i])
        else
            @inbounds t = x[o+i] + r[o+i]
            @inbounds z[o+i] = t
            Float32(t)
        end
    end
    s = 0.0f0
    for k in 1:NV
        s += v[k]
    end
    μ = simd_reduce(+, s) / d
    q = 0.0f0
    for k in 1:NV
        lane + Int32(SIMD) * Int32(k - 1) <= d && (q += abs2(v[k] - μ))
    end
    inv_σ = 1.0f0 / sqrt(simd_reduce(+, q) / d + eps)
    for k in 1:NV
        i = lane + Int32(SIMD) * Int32(k - 1)
        i <= d || continue
        @inbounds t = (v[k] - μ) * inv_σ * Float32(w[i])
        b === nothing || (@inbounds t += Float32(b[i]))
        @inbounds y[o+i] = t
    end
    return
end

function launch_layernorm!(y, z, x, r, ln::LayerNorm)
    d = size(x, 1)
    launch_reduce!(layernorm_kernel, length(x) ÷ d, y, z, x, r, ln.weight, ln.bias, Int32(d), ln.eps, Val(cld(d, SIMD)))
end

function (ln::LayerNorm)(x::MtlArray{T}) where {T}
    y = pooled(T, size(x))
    launch_layernorm!(y, nothing, x, nothing, ln)
    y
end

function Laya.residual_norm(x::MtlArray{T}, r::MtlArray{T}, ln::LayerNorm) where {T}
    z, y = pooled(T, size(x)), pooled(T, size(x))
    launch_layernorm!(y, z, x, r, ln)
    Laya.release!(r)
    z, y
end

# Masked softmax over keys, one column of S `(Lk, Lq, H*B)` per simdgroup; `mask` is
# `(Lk, Mq, B)` with `Mq` either `Lq` or 1 (`true` keeps a key).
function masked_softmax_kernel(P, S, mask, Lk, Lq, H, Mq, usemask, ncol)
    lane, col = lane_col()
    col <= ncol || return
    n, qi = divmod1(col, Lq)
    b = (n - Int32(1)) ÷ H + Int32(1)
    o = (col - Int32(1)) * Lk
    mo = ((Mq == Int32(1) ? Int32(1) : qi) - Int32(1)) * Lk + (b - Int32(1)) * Lk * Mq
    keep(j) = !usemask || @inbounds mask[mo+j]
    m = -Inf32
    for j in lane:Int32(SIMD):Lk
        keep(j) && (@inbounds m = max(m, S[o+j]))
    end
    m = simd_reduce(max, m)
    t = 0.0f0
    for j in lane:Int32(SIMD):Lk
        @inbounds t += keep(j) ? fast_exp(S[o+j] - m) : 0.0f0
    end
    r = 1.0f0 / simd_reduce(+, t)
    for j in lane:Int32(SIMD):Lk
        @inbounds P[o+j] = keep(j) ? fast_exp(S[o+j] - m) * r : 0.0f0
    end
    return
end

# ---------------------------------------------------------------------------- elementwise

function bias_kernel(y, b, m, n)
    i, j = grid2()
    (i <= m && j <= n) && (@inbounds y[i+(j-Int32(1))*m] += b[i])
    return
end

function gelu_gate_kernel(out, y, n, ncol)
    i, j = grid2()
    if i <= n && j <= ncol
        o = (j - Int32(1)) * Int32(2) * n
        @inbounds out[i+(j-Int32(1))*n] = Laya.gelu(y[o+i]) * y[o+n+i]
    end
    return
end

function Laya.gelu_gate(y::MtlArray{T}) where {T}
    n = size(y, 1) ÷ 2
    out = pooled(T, (n, size(y)[2:end]...))
    launch!(gelu_gate_kernel, (n, length(out) ÷ n), out, y, n, length(out) ÷ n)
    out
end

# One kernel per activation, on a 2-D grid `(n, 1)`: in this module, a kernel taking the
# function as an argument or reading `thread_position_in_grid_1d` crashed LLVM's inliner
# (see docs/agents/workarounds.md).
function relu_kernel(y, x, n)
    i, _ = grid2()
    i <= n && (@inbounds y[i] = Laya.relu(x[i]))
    return
end
function gelu_kernel(y, x, n)
    i, _ = grid2()
    i <= n && (@inbounds y[i] = Laya.gelu(x[i]))
    return
end

for (f, kernel) in ((:(Laya.relu), :relu_kernel), (:(Laya.gelu), :gelu_kernel))
    @eval function Laya.elementwise(::typeof($f), x::MtlArray{T}) where {T}
        y = pooled(T, size(x))
        launch!($kernel, (length(x), 1), y, x, length(x))
        y
    end
end

function add_kernel(z, x, y, n)
    i, _ = grid2()
    i <= n && (@inbounds z[i] = x[i] + y[i])
    return
end

function Laya.residual(x::MtlArray{T}, y::MtlArray{T}) where {T}
    z = pooled(T, size(x))
    launch!(add_kernel, (length(x), 1), z, x, y, length(x))
    Laya.release!(y)
    z
end

function add_columns_kernel(z, x, v, d, L, B)
    i, l, b = grid3()
    (i <= d && l <= L) && (@inbounds z[i+d*((l-Int32(1))+L*(b-Int32(1)))] = x[i+d*((l-Int32(1))+L*(b-Int32(1)))] + v[i+d*(b-Int32(1))])
    return
end

function Laya.add_columns(x::MtlArray{T,3}, v::MtlMatrix{T}) where {T}
    d, L, B = size(x)
    z = pooled(T, size(x))
    launch!(add_columns_kernel, (d, L, B), z, x, v, d, L, B)
    z
end

# y `(d, n)` = E[:, idx] for device indices `idx` (1-based); also `columns_at` with a stride.
function gather_kernel(y, E, idx, d, n, j, L)
    i, k = grid2()
    (i <= d && k <= n) && (@inbounds y[i+d*(k-Int32(1))] = E[i+d*((idx === nothing ? j - Int32(1) + L * (k - Int32(1)) : idx[k] - Int32(1)))])
    return
end

function Laya.gather_columns(E::MtlMatrix{T}, idx::AbstractVector{<:Integer}) where {T}
    d, n = size(E, 1), length(idx)
    di = Laya.on_device_of(E, Int32.(idx))
    y = pooled(T, (d, n))
    launch!(gather_kernel, (d, n), y, E, di, d, n, 0, 0)
    Laya.release!(di)
    y
end

function Laya.columns_at(h::MtlArray{T,3}, j::Integer) where {T}
    d, L, B = size(h)
    y = pooled(T, (d, B))
    launch!(gather_kernel, (d, B), y, h, nothing, d, B, j, L)
    y
end

# ---------------------------------------------------------------------------- attention

# cos/sin tables `(half, L)` per (dtype, head_dim, L, base), computed on the host exactly as
# `Laya.rope` does and kept on the device.
const ROPE_TABLES = Dict{Tuple{DataType,Int,Int,Float64},Tuple{MtlArray,MtlArray}}()
const ROPE_LOCK = ReentrantLock()

function rope_tables(::Type{T}, hd::Int, L::Int, base::Real) where {T}
    @lock ROPE_LOCK get!(ROPE_TABLES, (T, hd, L, Float64(base))) do
        half = hd ÷ 2
        lb = log2(Float32(base))
        inv_freq = [exp2(-(Float32(i) / Float32(half)) * lb) for i in 0:half-1]
        θ = inv_freq .* reshape(Float32.(0:L-1), 1, L)
        (MtlArray(T.(cos.(θ))), MtlArray(T.(sin.(θ))))
    end
end

# qkv `(hd, H, 3, L, B)` -> q, k, v `(hd, L, H*B)` (heads batched for MPSGraph), with RoPE on q
# and k (same arithmetic as `Laya.rope`: `[x1 c - x2 s; x1 s + x2 c]`) and the attention
# `scale` folded into q. Grid `(hd, L, H*B)`.
function split_rope_kernel(q, k, v, qkv, c, s, hd, H, L, userope, scale)
    i, l, n = grid3()                     # n = (b - 1) * H + h
    (i <= hd && l <= L) || return
    b, h = divmod1(n, H)
    idx = i + hd * ((l - Int32(1)) + L * (n - Int32(1)))
    base = hd * ((h - Int32(1)) + H * Int32(3) * ((l - Int32(1)) + L * (b - Int32(1))))
    src(p, j) = j + base + hd * H * p
    @inbounds v[idx] = qkv[src(Int32(2), i)]
    if userope
        half = hd ÷ Int32(2)
        if i <= half
            @inbounds cc, ss = c[i+half*(l-Int32(1))], s[i+half*(l-Int32(1))]
            @inbounds q[idx] = (qkv[src(Int32(0), i)] * cc - qkv[src(Int32(0), i + half)] * ss) * scale
            @inbounds k[idx] = qkv[src(Int32(1), i)] * cc - qkv[src(Int32(1), i + half)] * ss
        else
            j = i - half
            @inbounds cc, ss = c[j+half*(l-Int32(1))], s[j+half*(l-Int32(1))]
            @inbounds q[idx] = (qkv[src(Int32(0), j)] * ss + qkv[src(Int32(0), i)] * cc) * scale
            @inbounds k[idx] = qkv[src(Int32(1), j)] * ss + qkv[src(Int32(1), i)] * cc
        end
    else
        @inbounds q[idx] = qkv[src(Int32(0), i)] * scale
        @inbounds k[idx] = qkv[src(Int32(1), i)]
    end
    return
end

# out `(hd, L, H*B)` -> `(hd, H, L, B)` = `(d, L, B)`, the input layout of the output
# projection. Grid `(d, L, B)`.
function merge_heads_kernel(y, out, hd, H, L)
    i, l, b = grid3()
    (i <= hd * H && l <= L) || return
    h, ii = divmod1(i, hd)
    @inbounds y[i+hd*H*((l-Int32(1))+L*(b-Int32(1)))] = out[ii+hd*((l-Int32(1))+L*((h-Int32(1))+H*(b-Int32(1))))]
    return
end

function Laya.qkv_attention(qkv::MtlArray{T,3}, H::Integer, base, am, scale::Real) where {T}
    d, L, B = size(qkv, 1) ÷ 3, size(qkv, 2), size(qkv, 3)
    mask = Laya.dense_mask(am)        # the kernels read the dense mask; masked tiles are skipped by it
    d ÷ H == FA_HD || return attention_unfused(qkv, H, base, mask, scale)
    y = pooled(T, d, L, B)
    usemask = mask isa AbstractArray
    m = usemask ? mask : y
    Mq = usemask ? size(mask, 2) : 1
    # local attention: tiles outside the window are not visited (`window < 0`: all tiles).
    # `valid` is passed for every AttentionMask so that all layers share one compiled kernel.
    valid = am isa Laya.AttentionMask ? am.valid : m
    window = am isa Laya.AttentionMask && am.window !== nothing ? Int32(am.window) : Int32(-1)
    c, s = base === nothing ? (qkv, qkv) : rope_tables(T, FA_HD, L, base)
    @metal threads=FA_THREADS groups=(cld(L, FA_BQ), H, B) attention_kernel(y, qkv, c, s, m, valid, window, Int32(H), Int32(L), Int32(Mq), usemask, base !== nothing, Float32(scale))
    y
end

# ---------------------------------------------------------------------------- fused attention

# One kernel for the whole attention of a head_dim-64 model: a threadgroup of 8 simdgroups
# takes 64 queries of one head and walks the keys in tiles of 32 with an online softmax.
# The Q tile is staged once and kept as simdgroup matrix fragments; each K tile is staged in
# threadgroup memory (RoPE applied on the way in), S = KᵀQ and O += V P run on the 8x8
# simdgroup matrix units, and the output goes out from registers in the layout of the output
# projection. Nothing but the output touches device memory: no S, P or head-permuted copies.
# A tile whose mask is all false (padding) is skipped, and under local attention the tiles
# outside the window are not visited. Threadgroup memory is kept to 16 KB (K, then V share
# one buffer): the matrix products are bound by threadgroup-memory loads, and 64 queries per
# threadgroup halve those per query while still fitting two threadgroups on a core.
const FA_HD = 64
const FA_SG = 8
const FA_BQ = 8FA_SG
const FA_BK = 32
const FA_THREADS = 32FA_SG

@inline zfrag() = Metal.simdgroup_matrix_init_filled(0.0f0)
@inline barrier() = threadgroup_barrier(Metal.MemoryFlagThreadGroup)
@inline qkv_offset(p, l, h, H, L, b) =
    Int32(FA_HD) * ((h - Int32(1)) + H * (p + Int32(3) * ((l - Int32(1)) + L * (b - Int32(1)))))

# 32 rows of q or k (p = 0, 1) from `l0 + 1` into `tile` `(FA_HD, 32)`, RoPE and scale applied
@inline function stage_rope!(tile, qkv, c, s, p, l0, L, h, H, b, userope, scale, t)
    half = Int32(FA_HD ÷ 2)
    for idx in t:Int32(FA_THREADS):Int32(half * FA_BK)
        j = (idx - Int32(1)) & (half - Int32(1)) + Int32(1)
        ll = (idx - Int32(1)) >> Int32(5) + Int32(1)
        l = l0 + ll
        o = (ll - Int32(1)) * Int32(FA_HD)
        if l <= L
            src = qkv_offset(p, l, h, H, L, b)
            @inbounds x1 = Float32(qkv[src+j])
            @inbounds x2 = Float32(qkv[src+j+half])
            if userope
                @inbounds cc = Float32(c[j+half*(l-Int32(1))])
                @inbounds ss = Float32(s[j+half*(l-Int32(1))])
                @inbounds tile[o+j] = (x1 * cc - x2 * ss) * scale
                @inbounds tile[o+j+half] = (x1 * ss + x2 * cc) * scale
            else
                @inbounds tile[o+j] = x1 * scale
                @inbounds tile[o+j+half] = x2 * scale
            end
        else
            @inbounds tile[o+j] = 0.0f0
            @inbounds tile[o+j+half] = 0.0f0
        end
    end
end

@inline function stage_v!(tile, qkv, l0, L, h, H, b, t)
    for idx in t:Int32(FA_THREADS):Int32(FA_HD * FA_BK)
        i = (idx - Int32(1)) & Int32(FA_HD - 1) + Int32(1)
        ll = (idx - Int32(1)) >> Int32(6) + Int32(1)
        l = l0 + ll
        if l <= L
            @inbounds tile[idx] = Float32(qkv[qkv_offset(Int32(2), l, h, H, L, b)+i])
        else
            @inbounds tile[idx] = 0.0f0
        end
    end
end

@inline mma_k(acc, Kt, hb, q) =
    ntuple(kb -> simdgroup_multiply_accumulate(simdgroup_load(Kt, (1 + 8hb, 1 + 8(kb - 1)), Val(false)), q, acc[kb]), Val(4))
@inline mma_v(acc, Vt, kb, p) =
    ntuple(hb -> simdgroup_multiply_accumulate(simdgroup_load(Vt, (1 + 8(hb - 1), 1 + 8kb), Val(true)), p, acc[hb]), Val(8))
@inline load_q(Qt, qo) = ntuple(hb -> simdgroup_load(Qt, (1 + 8(hb - 1), qo), Val(true)), Val(8))

# S `(FA_BK, FA_BQ)` = Ktᵀ Q for this simdgroup's query block, with Q fragments `qf` in registers.
@inline function s_tile!(St, Kt, qf, qo)
    acc = (zfrag(), zfrag(), zfrag(), zfrag())
    for hb in 0:7
        acc = mma_k(acc, Kt, hb, qf[hb+1])
    end
    for kb in 1:4
        simdgroup_store(acc[kb], St, (1 + 8(kb - 1), qo), Val(true))
    end
end

# Lane `λ` (0-based) of a simdgroup holds, in every O fragment, head-dim row `ri` of the
# simdgroup's queries `cq` and `cq + 1` (`workarounds.md` item 17): the O rescale and the
# output need no trip through threadgroup memory.
@inline frag_query(λ) = Int32(2) * (λ & Int32(1)) + Int32(4) * ((λ >> Int32(3)) & Int32(1))
@inline frag_row(λ) = ((λ >> Int32(1)) & Int32(3)) + Int32(4) * ((λ >> Int32(4)) & Int32(1))
@inline scale_frag(f, a1, a2) = ntuple(i -> i == 1 ? VecElement(f[1].value * a1) : i == 2 ? VecElement(f[2].value * a2) : f[i], Val(64))
# a per-query value from the 4 threads that own the query in the softmax (1-based lanes)
@inline query_value(x, cq) = (simd_shuffle(x, Int16(4) * Int16(cq) + Int16(1)), simd_shuffle(x, Int16(4) * Int16(cq) + Int16(5)))

@inline function key_bits(mask, usemask, k0, part, qi, L, Mq, b)
    bits = Int32(0)
    if qi <= L
        mo = L * ((Mq == Int32(1) ? Int32(0) : qi - Int32(1)) + Mq * (b - Int32(1)))
        for j in Int32(1):Int32(8)
            kk = k0 + Int32(8) * part + j
            keep = kk <= L && (!usemask || @inbounds mask[mo+kk])
            keep && (bits |= Int32(1) << (j - Int32(1)))
        end
    end
    bits
end

@inline function tile_kept(flags, bits, t, sg)
    sg_any = simd_vote_any(simd_ballot(bits != Int32(0)))
    t <= Int32(FA_SG) && (@inbounds flags[t] = Int32(0))
    barrier()
    sg_any && (@inbounds flags[sg] = Int32(1))
    barrier()
    kept = Int32(0)
    for i in 1:FA_SG
        @inbounds kept |= flags[i]
    end
    kept != Int32(0)
end

# Key tiles the queries `q0 + 1 : q0 + FA_BQ` can see under local attention of half-width
# `window` (all tiles if `window < 0` or a query of the block is padding: padded queries keep
# every valid key). Uniform over the threadgroup.
@inline function tile_range(valid, window, q0, L, b, ntiles)
    window < Int32(0) && return Int32(0), ntiles - Int32(1)
    pad = false
    for q in q0+Int32(thread_index_in_simdgroup()):Int32(32):min(q0 + Int32(FA_BQ), L)
        pad |= (@inbounds valid[q+L*(b-Int32(1))]) == false
    end
    simd_ballot(pad) != 0 && return Int32(0), ntiles - Int32(1)
    lo = max(q0 - window, Int32(0)) ÷ Int32(FA_BK)
    hi = min((min(q0 + Int32(FA_BQ), L) - Int32(1) + window) ÷ Int32(FA_BK), ntiles - Int32(1))
    lo, hi
end

# Single pass, online softmax. Threadgroup memory: `buf` (FA_HD, 32) for the Q staging (32
# queries at a time), then each K tile and each V tile in turn; `St` (FA_BK, FA_BQ) for S then P.
function attention_kernel(y, qkv, c, s, mask, valid, window, H, L, Mq, usemask, userope, scale)
    buf = MtlThreadGroupArray(Float32, (FA_HD, FA_BK))
    St = MtlThreadGroupArray(Float32, (FA_BK, FA_BQ))
    flags = MtlThreadGroupArray(Int32, FA_SG)
    g = threadgroup_position_in_grid_3d()
    q0 = (Int32(g.x) - Int32(1)) * Int32(FA_BQ)
    h, b = Int32(g.y), Int32(g.z)
    t = Int32(thread_position_in_threadgroup_1d())
    sg = Int32(simdgroup_index_in_threadgroup())
    qo = 1 + 8 * Int(sg - Int32(1))
    col = (t - Int32(1)) >> Int32(2) + Int32(1)
    part = (t - Int32(1)) & Int32(3)
    qi = q0 + col
    so = Int32(8) * part + Int32(FA_BK) * (col - Int32(1))
    ntiles = (L + Int32(FA_BK - 1)) ÷ Int32(FA_BK)
    λ = Int32(thread_index_in_simdgroup()) - Int32(1)
    cq = frag_query(λ)

    # Q in two halves of 32 queries through `buf`
    stage_rope!(buf, qkv, c, s, Int32(0), q0, L, h, H, b, userope, scale, t)
    barrier()
    qf = load_q(buf, 1 + 8 * ((Int(sg) - 1) & 3))
    barrier()
    stage_rope!(buf, qkv, c, s, Int32(0), q0 + Int32(32), L, h, H, b, userope, scale, t)
    barrier()
    qn = load_q(buf, 1 + 8 * ((Int(sg) - 1) & 3))
    qf = sg > Int32(4) ? qn : qf
    barrier()

    m = -Inf32
    l = 0.0f0
    acc = (zfrag(), zfrag(), zfrag(), zfrag(), zfrag(), zfrag(), zfrag(), zfrag())
    t0, t1 = tile_range(valid, window, q0, L, b, ntiles)
    for tile in t0:t1
        k0 = tile * Int32(FA_BK)
        bits = key_bits(mask, usemask, k0, part, qi, L, Mq, b)
        tile_kept(flags, bits, t, sg) || continue
        stage_rope!(buf, qkv, c, s, Int32(1), k0, L, h, H, b, userope, 1.0f0, t)
        barrier()
        s_tile!(St, buf, qf, qo)
        barrier()
        # softmax statistics of this thread's 8 keys, combined over the 4 threads of the query
        mt = -Inf32
        for j in Int32(1):Int32(8)
            (bits >> (j - Int32(1))) & Int32(1) == Int32(1) && (@inbounds mt = max(mt, St[so+j]))
        end
        mt = max(mt, simd_shuffle_xor(mt, Int16(1)))
        mt = max(mt, simd_shuffle_xor(mt, Int16(2)))
        mn = max(m, mt)
        α = mn == m ? 1.0f0 : fast_exp(m - mn)
        p = 0.0f0
        for j in Int32(1):Int32(8)
            keep = (bits >> (j - Int32(1))) & Int32(1) == Int32(1)
            @inbounds e = keep ? fast_exp(St[so+j] - mn) : 0.0f0
            @inbounds St[so+j] = e
            p += e
        end
        p += simd_shuffle_xor(p, Int16(1))
        p += simd_shuffle_xor(p, Int16(2))
        l = l * α + p
        m = mn
        a1, a2 = query_value(α, cq)
        acc = map(f -> scale_frag(f, a1, a2), acc)
        stage_v!(buf, qkv, k0, L, h, H, b, t)       # K is no longer read: the barrier above
        barrier()
        for kb in 0:3
            pf = simdgroup_load(St, (1 + 8kb, qo), Val(true))
            acc = mma_v(acc, buf, kb, pf)
        end
        barrier()
    end
    i1, i2 = query_value(l > 0.0f0 ? 1.0f0 / l : 0.0f0, cq)
    lq = q0 + Int32(qo - 1) + cq + Int32(1)
    ri = frag_row(λ)
    for hb in 1:8
        o = Int32(8(hb - 1)) + ri + Int32(1) + Int32(FA_HD) * ((h - Int32(1)) + H * ((lq - Int32(1)) + L * (b - Int32(1))))
        lq <= L && (@inbounds y[o] = acc[hb][1].value * i1)
        lq < L && (@inbounds y[o+Int32(FA_HD)*H] = acc[hb][2].value * i2)
    end
    return
end

# ---------------------------------------------------------------------------- unfused attention

# For other head sizes: split + RoPE, batched MPSGraph products, masked softmax, merge.
function attention_unfused(qkv::MtlArray{T,3}, H::Integer, base, mask, scale::Real) where {T}
    d, L, B = size(qkv, 1) ÷ 3, size(qkv, 2), size(qkv, 3)
    hd, N = d ÷ H, H * B
    q, k, v = pooled(T, hd, L, N), pooled(T, hd, L, N), pooled(T, hd, L, N)
    c, s = base === nothing ? (q, q) : rope_tables(T, hd, L, base)
    launch!(split_rope_kernel, (hd, L, N), q, k, v, qkv, c, s, hd, H, L, base !== nothing, T(scale))
    S = pooled(Float32, L, L, N)
    matmul!(S, k, q, true, false, L, L, hd; batch=N)                # (L_k, L_q, H*B)
    P = pooled(T, L, L, N)
    usemask = mask isa AbstractArray
    m = usemask ? mask : S
    Mq = usemask ? size(mask, 2) : 1
    launch_reduce!(masked_softmax_kernel, L * N, P, S, m, Int32(L), Int32(L), Int32(H), Int32(Mq), usemask)
    out = pooled(T, hd, L, N)
    matmul!(out, v, P, false, false, hd, L, L; batch=N)            # (hd, L_q, H*B)
    y = pooled(T, d, L, B)
    launch!(merge_heads_kernel, (d, L, B), y, out, hd, H, L)
    Laya.release!(q, k, v, S, P, out)
    y
end

# ---------------------------------------------------------------------------- precompilation

# The first `predict` on a Metal model spends most of its time in Julia compilation (92.9% of
# an 8.7 s call; the forward itself is ~23 ms) and nothing on the CPU path warms it. Run a
# tiny forward here so the whole Metal path (model load, every kernel, MPSGraph products,
# `predict`) lives in the extension image. Metal's `__init__` is skipped while precompiling,
# so the MetalPerformanceShadersGraph framework is loaded by hand; Metal discards the command
# buffers its kernels are encoded into while precompiling, so nothing actually runs on the
# GPU. The extension's own pools hold session-local buffers and are emptied before the image
# is written (Metal's own caches are process-local and never serialized). Only Apple silicon
# can run the workload: on other platforms the framework and the ObjectiveC runtime are
# absent, so loading it would abort the extension's precompilation (Metal is in the test
# environment everywhere).
@setup_workload begin
    @compile_workload begin
        if ccall(:jl_generating_output, Cint, ()) != 0 && Sys.isapple() && Sys.ARCH === :aarch64
            Metal.load_framework("MetalPerformanceShadersGraph")
            Metal.initialized[] = true
            try
                mktempdir() do root
                    # hidden 1024 / heads 16 / head_dim 64 and two head layers, as the shipped
                    # checkpoints, so the fused attention, the `Val`-specialized LayerNorm and
                    # the head are the real specializations.
                    dir = Laya.write_tiny_checkpoint(joinpath(root, "metal"); hidden_size=1024,
                        num_attention_heads=16, intermediate_size=64, num_hidden_layers=2, head_layers=2)
                    agent = Laya.load(dir; backend=MetalBackend())
                    questions = Dict(
                        "topic" => Dict("type" => "choice", "instructions" => "Choose", "criteria" => ["a", "b", "c"]),
                        "team" => Dict("type" => "choice", "instructions" => "Route", "criteria" => Dict("x" => "first", "y" => "second")),
                        "level" => Dict("type" => "score", "instructions" => "Level", "criteria" => ["low", "high"]),
                        "yes" => Dict("type" => "noul", "instructions" => "Is this true?"),
                    )
                    Laya.predict(agent, "hello world", questions)
                    Laya.predict(agent, Dict("body" => "hello", "items" => [1, 2.5, nothing, true]), questions)
                end
            finally
                empty!(POOL)
                empty!(GRAVE)
                empty!(UPLOAD_FREE)
                empty!(UPLOAD_PENDING)
                empty!(ROPE_TABLES)
                POOL_BYTES[] = 0
                GRAVE_BYTES[] = 0
                UPLOAD_PENDING_BYTES[] = 0
                Metal.initialized[] = false
            end
        end
    end
end

end
