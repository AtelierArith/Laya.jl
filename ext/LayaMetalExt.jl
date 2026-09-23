module LayaMetalExt

using Laya: Laya, LayerNorm, Linear
using Metal: Metal, MetalBackend, MtlArray, MtlMatrix, MtlThreadGroupArray, @metal, threadgroup_barrier,
    simd_shuffle_xor, simd_ballot, simd_vote_any, thread_index_in_simdgroup, simdgroup_index_in_threadgroup,
    thread_position_in_threadgroup_1d, threadgroup_position_in_grid_1d, threadgroup_position_in_grid_3d,
    thread_position_in_grid_2d, thread_position_in_grid_3d,
    simdgroup_load, simdgroup_store, simdgroup_multiply_accumulate
using Metal: GPUArrays, MPS
# Metal.jl's cached matmul graphs (internal names, pinned by the `[compat]` bound on Metal)
using Metal.MPSGraphs: MatmulGraphKey, CachedMatmulGraph, _matmul_graph_cache, _matmul_graph_cache_lock,
    MPSGraphTensor, MPSGraphTensorData, default_exec_desc
using Metal.ObjectiveC.Foundation: @autoreleasepool, NSDictionary, nil

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
# column is first replaced by the residual sum `x + r`, which is also written to `z`.
function layernorm_kernel(y, z, x, r, w, b, d, eps, ncol)
    lane, col = lane_col()
    col <= ncol || return
    o = (col - Int32(1)) * d
    s = 0.0f0
    for i in lane:Int32(SIMD):d
        if r === nothing
            @inbounds s += Float32(x[o+i])
        else
            @inbounds v = x[o+i] + r[o+i]
            @inbounds z[o+i] = v
            s += Float32(v)
        end
    end
    μ = simd_reduce(+, s) / d
    src = r === nothing ? x : z
    q = 0.0f0
    for i in lane:Int32(SIMD):d
        @inbounds q += abs2(Float32(src[o+i]) - μ)
    end
    inv_σ = 1.0f0 / sqrt(simd_reduce(+, q) / d + eps)
    for i in lane:Int32(SIMD):d
        @inbounds t = (Float32(src[o+i]) - μ) * inv_σ * Float32(w[i])
        b === nothing || (@inbounds t += Float32(b[i]))
        @inbounds y[o+i] = t
    end
    return
end

function launch_layernorm!(y, z, x, r, ln::LayerNorm)
    d = size(x, 1)
    launch_reduce!(layernorm_kernel, length(x) ÷ d, y, z, x, r, ln.weight, ln.bias, Int32(d), ln.eps)
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

function Laya.qkv_attention(qkv::MtlArray{T,3}, H::Integer, base, mask, scale::Real) where {T}
    d, L, B = size(qkv, 1) ÷ 3, size(qkv, 2), size(qkv, 3)
    mask = Laya.dense_mask(mask)      # the kernels read the dense mask; masked tiles are skipped by it
    d ÷ H == FA_HD || return attention_unfused(qkv, H, base, mask, scale)
    y = pooled(T, d, L, B)
    usemask = mask isa AbstractArray
    m = usemask ? mask : y
    Mq = usemask ? size(mask, 2) : 1
    c, s = base === nothing ? (qkv, qkv) : rope_tables(T, FA_HD, L, base)
    @metal threads=FA_THREADS groups=(cld(L, FA_BQ), H, B) attention_kernel(y, qkv, c, s, m, Int32(H), Int32(L), Int32(Mq), usemask, base !== nothing, Float32(scale))
    y
end

# ---------------------------------------------------------------------------- fused attention

# One kernel for the whole attention of a head_dim-64 model: a threadgroup of 4 simdgroups
# takes 32 queries of one head and walks the keys in tiles of 32 with an online softmax.
# The Q tile is staged once and kept as simdgroup matrix fragments; each K tile is staged in
# threadgroup memory (RoPE applied on the way in), S = KᵀQ and O += V P run on the 8x8
# simdgroup matrix units, and the output goes out in the layout of the output projection.
# Nothing but the output touches device memory: no S, P or head-permuted copies. A tile
# whose mask is all false (outside the local window, padding) is skipped. Threadgroup memory
# is kept to 12 KB (K, then the O rescale, then V share one buffer): with more, fewer
# threadgroups fit on a GPU core and the barriers are no longer hidden (20 KB ran 1.7x
# slower).
const FA_HD = 64
const FA_BQ = 32
const FA_BK = 32
const FA_THREADS = 128

@inline zfrag() = Metal.simdgroup_matrix_init_filled(0.0f0)
@inline barrier() = threadgroup_barrier(Metal.MemoryFlagThreadGroup)
@inline qkv_offset(p, l, h, H, L, b) =
    Int32(FA_HD) * ((h - Int32(1)) + H * (p + Int32(3) * ((l - Int32(1)) + L * (b - Int32(1)))))

@inline function stage_rope!(tile, qkv, c, s, p, l0, L, h, H, b, userope, scale, t)
    half = Int32(FA_HD ÷ 2)
    for idx in t:Int32(FA_THREADS):Int32(half * FA_BQ)
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
@inline store_o!(acc, buf, qo) = (ntuple(hb -> simdgroup_store(acc[hb], buf, (1 + 8(hb - 1), qo), Val(true)), Val(8)); nothing)
@inline load_o(buf, qo) = ntuple(hb -> simdgroup_load(buf, (1 + 8(hb - 1), qo), Val(true)), Val(8))

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
    t <= Int32(4) && (@inbounds flags[t] = Int32(0))
    barrier()
    sg_any && (@inbounds flags[sg] = Int32(1))
    barrier()
    @inbounds (flags[1] | flags[2] | flags[3] | flags[4]) != Int32(0)
end

# Single pass, online softmax. Threadgroup memory: `buf` (FA_HD, 32) for the Q staging, then
# each K tile, the O rescale and each V tile in turn; `St` (FA_BK, FA_BQ) for S then P.
function attention_kernel(y, qkv, c, s, mask, H, L, Mq, usemask, userope, scale)
    buf = MtlThreadGroupArray(Float32, (FA_HD, FA_BK))
    St = MtlThreadGroupArray(Float32, (FA_BK, FA_BQ))
    flags = MtlThreadGroupArray(Int32, 4)
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

    stage_rope!(buf, qkv, c, s, Int32(0), q0, L, h, H, b, userope, scale, t)
    barrier()
    qf = load_q(buf, qo)
    barrier()

    m = -Inf32
    l = 0.0f0
    acc = (zfrag(), zfrag(), zfrag(), zfrag(), zfrag(), zfrag(), zfrag(), zfrag())
    for tile in Int32(0):ntiles-Int32(1)
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
        # rescale O by α (per query column) through buf, K is no longer needed
        store_o!(acc, buf, qo)
        barrier()
        for i in part*Int32(16)+Int32(1):part*Int32(16)+Int32(16)
            @inbounds buf[i+Int32(FA_HD)*(col-Int32(1))] *= α
        end
        barrier()
        acc = load_o(buf, qo)
        barrier()
        stage_v!(buf, qkv, k0, L, h, H, b, t)
        barrier()
        for kb in 0:3
            pf = simdgroup_load(St, (1 + 8kb, qo), Val(true))
            acc = mma_v(acc, buf, kb, pf)
        end
        barrier()
    end
    inv_l = l > 0.0f0 ? 1.0f0 / l : 0.0f0
    store_o!(acc, buf, qo)
    barrier()
    for i in part*Int32(16)+Int32(1):part*Int32(16)+Int32(16)
        lq = q0 + col
        if lq <= L
            @inbounds y[i+Int32(FA_HD)*((h-Int32(1))+H*((lq-Int32(1))+L*(b-Int32(1))))] = buf[i+Int32(FA_HD)*(col-Int32(1))] * inv_l
        end
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

end
