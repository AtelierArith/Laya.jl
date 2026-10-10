module LayaCUDAExt

using Laya: Laya, LayerNorm, Linear
using CUDA: CUDA, CUDABackend, CuArray, CuMatrix, @cuda, threadIdx, blockIdx, blockDim, shfl_xor_sync, i32
using CUDA: cuBLAS
using CUDA.CUDACore: CUDACore, GPUArrays, DeviceMemory, HostMemory, Managed
using LinearAlgebra: Transpose, transpose

# `Laya.load(repo; backend=CUDABackend())`. The model code in Laya is generic over array
# types; this extension moves the weights to the GPU and specializes the hot spots for
# `CuArray`: cuBLAS products (`cublasGemmEx`) for the linear layers, a fused attention
# kernel, warp-per-column LayerNorm, and elementwise kernels. Intermediates
# come from a buffer pool and host arrays go up through pinned staging memory, so a warm
# forward pass allocates no device memory and waits for the GPU only for its two downloads.

function Laya.load_backend_model(::CUDABackend, dir::AbstractString, ::Type{T}) where {T}
    T in (Float32, Float16) || throw(ArgumentError("CUDABackend supports Float32 and Float16, not $T"))
    CUDA.functional() || throw(ArgumentError("CUDABackend needs a functional CUDA GPU (see `CUDA.versioninfo()`)"))
    Laya.adapt_arrays(ToDevice(), first(Laya.load_model(dir; dtype=T)))
end

# Moves arrays to the GPU, linear weights transposed: `transpose(Wt)` with `Wt` `(out, in)`
# is still the `(in, out)` matrix the generic code expects, and cuBLAS's NN kernels are
# 10-15% faster than TN for these shapes with 93 or >4000 tokens (no slower otherwise).
struct ToDevice end
(::ToDevice)(x::AbstractArray) = CuArray(x)
Laya.adapt_arrays(to::ToDevice, l::Linear) =
    Linear(transpose(CuArray(permutedims(l.weight))), Laya.adapt_arrays(to, l.bias))

# ---------------------------------------------------------------------------- buffer pool

# Intermediates come from a pool of device buffers keyed by (stream, byte size). Work on one
# stream runs in submission order, so a buffer released by queued work can be handed out at
# once to later work on the same stream. Pool buffers return through their DataRef finalizer:
# `release!` (or the GC, for arrays that are dropped instead) gives them back. Other arrays
# are freed with `CUDA.unsafe_free!`, which is stream-ordered too. The pool's free buffers are
# returned to CUDA when they exceed a quarter of the device memory.
const PoolKey = Tuple{UInt,Int}
const POOL = Dict{PoolKey,Vector{Managed{DeviceMemory}}}()
const POOL_LOCK = Threads.SpinLock()
const POOL_BYTES = Ref(0)
const POOL_MISSES = Ref(0)      # allocations the pool could not serve (for tests)
const POOL_PAGE = 1 << 16

# The pool is also entered from finalizers: keep them off while holding the lock.
function locked(f)
    GC.enable_finalizers(false)
    lock(POOL_LOCK)
    try
        f()
    finally
        unlock(POOL_LOCK)
        GC.enable_finalizers(true)
    end
end

stream_id(s=CUDA.stream()) = UInt(s.handle)
pool_limit() = Int(CUDA.totalmem(CUDA.device())) ÷ 4

"""The finalizer of a pool buffer's reference: the last owner hands the buffer back."""
struct PoolReturn
    key::PoolKey
end
function (f::PoolReturn)(mem::Managed{DeviceMemory})
    locked() do
        push!(get!(Vector{Managed{DeviceMemory}}, POOL, f.key), mem)
        POOL_BYTES[] += f.key[2]
    end
    nothing
end

"""An uninitialized `CuArray{T}` of size `dims`, from the pool when possible."""
function pooled(::Type{T}, dims::Dims{N}) where {T,N}
    bytes = cld(max(prod(dims) * sizeof(T), 1), POOL_PAGE) * POOL_PAGE
    key = (stream_id(), bytes)
    mem = locked() do
        free = get(POOL, key, nothing)
        if free === nothing || isempty(free)
            nothing
        else
            POOL_BYTES[] -= bytes
            pop!(free)
        end
    end
    if mem === nothing
        POOL_MISSES[] += 1
        POOL_BYTES[] > pool_limit() && evict!()
        mem = CUDACore.pool_alloc(DeviceMemory, bytes)
    end
    CuArray{T,N}(GPUArrays.DataRef(PoolReturn(key), mem), dims; maxsize=bytes)
end
pooled(::Type{T}, dims::Integer...) where {T} = pooled(T, Int.(dims))

"""Return the pool's free buffers to CUDA (stream-ordered, so queued work is unaffected)."""
function evict!()
    mems = locked() do
        mems = reduce(vcat, values(POOL); init=Managed{DeviceMemory}[])
        empty!(POOL)
        POOL_BYTES[] = 0
        mems
    end
    foreach(CUDACore.pool_free, mems)
    nothing
end

# Dropping the last reference to a pool buffer returns it (see `PoolReturn`); other buffers
# go back to CUDA's stream-ordered allocator; a view only drops its reference.
function Laya.release_one!(x::CuArray)
    x.data.freed || CUDA.unsafe_free!(x)
    nothing
end

# ---------------------------------------------------------------------------- host <-> device

# Copies from pageable host memory wait for the whole stream, so host arrays (token ids,
# masks, indices, pooled features) are first copied into pinned staging memory, from which
# `cuMemcpyHtoDAsync` runs in stream order. Staging space is handed out bump-allocator style
# and recycled once the stream has been synchronized: by a download, or when it runs out.
mutable struct Staging
    mem::HostMemory
    offset::Int
end
const STAGING = Dict{UInt,Staging}()
const STAGING_LOCK = ReentrantLock()
const STAGING_MIN = 8 << 20

"""Pinned staging space for `bytes` bytes on the current stream: (host pointer, staging)."""
function staging!(bytes::Int)
    s = CUDA.stream()
    st = @lock STAGING_LOCK get!(() -> Staging(CUDACore.alloc(HostMemory, STAGING_MIN), 0), STAGING, stream_id(s))
    need = cld(bytes, 256) * 256
    if st.offset + need > sizeof(st.mem)
        CUDA.synchronize(s)                    # every staged copy has completed
        st.offset = 0
        if need > sizeof(st.mem)
            CUDACore.free(st.mem)
            st.mem = CUDACore.alloc(HostMemory, max(need, 2 * sizeof(st.mem)))
        end
    end
    ptr = pointer(st.mem) + st.offset
    st.offset += need
    ptr
end

"""Copy the host array `xa` into the device array `y` (same length), in stream order."""
function upload!(y::CuArray{T}, xa::Array{T}) where {T}
    n = sizeof(xa)
    n == 0 && return y
    ptr = staging!(n)
    GC.@preserve xa unsafe_copyto!(convert(Ptr{UInt8}, ptr), convert(Ptr{UInt8}, pointer(xa)), n)
    CUDACore.cuMemcpyHtoDAsync_v2(pointer(y), ptr, n, CUDA.stream())
    y
end

Laya.on_device_of(::CuArray, x::AbstractArray{T,N}) where {T,N} =
    (xa = convert(Array{T,N}, x); upload!(pooled(T, size(xa)), xa))

# A download waits for the stream, so the staging space can be reused.
function Laya.to_host(x::CuArray)
    CUDA.synchronize()
    a = Array(x)
    st = @lock STAGING_LOCK get(STAGING, stream_id(), nothing)
    st === nothing || (st.offset = 0)
    a
end

# ---------------------------------------------------------------------------- matrix products

# alpha = 1, beta = 0 as device scalars (cuBLAS runs in device pointer mode in CUDA.jl): a
# host scalar would be uploaded on every call.
const SCALARS = Dict{CUDA.CuContext,CuArray{Float32,1}}()
const SCALARS_LOCK = ReentrantLock()
scalars() = @lock SCALARS_LOCK get!(() -> CuArray(Float32[1, 0]), SCALARS, CUDA.context())

# Float16 inputs accumulate in Float32; Float32 follows `CUDA.math_mode()` (pedantic/default
# FP32 unless the user opts into TF32 with `CUDA.math_mode!(CUDA.FAST_MATH)`).
compute_type(::Type{Float16}, m, k, n) = cuBLAS.CUBLAS_COMPUTE_32F
compute_type(::Type{Float32}, m, k, n) = cuBLAS.gemmExComputeType(Float32, Float32, Float32, m, k, n)

"""
    gemm!(C, A, B, tA, tB, m, n, k; batch=1)

`C = op(A) * op(B)` for column-major `C` `(m, n)` and `op(A)` `(m, k)`, `batch` times over
consecutive matrices, whatever the arrays' own shapes (`cublasGemmEx` /
`cublasGemmStridedBatchedEx`). `C` may be Float32 for Float16 inputs.
"""
function gemm!(C::CuArray, A::CuArray{T}, B::CuArray{T}, tA::Char, tB::Char, m::Integer, n::Integer, k::Integer;
               batch::Integer=1) where {T}
    one_zero = scalars()
    α, β = pointer(one_zero, 1), pointer(one_zero, 2)
    lda, ldb = tA == 'N' ? m : k, tB == 'N' ? k : n
    ct = compute_type(T, m, k, n)
    if batch == 1
        cuBLAS.cublasGemmEx(cuBLAS.handle(), tA, tB, m, n, k, α, A, T, lda, B, T, ldb, β, C, eltype(C), m, ct,
            cuBLAS.CUBLAS_GEMM_DEFAULT)
    else
        cuBLAS.cublasGemmStridedBatchedEx(cuBLAS.handle(), tA, tB, m, n, k, α, A, T, lda, m * k, B, T, ldb, k * n,
            β, C, eltype(C), m, m * n, batch, ct, cuBLAS.CUBLAS_GEMM_DEFAULT)
    end
    C
end

function (l::Linear{<:CuMatrix{T}})(x::CuArray{T}) where {T}
    k, m = size(l.weight)                          # weight is (in, out)
    n = length(x) ÷ k
    y = pooled(T, (m, size(x)[2:end]...))
    gemm!(y, l.weight, x, 'T', 'N', m, n, k)
    l.bias === nothing || launch!(bias_kernel, m * n, y, l.bias, m)
    y
end

function (l::Linear{<:Transpose{T,<:CuMatrix{T}}})(x::CuArray{T}) where {T}
    k, m = size(l.weight)                          # weight is transpose of (out, in)
    n = length(x) ÷ k
    y = pooled(T, (m, size(x)[2:end]...))
    gemm!(y, parent(l.weight), x, 'N', 'N', m, n, k)
    l.bias === nothing || launch!(bias_kernel, m * n, y, l.bias, m)
    y
end

# ---------------------------------------------------------------------------- kernels

"""Run `kernel(args..., n)` with one thread per item `1:n`; integer arguments become `Int32`."""
function launch!(kernel, n::Integer, args...)
    n == 0 && return
    args32 = map(a -> a isa Integer && !(a isa Bool) ? Int32(a) : a, args)
    threads = 256
    @cuda threads=threads blocks=cld(n, threads) kernel(args32..., Int32(n))
    nothing
end

@inline item() = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
@inline divrem1(a::Int32, n::Int32) = (q = (a - 1i32) ÷ n; (q + 1i32, a - q * n))   # (cld(a, n), mod1(a, n))

function bias_kernel(y, b, m, n)
    i = item()
    i <= n && (@inbounds y[i] += b[(i-1i32)%m+1i32])
    return
end

function gelu_gate_kernel(out, y, h, n)
    i = item()
    if i <= n
        j, r = divrem1(i, h)
        o = (j - 1i32) * 2i32 * h
        @inbounds out[i] = Laya.gelu(y[o+r]) * y[o+h+r]
    end
    return
end

function Laya.gelu_gate(y::CuArray{T}) where {T}
    h = size(y, 1) ÷ 2
    out = pooled(T, (h, size(y)[2:end]...))
    launch!(gelu_gate_kernel, length(out), out, y, h)
    out
end

function relu_kernel(y, x, n)
    i = item()
    i <= n && (@inbounds y[i] = Laya.relu(x[i]))
    return
end
function gelu_kernel(y, x, n)
    i = item()
    i <= n && (@inbounds y[i] = Laya.gelu(x[i]))
    return
end

for (f, kernel) in ((:(Laya.relu), :relu_kernel), (:(Laya.gelu), :gelu_kernel))
    @eval function Laya.elementwise(::typeof($f), x::CuArray{T}) where {T}
        y = pooled(T, size(x))
        launch!($kernel, length(x), y, x)
        y
    end
end

function add_kernel(z, x, y, n)
    i = item()
    i <= n && (@inbounds z[i] = x[i] + y[i])
    return
end

function Laya.residual(x::CuArray{T}, y::CuArray{T}) where {T}
    z = pooled(T, size(x))
    launch!(add_kernel, length(x), z, x, y)
    Laya.release!(y)
    z
end

function add_columns_kernel(z, x, v, d, L, n)
    i = item()
    if i <= n
        col, r = divrem1(i, d)
        b = (col - 1i32) ÷ L
        @inbounds z[i] = x[i] + v[r+d*b]
    end
    return
end

function Laya.add_columns(x::CuArray{T,3}, v::CuMatrix{T}) where {T}
    d, L, _ = size(x)
    z = pooled(T, size(x))
    launch!(add_columns_kernel, length(x), z, x, v, d, L)
    z
end

# y `(d, m)` = columns of E: `idx[k]` (1-based) or, with `idx === nothing`, `j + L (k - 1)`.
function gather_kernel(y, E, idx, d, j, L, n)
    i = item()
    if i <= n
        k, r = divrem1(i, d)
        c = idx === nothing ? j + L * (k - 1i32) : @inbounds idx[k]
        @inbounds y[i] = E[r+d*(c-1i32)]
    end
    return
end

function Laya.gather_columns(E::CuMatrix{T}, idx::AbstractVector{<:Integer}) where {T}
    d = size(E, 1)
    di = Laya.on_device_of(E, Int32.(idx))
    y = pooled(T, (d, length(idx)))
    launch!(gather_kernel, length(y), y, E, di, d, 0, 0)
    Laya.release!(di)
    y
end

function Laya.columns_at(h::CuArray{T,3}, j::Integer) where {T}
    d, L, B = size(h)
    y = pooled(T, (d, B))
    launch!(gather_kernel, length(y), y, h, nothing, d, j, L)
    y
end

# ---------------------------------------------------------------------------- reductions

# LayerNorm and softmax reduce over one column per warp (shuffles, no shared memory),
# `REDUCE_COLS` columns per block.
const WARP = 32
const REDUCE_COLS = 8

@inline function warp_reduce(op, v::Float32)
    v = op(v, shfl_xor_sync(0xffffffff, v, 16i32))
    v = op(v, shfl_xor_sync(0xffffffff, v, 8i32))
    v = op(v, shfl_xor_sync(0xffffffff, v, 4i32))
    v = op(v, shfl_xor_sync(0xffffffff, v, 2i32))
    v = op(v, shfl_xor_sync(0xffffffff, v, 1i32))
    v
end

"""This thread's lane (1-based) and the column its warp reduces."""
@inline function lane_col()
    t = threadIdx().x - 1i32
    (t % Int32(WARP) + 1i32, (blockIdx().x - 1i32) * Int32(REDUCE_COLS) + t ÷ Int32(WARP) + 1i32)
end

function launch_reduce!(kernel, ncol::Integer, args...)
    ncol == 0 && return
    @cuda threads=WARP*REDUCE_COLS blocks=cld(ncol, REDUCE_COLS) kernel(args..., Int32(ncol))
    nothing
end

# Statistics in Float32 (also for Float16). With `r !== nothing` the column is first replaced
# by the residual sum `x + r`, which is also written to `z`. Each lane keeps its `NV` elements
# of the column in registers, so the column is read once.
function layernorm_kernel(y, z, x, r, w, b, d, eps, ::Val{NV}, ncol) where {NV}
    lane, col = lane_col()
    col <= ncol || return
    o = (col - 1i32) * d
    v = ntuple(Val(NV)) do k
        i = lane + Int32(WARP) * Int32(k - 1)
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
    μ = warp_reduce(+, s) / d
    q = 0.0f0
    for k in 1:NV
        lane + Int32(WARP) * Int32(k - 1) <= d && (q += abs2(v[k] - μ))
    end
    inv_σ = 1.0f0 / sqrt(warp_reduce(+, q) / d + eps)
    for k in 1:NV
        i = lane + Int32(WARP) * Int32(k - 1)
        i <= d || continue
        @inbounds t = (v[k] - μ) * inv_σ * Float32(w[i])
        b === nothing || (@inbounds t += Float32(b[i]))
        @inbounds y[o+i] = t
    end
    return
end

function launch_layernorm!(y, z, x, r, ln::LayerNorm)
    d = size(x, 1)
    launch_reduce!(layernorm_kernel, length(x) ÷ d, y, z, x, r, ln.weight, ln.bias, Int32(d), ln.eps, Val(cld(d, WARP)))
end

function (ln::LayerNorm)(x::CuArray{T}) where {T}
    y = pooled(T, size(x))
    launch_layernorm!(y, nothing, x, nothing, ln)
    y
end

function Laya.residual_norm(x::CuArray{T}, r::CuArray{T}, ln::LayerNorm) where {T}
    z, y = pooled(T, size(x)), pooled(T, size(x))
    launch_layernorm!(y, z, x, r, ln)
    Laya.release!(r)
    z, y
end

# Masked softmax over keys, one column of S `(Lk, Lq, H*B)` per warp; `mask` is `(Lk, Mq, B)`
# with `Mq` either `Lq` or 1 (`true` keeps a key). `P` may alias `S`.
function masked_softmax_kernel(P, S, mask, Lk, Lq, H, Mq, usemask, ncol)
    lane, col = lane_col()
    col <= ncol || return
    n, qi = divrem1(col, Lq)
    b = (n - 1i32) ÷ H + 1i32
    o = (col - 1i32) * Lk
    mo = ((Mq == 1i32 ? 1i32 : qi) - 1i32) * Lk + (b - 1i32) * Lk * Mq
    keep(j) = !usemask || @inbounds mask[mo+j]
    m = -Inf32
    for j in lane:Int32(WARP):Lk
        keep(j) && (@inbounds m = max(m, Float32(S[o+j])))
    end
    m = warp_reduce(max, m)
    t = 0.0f0
    for j in lane:Int32(WARP):Lk
        @inbounds t += keep(j) ? exp(Float32(S[o+j]) - m) : 0.0f0
    end
    r = 1.0f0 / warp_reduce(+, t)
    for j in lane:Int32(WARP):Lk
        @inbounds P[o+j] = keep(j) ? exp(Float32(S[o+j]) - m) * r : 0.0f0
    end
    return
end

# ---------------------------------------------------------------------------- attention

# cos/sin tables `(half, L)` per (dtype, head_dim, L, base, stream), computed on the host
# exactly as `Laya.rope` does and kept on the device.
const ROPE_TABLES = Dict{Tuple{DataType,Int,Int,Float64,UInt},Tuple{CuArray,CuArray}}()
const ROPE_LOCK = ReentrantLock()

function rope_tables(::Type{T}, hd::Int, L::Int, base::Real) where {T}
    @lock ROPE_LOCK get!(ROPE_TABLES, (T, hd, L, Float64(base), stream_id())) do
        half = hd ÷ 2
        lb = log2(Float32(base))
        inv_freq = [exp2(-(Float32(i) / Float32(half)) * lb) for i in 0:half-1]
        θ = inv_freq .* reshape(Float32.(0:L-1), 1, L)
        c, s = T.(cos.(θ)), T.(sin.(θ))
        (upload!(CuArray{T}(undef, size(c)), c), upload!(CuArray{T}(undef, size(s)), s))
    end::Tuple{CuArray{T,2},CuArray{T,2}}
end

# qkv `(hd, H, 3, L, B)` -> q, k, v `(hd, L, H*B)` (heads batched for cuBLAS), with RoPE on q
# and k (same arithmetic as `Laya.rope`: `[x1 c - x2 s; x1 s + x2 c]`) and the attention
# `scale` folded into q.
function split_rope_kernel(q, k, v, qkv, c, s, hd, H, L, userope, scale, total)
    idx = item()
    idx <= total || return
    col, i = divrem1(idx, hd)                 # col = l + L (n - 1)
    n, l = divrem1(col, L)                    # n = h + H (b - 1)
    b, h = divrem1(n, H)
    base = hd * ((h - 1i32) + H * 3i32 * ((l - 1i32) + L * (b - 1i32)))
    src(p, j) = j + base + hd * H * p
    @inbounds v[idx] = qkv[src(2i32, i)]
    if userope
        half = hd ÷ 2i32
        if i <= half
            @inbounds cc, ss = c[i+half*(l-1i32)], s[i+half*(l-1i32)]
            @inbounds q[idx] = (qkv[src(0i32, i)] * cc - qkv[src(0i32, i + half)] * ss) * scale
            @inbounds k[idx] = qkv[src(1i32, i)] * cc - qkv[src(1i32, i + half)] * ss
        else
            j = i - half
            @inbounds cc, ss = c[j+half*(l-1i32)], s[j+half*(l-1i32)]
            @inbounds q[idx] = (qkv[src(0i32, j)] * ss + qkv[src(0i32, i)] * cc) * scale
            @inbounds k[idx] = qkv[src(1i32, j)] * ss + qkv[src(1i32, i)] * cc
        end
    else
        @inbounds q[idx] = qkv[src(0i32, i)] * scale
        @inbounds k[idx] = qkv[src(1i32, i)]
    end
    return
end

# out `(hd, L, H*B)` -> `(hd, H, L, B)` = `(d, L, B)`, the input layout of the output projection.
function merge_heads_kernel(y, out, hd, H, L, total)
    idx = item()
    idx <= total || return
    col, i = divrem1(idx, hd * H)             # col = l + L (b - 1)
    h, ii = divrem1(i, hd)
    b, l = divrem1(col, L)
    @inbounds y[idx] = out[ii+hd*((l-1i32)+L*((h-1i32)+H*(b-1i32)))]
    return
end

function Laya.qkv_attention(qkv::CuArray{T,3}, H::Integer, base, am, scale::Real) where {T}
    # the fused kernel reads the mask's structure (`valid`, `window`), not the dense mask
    (am isa Laya.AttentionMask && size(qkv, 1) ÷ 3H == FA_HD && FUSED[]) ||
        return attention_unfused(qkv, H, base, Laya.dense_mask(am), scale)
    d, L, B = size(qkv, 1) ÷ 3, size(qkv, 2), size(qkv, 3)
    y = pooled(T, d, L, B)
    window = am.window === nothing ? -1 : am.window
    c, s = base === nothing ? (qkv, qkv) : rope_tables(T, FA_HD, L, base)
    @cuda threads=FA_NT blocks=(cld(L, FA_BQ), H, B) attention_kernel(y, qkv, c, s, am.valid, Int32(window), Int32(H),
        Int32(L), base !== nothing, Float32(scale))
    y
end

const FUSED = Ref(true)

# ---------------------------------------------------------------------------- fused attention

# One kernel for the whole attention of a head_dim-64 model (FP32 arithmetic, also for
# Float16 inputs): a block of 128 threads takes 64 queries of one head and walks the keys in
# tiles of 32 with an online softmax. Q (RoPE and scale applied) stays in shared memory; each
# K tile is staged with RoPE on the way in, S = KᵀQ and O += V P run as register-tiled outer
# products (4 queries x 4 keys and 4 queries x 8 head-dim rows per thread, 128-bit shared
# loads), and the output is written in the layout of the output projection. Nothing but the
# output touches device memory: no S, P or head-permuted copies. Tiles whose keys are all
# masked for the block's queries (padding, and outside the local window) are skipped.
# Shared memory is 32 KB (three blocks per SM): Q `(64, 64)`, the K tile `(32, 64)` (then
# P `(64, 32)`) and the V tile `(64, 32)`, unpadded; Q, K and P are stored with their
# float4 groups XOR-swizzled, so the 128-bit reads are conflict-free and staging is not
# serialized on one bank.
const FA_HD = 64
const FA_BQ = 64
const FA_BK = 32
const FA_NT = 128

# 0-based positions in the shared tiles: Q (query q, dim d), K (key k, dim d), P (query q, key k)
@inline qpos(q, d) = 4i32 * ((q >> 2i32) ⊻ (d & 15i32)) + (q & 3i32) + Int32(FA_BQ) * d
@inline kpos(k, d) = 4i32 * ((k >> 2i32) ⊻ (d & 7i32)) + (k & 3i32) + Int32(FA_BK) * d
@inline ppos(q, k) = 4i32 * ((q >> 2i32) ⊻ (k & 15i32)) + (q & 3i32) + Int32(FA_BQ) * k

const Vec4 = NTuple{4,VecElement{Float32}}

# 4 consecutive floats at 0-based offset `o` (a multiple of 4) of a shared array
@inline function ld4(A, o::Int32)
    p = reinterpret(Core.LLVMPtr{Vec4,CUDA.AS.Shared}, pointer(A))
    v = unsafe_load(p, o ÷ 4i32 + 1i32, Val(16))
    (v[1].value, v[2].value, v[3].value, v[4].value)
end
@inline function st4!(A, o::Int32, x::NTuple{4,Float32})
    p = reinterpret(Core.LLVMPtr{Vec4,CUDA.AS.Shared}, pointer(A))
    unsafe_store!(p, map(VecElement, x), o ÷ 4i32 + 1i32, Val(16))
end

@inline fa_offset(p, l, h, H, L, b) = Int32(FA_HD) * ((h - 1i32) + H * (p + 3i32 * ((l - 1i32) + L * (b - 1i32))))

# `n` rows (tokens `l0 + 1 : l0 + n`) of q or k (p = 0, 1) into `tile` at `pos(row, dim)`
# (0-based), RoPE and `scale` applied (same arithmetic as `Laya.rope`); zeros past `L`.
@inline function stage_rope!(tile, pos, qkv, c, s, p, l0, n, L, h, H, b, userope, scale, t)
    half = Int32(FA_HD ÷ 2)
    for idx in t:Int32(FA_NT):n*half
        j = (idx - 1i32) % half + 1i32
        row = (idx - 1i32) ÷ half + 1i32
        l = l0 + row
        x1 = x2 = 0.0f0
        if l <= L
            src = fa_offset(p, l, h, H, L, b)
            @inbounds x1 = Float32(qkv[src+j])
            @inbounds x2 = Float32(qkv[src+j+half])
        end
        if userope && l <= L
            @inbounds cc = Float32(c[j+half*(l-1i32)])
            @inbounds ss = Float32(s[j+half*(l-1i32)])
            @inbounds tile[pos(row-1i32, j-1i32)+1i32] = (x1 * cc - x2 * ss) * scale
            @inbounds tile[pos(row-1i32, j+half-1i32)+1i32] = (x1 * ss + x2 * cc) * scale
        else
            @inbounds tile[pos(row-1i32, j-1i32)+1i32] = x1 * scale
            @inbounds tile[pos(row-1i32, j+half-1i32)+1i32] = x2 * scale
        end
    end
end

@inline function stage_v!(tile, qkv, l0, L, h, H, b, t)
    for idx in t:Int32(FA_NT):Int32(FA_HD * FA_BK)
        i = (idx - 1i32) % Int32(FA_HD) + 1i32
        l = l0 + (idx - 1i32) ÷ Int32(FA_HD) + 1i32
        @inbounds tile[idx] = l <= L ? Float32(qkv[fa_offset(2i32, l, h, H, L, b)+i]) : 0.0f0
    end
end

@inline fa_max(x) = (x = max(x, shfl_xor_sync(0xffffffff, x, 1i32)); x = max(x, shfl_xor_sync(0xffffffff, x, 2i32));
                     max(x, shfl_xor_sync(0xffffffff, x, 4i32)))
@inline fa_sum(x) = (x += shfl_xor_sync(0xffffffff, x, 1i32); x += shfl_xor_sync(0xffffffff, x, 2i32);
                     x + shfl_xor_sync(0xffffffff, x, 4i32))

# `valid` `(L, B)` and `window` (< 0: global attention) describe the key mask as in
# `Laya.AttentionMask`: a valid query keeps the valid keys within `window`, a padded query
# keeps every valid key.
function attention_kernel(y, qkv, c, s, valid, window, H, L, userope, scale)
    Qs = CUDA.CuStaticSharedArray(Float32, FA_BQ * FA_HD)
    Ks = CUDA.CuStaticSharedArray(Float32, FA_BK * FA_HD)   # K tile, then P
    Vs = CUDA.CuStaticSharedArray(Float32, FA_HD * FA_BK)
    q0 = (blockIdx().x - 1i32) * Int32(FA_BQ)
    h, b = blockIdx().y, blockIdx().z
    t = threadIdx().x
    lane = (t - 1i32) % 32i32
    kg = lane % 8i32                               # keys 4kg+1:4kg+4 of S, rows 8kg+1:8kg+8 of O
    qg = lane ÷ 8i32 + 4i32 * ((t - 1i32) ÷ 32i32) # queries 4qg+1:4qg+4 (the 8 threads of a qg share a warp)
    vo = L * (b - 1i32)

    stage_rope!(Qs, qpos, qkv, c, s, 0i32, q0, Int32(FA_BQ), L, h, H, b, userope, scale, t)

    # this thread's queries: index, and whether the window applies (valid query)
    qis = ntuple(i -> (@inline; q0 + 4i32 * qg + Int32(i)), Val(4))
    win = ntuple(i -> (@inline; window >= 0i32 && qis[i] <= L && (@inbounds valid[vo+qis[i]])), Val(4))
    # key tiles to visit: under local attention, those within the window of the block's
    # queries, unless one of them is padding (padded queries keep every valid key)
    ntiles = (L + Int32(FA_BK) - 1i32) ÷ Int32(FA_BK)
    lastq = min(q0 + Int32(FA_BQ), L)
    padded = CUDA.sync_threads_or(any(ntuple(i -> (@inline; qis[i] <= L && !win[i]), Val(4))))
    t0, t1 = if window < 0i32 || padded
        0i32, ntiles - 1i32
    else
        max(q0 - window, 0i32) ÷ Int32(FA_BK), min((lastq - 1i32 + window) ÷ Int32(FA_BK), ntiles - 1i32)
    end
    m = (-Inf32, -Inf32, -Inf32, -Inf32)
    l = (0.0f0, 0.0f0, 0.0f0, 0.0f0)
    acc = ntuple(_ -> (@inline; 0.0f0), Val(32))              # O[8kg + r, 4qg + i] at r + 8 (i - 1)
    for tile in t0:t1
        k0 = tile * Int32(FA_BK)
        kks = ntuple(j -> (@inline; k0 + 4i32 * kg + Int32(j)), Val(4))
        kv = ntuple(j -> (@inline; kks[j] <= L && (@inbounds valid[vo+kks[j]])), Val(4))
        bits = 0i32                                # (key j, query i) kept at bit j + 4 (i - 1) - 1
        for i in 1:4, j in 1:4
            keep = qis[i] <= L && kv[j] && (!win[i] || abs(qis[i] - kks[j]) <= window)
            bits |= Int32(keep) << Int32(j + 4i - 5)
        end
        CUDA.sync_threads_or(bits != 0i32) || continue   # also orders the previous tile's reads
        stage_rope!(Ks, kpos, qkv, c, s, 1i32, k0, Int32(FA_BK), L, h, H, b, userope, 1.0f0, t)
        stage_v!(Vs, qkv, k0, L, h, H, b, t)
        CUDA.sync_threads()
        S = ntuple(_ -> (@inline; 0.0f0), Val(16))            # (key j, query i) at j + 4 (i - 1)
        for d in 0i32:Int32(FA_HD)-1i32
            qv = ld4(Qs, qpos(4i32 * qg, d))
            kv4 = ld4(Ks, kpos(4i32 * kg, d))
            S = let S = S
                ntuple(e -> (@inline; fma(qv[(e-1)÷4+1], kv4[(e-1)%4+1], S[e])), Val(16))
            end
        end
        # online softmax over this tile; the 8 threads of a query reduce with shuffles (`let`:
        # closures must not capture the loop's reassigned variables, which would box them)
        Sm = let S = S, bits = bits
            ntuple(e -> (@inline; (bits >> Int32(e - 1)) & 1i32 == 1i32 ? S[e] : -Inf32), Val(16))
        end
        mn = let m = m
            ntuple(i -> (@inline; max(m[i], fa_max(max(Sm[4i-3], Sm[4i-2], Sm[4i-1], Sm[4i])))), Val(4))
        end
        α = let m = m
            ntuple(i -> (@inline; mn[i] == m[i] ? 1.0f0 : exp(m[i] - mn[i])), Val(4))
        end
        E = ntuple(e -> (@inline; exp(Sm[e] - mn[(e-1)÷4+1])), Val(16))
        P = ntuple(e -> (@inline; Sm[e] == -Inf32 ? 0.0f0 : E[e]), Val(16))   # masked keys (also when the row max is -Inf)
        l = let l = l
            ntuple(i -> (@inline; l[i] * α[i] + fa_sum(P[4i-3] + P[4i-2] + P[4i-1] + P[4i])), Val(4))
        end
        m = mn
        acc = let acc = acc
            ntuple(e -> (@inline; acc[e] * α[(e-1)÷8+1]), Val(32))
        end
        CUDA.sync_threads()                        # every thread is done with the K tile
        for j in 1:4
            st4!(Ks, ppos(4i32 * qg, 4i32 * kg + Int32(j - 1)), (P[j], P[j+4], P[j+8], P[j+12]))
        end
        CUDA.sync_threads()
        for k in 0i32:Int32(FA_BK)-1i32
            pv = ld4(Ks, ppos(4i32 * qg, k))
            v1 = ld4(Vs, 8i32 * kg + Int32(FA_HD) * k)
            v2 = ld4(Vs, 8i32 * kg + 4i32 + Int32(FA_HD) * k)
            acc = let acc = acc
                ntuple(Val(32)) do e
                    @inline
                    r, i = (e - 1) % 8 + 1, (e - 1) ÷ 8 + 1
                    fma(r <= 4 ? v1[r] : v2[r-4], pv[i], acc[e])
                end
            end
        end
    end
    let l = l, acc = acc
        ntuple(Val(4)) do i                        # unrolled: no dynamic indexing into `acc`
        @inline
        if qis[i] <= L
            inv_l = l[i] > 0.0f0 ? 1.0f0 / l[i] : 0.0f0
            o = 8i32 * kg + Int32(FA_HD) * ((h - 1i32) + H * ((qis[i] - 1i32) + L * (b - 1i32)))
            ntuple(Val(8)) do r
                @inline
                @inbounds y[o+Int32(r)] = acc[r+8(i-1)] * inv_l
            end
        end
        nothing
        end
    end
    return
end

# ---------------------------------------------------------------------------- unfused attention

# For other head sizes: split + RoPE, strided-batched cuBLAS products, masked softmax, merge.
function attention_unfused(qkv::CuArray{T,3}, H::Integer, base, mask, scale::Real) where {T}
    d, L, B = size(qkv, 1) ÷ 3, size(qkv, 2), size(qkv, 3)
    hd, N = d ÷ H, H * B
    q, k, v = pooled(T, hd, L, N), pooled(T, hd, L, N), pooled(T, hd, L, N)
    c, s = base === nothing ? (q, q) : rope_tables(T, hd, L, base)
    launch!(split_rope_kernel, hd * L * N, q, k, v, qkv, c, s, hd, H, L, base !== nothing, T(scale))
    S = pooled(Float32, L, L, N)
    gemm!(S, k, q, 'T', 'N', L, L, hd; batch=N)                    # (L_k, L_q, H*B)
    P = T === Float32 ? S : pooled(T, L, L, N)
    usemask = mask isa AbstractArray
    m = usemask ? mask : S
    Mq = usemask ? size(mask, 2) : 1
    launch_reduce!(masked_softmax_kernel, L * N, P, S, m, Int32(L), Int32(L), Int32(H), Int32(Mq), usemask)
    out = pooled(T, hd, L, N)
    gemm!(out, v, P, 'N', 'N', hd, L, L; batch=N)                  # (hd, L_q, H*B)
    y = pooled(T, d, L, B)
    launch!(merge_heads_kernel, length(y), y, out, hd, H, L)
    Laya.release!(q, k, v, S, out)
    P === S || Laya.release!(P)
    y
end

end
