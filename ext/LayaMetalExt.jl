module LayaMetalExt

using Laya: Laya, LayerNorm
using Metal: Metal, MetalBackend, MtlArray, MtlThreadGroupArray, @metal, threadgroup_barrier,
    thread_position_in_grid_1d, thread_position_in_threadgroup_1d, threadgroup_position_in_grid_1d,
    threadgroup_position_in_grid_2d
using Metal.MPSGraphs: graph_matmul!

# `Laya.load(repo; backend=MetalBackend())`. The model code in Laya is generic over array
# types; this extension moves the weights to the GPU and specializes the hot spots for
# `MtlArray`: fused kernels for LayerNorm, GeGLU and the attention glue, and batched MPSGraph
# matmuls for all heads at once. Matrix products of `Linear` go through Metal.jl's MPS path.

function Laya.load_backend_model(::MetalBackend, dir::AbstractString, ::Type{T}) where {T}
    T in (Float32, Float16) || throw(ArgumentError("MetalBackend supports Float32 and Float16, not $T"))
    Laya.adapt_arrays(MtlArray, first(Laya.load_model(dir; dtype=T)))
end

Laya.release_one!(x::MtlArray) = Metal.unsafe_free!(x)

"""Run `kernel(args...)` on `n` threads (one per work item); integer arguments become `Int32`."""
function launch!(kernel, n::Integer, args...; threads::Integer=256)
    args32 = map(a -> a isa Integer && !(a isa Bool) ? Int32(a) : a, args)
    @metal threads=threads groups=cld(n, threads) kernel(args32...)
end

# Index math stays in Int32: 64-bit division is slow on Apple GPUs.
@inline tid() = Int32(thread_position_in_grid_1d())
@inline divmod1(a::Int32, n::Int32) = (q = (a - Int32(1)) ÷ n; (q + Int32(1), a - q * n))   # (cld(a, n), mod1(a, n))

# ---------------------------------------------------------------------------- LayerNorm

const REDUCE_THREADS = 256   # threads per threadgroup for per-column reductions (power of 2)

"""Tree reduction of `v` over a threadgroup of `NT` threads; every thread gets the result."""
@inline function threadgroup_reduce(op, v::Float32, tid::Int32, ::Val{NT}=Val(REDUCE_THREADS)) where {NT}
    shared = MtlThreadGroupArray(Float32, NT)
    @inbounds shared[tid] = v
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    s = Int32(NT ÷ 2)
    while s > Int32(0)
        tid <= s && (@inbounds shared[tid] = op(shared[tid], shared[tid+s]))
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        s ÷= Int32(2)
    end
    r = @inbounds shared[1]
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    r
end

# One threadgroup per column; statistics in Float32 as MLX's fast.layer_norm (also for
# Float16). With `r !== nothing` the column is first replaced by the residual sum `x + r`,
# which is also written to `z`.
function layernorm_kernel(y, z, x, r, w, b, d, eps)
    col = Int32(threadgroup_position_in_grid_1d())
    tid = Int32(thread_position_in_threadgroup_1d())
    o = (col - Int32(1)) * d
    s = 0.0f0
    for i in tid:Int32(REDUCE_THREADS):d
        if r === nothing
            @inbounds s += Float32(x[o+i])
        else
            @inbounds v = x[o+i] + r[o+i]
            @inbounds z[o+i] = v
            s += Float32(v)
        end
    end
    μ = threadgroup_reduce(+, s, tid) / d
    src = r === nothing ? x : z
    q = 0.0f0
    for i in tid:Int32(REDUCE_THREADS):d
        @inbounds q += abs2(Float32(src[o+i]) - μ)
    end
    inv_σ = 1.0f0 / sqrt(threadgroup_reduce(+, q, tid) / d + eps)
    for i in tid:Int32(REDUCE_THREADS):d
        @inbounds t = (Float32(src[o+i]) - μ) * inv_σ * Float32(w[i])
        b === nothing || (@inbounds t += Float32(b[i]))
        @inbounds y[o+i] = t
    end
    return
end

function launch_layernorm!(y, z, x, r, ln::LayerNorm)
    d = size(x, 1)
    ncol = length(x) ÷ d
    @metal threads=REDUCE_THREADS groups=ncol layernorm_kernel(y, z, x, r, ln.weight, ln.bias, Int32(d), ln.eps)
end

function (ln::LayerNorm)(x::MtlArray{T}) where {T}
    y = similar(x)
    launch_layernorm!(y, nothing, x, nothing, ln)
    y
end

function Laya.residual_norm(x::MtlArray{T}, r::MtlArray{T}, ln::LayerNorm) where {T}
    z, y = similar(x), similar(x)
    launch_layernorm!(y, z, x, r, ln)
    Laya.release!(r)
    z, y
end

# ---------------------------------------------------------------------------- GeGLU

function gelu_gate_kernel(out, y, n, total)
    idx = tid()
    if idx <= total
        j, i = divmod1(idx, n)
        o = (j - Int32(1)) * Int32(2) * n
        @inbounds out[idx] = Laya.gelu(y[o+i]) * y[o+n+i]
    end
    return
end

function Laya.gelu_gate(y::MtlArray{T}) where {T}
    n = size(y, 1) ÷ 2
    out = MtlArray{T}(undef, n, size(y)[2:end]...)
    launch!(gelu_gate_kernel, length(out), out, y, n, length(out))
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

# qkv `(hd, H, 3, L, B)` -> q, k, v `(hd, L, H*B)` (heads batched for MPSGraph), with RoPE on
# q and k (same arithmetic as `Laya.rope`: `[x1 c - x2 s; x1 s + x2 c]`).
function split_rope_kernel(q, k, v, qkv, c, s, hd, H, L, total, userope)
    idx = tid()
    if idx <= total
        r, i = divmod1(idx, hd)           # q/k/v are (hd, L, H, B) in memory
        n, l = divmod1(r, L)              # n = (b - 1) * H + h
        b, h = divmod1(n, H)
        base = hd * ((h - Int32(1)) + H * Int32(3) * ((l - Int32(1)) + L * (b - Int32(1))))
        src(p, j) = j + base + hd * H * p
        @inbounds v[idx] = qkv[src(Int32(2), i)]
        if userope
            half = hd ÷ Int32(2)
            if i <= half
                @inbounds cc, ss = c[i+half*(l-Int32(1))], s[i+half*(l-Int32(1))]
                @inbounds q[idx] = qkv[src(Int32(0), i)] * cc - qkv[src(Int32(0), i + half)] * ss
                @inbounds k[idx] = qkv[src(Int32(1), i)] * cc - qkv[src(Int32(1), i + half)] * ss
            else
                j = i - half
                @inbounds cc, ss = c[j+half*(l-Int32(1))], s[j+half*(l-Int32(1))]
                @inbounds q[idx] = qkv[src(Int32(0), j)] * ss + qkv[src(Int32(0), i)] * cc
                @inbounds k[idx] = qkv[src(Int32(1), j)] * ss + qkv[src(Int32(1), i)] * cc
            end
        else
            @inbounds q[idx] = qkv[src(Int32(0), i)]
            @inbounds k[idx] = qkv[src(Int32(1), i)]
        end
    end
    return
end

# Masked softmax over keys, one threadgroup per (query, head, batch) column of S
# `(Lk, Lq, H*B)`; `mask` is `(Lk, Mq, B)` with `Mq` either `Lq` or 1 (`true` keeps a key).
function masked_softmax_kernel(P, S, mask, Lk, Lq, H, Mq, usemask)
    col = Int32(threadgroup_position_in_grid_1d())
    tid = Int32(thread_position_in_threadgroup_1d())
    n, qi = divmod1(col, Lq)
    b = (n - Int32(1)) ÷ H + Int32(1)
    o = (col - Int32(1)) * Lk
    mo = ((Mq == Int32(1) ? Int32(1) : qi) - Int32(1)) * Lk + (b - Int32(1)) * Lk * Mq
    keep(j) = !usemask || @inbounds mask[mo+j]
    m = -Inf32
    for j in tid:Int32(REDUCE_THREADS):Lk
        keep(j) && (@inbounds m = max(m, S[o+j]))
    end
    m = threadgroup_reduce(max, m, tid)
    t = 0.0f0
    for j in tid:Int32(REDUCE_THREADS):Lk
        @inbounds t += keep(j) ? exp(S[o+j] - m) : 0.0f0
    end
    r = 1.0f0 / threadgroup_reduce(+, t, tid)
    for j in tid:Int32(REDUCE_THREADS):Lk
        @inbounds P[o+j] = keep(j) ? exp(S[o+j] - m) * r : 0.0f0
    end
    return
end

# out `(hd, L, H*B)` -> `(hd, H, L, B)` = `(d, L, B)`, the input layout of the output projection.
function merge_heads_kernel(y, out, hd, H, L, total)
    idx = tid()
    if idx <= total
        r, i = divmod1(idx, hd)           # y is (hd, H, L, B)
        n, h = divmod1(r, H)
        b, l = divmod1(n, L)
        @inbounds y[idx] = out[i+hd*((l-Int32(1))+L*((h-Int32(1))+H*(b-Int32(1))))]
    end
    return
end

# Fused (flash) attention when the mask's structure is known. Off by default: this scalar
# version is 2-5x slower than the batched-matmul path below (see issue #1); it becomes the
# default once it uses simdgroup matrices.
const FLASH_ATTENTION = Ref(false)

function Laya.qkv_attention(qkv::MtlArray{T,3}, H::Integer, base, mask, scale::Real) where {T}
    d, L, B = size(qkv, 1) ÷ 3, size(qkv, 2), size(qkv, 3)
    hd, N = d ÷ H, H * B
    q, k, v = (MtlArray{T}(undef, hd, L, N) for _ in 1:3)
    c, s = base === nothing ? (q, q) : rope_tables(T, hd, L, base)
    launch!(split_rope_kernel, length(q), q, k, v, qkv, c, s, hd, H, L, length(q), base !== nothing)
    if FLASH_ATTENTION[] && mask isa Laya.AttentionMask
        y = flash_attention(q, k, v, mask, H, B, scale)
        Laya.release!(q, k, v)
        return y
    end
    mask = Laya.dense_mask(mask)
    S = MtlArray{Float32}(undef, L, L, N)
    graph_matmul!(S, k, q, Float32(scale), false, 'T', 'N')             # (L_k, L_q, H*B)
    P = MtlArray{T}(undef, L, L, N)
    usemask = mask isa AbstractArray
    m = usemask ? mask : S
    Mq = usemask ? size(mask, 2) : 1
    @metal threads=REDUCE_THREADS groups=L*N masked_softmax_kernel(P, S, m, Int32(L), Int32(L), Int32(H), Int32(Mq), usemask)
    out = MtlArray{T}(undef, hd, L, N)
    graph_matmul!(out, v, P, true, false, 'N', 'N')                    # (hd, L_q, H*B)
    y = MtlArray{T}(undef, d, L, B)
    launch!(merge_heads_kernel, length(y), y, out, hd, H, L, length(y))
    Laya.release!(q, k, v, S, P, out)
    y
end


# ---------------------------------------------------------------------------- flash attention

const FLASH_TQ = 64   # queries per threadgroup, one per thread
const FLASH_TK = 32   # keys per tile staged in threadgroup memory

# Fused attention for q, k, v `(hd, L, H*B)`: each thread owns one query and runs an online
# softmax over the key tiles that the threadgroup stages in threadgroup memory, so the score
# matrix never reaches device memory. Only tiles that intersect some query's key range are
# visited: for local attention (`window >= 0`) a valid query keeps the valid keys within
# `window`, a padded query all valid keys (the semantics of `Laya.attention_masks`). The
# output is written in the `(d, L, B)` layout of the output projection.
function flash_attention_kernel(y, q, k, v, valid, ::Val{HD}, L, H, window, scale) where {HD}
    tid = Int32(thread_position_in_threadgroup_1d())
    g = threadgroup_position_in_grid_2d()
    qb, n = Int32(g.x), Int32(g.y)
    b = (n - Int32(1)) ÷ H + Int32(1)
    h = n - (b - Int32(1)) * H
    i = (qb - Int32(1)) * Int32(FLASH_TQ) + tid
    active = i <= L
    Kt = MtlThreadGroupArray(eltype(k), (HD, FLASH_TK))
    Vt = MtlThreadGroupArray(eltype(v), (HD, FLASH_TK))
    vo = L * (b - Int32(1))                      # column offset of this batch row in `valid`
    qo = Int32(HD) * ((i - Int32(1)) + L * (n - Int32(1)))
    qv = ntuple(d -> active ? @inbounds(Float32(q[qo+d])) * scale : 0.0f0, Val(HD))
    local_ = window >= Int32(0) && active && @inbounds(valid[vo+i])
    lo = active ? (local_ ? max(Int32(1), i - window) : Int32(1)) : L + Int32(1)
    hi = active ? (local_ ? min(L, i + window) : L) : Int32(0)
    glo = Int32(threadgroup_reduce(min, Float32(lo), tid, Val(FLASH_TQ)))
    ghi = Int32(threadgroup_reduce(max, Float32(hi), tid, Val(FLASH_TQ)))
    m = -Inf32
    l = 0.0f0
    acc = ntuple(_ -> 0.0f0, Val(HD))
    kbase = Int32(HD) * L * (n - Int32(1))
    t0 = glo
    while t0 <= ghi
        e = tid
        while e <= Int32(HD * FLASH_TK)             # stage keys t0 .. t0 + TK - 1
            j, d = divmod1(e, Int32(HD))
            kj = t0 + j - Int32(1)
            if kj <= L
                @inbounds Kt[d, j] = k[kbase+Int32(HD)*(kj-Int32(1))+d]
                @inbounds Vt[d, j] = v[kbase+Int32(HD)*(kj-Int32(1))+d]
            end
            e += Int32(FLASH_TQ)
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        if active
            for j in Int32(1):Int32(FLASH_TK)
                kj = t0 + j - Int32(1)
                (lo <= kj <= hi && @inbounds(valid[vo+kj])) || continue
                s = 0.0f0
                for d in 1:HD
                    @inbounds s += qv[d] * Float32(Kt[d, j])
                end
                if s > m
                    c = exp(m - s)
                    acc = acc .* c
                    l *= c
                    m = s
                end
                p = exp(s - m)
                l += p
                prev = acc   # a fresh binding: capturing the reassigned `acc` would box it
                acc = ntuple(d -> @inbounds(prev[d] + p * Float32(Vt[d, j])), Val(HD))
            end
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        t0 += Int32(FLASH_TK)
    end
    if active
        r = 1.0f0 / l
        yo = Int32(HD) * ((h - Int32(1)) + H * ((i - Int32(1)) + L * (b - Int32(1))))
        for d in 1:HD
            @inbounds y[yo+d] = acc[d] * r
        end
    end
    return
end

function flash_attention(q::MtlArray{T,3}, k, v, mask::Laya.AttentionMask, H::Integer, B::Integer, scale::Real) where {T}
    hd, L, N = size(q)
    y = MtlArray{T}(undef, hd * H, L, B)
    window = mask.window === nothing ? Int32(-1) : Int32(mask.window)
    @metal threads=FLASH_TQ groups=(cld(L, FLASH_TQ), N) flash_attention_kernel(
        y, q, k, v, mask.valid, Val(hd), Int32(L), Int32(H), window, Float32(scale))
    y
end

end
