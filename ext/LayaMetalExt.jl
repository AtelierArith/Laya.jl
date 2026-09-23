module LayaMetalExt

using Laya: Laya, LayerNorm
using Metal: Metal, MetalBackend, MtlArray, @metal, thread_position_in_grid_1d
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

# One thread per column; statistics in Float32 as MLX's fast.layer_norm (also for Float16).
function layernorm_kernel(y, x, w, b, d, ncol, eps)
    j = tid()
    if j <= ncol
        o = (j - Int32(1)) * d
        s = 0.0f0
        for i in Int32(1):d
            @inbounds s += Float32(x[o+i])
        end
        μ = s / d
        v = 0.0f0
        for i in Int32(1):d
            @inbounds v += abs2(Float32(x[o+i]) - μ)
        end
        r = 1.0f0 / sqrt(v / d + eps)
        for i in Int32(1):d
            @inbounds t = (Float32(x[o+i]) - μ) * r * Float32(w[i])
            b === nothing || (@inbounds t += Float32(b[i]))
            @inbounds y[o+i] = t
        end
    end
    return
end

function (ln::LayerNorm)(x::MtlArray{T}) where {T}
    d = size(x, 1)
    ncol = length(x) ÷ d
    y = similar(x)
    launch!(layernorm_kernel, ncol, y, x, ln.weight, ln.bias, d, ncol, ln.eps; threads=64)
    y
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

# Masked softmax over keys, one thread per (query, head, batch) column of S `(Lk, Lq, H*B)`;
# `mask` is `(Lk, Mq, B)` with `Mq` either `Lq` or 1 (`true` keeps a key).
function masked_softmax_kernel(P, S, mask, Lk, Lq, H, Mq, ncol, usemask)
    col = tid()
    if col <= ncol
        n, qi = divmod1(col, Lq)
        b = (n - Int32(1)) ÷ H + Int32(1)
        o = (col - Int32(1)) * Lk
        mo = ((Mq == Int32(1) ? Int32(1) : qi) - Int32(1)) * Lk + (b - Int32(1)) * Lk * Mq
        keep(j) = !usemask || @inbounds mask[mo+j]
        m = -Inf32
        for j in Int32(1):Lk
            keep(j) && (@inbounds m = max(m, S[o+j]))
        end
        t = 0.0f0
        for j in Int32(1):Lk
            @inbounds t += keep(j) ? exp(S[o+j] - m) : 0.0f0
        end
        r = 1.0f0 / t
        for j in Int32(1):Lk
            @inbounds P[o+j] = keep(j) ? exp(S[o+j] - m) * r : 0.0f0
        end
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

function Laya.qkv_attention(qkv::MtlArray{T,3}, H::Integer, base, mask, scale::Real) where {T}
    d, L, B = size(qkv, 1) ÷ 3, size(qkv, 2), size(qkv, 3)
    hd, N = d ÷ H, H * B
    q, k, v = (MtlArray{T}(undef, hd, L, N) for _ in 1:3)
    c, s = base === nothing ? (q, q) : rope_tables(T, hd, L, base)
    launch!(split_rope_kernel, length(q), q, k, v, qkv, c, s, hd, H, L, length(q), base !== nothing)
    S = MtlArray{Float32}(undef, L, L, N)
    graph_matmul!(S, k, q, Float32(scale), false, 'T', 'N')             # (L_k, L_q, H*B)
    P = MtlArray{T}(undef, L, L, N)
    usemask = mask isa AbstractArray
    m = usemask ? mask : S
    Mq = usemask ? size(mask, 2) : 1
    launch!(masked_softmax_kernel, L * N, P, S, m, L, L, H, Mq, L * N, usemask)
    out = MtlArray{T}(undef, hd, L, N)
    graph_matmul!(out, v, P, true, false, 'N', 'N')                    # (hd, L_q, H*B)
    y = MtlArray{T}(undef, d, L, B)
    launch!(merge_heads_kernel, length(y), y, out, hd, H, L, length(y))
    Laya.release!(q, k, v, S, P, out)
    y
end

end
