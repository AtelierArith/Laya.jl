# Building blocks on plain Julia arrays. Activations have size `(features, L, B)`: the
# row-major `(B, L, features)` layout of the Python implementation with axes reversed.

# Host <-> device glue. The model code is written against `AbstractArray`; weights may live in
# any array type (e.g. `Metal.MtlArray`) while inputs, masks and outputs are host `Array`s.

"""
    on_device_of(ref, x) -> array

`x` (a host array) in the same kind of array as `ref`: `x` itself when `ref` is an `Array`.
"""
on_device_of(::Array, x::AbstractArray) = x
on_device_of(ref::AbstractArray, x::AbstractArray) = copyto!(similar(ref, eltype(x), size(x)), convert(Array, x))

"""`x` as a host `Array` (no copy when it already is one)."""
to_host(x::Array) = x
to_host(x::AbstractArray) = Array(x)

"""
    release!(arrays...)

Hint that intermediate `arrays` are no longer used. A no-op for host arrays; device backends
return the memory to their pool at once (Julia's GC does not see device memory pressure).
"""
release!(xs...) = foreach(release_one!, xs)
release_one!(::AbstractArray) = nothing

"""`x .+ y` that frees `y` (a temporary) afterwards."""
residual(x, y) = (z = x .+ y; release!(y); z)

"""
    residual_norm(x, y, ln) -> (z, ln(z))

The residual sum `z = x .+ y` and its normalization, freeing `y`. Device backends fuse both
into one kernel.
"""
residual_norm(x, y, ln) = (z = residual(x, y); (z, ln(z)))
residual_norm(x, y, ::Nothing) = (residual(x, y), nothing)

"""`E[:, idx]` for a host index vector `idx`, on `E`'s device."""
gather_columns(E::AbstractMatrix, idx::AbstractVector{<:Integer}) = E[:, on_device_of(E, Int32.(idx))]
gather_columns(E::Matrix, idx::AbstractVector{<:Integer}) = E[:, idx]

"""Linear layer; `weight` is stored `(in, out)`, i.e. PyTorch/MLX `(out, in)` reversed."""
struct Linear{M<:AbstractMatrix,V<:Union{Nothing,AbstractVector}}
    weight::M
    bias::V
end

function (l::Linear)(x::AbstractArray)
    X = reshape(x, size(x, 1), :)
    Y = transpose(l.weight) * X
    l.bias === nothing || (Y .+= l.bias)
    reshape(Y, size(Y, 1), size(x)[2:end]...)
end

struct LayerNorm{V<:AbstractVector,B<:Union{Nothing,AbstractVector}}
    weight::V
    bias::B
    eps::Float32
end

function (ln::LayerNorm)(x::AbstractArray{T}) where {T}
    μ = sum(x; dims=1) ./ size(x, 1)
    xc = x .- μ
    σ² = sum(abs2, xc; dims=1) ./ size(x, 1)
    y = xc .* inv.(sqrt.(σ² .+ T(ln.eps))) .* ln.weight
    ln.bias === nothing ? y : y .+ ln.bias
end

"""Column-wise softmax (over the first axis), as `mx.softmax(..., axis=-1)`."""
function softmax(x::AbstractArray; dims=1)
    e = exp.(x .- maximum(x; dims))
    e ./ sum(e; dims)
end

"""
    rope(x, base)

Non-traditional RoPE (`mx.fast.rope(..., traditional=False, scale=1, offset=0)`) on `x` of
size `(head_dim, heads, L, B)`: the first and second halves of each head are rotated as
pairs, with angle `position * base^(-2i / head_dim)`.
"""
function rope(x::AbstractArray{T,4}, base::Real) where {T}
    hd, _, L, _ = size(x)
    half = hd ÷ 2
    # MLX: inv_freq = exp2(-(i / half) * log2(base)), evaluated in Float32.
    lb = log2(Float32(base))
    inv_freq = [exp2(-(Float32(i) / Float32(half)) * lb) for i in 0:half-1]
    θ = inv_freq .* reshape(Float32.(0:L-1), 1, L)          # (half, L)
    c = on_device_of(x, reshape(T.(cos.(θ)), half, 1, L, 1))
    s = on_device_of(x, reshape(T.(sin.(θ)), half, 1, L, 1))
    x1 = @view x[1:half, :, :, :]
    x2 = @view x[half+1:hd, :, :, :]
    vcat(x1 .* c .- x2 .* s, x1 .* s .+ x2 .* c)
end

"""
    qkv_attention(qkv, heads, rope_base, mask, scale) -> (d, L, B)

Multi-head self-attention from the fused projection `qkv` `(3d, L, B)` (`[q; k; v]` along the
first axis), with RoPE on `q` and `k` unless `rope_base === nothing`. The unit that device
backends specialize (e.g. one fused kernel for the split, RoPE and head layout).
"""
function qkv_attention(qkv::AbstractArray{T,3}, H::Integer, base, mask, scale::Real) where {T}
    d, L, B = size(qkv, 1) ÷ 3, size(qkv, 2), size(qkv, 3)
    x = reshape(qkv, d ÷ H, H, 3, L, B)
    q, k, v = x[:, :, 1, :, :], x[:, :, 2, :, :], x[:, :, 3, :, :]
    if base !== nothing
        q, k = rope(q, base), rope(k, base)
    end
    reshape(attention(q, k, v, mask, scale), d, L, B)
end

"""
    attention(q, k, v, mask, scale)

Scaled dot-product attention for `q, k, v` of size `(head_dim, heads, L, B)`. `mask` is a
Bool array broadcastable to `(L_k, L_q, B)`; `true` keeps a key. Returns
`(head_dim, heads, L, B)`.
"""
function attention(q::AbstractArray{T,4}, k, v, mask, scale::Real) where {T}
    hd, H, L, B = size(q)
    qp, kp, vp = (permutedims(a, (1, 3, 2, 4)) for a in (q, k, v))  # (hd, L, H, B)
    out = similar(qp)
    m = mask isa AbstractArray ? mask : fill(true, 1, 1, B)
    for b in 1:B, h in 1:H
        Q = @view qp[:, :, h, b]
        K = @view kp[:, :, h, b]
        V = @view vp[:, :, h, b]
        S = (transpose(K) * Q) .* T(scale)                     # (L_k, L_q)
        S = ifelse.(view(m, :, :, min(b, size(m, 3))), S, T(-Inf))
        out[:, :, h, b] = V * softmax(S; dims=1)
    end
    permutedims(out, (1, 3, 2, 4))
end
