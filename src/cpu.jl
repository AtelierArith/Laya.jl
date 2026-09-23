# CPU fast paths for plain `Array`s: fused, allocation-light loops threaded over columns
# (start Julia with `-t auto`). The generic broadcast versions in layers.jl stay the
# reference semantics and are what other array types (e.g. GPU arrays) use.

function (ln::LayerNorm)(x::Array{T}) where {T}
    d = size(x, 1)
    X = reshape(x, d, :)
    Y = similar(X)
    w, b, eps = ln.weight, ln.bias, T(ln.eps)
    Threads.@threads for j in axes(X, 2)
        col = view(X, :, j)
        μ = sum(col) / d
        σ² = sum(v -> abs2(v - μ), col) / d
        r = inv(sqrt(σ² + eps))
        if b === nothing
            @inbounds @simd for i in 1:d
                Y[i, j] = (X[i, j] - μ) * r * w[i]
            end
        else
            @inbounds @simd for i in 1:d
                Y[i, j] = (X[i, j] - μ) * r * w[i] + b[i]
            end
        end
    end
    reshape(Y, size(x))
end

"""`gelu(value) .* gate` for the GeGLU MLP, where `y = [value; gate]` along the first axis."""
function gelu_gate(y::AbstractArray)
    n = size(y, 1) ÷ 2
    gelu.(view(y, 1:n, ntuple(_ -> Colon(), ndims(y) - 1)...)) .* view(y, n+1:2n, ntuple(_ -> Colon(), ndims(y) - 1)...)
end

function gelu_gate(y::Array{T}) where {T}
    n = size(y, 1) ÷ 2
    Y = reshape(y, 2n, :)
    out = Matrix{T}(undef, n, size(Y, 2))
    Threads.@threads for j in axes(Y, 2)
        @inbounds @simd for i in 1:n
            out[i, j] = gelu(Y[i, j]) * Y[n+i, j]
        end
    end
    reshape(out, n, size(y)[2:end]...)
end

"""In-place masked softmax over the first axis of `S` (`true` in `mask` keeps a key)."""
function masked_softmax!(S::AbstractMatrix{T}, mask::AbstractMatrix{Bool}) where {T}
    broadcast_q = size(mask, 2) == 1
    @inbounds for i in axes(S, 2)
        mi = broadcast_q ? 1 : i
        m = T(-Inf)
        for j in axes(S, 1)
            mask[j, mi] ? (m = max(m, S[j, i])) : (S[j, i] = T(-Inf))
        end
        s = zero(T)
        for j in axes(S, 1)
            e = exp(S[j, i] - m)
            S[j, i] = e
            s += e
        end
        r = inv(s)
        for j in axes(S, 1)
            S[j, i] *= r
        end
    end
    S
end

# Heads read straight from the (head_dim, heads, L, B) layout through strided views, so no
# permutation copies are made; each thread owns one score buffer.
function attention(q::Array{T,4}, k::Array{T,4}, v::Array{T,4}, mask, scale::Real) where {T}
    hd, H, L, B = size(q)
    out = similar(q)
    m = mask isa AbstractArray ? mask : fill(true, L, 1, B)
    buffers = [Matrix{T}(undef, L, L) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for idx in 1:H*B
        h, b = mod1(idx, H), cld(idx, H)
        S = buffers[Threads.threadid()]
        mul!(S, transpose(view(k, :, h, :, b)), view(q, :, h, :, b), T(scale), zero(T))
        masked_softmax!(S, view(m, :, :, min(b, size(m, 3))))
        mul!(view(out, :, h, :, b), view(v, :, h, :, b), S)
    end
    out
end
