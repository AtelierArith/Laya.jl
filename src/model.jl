# ModernBERT encoder and Laya decision heads (inference only), following
# `laya_mlx/model.py`. Token ids and marker positions are 0-based, as in Python.

# The model structs carry their layers' concrete types as parameters (`T` is the element
# type), so that the forward pass is inferred end to end: a field of an abstract type would
# cost a dynamic dispatch at every use. All encoder layers share one type: the first layer's
# `attn_norm` is `nothing` (identity), typed as `Union{Nothing, N}` with `N` the others' type.
struct EncoderLayer{T,N<:LayerNorm,A<:Linear,B<:Linear,C<:Linear,D<:Linear}
    kind::Symbol
    attn_norm::Union{Nothing,N}   # identity for the first layer
    Wqkv::A
    Wo::B
    num_heads::Int
    base::Float64
    mlp_norm::N
    Wi::C
    Wo_mlp::D
end
EncoderLayer{T}(kind, attn_norm::Union{Nothing,N}, Wqkv::A, Wo::B, num_heads, base, mlp_norm::N, Wi::C,
    Wo_mlp::D) where {T,N<:LayerNorm,A<:Linear,B<:Linear,C<:Linear,D<:Linear} =
    EncoderLayer{T,N,A,B,C,D}(kind, attn_norm, Wqkv, Wo, num_heads, base, mlp_norm, Wi, Wo_mlp)

function (layer::EncoderLayer)(x, mask)
    h = layer.attn_norm === nothing ? x : layer.attn_norm(x)
    first(layer(x, h, mask, nothing))
end

# `h` is `attn_norm(x)` (or `x` itself for the first layer), computed by the previous layer;
# returns the output `y` and `next_norm(y)` (`nothing` when `next_norm === nothing`), so the
# residual sum at the end of a layer and the next normalization run as one fused step.
function (layer::EncoderLayer{T})(x, h, mask, next_norm) where {T}
    d, L, B = size(x)
    H = layer.num_heads
    hd = d ÷ H
    qkv = layer.Wqkv(h)
    h === x || release!(h)
    a = qkv_attention(qkv, H, layer.base, mask, T(hd)^T(-0.5))
    release!(qkv)
    x1, h1 = residual_norm(x, layer.Wo(a), layer.mlp_norm)
    release!(a)
    u = layer.Wi(h1)
    release!(h1)
    g = gelu_gate(u)
    release!(u)
    y, hn = residual_norm(x1, layer.Wo_mlp(g), next_norm)
    release!(g, x1)
    y, hn
end

struct ModernBert{T,E<:AbstractMatrix{T},N1<:LayerNorm,L<:EncoderLayer{T},N2<:LayerNorm}
    config::EncoderConfig
    tok_embeddings::E                        # (hidden, vocab)
    embed_norm::N1
    layers::Vector{L}
    final_norm::N2
end
ModernBert{T}(config, tok_embeddings::E, embed_norm::N1, layers::Vector{L}, final_norm::N2) where {T,E,N1,L,N2} =
    ModernBert{T,E,N1,L,N2}(config, tok_embeddings, embed_norm, layers, final_norm)

"""
    attention_masks(attention_mask, window) -> (full, sliding)

Bool key masks of size `(L_k, L_q, B)`. Local attention keeps `|i - j| <= window ÷ 2`
(inclusive). Padded queries may see valid keys so no softmax row is fully masked; they are
never used as keys or pooled outputs.
"""
function attention_masks(attention_mask::AbstractMatrix{Bool}, window::Integer)
    L, B = size(attention_mask)
    valid_k = reshape(attention_mask, L, 1, B)
    valid_q = reshape(attention_mask, 1, L, B)
    full = valid_k .& trues(1, L, 1)
    pos = 0:L-1
    near = abs.(pos .- transpose(pos)) .<= window ÷ 2
    sliding = (near .| .!valid_q) .& valid_k
    (full=full, sliding=sliding)
end

embed(E::AbstractMatrix, ids::AbstractArray{<:Integer}) =
    reshape(gather_columns(E, vec(ids) .+ 1), size(E, 1), size(ids)...)

function (m::ModernBert)(input_ids, attention_mask; trace=nothing)
    x = m.embed_norm(embed(m.tok_embeddings, input_ids))
    trace === nothing || (trace["embeddings"] = x)
    dense = attention_masks(attention_mask, m.config.local_attention)
    valid = on_device_of(x, attention_mask)
    masks = (full=AttentionMask(on_device_of(x, dense.full), valid, nothing),
             sliding=AttentionMask(on_device_of(x, dense.sliding), valid, m.config.local_attention ÷ 2))
    trace === nothing || (trace["mask_full"] = masks.full.dense; trace["mask_sliding"] = masks.sliding.dense)
    first_norm = m.layers[1].attn_norm
    h = first_norm === nothing ? x : first_norm(x)
    for (i, layer) in enumerate(m.layers)
        next_norm = i < length(m.layers) ? m.layers[i+1].attn_norm : m.final_norm
        y, h = layer(x, h, layer.kind === :full_attention ? masks.full : masks.sliding, next_norm)
        trace === nothing ? release!(x) : (trace["layer_$(i-1)"] = y)
        x = y
    end
    trace === nothing && release!(x, masks.full.dense, masks.sliding)
    h    # final_norm of the last layer's output
end

struct HeadLayer{N1<:LayerNorm,A<:Linear,B<:Linear,N2<:LayerNorm,C<:Linear,D<:Linear}
    num_heads::Int
    norm1::N1
    in_proj::A
    out_proj::B
    norm2::N2
    linear1::C
    linear2::D
end

function (layer::HeadLayer)(x::AbstractArray{T}, mask) where {T}
    d, L, B = size(x)
    H = layer.num_heads
    hd = d ÷ H
    n = layer.norm1(x)
    qkv = layer.in_proj(n)
    release!(n)
    a = qkv_attention(qkv, H, nothing, mask, T(hd)^T(-0.5))
    release!(qkv)
    x1, h = residual_norm(x, layer.out_proj(a), layer.norm2)
    release!(a)
    # PyTorch TransformerEncoderLayer defaults to ReLU; the encoder and scorer use GELU.
    u = layer.linear1(h)
    release!(h)
    r = elementwise(relu, u)
    release!(u)
    y = residual(x1, layer.linear2(r))
    release!(x1, r)
    y
end

"""
    DecisionModel{T}

The Laya network: a ModernBERT encoder, the decision-head Transformer layers, the option
scorer and the action head, with parameters of element type `T`. Load one with
[`load_model`](@ref) (or through [`load`](@ref)); call it on a [`collate`](@ref)d batch.
Its weights may live in any array type (see [`adapt_arrays`](@ref)).
"""
struct DecisionModel{T,M<:ModernBert{T},H<:HeadLayer,E<:AbstractMatrix{T},N<:LayerNorm,S1<:Linear,S2<:Linear,
                     A1<:Linear,A2<:Linear}
    encoder::M
    head::Vector{H}
    type_emb::E                              # (hidden, 3)
    scorer_norm::N
    scorer1::S1
    scorer2::S2
    act1::A1
    act2::A2
end
DecisionModel{T}(encoder::M, head::Vector{H}, type_emb::E, scorer_norm::N, scorer1::S1, scorer2::S2, act1::A1,
    act2::A2) where {T,M,H,E,N,S1,S2,A1,A2} =
    DecisionModel{T,M,H,E,N,S1,S2,A1,A2}(encoder, head, type_emb, scorer_norm, scorer1, scorer2, act1, act2)

Base.eltype(::DecisionModel{T}) where {T} = T

"""
    (model::DecisionModel)(batch; trace=nothing) -> (logits, action)

`batch` holds `input_ids` `(L, B)`, `attention_mask` `(L, B)`, `marker_pos` `(K, B)`,
`marker_mask` `(K, B)` and `qtype` `(B,)`, as produced by `collate`, as host arrays. Returns
Float32 host `Array`s `logits` `(K, B)` and `action` `(n_actions, B)`, whatever array type the
weights live in (see [`on_device_of`](@ref)); the pooled features are computed on the host. Pass a `Dict` as `trace` to record every
intermediate activation under the same keys as `LayaMLXReference.trace`.
"""
function (m::DecisionModel{T})(batch::AbstractDict; trace=nothing) where {T}
    # The batch is a `Dict{String, Array}`: fix the element types here, so the rest is inferred.
    ids = convert(Matrix{Int32}, batch["input_ids"])::Matrix{Int32}
    mask = convert(Matrix{Bool}, batch["attention_mask"])::Matrix{Bool}
    marker_pos = convert(Matrix{Int32}, batch["marker_pos"])::Matrix{Int32}
    marker_mask = convert(Matrix{Bool}, batch["marker_mask"])::Matrix{Bool}
    qtype = convert(Vector{Int32}, batch["qtype"])::Vector{Int32}
    B = size(ids, 2)

    # Intermediates are released as soon as they are dead (a no-op on the CPU), except in
    # trace mode, where they are recorded.
    done!(xs...) = trace === nothing ? release!(xs...) : nothing
    e = m.encoder(ids, mask; trace)
    trace === nothing || (trace["encoder"] = e)
    te = gather_columns(m.type_emb, qtype .+ 1)
    h = add_columns(e, te)
    done!(e); release!(te)
    trace === nothing || (trace["typed"] = h)
    hmask = AttentionMask(on_device_of(h, reshape(mask, size(mask, 1), 1, B)), on_device_of(h, mask), nothing)
    for (i, layer) in enumerate(m.head)
        hi = layer(h, hmask)
        done!(h)
        h = hi
        trace === nothing || (trace["head_$(i-1)"] = h)
    end
    release!(hmask)

    d, L, K = size(h, 1), size(h, 2), size(marker_pos, 1)
    columns = [(b - 1) * L + max(marker_pos[j, b], 0) + 1 for j in 1:K, b in 1:B]
    markers = reshape(gather_columns(reshape(h, d, :), vec(columns)), d, K, B)
    s0 = m.scorer_norm(markers)
    s1 = m.scorer1(s0)
    g1 = elementwise(gelu, s1)
    s2 = m.scorer2(g1)
    h1 = columns_at(h, 1)                     # gathered before the first download, so that
    logits = Float32.(dropdims(to_host(s2); dims=1))   # the second one finds the queue idle
    release!(markers, s0, s1, g1, s2)
    logits = ifelse.(marker_mask, logits, -1.0f4)
    p = softmax(logits; dims=1)
    k = Float32.(max.(sum(marker_mask; dims=1), 2))
    entropy = -sum(p .* log.(max.(p, 1.0f-9)); dims=1) ./ log.(k)
    # The public runtime pads to at least two marker slots for one-option choices.
    top = mapslices(c -> partialsort(c, 1:2; rev=true), p; dims=1)   # (2, B): best, second
    features = vcat(top[1:1, :], top[1:1, :] .- top[2:2, :], entropy, k ./ 255.0f0)
    pooled = vcat(Float32.(to_host(h1)), features)
    done!(h); release!(h1)
    pin = on_device_of(e, T.(pooled))
    a1 = m.act1(pin)
    g2 = elementwise(gelu, a1)
    a2 = m.act2(g2)
    action = Float32.(to_host(a2))
    release!(pin, a1, g2, a2)
    if trace !== nothing
        trace["logits"] = logits
        trace["action"] = action
    end
    logits, action
end

"""
    adapt_arrays(to, model) -> model

`model` with every weight array replaced by `to(array)` (e.g. `Metal.MtlArray`), keeping the
structure and configuration; device backends use it to move a CPU-loaded model.
"""
adapt_arrays(to, x::AbstractArray) = to(x)
adapt_arrays(to, ::Nothing) = nothing
adapt_arrays(to, l::Linear) = Linear(adapt_arrays(to, l.weight), adapt_arrays(to, l.bias))
adapt_arrays(to, ln::LayerNorm) = LayerNorm(adapt_arrays(to, ln.weight), adapt_arrays(to, ln.bias), ln.eps)
adapt_arrays(to, l::EncoderLayer{T}) where {T} = EncoderLayer{T}(l.kind, adapt_arrays(to, l.attn_norm),
    adapt_arrays(to, l.Wqkv), adapt_arrays(to, l.Wo), l.num_heads, l.base, adapt_arrays(to, l.mlp_norm),
    adapt_arrays(to, l.Wi), adapt_arrays(to, l.Wo_mlp))
adapt_arrays(to, m::ModernBert{T}) where {T} = ModernBert{T}(m.config, adapt_arrays(to, m.tok_embeddings),
    adapt_arrays(to, m.embed_norm), [adapt_arrays(to, l) for l in m.layers], adapt_arrays(to, m.final_norm))
adapt_arrays(to, l::HeadLayer) = HeadLayer(l.num_heads, adapt_arrays(to, l.norm1), adapt_arrays(to, l.in_proj),
    adapt_arrays(to, l.out_proj), adapt_arrays(to, l.norm2), adapt_arrays(to, l.linear1), adapt_arrays(to, l.linear2))
adapt_arrays(to, m::DecisionModel{T}) where {T} = DecisionModel{T}(adapt_arrays(to, m.encoder),
    [adapt_arrays(to, l) for l in m.head], adapt_arrays(to, m.type_emb), adapt_arrays(to, m.scorer_norm),
    adapt_arrays(to, m.scorer1), adapt_arrays(to, m.scorer2), adapt_arrays(to, m.act1), adapt_arrays(to, m.act2))
