# ModernBERT encoder and Laya decision heads (inference only), following
# `laya_mlx/model.py`. Token ids and marker positions are 0-based, as in Python.

struct EncoderLayer{T}
    kind::Symbol
    attn_norm::Union{Nothing,LayerNorm}   # identity for the first layer
    Wqkv::Linear
    Wo::Linear
    num_heads::Int
    base::Float64
    mlp_norm::LayerNorm
    Wi::Linear
    Wo_mlp::Linear
end

function (layer::EncoderLayer{T})(x, mask) where {T}
    d, L, B = size(x)
    H = layer.num_heads
    hd = d ÷ H
    h = layer.attn_norm === nothing ? x : layer.attn_norm(x)
    qkv = reshape(layer.Wqkv(h), hd, H, 3, L, B)
    q = rope(qkv[:, :, 1, :, :], layer.base)
    k = rope(qkv[:, :, 2, :, :], layer.base)
    v = qkv[:, :, 3, :, :]
    a = attention(q, k, v, mask, T(hd)^T(-0.5))
    x = x .+ layer.Wo(reshape(a, d, L, B))
    x .+ layer.Wo_mlp(gelu_gate(layer.Wi(layer.mlp_norm(x))))
end

struct ModernBert{T}
    config::EncoderConfig
    tok_embeddings::Matrix{T}                # (hidden, vocab)
    embed_norm::LayerNorm
    layers::Vector{EncoderLayer{T}}
    final_norm::LayerNorm
end

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

embed(E::AbstractMatrix, ids::AbstractArray{<:Integer}) = reshape(E[:, vec(ids) .+ 1], size(E, 1), size(ids)...)

function (m::ModernBert)(input_ids, attention_mask; trace=nothing)
    x = m.embed_norm(embed(m.tok_embeddings, input_ids))
    trace === nothing || (trace["embeddings"] = x)
    masks = attention_masks(attention_mask, m.config.local_attention)
    trace === nothing || (trace["mask_full"] = masks.full; trace["mask_sliding"] = masks.sliding)
    for (i, layer) in enumerate(m.layers)
        x = layer(x, layer.kind === :full_attention ? masks.full : masks.sliding)
        trace === nothing || (trace["layer_$(i-1)"] = x)
    end
    m.final_norm(x)
end

struct HeadLayer
    num_heads::Int
    norm1::LayerNorm
    in_proj::Linear
    out_proj::Linear
    norm2::LayerNorm
    linear1::Linear
    linear2::Linear
end

function (layer::HeadLayer)(x::AbstractArray{T}, mask) where {T}
    d, L, B = size(x)
    H = layer.num_heads
    hd = d ÷ H
    qkv = reshape(layer.in_proj(layer.norm1(x)), hd, H, 3, L, B)
    a = attention(qkv[:, :, 1, :, :], qkv[:, :, 2, :, :], qkv[:, :, 3, :, :], mask, T(hd)^T(-0.5))
    x = x .+ layer.out_proj(reshape(a, d, L, B))
    # PyTorch TransformerEncoderLayer defaults to ReLU; the encoder and scorer use GELU.
    x .+ layer.linear2(relu.(layer.linear1(layer.norm2(x))))
end

struct DecisionModel{T}
    encoder::ModernBert{T}
    head::Vector{HeadLayer}
    type_emb::Matrix{T}                      # (hidden, 3)
    scorer_norm::LayerNorm
    scorer1::Linear
    scorer2::Linear
    act1::Linear
    act2::Linear
end

Base.eltype(::DecisionModel{T}) where {T} = T

"""
    (model::DecisionModel)(batch; trace=nothing) -> (logits, action)

`batch` holds `input_ids` `(L, B)`, `attention_mask` `(L, B)`, `marker_pos` `(K, B)`,
`marker_mask` `(K, B)` and `qtype` `(B,)`, as produced by `collate`. Returns Float32
`logits` `(K, B)` and `action` `(n_actions, B)`. Pass a `Dict` as `trace` to record every
intermediate activation under the same keys as `LayaMLXReference.trace`.
"""
function (m::DecisionModel{T})(batch::AbstractDict; trace=nothing) where {T}
    ids, mask = batch["input_ids"], Bool.(batch["attention_mask"])
    marker_pos, marker_mask = batch["marker_pos"], Bool.(batch["marker_mask"])
    qtype = batch["qtype"]
    B = size(ids, 2)

    h = m.encoder(ids, mask; trace)
    trace === nothing || (trace["encoder"] = h)
    h = h .+ reshape(m.type_emb[:, qtype .+ 1], :, 1, B)
    trace === nothing || (trace["typed"] = h)
    hmask = reshape(mask, size(mask, 1), 1, B)
    for (i, layer) in enumerate(m.head)
        h = layer(h, hmask)
        trace === nothing || (trace["head_$(i-1)"] = h)
    end

    K = size(marker_pos, 1)
    markers = similar(h, size(h, 1), K, B)
    for b in 1:B, j in 1:K
        markers[:, j, b] = @view h[:, max(marker_pos[j, b], 0)+1, b]
    end
    logits = Float32.(dropdims(m.scorer2(gelu.(m.scorer1(m.scorer_norm(markers)))); dims=1))
    logits = ifelse.(marker_mask, logits, -1.0f4)
    p = softmax(logits; dims=1)
    k = Float32.(max.(sum(marker_mask; dims=1), 2))
    entropy = -sum(p .* log.(max.(p, 1.0f-9)); dims=1) ./ log.(k)
    # The public runtime pads to at least two marker slots for one-option choices.
    top = mapslices(c -> partialsort(c, 1:2; rev=true), p; dims=1)   # (2, B): best, second
    features = vcat(top[1:1, :], top[1:1, :] .- top[2:2, :], entropy, k ./ 255.0f0)
    pooled = vcat(Float32.(h[:, 1, :]), features)
    action = Float32.(m.act2(gelu.(m.act1(T.(pooled)))))
    if trace !== nothing
        trace["logits"] = logits
        trace["action"] = action
    end
    logits, action
end
