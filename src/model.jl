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
    qkv = layer.Wqkv(h)
    h === x || release!(h)
    a = qkv_attention(qkv, H, layer.base, mask, T(hd)^T(-0.5))
    release!(qkv)
    x1 = residual(x, layer.Wo(a))
    release!(a)
    u = layer.Wi(layer.mlp_norm(x1))
    g = gelu_gate(u)
    release!(u)
    y = residual(x1, layer.Wo_mlp(g))
    release!(g, x1)
    y
end

struct ModernBert{T}
    config::EncoderConfig
    tok_embeddings::AbstractMatrix{T}        # (hidden, vocab)
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

embed(E::AbstractMatrix, ids::AbstractArray{<:Integer}) =
    reshape(gather_columns(E, vec(ids) .+ 1), size(E, 1), size(ids)...)

function (m::ModernBert)(input_ids, attention_mask; trace=nothing)
    x = m.embed_norm(embed(m.tok_embeddings, input_ids))
    trace === nothing || (trace["embeddings"] = x)
    masks = map(mk -> on_device_of(x, mk), attention_masks(attention_mask, m.config.local_attention))
    trace === nothing || (trace["mask_full"] = masks.full; trace["mask_sliding"] = masks.sliding)
    for (i, layer) in enumerate(m.layers)
        y = layer(x, layer.kind === :full_attention ? masks.full : masks.sliding)
        trace === nothing ? release!(x) : (trace["layer_$(i-1)"] = y)
        x = y
    end
    trace === nothing && release!(masks...)
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
    a = qkv_attention(layer.in_proj(layer.norm1(x)), H, nothing, mask, T(hd)^T(-0.5))
    x = x .+ layer.out_proj(a)
    # PyTorch TransformerEncoderLayer defaults to ReLU; the encoder and scorer use GELU.
    x .+ layer.linear2(relu.(layer.linear1(layer.norm2(x))))
end

struct DecisionModel{T}
    encoder::ModernBert{T}
    head::Vector{HeadLayer}
    type_emb::AbstractMatrix{T}              # (hidden, 3)
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
`marker_mask` `(K, B)` and `qtype` `(B,)`, as produced by `collate`, as host arrays. Returns
Float32 host `Array`s `logits` `(K, B)` and `action` `(n_actions, B)`, whatever array type the
weights live in (see [`on_device_of`](@ref)); the pooled features are computed on the host. Pass a `Dict` as `trace` to record every
intermediate activation under the same keys as `LayaMLXReference.trace`.
"""
function (m::DecisionModel{T})(batch::AbstractDict; trace=nothing) where {T}
    ids, mask = batch["input_ids"], Bool.(batch["attention_mask"])
    marker_pos, marker_mask = batch["marker_pos"], Bool.(batch["marker_mask"])
    qtype = batch["qtype"]
    B = size(ids, 2)

    h = m.encoder(ids, mask; trace)
    trace === nothing || (trace["encoder"] = h)
    h = h .+ reshape(gather_columns(m.type_emb, qtype .+ 1), :, 1, B)
    trace === nothing || (trace["typed"] = h)
    hmask = on_device_of(h, reshape(mask, size(mask, 1), 1, B))
    for (i, layer) in enumerate(m.head)
        h = layer(h, hmask)
        trace === nothing || (trace["head_$(i-1)"] = h)
    end

    d, L, K = size(h, 1), size(h, 2), size(marker_pos, 1)
    columns = [(b - 1) * L + max(marker_pos[j, b], 0) + 1 for j in 1:K, b in 1:B]
    markers = reshape(gather_columns(reshape(h, d, :), vec(columns)), d, K, B)
    logits = to_host(Float32.(dropdims(m.scorer2(gelu.(m.scorer1(m.scorer_norm(markers)))); dims=1)))
    logits = ifelse.(marker_mask, logits, -1.0f4)
    p = softmax(logits; dims=1)
    k = Float32.(max.(sum(marker_mask; dims=1), 2))
    entropy = -sum(p .* log.(max.(p, 1.0f-9)); dims=1) ./ log.(k)
    # The public runtime pads to at least two marker slots for one-option choices.
    top = mapslices(c -> partialsort(c, 1:2; rev=true), p; dims=1)   # (2, B): best, second
    features = vcat(top[1:1, :], top[1:1, :] .- top[2:2, :], entropy, k ./ 255.0f0)
    pooled = vcat(to_host(Float32.(h[:, 1, :])), features)
    action = to_host(Float32.(m.act2(gelu.(m.act1(on_device_of(h, T.(pooled)))))))
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
