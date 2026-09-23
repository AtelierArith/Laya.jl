# Laya DecisionModel forward pass (port of extern/laya-mlx/laya_mlx/model.py).
# All shapes and axes in this file are MLX (row-major, 0-based) ones, as in Python.

using JSON

const MXA = MLXArray

# ---------------------------------------------------------------------------- config

struct EncoderConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    norm_eps::Float64
    norm_bias::Bool
    attention_bias::Bool
    mlp_bias::Bool
    local_attention::Int
    layer_types::Vector{String}
    rope_bases::Vector{Float64}     # per layer
end

head_dim(c::EncoderConfig) = c.hidden_size ÷ c.num_attention_heads

"""Mirror of `EncoderConfig.from_dict` (defaults and validation included)."""
function EncoderConfig(d::AbstractDict)
    g(k, default) = something(get(d, k, default), default)
    get(d, "model_type", "modernbert") == "modernbert" ||
        throw(ArgumentError("Unsupported encoder: $(d["model_type"]); expected modernbert"))
    get(d, "hidden_activation", "gelu") == "gelu" ||
        throw(ArgumentError("Unsupported encoder activation: $(d["hidden_activation"])"))
    D, H, n = Int(d["hidden_size"]), Int(d["num_attention_heads"]), Int(d["num_hidden_layers"])
    (D % H != 0 || (D ÷ H) % 2 != 0) &&
        throw(ArgumentError("ModernBERT requires an even, integral attention head dimension"))
    every = Int(g("global_attn_every_n_layers", 3))
    types = get(d, "layer_types", nothing)
    types = types === nothing ? [i % every == 0 ? "full_attention" : "sliding_attention" for i in 0:n-1] :
        String.(collect(types))
    (length(types) != n || !issubset(types, ("full_attention", "sliding_attention"))) &&
        throw(ArgumentError("Invalid ModernBERT layer_types"))
    rp = something(get(d, "rope_parameters", nothing), Dict())
    fallback = Dict("full_attention" => Float64(g("global_rope_theta", 160000.0)),
                    "sliding_attention" => Float64(g("local_rope_theta", 10000.0)))
    bases = map(types) do kind
        p = get(rp, kind, Dict())
        get(p, "rope_type", "default") == "default" ||
            throw(ArgumentError("Only default (unscaled) ModernBERT RoPE is supported"))
        Float64(get(p, "rope_theta", fallback[kind]))
    end
    EncoderConfig(Int(d["vocab_size"]), D, Int(d["intermediate_size"]), n, H,
                  Float64(g("norm_eps", 1e-5)), g("norm_bias", false), g("attention_bias", false),
                  g("mlp_bias", false), Int(g("local_attention", 128)), types, bases)
end

# ---------------------------------------------------------------------------- weights

"""Same renaming as `sanitize_weights` in model.py."""
function sanitize_weights(weights::AbstractDict)
    out = Dict{String,valtype(weights)}()
    for (name, v) in weights
        name = replace(name, ".in_proj_weight" => ".in_proj.weight", ".in_proj_bias" => ".in_proj.bias")
        for p in ("scorer", "act_head")
            if startswith(name, p * ".") && !startswith(name, p * ".layers.")
                name = p * ".layers." * name[length(p)+2:end]
            end
        end
        haskey(out, name) && throw(ArgumentError("Duplicate checkpoint parameter after conversion: $name"))
        out[name] = v
    end
    out
end

"""Every parameter `DecisionModel(cfg, agent_cfg)` owns, with its MLX shape."""
function parameter_shapes(c::EncoderConfig, head_layers::Integer, n_actions::Integer)
    D, I = c.hidden_size, c.intermediate_size
    s = Dict{String,Tuple}()
    ln(p, bias) = (s[p*".weight"] = (D,); bias && (s[p*".bias"] = (D,)))
    lin(p, o, i, bias) = (s[p*".weight"] = (o, i); bias && (s[p*".bias"] = (o,)))
    s["encoder.embeddings.tok_embeddings.weight"] = (c.vocab_size, D)
    ln("encoder.embeddings.norm", c.norm_bias)
    for i in 0:c.num_hidden_layers-1
        p = "encoder.layers.$i."
        i > 0 && ln(p * "attn_norm", c.norm_bias)
        lin(p * "attn.Wqkv", 3D, D, c.attention_bias)
        lin(p * "attn.Wo", D, D, c.attention_bias)
        ln(p * "mlp_norm", c.norm_bias)
        lin(p * "mlp.Wi", 2I, D, c.mlp_bias)
        lin(p * "mlp.Wo", D, I, c.mlp_bias)
    end
    ln("encoder.final_norm", c.norm_bias)
    for i in 0:head_layers-1
        p = "head.layers.$i."
        lin(p * "self_attn.in_proj", 3D, D, true)
        lin(p * "self_attn.out_proj", D, D, true)
        ln(p * "norm1", true); ln(p * "norm2", true)
        lin(p * "linear1", 4D, D, true)
        lin(p * "linear2", D, 4D, true)
    end
    s["type_emb.weight"] = (3, D)
    ln("scorer.layers.0", true)
    lin("scorer.layers.1", D, D, true)
    lin("scorer.layers.3", 1, D, true)
    lin("act_head.layers.0", 256, D + 4, true)
    lin("act_head.layers.2", n_actions, 256, true)
    s["temperature"] = (3,)
    s
end

# ---------------------------------------------------------------------------- model

struct LayaModel
    config::EncoderConfig
    agent_config::Dict{String,Any}
    head_layers::Int
    weights::Dict{String,MLXArray}
    dtype::Any
    device::Symbol
end

Base.show(io::IO, m::LayaModel) = print(io, "LayaModel(", m.config.num_hidden_layers, " layers, d=",
    m.config.hidden_size, ", ", m.dtype, ", ", m.device, ")")

"""
    load(dir; dtype=Float32, device=:gpu) -> LayaModel

Load `model.safetensors`, `encoder/config.json` and `rl_agent_config.json` from `dir`,
sanitize parameter names, strictly check names and shapes, and cast to `dtype`
(`Float32`, `Float16` or `MX.BFloat16`).
"""
function load(dir::AbstractString; dtype=Float32, device::Symbol=:gpu)
    enc = EncoderConfig(JSON.parsefile(joinpath(dir, "encoder", "config.json")))
    cfg = Dict{String,Any}(JSON.parsefile(joinpath(dir, "rl_agent_config.json")))
    haskey(cfg, "encoder") && haskey(cfg, "head_layers") ||
        throw(ArgumentError("Laya config must specify encoder and head_layers"))
    nh = Int(cfg["head_layers"])
    expected = parameter_shapes(enc, nh, length(get(cfg, "act_costs", Dict())) + 1)
    raw = sanitize_weights(MX.load_safetensors(joinpath(dir, "model.safetensors")))
    missing_ = setdiff(keys(expected), keys(raw))
    extra = setdiff(keys(raw), keys(expected))
    wrong = [k => (shape(raw[k]), expected[k]) for k in intersect(keys(raw), keys(expected))
             if shape(raw[k]) != expected[k]]
    isempty(missing_) && isempty(extra) && isempty(wrong) || throw(ArgumentError(
        "checkpoint mismatch: missing=$(sort!(collect(missing_))) unexpected=$(sort!(collect(extra))) shape=$wrong"))
    W = with_device(device) do
        Dict(k => astype(v, dtype) for (k, v) in raw)
    end
    eval!(values(W)...)
    for (k, v) in raw   # drop the file-dtype copies now instead of at the next GC
        v === W[k] || MX.free!(v)
    end
    LayaModel(enc, cfg, nh, W, dtype, device)
end

# ---- building blocks

function linear(m::LayaModel, p, x)
    w = MX.transpose(m.weights[p*".weight"])
    b = get(m.weights, p * ".bias", nothing)
    b === nothing ? MX.matmul(x, w) : MX.addmm(b, x, w)
end

layer_norm(m::LayaModel, p, x, eps=1e-5) =
    MX.layer_norm(x, m.weights[p*".weight"], get(m.weights, p * ".bias", nothing), eps)

"""Shared attention body; `base === nothing` skips RoPE (decision head)."""
function attention(m, qkvp, op, x, mask, H; base=nothing)
    b, L, D = shape(x)
    hd = D ÷ H
    qkv = MX.reshape(linear(m, qkvp, x), b, L, 3, H, hd)
    q, k, v = (MX.transpose(MX.index(qkv, 2, i), 0, 2, 1, 3) for i in 0:2)
    if base !== nothing
        q = MX.rope(q, hd; base); k = MX.rope(k, hd; base)
    end
    out = MX.sdpa(q, k, v; scale=hd^-0.5, mask)
    linear(m, op, MX.reshape(MX.transpose(out, 0, 2, 1, 3), b, L, -1))
end

function encoder_layer(m::LayaModel, i, x, mask)
    c, p = m.config, "encoder.layers.$i."
    h = i == 0 ? x : layer_norm(m, p * "attn_norm", x, c.norm_eps)
    x = x + attention(m, p * "attn.Wqkv", p * "attn.Wo", h, mask, c.num_attention_heads; base=c.rope_bases[i+1])
    value, gate = MX.split(linear(m, p * "mlp.Wi", layer_norm(m, p * "mlp_norm", x, c.norm_eps)), 2; axis=-1)
    x + linear(m, p * "mlp.Wo", MX.gelu(value) * gate)
end

function head_layer(m::LayaModel, i, x, mask)
    p = "head.layers.$i."
    H = max(1, m.config.hidden_size ÷ 64)
    x = x + attention(m, p * "self_attn.in_proj", p * "self_attn.out_proj", layer_norm(m, p * "norm1", x), mask, H)
    x + linear(m, p * "linear2", MX.relu(linear(m, p * "linear1", layer_norm(m, p * "norm2", x))))
end

"""Boolean key masks `(full, sliding)` exactly as `attention_masks` in model.py."""
function attention_masks(mask::MXA, window::Integer)
    b, L = shape(mask)
    valid = astype(mask, Bool)
    full = MX.reshape(valid, b, 1, 1, L)
    pos = MX.arange(L)
    loc = MX.less_equal(abs(MX.reshape(pos, L, 1) - MX.reshape(pos, 1, L)), window ÷ 2)
    loc = (MX.reshape(loc, 1, 1, L, L) | ~MX.reshape(valid, b, 1, L, 1)) & full
    full, loc
end

"""`model(batch)` is `forward(model, batch)`: `(logits, action)` as Float32 Julia arrays."""
(m::LayaModel)(batch::AbstractDict; trace::Bool=false) = forward(m, batch; trace)

tojulia(x::MXA) = Array(eltype(x) == Bool ? x : astype(x, Float32))

"""
    forward(model, batch; trace=false) -> (logits, action[, trace])

`batch` is the Dict from `LayaMLXReference.collate` (Julia layout, e.g. `input_ids` is
`(L, n)`). Returns Float32 Julia arrays `logits` `(K, n)` and `action` `(A, n)`; with
`trace=true` also a Dict with the same keys as the reference `trace`.
"""
function forward(m::LayaModel, batch::AbstractDict; trace::Bool=false)
    # every intermediate MLXArray is freed on return (device memory is invisible to Julia's GC)
    MX.scoped(() -> with_device(() -> _forward(m, batch, trace), m.device))
end

function _forward(m::LayaModel, batch::AbstractDict, trace::Bool)
    begin
        ids = MXA(Int32.(batch["input_ids"]))
        amask = MXA(Bool.(batch["attention_mask"]))
        mpos = MXA(Int32.(batch["marker_pos"]))
        mmask = MXA(Bool.(batch["marker_mask"]))
        qtype = MXA(Int32.(vec(batch["qtype"])))
        tr = Dict{String,MXA}()
        rec(k, x) = (trace && (tr[k] = x); x)
        W, c = m.weights, m.config

        x = rec("embeddings", layer_norm(m, "encoder.embeddings.norm",
            MX.embedding(W["encoder.embeddings.tok_embeddings.weight"], ids), c.norm_eps))
        full, sliding = attention_masks(amask, c.local_attention)
        rec("mask_full", full); rec("mask_sliding", sliding)
        for i in 0:c.num_hidden_layers-1
            x = rec("layer_$i", encoder_layer(m, i, x, c.layer_types[i+1] == "full_attention" ? full : sliding))
        end
        h = rec("encoder", layer_norm(m, "encoder.final_norm", x, c.norm_eps))
        b, L, D = shape(h)
        h = rec("typed", h + MX.reshape(MX.embedding(W["type_emb.weight"], qtype), b, 1, D))
        hmask = MX.reshape(astype(amask, Bool), b, 1, 1, L)
        for i in 0:m.head_layers-1
            h = rec("head_$i", head_layer(m, i, h, hmask))
        end

        K = shape(mpos)[2]
        flat = MX.reshape(MX.arange(b), b, 1) * L + MX.maximum(mpos, 0)
        markers = MX.take(MX.reshape(h, b * L, D), flat, 0)
        s = linear(m, "scorer.layers.1", layer_norm(m, "scorer.layers.0", markers))
        logits = astype(MX.squeeze(linear(m, "scorer.layers.3", MX.gelu(s)), -1), Float32)
        logits = MX.where(mmask, logits, -1e4)
        p = MX.softmax(logits; axis=-1)
        k = astype(MX.maximum(MX.sum(mmask; axis=-1), 2), Float32)
        entropy = -MX.sum(p * log(MX.maximum(p, 1e-9)); axis=-1) / log(k)
        top = MX.slice(MX.sort(p; axis=-1), [0, K - 2], [b, K])
        t0, t1 = MX.index(top, 1, 0), MX.index(top, 1, 1)
        features = MX.stack([t1, t1 - t0, entropy, k / 255.0]; axis=-1)
        pooled = MX.concatenate([astype(MX.index(h, 1, 0), Float32), features]; axis=-1)
        a = linear(m, "act_head.layers.0", astype(pooled, m.dtype))
        action = astype(linear(m, "act_head.layers.2", MX.gelu(a)), Float32)
        rec("logits", logits); rec("action", action)

        eval!(logits, action, values(tr)...)
        out = (tojulia(logits), tojulia(action))
        trace ? (out..., Dict(k => tojulia(v) for (k, v) in tr)) : out
    end
end
