"""
    sanitize_weights(weights) -> Dict

Map upstream PyTorch parameter names to MLX names, as `laya_mlx.model.sanitize_weights`.
MLX-format checkpoints pass through unchanged.
"""
function sanitize_weights(weights::AbstractDict)
    result = Dict{String,Array}()
    for (name, value) in weights
        name = replace(name, ".in_proj_weight" => ".in_proj.weight", ".in_proj_bias" => ".in_proj.bias")
        for prefix in ("scorer", "act_head")
            if startswith(name, prefix * ".") && !startswith(name, prefix * ".layers.")
                name = prefix * ".layers." * name[length(prefix)+2:end]
            end
        end
        haskey(result, name) && error("Duplicate checkpoint parameter after conversion: $name")
        result[name] = value
    end
    result
end

# Strict consumption of checkpoint tensors: every tensor must be used exactly once with the
# expected (reversed) size.
struct WeightReader{T}
    weights::Dict{String,Array}
end

function (r::WeightReader{T})(name::AbstractString, dims::Integer...) where {T}
    haskey(r.weights, name) || error("Missing checkpoint parameter: $name")
    w = pop!(r.weights, name)
    size(w) == dims || error("Shape mismatch for $name: checkpoint $(reverse(size(w))), expected $(reverse(dims))")
    Array{T}(w)
end

linear(r, name, nin, nout; bias=true) =
    Linear(r(name * ".weight", nin, nout), bias ? r(name * ".bias", nout) : nothing)
layernorm(r, name, d; bias=true, eps=1e-5) =
    LayerNorm(r(name * ".weight", d), bias ? r(name * ".bias", d) : nothing, Float32(eps))

function DecisionModel{T}(cfg::EncoderConfig, agent_cfg::AbstractDict, weights::AbstractDict) where {T}
    r = WeightReader{T}(sanitize_weights(weights))
    model = DecisionModel{T}(cfg, agent_cfg, r)
    # Checkpoint buffer; calibration uses the JSON config instead.
    haskey(r.weights, "temperature") && r("temperature", 3)
    isempty(r.weights) || error("Unexpected checkpoint parameters: $(join(sort(collect(keys(r.weights))), ", "))")
    model
end

# `r(name, dims...)` returns the `Array{T}` parameter `name` of Julia size `dims`.
function DecisionModel{T}(cfg::EncoderConfig, agent_cfg::AbstractDict, r) where {T}
    d, I = cfg.hidden_size, cfg.intermediate_size
    nb = (bias=cfg.norm_bias, eps=cfg.norm_eps)
    layers = map(0:cfg.num_hidden_layers-1) do i
        p = "encoder.layers.$i."
        kind = cfg.layer_types[i+1]
        EncoderLayer{T}(
            kind,
            i == 0 ? nothing : layernorm(r, p * "attn_norm", d; nb...),
            linear(r, p * "attn.Wqkv", d, 3d; bias=cfg.attention_bias),
            linear(r, p * "attn.Wo", d, d; bias=cfg.attention_bias),
            cfg.num_attention_heads,
            rope_base(cfg, kind),
            layernorm(r, p * "mlp_norm", d; nb...),
            linear(r, p * "mlp.Wi", d, 2I; bias=cfg.mlp_bias),
            linear(r, p * "mlp.Wo", I, d; bias=cfg.mlp_bias),
        )
    end
    encoder = ModernBert{T}(
        cfg,
        r("encoder.embeddings.tok_embeddings.weight", d, cfg.vocab_size),
        layernorm(r, "encoder.embeddings.norm", d; nb...),
        layers,
        layernorm(r, "encoder.final_norm", d; nb...),
    )
    nheads = max(1, d ÷ 64)
    d % nheads == 0 || error("Decision head dimensions must be divisible by its head count")
    head = map(0:get(agent_cfg, "head_layers", 2)-1) do i
        p = "head.layers.$i."
        HeadLayer(
            nheads,
            layernorm(r, p * "norm1", d),
            linear(r, p * "self_attn.in_proj", d, 3d),
            linear(r, p * "self_attn.out_proj", d, d),
            layernorm(r, p * "norm2", d),
            linear(r, p * "linear1", d, 4d),
            linear(r, p * "linear2", 4d, d),
        )
    end
    nact = length(get(agent_cfg, "act_costs", Dict())) + 1
    DecisionModel{T}(
        encoder,
        head,
        r("type_emb.weight", d, 3),
        layernorm(r, "scorer.layers.0", d),
        linear(r, "scorer.layers.1", d, d),
        linear(r, "scorer.layers.3", d, 1),
        linear(r, "act_head.layers.0", d + 4, 256),
        linear(r, "act_head.layers.2", 256, nact),
    )
end

"""
    load_model(dir; dtype=Float32) -> (model, encoder_config, agent_config)

Load `model.safetensors`, `encoder/config.json` and `rl_agent_config.json` from a Laya
checkpoint directory (upstream or MLX-converted).
"""
function load_model(dir::AbstractString; dtype::Type{T}=Float32) where {T}
    for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json")
        isfile(joinpath(dir, name)) || throw(ArgumentError("Not a complete Laya checkpoint: $(joinpath(dir, name)) is missing"))
    end
    agent_cfg = JSON.parsefile(joinpath(dir, "rl_agent_config.json"))
    cfg = EncoderConfig(JSON.parsefile(joinpath(dir, "encoder", "config.json")))
    weights = load_safetensors(joinpath(dir, "model.safetensors"))
    DecisionModel{T}(cfg, agent_cfg, weights), cfg, agent_cfg
end
