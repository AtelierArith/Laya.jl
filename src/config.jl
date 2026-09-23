"""
    EncoderConfig

ModernBERT encoder settings read from `encoder/config.json`. Mirrors `EncoderConfig` in
`laya_mlx/model.py`, including its validation.
"""
Base.@kwdef struct EncoderConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    model_type::String = "modernbert"
    norm_eps::Float64 = 1e-5
    norm_bias::Bool = false
    attention_bias::Bool = false
    mlp_bias::Bool = false
    hidden_activation::String = "gelu"
    local_attention::Int = 128
    global_attn_every_n_layers::Int = 3
    global_rope_theta::Float64 = 160000.0
    local_rope_theta::Float64 = 10000.0
    max_position_embeddings::Int = 8192
    layer_types::Vector{Symbol}
    rope_parameters::Dict{String,Any} = Dict{String,Any}()
end

head_dim(cfg::EncoderConfig) = cfg.hidden_size ÷ cfg.num_attention_heads

function rope_base(cfg::EncoderConfig, kind::Symbol)
    fallback = kind === :full_attention ? cfg.global_rope_theta : cfg.local_rope_theta
    params = get(cfg.rope_parameters, String(kind), nothing)
    Float64(params === nothing ? fallback : get(params, "rope_theta", fallback))
end

function EncoderConfig(value::AbstractDict)
    names = fieldnames(EncoderConfig)
    kw = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in value if Symbol(k) in names && v !== nothing)
    get(kw, :model_type, "modernbert") == "modernbert" ||
        throw(ArgumentError("Unsupported encoder: $(repr(kw[:model_type])); expected modernbert"))
    get(kw, :hidden_activation, "gelu") == "gelu" ||
        throw(ArgumentError("Unsupported encoder activation: $(repr(kw[:hidden_activation]))"))
    hidden, heads = kw[:hidden_size], kw[:num_attention_heads]
    (hidden % heads == 0 && (hidden ÷ heads) % 2 == 0) ||
        throw(ArgumentError("ModernBERT requires an even, integral attention head dimension"))
    nlayers = kw[:num_hidden_layers]
    every = get(kw, :global_attn_every_n_layers, 3)
    types = if haskey(kw, :layer_types)
        Symbol.(kw[:layer_types])
    else
        [i % every == 0 ? :full_attention : :sliding_attention for i in 0:nlayers-1]
    end
    (length(types) == nlayers && types ⊆ (:full_attention, :sliding_attention)) ||
        throw(ArgumentError("Invalid ModernBERT layer_types"))
    kw[:layer_types] = types
    rope = Dict{String,Any}(String(k) => v for (k, v) in get(kw, :rope_parameters, Dict()))
    kw[:rope_parameters] = rope
    for kind in unique(types)
        params = get(rope, String(kind), Dict())
        get(params, "rope_type", "default") == "default" ||
            throw(ArgumentError("Only default (unscaled) ModernBERT RoPE is supported"))
    end
    EncoderConfig(; kw...)
end
