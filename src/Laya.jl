"""
    Laya

Pure-Julia inference for Laya typed-decision models, ported from `laya-mlx`.
"""
module Laya

using Downloads: Downloads
using JSON
using LinearAlgebra
using Scratch: @get_scratch!

export Agent, DecisionModel, EncoderConfig, Tokenizer, load, load_model, load_safetensors, predict, system_one

include("safetensors.jl")
include("mathfns.jl")
include("config.jl")
include("layers.jl")
include("cpu.jl")
include("model.jl")
include("weights.jl")
include("tokenizer.jl")
include("prompt.jl")
include("agent.jl")

end # module Laya
