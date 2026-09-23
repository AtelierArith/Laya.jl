"""
    LayaMLX

Laya decision model (ModernBERT encoder + decision head) running on MLX through the
mlx-c C API. `LibMLX` holds the Clang.jl-generated bindings, `MX` a thin array/op layer,
and `model.jl` the forward pass of `laya_mlx/model.py`.
"""
module LayaMLX

include("LibMLX.jl")
include("mlx.jl")
using .MX

include("model.jl")

export MX, MLXArray, MLXError, LayaModel, load, forward

end # module LayaMLX
