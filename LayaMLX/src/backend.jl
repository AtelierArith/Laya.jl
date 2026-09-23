import Laya

"""
    MLXBackend(device=:gpu)

Laya backend that runs the model forward on MLX (`:gpu` or `:cpu`):
`Laya.load(repo; backend=MLXBackend())`. The tokenizer, prompts and calibration stay in `Laya`.
"""
struct MLXBackend <: Laya.Backend
    device::Symbol
end
MLXBackend() = MLXBackend(:gpu)

Laya.load_backend_model(b::MLXBackend, dir::AbstractString, ::Type{T}) where {T} = load(dir; dtype=T, device=b.device)
