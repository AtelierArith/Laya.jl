# Where the model forward runs. Every backend serves the same `Agent`/`predict` API; `load`
# dispatches on the backend's type through `load_backend_model`. Optional backends live in
# package extensions (ext/): Accelerate is declared here, while the GPU backend is Metal.jl's
# own `MetalBackend()` (the JuliaGPU convention, as `CUDABackend()` for CUDA.jl).

"""
    Backend

Supertype of Laya's own backends. Pass a backend to [`load`](@ref) as `backend`:

| backend | runs on | needs |
|---|---|---|
| [`CPUBackend()`](@ref CPUBackend) | CPU, Julia's current BLAS | – |
| [`AccelerateBackend()`](@ref AccelerateBackend) | CPU, Apple Accelerate BLAS | `using AppleAccelerate` |
| `MetalBackend()` | Apple GPU (Metal.jl's backend type) | `using Metal` |
| `LayaMLX.MLXBackend()` | Apple GPU via MLX | the `LayaMLX` package |

A new backend only needs a method of [`load_backend_model`](@ref) for its type.
"""
abstract type Backend end

"""
    CPUBackend()

The pure-Julia [`DecisionModel`](@ref) on the CPU with whatever BLAS Julia currently uses
(OpenBLAS by default). Start Julia with `-t auto` for threaded attention and normalization.
"""
struct CPUBackend <: Backend end

"""
    AccelerateBackend()

[`CPUBackend`](@ref) with BLAS forwarded to Apple's Accelerate framework (3-5× faster matrix
products on Apple silicon). Requires `using AppleAccelerate`. The forwarding is process-wide,
so every CPU model in the process uses Accelerate afterwards.
"""
struct AccelerateBackend <: Backend end

"""
    load_backend_model(backend, dir, dtype) -> model

Load the checkpoint in `dir` with element type `dtype` for `backend`. `model(batch)` takes a
[`collate`](@ref)d batch and returns host `Array`s `logits` `(K, n)` and `action` `(A, n)`,
like [`DecisionModel`](@ref).
"""
function load_backend_model end

function load_backend_model(::CPUBackend, dir::AbstractString, ::Type{T}) where {T}
    accelerate_hint()
    first(load_model(dir; dtype=T))
end

# Less specific than the extension's method, which replaces it once AppleAccelerate is loaded.
load_backend_model(::AccelerateBackend, _, _) =
    throw(ArgumentError("AccelerateBackend needs `using AppleAccelerate` (install it with `Pkg.add(\"AppleAccelerate\")`)"))

load_backend_model(backend, _, T) =
    throw(ArgumentError("Laya has no model for backend $(repr(backend)) with dtype $T; for the GPU use `using Metal` and `MetalBackend()`"))

const ACCELERATE_HINT_SHOWN = Ref(false)

# On Apple silicon, BLAS through Accelerate runs the model's matrix products ~4-5x faster
# than the bundled OpenBLAS. AppleAccelerate is an optional (weak) dependency.
function accelerate_hint()
    (Sys.isapple() && Sys.ARCH === :aarch64 && !ACCELERATE_HINT_SHOWN[]) || return
    ACCELERATE_HINT_SHOWN[] = true
    uses_accelerate() ||
        @info "On Apple silicon, `using AppleAccelerate` and `backend=AccelerateBackend()` make Laya CPU inference ~3-5x faster."
end

uses_accelerate() = any(lib -> occursin("Accelerate", lib.libname), LinearAlgebra.BLAS.get_config().loaded_libs)
