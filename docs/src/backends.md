# Backends

[`load`](@ref) chooses where the model forward runs with its `backend` keyword, by dispatch on
the backend's type. Everything else (`predict`, the tokenizer, prompts, calibration and the
result) is the same Julia code for every backend.

```julia
agent = Laya.load("aac6fef/laya-mlx"; backend=CPUBackend())         # default

using AppleAccelerate                                               # CPU, Accelerate BLAS
agent = Laya.load("aac6fef/laya-mlx"; backend=AccelerateBackend())

using Metal                                                         # Apple GPU
agent = Laya.load("aac6fef/laya-mlx"; dtype=Float16, backend=MetalBackend())

using LayaMLX                                                       # Apple GPU via MLX
agent = Laya.load("aac6fef/laya-mlx"; backend=MLXBackend())
```

| backend | runs on | package |
|---|---|---|
| [`CPUBackend`](@ref)`()` | CPU (any OS) | – |
| [`AccelerateBackend`](@ref)`()` | CPU, BLAS through Apple Accelerate | AppleAccelerate (weak dependency) |
| `MetalBackend()` | Apple GPU, `Float32` or `Float16` | Metal.jl (weak dependency; Metal.jl's own backend type) |
| `MLXBackend()` | Apple GPU through mlx-c | `LayaMLX` (in `LayaMLX/` of the repository) |

## CPU

The default. Start Julia with `-t auto`: attention and normalization are threaded over
columns. On Apple silicon, [`AccelerateBackend`](@ref) makes the matrix products 3-5× faster.
It forwards BLAS process-wide, so every CPU model in the process uses Accelerate afterwards.

## Metal

`MetalBackend()` runs the same [`DecisionModel`](@ref) code as the CPU with its weights in
`MtlArray`s ([`Laya.adapt_arrays`](@ref)). The model code is generic over array types; the
extension `LayaMetalExt` specializes a few hot spots for `MtlArray`:

- fused Metal kernels for LayerNorm (one threadgroup per column, statistics in `Float32`),
  fused with the preceding residual sum ([`Laya.residual_norm`](@ref)), including across
  layers; GeGLU; and the split, RoPE and head layout of attention;
- all heads and batch rows of attention in two batched MPSGraph matmuls, with the softmax in
  `Float32`;
- intermediates are returned to Metal.jl's pool as soon as they are dead
  ([`Laya.release!`](@ref)), since Julia's GC does not see GPU memory pressure.

## MLX

`LayaMLX` wraps the mlx-c C API and issues the same MLX kernels as the Python laya-mlx, so
its outputs are bit-identical to Python in `Float32` and `Float16`. It is not registered and
needs a local mlx-c build; see `LayaMLX/README.md` and `SETUP.md` in the repository.

## Writing a backend

A backend is any value with a method of [`Laya.load_backend_model`](@ref) for its type. The
method returns a callable `model(batch)` that maps a [`Laya.collate`](@ref)d host batch to
host `logits` `(K, n)` and `action` `(A, n)` arrays. For an array-type backend, the usual
route is `Laya.adapt_arrays(ArrayType, first(Laya.load_model(dir; dtype=T)))` plus methods
for [`Laya.qkv_attention`](@ref), `Laya.LayerNorm` and `Laya.gelu_gate` on that array type.
