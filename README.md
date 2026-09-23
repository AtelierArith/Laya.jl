# Laya.jl

Laya.jl runs [Laya](https://github.com/NandhaKishorM/laya) typed-decision models in Julia.
It answers `choice`, `score` and `noul` questions about a state in a single encoder forward
pass, with no token-by-token generation.

It is a port of [laya-mlx](https://github.com/mizorewww/laya-mlx) (Python/MLX) and keeps the
same checkpoints, prompt format, calibration and output schema.

- **`Laya`** (this package, `src/`):
  - Pure Julia on the CPU, so it runs on any OS.
  - Includes the safetensors reader, the ModernBERT encoder and decision head, the Hugging
    Face tokenizer and the prompt builder.
  - Depends only on JSON, Scratch and the standard libraries. Apple Accelerate (CPU) and
    Metal.jl (GPU) are optional package extensions.
- **`LayaMLX`** (`LayaMLX/`): the same forward pass on Apple's MLX (GPU), through the mlx-c C
  API. It is a separate package for Apple silicon that plugs into `Laya` as a backend, and
  `Laya` does not depend on it.

## Quick start

```julia
using Laya

agent = Laya.load("aac6fef/laya-mlx")   # CPU; downloaded on first use
result = predict(agent, "I was billed twice. Please refund the duplicate.",
    Dict("department" => Dict(
        "type" => "choice",
        "instructions" => "Who should handle this?",
        "criteria" => ["billing", "technical", "sales"])))
result["answers"]["department"]
# "choice" => "billing", "probabilities" => ("billing" => 0.8895, "technical" => 0.0833, "sales" => 0.0271),
# "confidence" => 0.6276, "action" => ("act_probability" => 1.0), "type" => "choice"
```

- **Question types**:
  - `choice`: probabilities over named options.
  - `score`: probabilities over ordered rubric levels, and the expected score.
  - `noul`: P(true) for a proposition.
- **`state`**: a string, or a dictionary or vector that is serialized as JSON.
- **Result**: `model`, `answers` and `usage`, as upstream Laya returns them.

### Backends

`load` picks where the model forward runs with `backend`. Everything else (`predict`, the
tokenizer, prompts, calibration and the result) is the same Julia code for every backend.

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
| `CPUBackend()` | CPU (any OS) | – |
| `AccelerateBackend()` | CPU, BLAS through Apple Accelerate | AppleAccelerate (weak dependency) |
| `MetalBackend()` | Apple GPU, Float32 or Float16 | Metal.jl (weak dependency; Metal.jl's own backend type) |
| `MLXBackend()` | Apple GPU through mlx-c | `LayaMLX` (`LayaMLX/`) |

- `AccelerateBackend` forwards BLAS process-wide, so every CPU model in the process uses
  Accelerate afterwards.
- `MetalBackend` runs the same `DecisionModel` code with its weights in `MtlArray`s. Laya
  specializes a few hot spots for `MtlArray` (fused Metal kernels for LayerNorm, GeGLU and the
  attention glue, and batched matmuls over all heads).
- A new backend only needs a method of `Laya.load_backend_model(backend, dir, dtype)`.

### Checkpoints

| repository | encoder | parameters |
|---|---|---:|
| [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx) | ModernBERT-large | 421M |
| [`aac6fef/laya-multilingual-mlx`](https://huggingface.co/aac6fef/laya-multilingual-mlx) | mmBERT-base | 322M |

These are the ones tested. `Laya.load` accepts either of the following:

- **A local directory**: it must contain `model.safetensors`, `rl_agent_config.json`,
  `encoder/config.json` and `tokenizer/`.
- **A Hugging Face repository id**: Laya looks in this order:
  1. the Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`),
  2. Laya's Scratch.jl space,
  3. otherwise it downloads the repository into that scratch space.

  Downloading respects `HF_HUB_OFFLINE`, `HF_TOKEN` and `HF_ENDPOINT`.

The `Float32` model of the 421M checkpoint needs about 3 GiB of memory.

## Accuracy and speed

Every backend is checked against the Python implementation (see `test/runtests.jl` and
`LayaMLX/test/runtests.jl`):

- **CPU**: every answer matches, and the relative error is below the noise MLX shows
  between its own float32 GPU and CPU results.
- **Metal**: the selected answers match. The relative error of the logits is about 1e-6
  in float32 and about 2e-3 in float16.
- **`LayaMLX`**: bit-identical (0.0 error) in float32 and float16, on both checkpoints.

Apple M4, `aac6fef/laya-mlx`, float32, end-to-end p50:

| backend | 1 question | 10 questions | 50 questions |
|---|---:|---:|---:|
| Python laya-mlx (MLX GPU) | 41 ms | 318 ms | 1575 ms |
| `LayaMLX` (MLX GPU via mlx-c) | 42 ms | 315 ms | 1599 ms |
| `Laya` (CPU, Accelerate) | 109 ms | 796 ms | 5366 ms |

`Laya` on Metal (float32) took 63 ms for 1 question and 367 ms for 10 questions in a first
run on a warm GPU (MLX: 42 ms and about 390 ms in the same state). For few questions, the
per-kernel launch overhead still dominates.

See `benchmark/` for the method and the raw results.

## Repository layout

| path | contents |
|---|---|
| `src/` | the `Laya` package |
| `test/` | tests against the Python reference |
| `ext/` | package extensions: `LayaAppleAccelerateExt` and `LayaMetalExt` |
| `LayaMLX/` | MLX backend. Clang.jl-generated mlx-c bindings, forward pass and benchmark (see `LayaMLX/README.md`). |
| `reference/` | `LayaMLXReference`: a PythonCall bridge to `extern/laya-mlx`, used only for testing |
| `benchmark/` | Python, CPU, Metal and MLX benchmarks (see `benchmark/README.md`) |
| `deps/` | the `mlx-c` submodule and `build.sh` |
| `extern/laya-mlx` | upstream laya-mlx (submodule): the Python reference for tests and benchmarks |

`SETUP.md` describes the development setup: the Python reference in `extern/laya-mlx`, the
mlx-c build and the checkpoints.

## Status

Not ported yet:

- the router, language detection, presets, email helpers and the shortlist

Known gaps:

- `MetalBackend` is slower than MLX for one or a few questions (kernel launch overhead).

## Acknowledgements

- [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) is by Convai Innovations and the
  Laya contributors. The original weights are on Hugging Face, e.g.
  [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya).
- [laya-mlx](https://github.com/mizorewww/laya-mlx) (Apache-2.0), the MLX port Laya.jl follows,
  and its converted checkpoints ([aac6fef](https://huggingface.co/aac6fef)) are by the laya-mlx
  contributors.

Laya.jl is an independent port and is not affiliated with either project.
