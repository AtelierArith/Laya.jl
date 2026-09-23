# Laya.jl

Laya.jl runs [Laya](https://github.com/mizorewww/laya-mlx) typed-decision models in Julia.
It answers `choice`, `score` and `noul` questions about a state in a single encoder forward
pass, with no token-by-token generation.

It is a port of [laya-mlx](https://github.com/mizorewww/laya-mlx) (Python/MLX) and keeps the
same checkpoints, prompt format, calibration and output schema.

- **`Laya`** (this package, `src/`):
  - Pure Julia on the CPU, so it runs on any OS.
  - Includes the safetensors reader, the ModernBERT encoder and decision head, the Hugging
    Face tokenizer and the prompt builder.
  - Depends only on JSON, Scratch and the standard libraries.
- **`LayaMLX`** (`LayaMLX/`): the same forward pass on Apple's MLX (GPU), through the mlx-c C
  API. It is a separate package for Apple silicon, and `Laya` does not depend on it.

## Quick start

```julia
using Laya
using AppleAccelerate   # optional, on Apple silicon: ~3-5× faster CPU inference

agent = Laya.load("aac6fef/laya-mlx")   # downloaded on first use
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

Both `Laya` and `LayaMLX` are checked against the Python implementation (see
`test/runtests.jl` and `LayaMLX/test/runtests.jl`):

- **`Laya` (CPU)**: every answer matches, and the relative error is below the noise MLX shows
  between its own float32 GPU and CPU results.
- **`LayaMLX`**: bit-identical (0.0 error) in float32 and float16, on both checkpoints.

Apple M4, `aac6fef/laya-mlx`, float32, end-to-end p50:

| backend | 1 question | 10 questions | 50 questions |
|---|---:|---:|---:|
| Python laya-mlx (MLX GPU) | 41 ms | 318 ms | 1575 ms |
| `LayaMLX` (MLX GPU via mlx-c) | 42 ms | 315 ms | 1599 ms |
| `Laya` (CPU, Accelerate) | 109 ms | 796 ms | 5366 ms |

See `benchmark/` for the method and the raw results.

## Repository layout

| path | contents |
|---|---|
| `src/` | the `Laya` package |
| `test/` | tests against the Python reference |
| `LayaMLX/` | MLX backend. Clang.jl-generated mlx-c bindings, forward pass and benchmark (see `LayaMLX/README.md`). |
| `reference/` | `LayaMLXReference`: a PythonCall bridge to `extern/laya-mlx`, used only for testing |
| `benchmark/` | Python, CPU and MLX benchmarks (see `benchmark/README.md`) |
| `deps/` | the `mlx-c` submodule and `build.sh` |

`SETUP.md` describes the development setup: the Python reference in `extern/laya-mlx`, the
mlx-c build and the checkpoints.

## Status

Not ported yet:

- the router, language detection, presets, email helpers and the shortlist
- a backend hook that lets `predict` run on `LayaMLX` (for now, `LayaMLX/bench.jl` copies the
  post-processing)
- a Metal.jl GPU backend

## Acknowledgements

Laya is by Convai Innovations. The MLX port and the converted checkpoints are by
[mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx) (Apache-2.0). Laya.jl is an
independent port and is not affiliated with either project.
