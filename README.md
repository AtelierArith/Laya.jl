# Laya.jl

[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://atelierarith.github.io/Laya.jl/dev/)
[![CI](https://github.com/AtelierArith/Laya.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/AtelierArith/Laya.jl/actions/workflows/CI.yml)

Laya.jl runs [Laya](https://github.com/NandhaKishorM/laya) typed-decision models in Julia.
It answers `choice`, `score` and `noul` questions about a state in a single encoder forward
pass, with no token-by-token generation.

It is a port of [laya-mlx](https://github.com/mizorewww/laya-mlx) (Python/MLX) and keeps the
same checkpoints, prompt format, calibration and output schema.

- **`Laya`** (this package, `src/`):
  - Pure Julia on the CPU, so it runs on any OS.
  - Includes the safetensors reader, the ModernBERT encoder and decision head, the Hugging
    Face tokenizer and the prompt builder.
  - Depends only on JSON, Scratch and the standard libraries. Apple Accelerate (CPU),
    Metal.jl (Apple GPU) and CUDA.jl (NVIDIA GPU) are optional package extensions.
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

using CUDA                                                          # NVIDIA GPU
agent = Laya.load("aac6fef/laya-mlx"; backend=CUDABackend())

using LayaMLX                                                       # Apple GPU via MLX
agent = Laya.load("aac6fef/laya-mlx"; backend=MLXBackend())
```

| backend | runs on | package |
|---|---|---|
| `CPUBackend()` | CPU (any OS) | – |
| `AccelerateBackend()` | CPU, BLAS through Apple Accelerate | AppleAccelerate (weak dependency) |
| `MetalBackend()` | Apple GPU, Float32 or Float16 | Metal.jl (weak dependency; Metal.jl's own backend type) |
| `CUDABackend()` | NVIDIA GPU, Float32 or Float16 | CUDA.jl 6 (weak dependency; CUDA.jl's own backend type) |
| `MLXBackend()` | Apple GPU through mlx-c | `LayaMLX` (`LayaMLX/`) |

- `AccelerateBackend` forwards BLAS process-wide, so every CPU model in the process uses
  Accelerate afterwards.
- `MetalBackend` runs the same `DecisionModel` code with its weights in `MtlArray`s. Laya
  specializes a few hot spots for `MtlArray`: one fused attention kernel (simdgroup matrix
  units, no score matrices in memory, masked tiles skipped), simdgroup LayerNorm and GeGLU
  kernels, MPSGraph matmuls for the linear layers, and a buffer pool so that a warm forward
  pass allocates no device memory.
- `CUDABackend` (`LayaCUDAExt`) does the same with `CuArray`s: cuBLAS `GemmEx` for the
  linear layers (weights stored transposed so cuBLAS runs its faster NN kernels; Float16
  accumulates in Float32; Float32 stays full FP32 unless you opt into TF32 with
  `CUDA.math_mode!(CUDA.FAST_MATH)`), a fused FP32 flash-attention kernel for head_dim 64
  (skips tiles outside the local window and padding), warp-per-column LayerNorm, a buffer
  pool and pinned staging for uploads: a warm forward allocates no device memory and waits
  for the GPU only at its two downloads. Pick the GPU with `CUDA.device!` or
  `CUDA_VISIBLE_DEVICES`.
- A new backend only needs a method of `Laya.load_backend_model(backend, dir, dtype)`.

### Checkpoints

| repository | encoder | parameters |
|---|---|---:|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (the default of `load()`) | ModernBERT-large | 421M |
| [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx) (MLX conversion of the above) | ModernBERT-large | 421M |
| [`aac6fef/laya-multilingual-mlx`](https://huggingface.co/aac6fef/laya-multilingual-mlx) | mmBERT-base | 322M |

- **Original checkpoints**: the upstream repository `convaiinnovations/laya` also holds the
  multilingual and typed-decisions checkpoints in subfolders:
  `Laya.load("convaiinnovations/laya"; subfolder="multilingual")`.
- **Downloads**: only the files of the requested checkpoint are downloaded, as upstream does.
- **Tested**: the original and the MLX-converted English checkpoints give the same answers.
  The MLX-converted checkpoints are the ones compared against the Python reference.

`Laya.load` accepts either of the following:

- **A local directory**: it must contain `model.safetensors`, `rl_agent_config.json`,
  `encoder/config.json` and `tokenizer/`.
- **A Hugging Face repository id**: Laya looks in this order:
  1. the Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`),
  2. Laya's Scratch.jl space,
  3. otherwise it downloads the checkpoint's files into that scratch space.

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
- **CUDA**: compared with the CPU backend (`LAYA_TEST_GROUPS=cuda`, no Python needed): the
  selected answers match, the logits agree to about 2e-6 relative in float32 (about 1e-3 in
  float16), and repeated forwards are bitwise identical.

On the CPU the heavy work is BLAS: each `Linear` layer is a single
`transpose(W) * X` GEMM (`src/layers.jl`) and attention uses `mul!`
(`src/cpu.jl`), while `Threads.@threads` only covers LayerNorm, the GeGLU gate
and the attention heads. The default BLAS thread count (the physical-core
count) is therefore the fastest CPU configuration; starting Julia with `-t auto`
does not help the CPU backend, and can be slower, because it lowers the BLAS
thread count without moving the projections off BLAS. On an Intel i9-9900K
(8 physical cores, OpenBLAS 0.3.30) a short (8-token) production forward took
about 94 ms with the default BLAS threads versus about 287 ms with `-t 8` and
`OPENBLAS_NUM_THREADS=1`.

Apple M2 Max (38 GPU cores, 96 GB), `aac6fef/laya-mlx`, float32, end-to-end p50
(tokenization, prompts, forward, calibration and results; model loading excluded). `short:N`
asks N questions about a short state (93 tokens); `long:N` uses a state that fills the
512-token context.

| backend | short:1 | short:10 | short:50 | long:1 | long:10 |
|---|---:|---:|---:|---:|---:|
| Python laya-mlx (MLX GPU) | 17 ms | 93 ms | 410 ms | 56 ms | 480 ms |
| `Laya` + `MetalBackend()` (GPU) | 16 ms | 88 ms | 386 ms | 53 ms | 453 ms |

- **How the GPU rows were measured**: on an idle machine (load average below 4 for two
  minutes), cooled for 120 s, in the order Metal, Metal (previous version), Python, Python,
  Metal (previous version), Metal with 60 s pauses in between, each in its own process. The
  table shows each backend's faster run (10 iterations after 2 warmups). The script and raw
  results are in `benchmark/results/gpu-fair-2026-09-23-m2max-3/`; the previous version
  (before the attention kernel's third round) ran 17 / 89 / 385 / 55 / 474 ms there. The
  earlier rounds on this machine are in `gpu-fair-2026-09-23-m2max-2/` and
  `gpu-fair-2026-09-23-m2max/` (Metal 23 / 97 / 402 / 66 / 501 ms, before the matmuls were
  batched and the uploads pooled).
- **Answers**: both backends selected the same answers on every workload.
- **Reading the results**: the Metal backend is 4-6% faster than MLX on every workload. The whole forward pass is a few command buffers
  (matmuls and kernels batched together), uploads never wait for the GPU, and a warm forward
  pass makes about a dozen small allocations. Other CPU load distorts the Metal numbers far
  more than the MLX ones (see `docs/agents/workarounds.md`).
- An earlier comparison on an Apple M4 (24 GiB), before the fused attention kernel and the
  buffer pool, is in `benchmark/results/gpu-fair-2026-09-23/`; there the Metal backend was
  1.1-2x slower than MLX. `LayaMLX` matched Python on that machine up to 50 short questions;
  the CPU backend with Accelerate took 109 ms / 796 ms / 5366 ms for `short:1/10/50`.

NVIDIA RTX 3060 (12 GB, sm_86; CUDA.jl 6.4, CUDA 13.4) vs the CPU backend on a Xeon
E5-2699 v3 (18 cores, OpenBLAS, `-t auto`), Linux, `aac6fef/laya-mlx`, end-to-end p50 as
above (`benchmark/bench_julia.jl`, CUDA: 15 iterations after 3 warmups, CPU: 10 after 2):

| backend | short:1 | short:10 | short:50 | long:1 | long:10 |
|---|---:|---:|---:|---:|---:|
| `CPUBackend()` (Float32) | 522 ms | 2931 ms | 11575 ms | 1857 ms | 14411 ms |
| `CUDABackend()` (Float32) | 17 ms | 114 ms | 470 ms | 75 ms | 597 ms |
| `CUDABackend()` (Float16) | 11 ms | 49 ms | 212 ms | 34 ms | 291 ms |

In Float32 about three quarters of the GPU time is cuBLAS SGEMM at ~7.5 TFLOP/s (see
`docs/agents/cuda.md`).

See `benchmark/` for the method and the raw results.

## Repository layout

| path | contents |
|---|---|
| `src/` | the `Laya` package |
| `test/` | tests against the Python reference |
| `ext/` | package extensions: `LayaAppleAccelerateExt`, `LayaMetalExt` and `LayaCUDAExt` |
| `LayaMLX/` | MLX backend. Clang.jl-generated mlx-c bindings, forward pass and benchmark (see `LayaMLX/README.md`). |
| `reference/` | `LayaMLXReference`: a PythonCall bridge to `extern/laya-mlx`, used only for testing |
| `benchmark/` | Python, CPU, Metal, CUDA and MLX benchmarks (see `benchmark/README.md`) |
| `deps/` | the `mlx-c` submodule and `build.sh` |
| `extern/laya-mlx` | upstream laya-mlx (submodule): the Python reference for tests and benchmarks |

`SETUP.md` describes the development setup: the Python reference in `extern/laya-mlx`, the
mlx-c build and the checkpoints.

## Status

Not ported yet:

- the router, language detection, presets, email helpers and the shortlist

Known gaps:

- `MetalBackend`'s attention kernel is still about 2x slower than MLX's
  `scaled_dot_product_attention` in the global-attention layers on 512-token inputs (3.9 vs
  2.0 ms per layer at 10 questions; the local-attention layers are faster than MLX's, which
  do not skip the tiles outside the window); the softmax goes through threadgroup memory,
  MLX's stays in registers.

## Acknowledgements

- [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) is by Convai Innovations and the
  Laya contributors. The original weights are on Hugging Face, e.g.
  [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya).
- [laya-mlx](https://github.com/mizorewww/laya-mlx) (Apache-2.0), the MLX port Laya.jl follows,
  and its converted checkpoints ([aac6fef](https://huggingface.co/aac6fef)) are by the laya-mlx
  contributors.

Laya.jl is an independent port and is not affiliated with either project.
