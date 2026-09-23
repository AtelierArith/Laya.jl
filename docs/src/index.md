# Laya.jl

Laya.jl runs [Laya](https://github.com/NandhaKishorM/laya) typed-decision models in Julia.
It answers `choice`, `score` and `noul` questions about a state in a single encoder forward
pass, with no token-by-token generation.

It is a port of [laya-mlx](https://github.com/mizorewww/laya-mlx) (Python/MLX) and keeps the
same checkpoints, prompt format, calibration and output schema.

- The core package `Laya` is pure Julia and runs on the CPU on any OS. Its only dependencies
  are JSON, Scratch, PrecompileTools and the standard libraries.
- Apple Accelerate (CPU) and Metal.jl (GPU) are optional package extensions, and the MLX
  backend is the separate package `LayaMLX`. See [Backends](backends.md).

## Installation

Laya.jl is not registered yet:

```julia
using Pkg
Pkg.add(url="https://github.com/AtelierArith/Laya.jl")
```

## Quick start

```julia
using Laya

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

## Questions and results

`predict(agent, state, questions)` takes:

- **`state`**: a string, or a dictionary or vector that is serialized as JSON (compatible
  with Python's `json.dumps`, so prompts match upstream token for token).
- **`questions`**: a dictionary from question ids to definitions:

| `"type"` | `"criteria"` | answer fields |
|---|---|---|
| `"choice"` | option labels (a vector), or labels with descriptions (a dictionary) | `choice`, `probabilities` |
| `"score"` | ordered rubric levels (a vector) | `score` (expected level), `legend`, `probabilities` |
| `"noul"` | optional `false`/`true` descriptions (a dictionary) | `noul` (P(true)) |

Every definition also has `"instructions"`. Every answer has `type`, `confidence` and
`action.act_probability`. The result also holds `model` and `usage` (`input_tokens`,
`output_tokens`), as upstream Laya returns them.

Questions are batched `batch_size` at a time (an option of [`load`](@ref)); batching does not
change the answers.
