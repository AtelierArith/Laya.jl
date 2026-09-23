# Benchmarks

Speed comparison between the pure-Julia runtime (`Laya`, CPU) and the Python `laya-mlx`
runtime (MLX, GPU) on the same checkpoint and workloads.

- Workloads follow `extern/laya-mlx/benchmarks/common.py`: `examples/state.json` with
  `examples/questions.json`; `short:N` asks N questions, `long:N` adds a 200×-repeated body
  so each sequence hits `max_len`. Defaults: `short:1,5,10,50` and `long:1,10`.
- Timing: wall clock, model loading excluded (GPU work synchronized on the Python side).
  - `prepare`: tokenization and prompt building.
  - `forward`: one collated batch of up to `batch_size` (default 64) questions.
  - `e2e`: full `predict` including calibration and result formatting.
- Backends: `python` (laya-mlx, MLX GPU), `julia` (Laya, CPU), `metal` (Laya, Metal.jl GPU)
  and `mlxc` (LayaMLX, MLX GPU via mlx-c); select with `BACKENDS` (default: `python julia mlxc`).
- Julia runs with `-t auto` (override with `JULIA_THREADS`); `Laya` loads AppleAccelerate,
  so BLAS calls go to Apple's Accelerate framework.
- Each backend/checkpoint runs in its own process, one after another, so only one model is
  resident at a time (~3 GiB for the 421M checkpoint in Julia Float32).

## Run

```bash
benchmark/run.sh                                  # aac6fef/laya-mlx, float32, all workloads
ITERATIONS=5 WARMUP=1 WORKLOADS=short:1,short:10 benchmark/run.sh
benchmark/run.sh aac6fef/laya-mlx aac6fef/laya-multilingual-mlx
```

Requirements: `extern/laya-mlx` with its venv, `deps/usr` (for `mlxc`) and the checkpoints
in the Hugging Face cache (see `../SETUP.md`); `run.sh` sets `HF_HUB_OFFLINE=1`.

GPU timings depend on the machine's thermal state. A backend that runs right after another
one can be up to 2× slower. Compare GPU backends from a cool machine, or run them again in
the reverse order.

Individual runs and the summary:

```bash
../extern/laya-mlx/.venv/bin/python benchmark/bench_python.py --model aac6fef/laya-mlx --output benchmark/results/py.json
julia --project=benchmark benchmark/bench_julia.jl --model aac6fef/laya-mlx --output benchmark/results/jl.json
julia --project=benchmark benchmark/compare.jl     # Markdown table of benchmark/results/*.json
```

`compare.jl` reports the Julia/Python e2e p50 ratio and whether the selected answers agree.
Results keep every timing sample.
