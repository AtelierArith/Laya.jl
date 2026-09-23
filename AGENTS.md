# AGENTS.md

Guidance for coding agents working in this repository. `README.md` covers what the project
is, and `SETUP.md` covers the development setup.

## Architecture rules

- **`Laya` (`src/`) is pure Julia and OS-independent.**
  - Hard dependencies are JSON, Scratch, Downloads and LinearAlgebra. Do not add Python,
    binary or platform-specific packages to `[deps]`.
  - Platform speedups are weak dependencies or package extensions:
    - `AppleAccelerate` is already a weak dependency. Laya only prints a hint; it never loads
      it.
    - A planned `LayaMetalExt` will use Metal.jl.
- **`LayaMLX/` is a separate package** that is not registered and uses a local dylib. It
  depends on `Laya`, never the other way round. It is not a weak dependency of `Laya`.
- **PythonCall appears only in `reference/` (`LayaMLXReference`) and in the test
  environments.**
- **The tokenizer stays pure Julia.**
- **Stay faithful to upstream.** Behaviour follows `extern/laya-mlx/laya_mlx/`: prompt
  format, `json.dumps`-compatible serialization, calibration, output schema and MLX-exact
  math (for example `mlx_erf`). When porting, read the Python source and mirror it. Do not
  redesign.
- **Model files**: `Laya` reads an existing Hugging Face cache and otherwise downloads into its
  Scratch.jl space. Prefer Julia-ecosystem conventions (Scratch.jl, Preferences.jl, stdlib
  Downloads) over copying Python's layouts.

## Machine limits (Apple M4, 24 GiB)

- **One checkpoint per process.** Run model-loading jobs (tests on real checkpoints,
  benchmarks) one after another, never in parallel. The 421M model is about 3 GiB in Julia
  Float32 plus the Python reference in the same process. `LAYAMLX_TEST_REPOS` accepts one
  repository per process.
- **Ask before starting long or memory-heavy runs**, such as the full benchmark or tests on
  real checkpoints.
- **GPU timings depend on the machine's thermal state.** A backend that runs right after
  another can be up to 2× slower. Before concluding that one GPU backend is slower, run the
  backends again in the reverse order from a cool machine.

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                          # Laya (needs extern/ + LocalPreferences)
LAYA_TEST_BLAS=accelerate julia --project=. -e 'using Pkg; Pkg.test()' # same, with Accelerate BLAS
julia --project=LayaMLX/test LayaMLX/test/runtests.jl                  # LayaMLX ops + tiny checkpoint
LAYAMLX_TEST_REPOS=aac6fef/laya-mlx LAYAMLX_TEST_DTYPES=float32 \
    julia --project=LayaMLX/test LayaMLX/test/runtests.jl              # one real checkpoint
julia --project=reference -e 'using Pkg; Pkg.test()'                   # the Python bridge
benchmark/run.sh                                                       # see benchmark/README.md
cd LayaMLX/gen && julia --project=. generator.jl                       # regenerate src/LibMLX.jl
deps/build.sh                                                          # rebuild mlx-c into deps/usr
```

Set `HF_HUB_OFFLINE=1` when the checkpoints are already cached, so nothing is downloaded by
accident.

## Code conventions

- **Array layout**: Julia arrays are column-major with axes reversed relative to NumPy/MLX.
  `(b, L, d)` in Python is `(d, L, b)` in Julia, in the same memory order. Token ids and
  marker positions stay 0-based, as in Python, wherever they cross the reference boundary.
- **`src/LibMLX.jl` is generated** by Clang.jl. Do not edit it by hand; change `LayaMLX/gen/`
  and regenerate.
- **mlx-c version**: `deps/mlx-c` is pinned at `ebc88f1`, which supports MLX v0.32.2. It must
  match the MLX in the `extern/laya-mlx` venv that `libmlxc.dylib` links against. If you move
  it, rebuild, regenerate the bindings and rerun the LayaMLX tests (they expect 0.0 error).
- **Style**: match the surrounding code. Keep comments short and only where they explain why,
  and keep docstrings on the public API.

## Repository hygiene

- `extern/`, `deps/usr/`, `Manifest.toml` and `LocalPreferences.toml` are gitignored.
  `LocalPreferences.toml` holds absolute paths for this machine.
- Do not commit absolute local paths. Benchmark JSON records only `basename`s for libraries.
- `extern/laya-mlx/uv.lock` has local changes (a PyPI index instead of a mirror). Leave them
  there and do not commit them.
