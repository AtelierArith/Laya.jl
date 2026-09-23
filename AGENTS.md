# AGENTS.md

Guidance for coding agents working in this repository. `README.md` covers what the project
is, and `SETUP.md` covers the development setup.

## Architecture rules

- **`Laya` (`src/`) is pure Julia and OS-independent.**
  - Hard dependencies are JSON, Scratch, Downloads, LinearAlgebra and PrecompileTools. Do not add Python,
    binary or platform-specific packages to `[deps]`.
  - Platform speedups are weak dependencies with package extensions in `ext/`, selected with
    `load(...; backend=...)` by dispatch on the backend type:
    - `AccelerateBackend()` → `LayaAppleAccelerateExt`. Laya only prints a hint; it never loads
      AppleAccelerate itself.
    - `MetalBackend()` (Metal.jl's own type) → `LayaMetalExt`.
  - The model code (`src/layers.jl`, `src/model.jl`) is generic over array types. Device
    backends move the weights with `adapt_arrays` and specialize hot spots by array type
    (`qkv_attention`, `LayerNorm`, `residual_norm`, `gelu_gate`, `release!`). `src/cpu.jl` does the same for
    `Array`. Host/device crossings go through `on_device_of`, `to_host` and `gather_columns`.
- **`LayaMetalExt` rules** (learned the hard way; details and minimal examples in
  `docs/agents/workarounds.md`):
  - `release!` never frees a device buffer: Metal.jl passes kernel arguments by GPU address,
    so freeing a buffer that queued work still reads is a use-after-free. Released buffers
    go to the pool (`pooled`, keyed by queue, element type and size) and are reused in
    queue order.
  - Matrix products read their output operand even with `beta = 0` (MPSGraph always,
    `MPSMatrixMultiplication` with `alpha != 1`), so a pool buffer must never hold NaN:
    new pool buffers are zeroed and the key includes the element type. Do not pass
    `alpha`; fold scales into an input.
  - Hand-encoded MPS kernels produced NaN about once in 100 forwards; the linear layers use
    Metal.jl's cached MPSGraph `graph_matmul!`.
  - Metal.jl's simdgroup and lane indices are 1-based; `Base.reshape` may return its
    argument, so never `unsafe_free!` a "view" without checking `===`.
  - After touching the extension, run the `backends` test group **and** a determinism
    stress (50+ identical forwards of `short:1`, `short:10`, `short:3`, `long:1`, `long:10`
    compared bitwise): the failures above were all transient and invisible to a single
    comparison against the reference.
  - Judge kernel changes by the full forward pass, not by microbenchmarks of one kernel:
    the GPU's clock state changes small-kernel timings 2-3×.
- **`LayaMLX/` is a separate package** that is not registered and uses a local dylib. It
  depends on `Laya`, never the other way round, and plugs in as `MLXBackend()`. It is not a
  weak dependency of `Laya`.
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

## Machine limits (Apple M4, 24 GiB; the README's numbers are from this machine)

- **One checkpoint per process.** Run model-loading jobs (tests on real checkpoints,
  benchmarks) one after another, never in parallel. The 421M model is about 3 GiB in Julia
  Float32 plus the Python reference in the same process. `LAYAMLX_TEST_REPOS` accepts one
  repository per process. Even on a machine with more memory, run GPU jobs one at a time:
  a concurrent job distorts every timing.
- **Ask before starting long or memory-heavy runs**, such as the full benchmark or tests on
  real checkpoints.
- **GPU timings depend on the machine's thermal state.** A backend that runs right after
  another can be up to 2× slower. Before concluding that one GPU backend is slower, run the
  backends again in the reverse order from a cool machine.

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                          # Laya (needs extern/ + LocalPreferences)
LAYA_TEST_BLAS=accelerate julia --project=. -e 'using Pkg; Pkg.test()' # same, with Accelerate BLAS
LAYA_TEST_GROUPS=aqua,smoke julia --project=. -e 'using Pkg; Pkg.test()' # what CI runs: no Python, no checkpoints
LAYA_TEST_METAL=1 LAYA_TEST_GROUPS=backends \
    julia --project=. -e 'using Pkg; Pkg.test()'                      # Metal backend vs reference
julia --project=LayaMLX/test LayaMLX/test/runtests.jl                  # LayaMLX ops + tiny checkpoint
LAYAMLX_TEST_REPOS=aac6fef/laya-mlx LAYAMLX_TEST_DTYPES=float32 \
    julia --project=LayaMLX/test LayaMLX/test/runtests.jl              # one real checkpoint
julia --project=reference -e 'using Pkg; Pkg.test()'                   # the Python bridge
benchmark/run.sh                                                       # see benchmark/README.md
BACKENDS=metal DTYPE=float16 benchmark/run.sh                          # Metal only
cd LayaMLX/gen && julia --project=. generator.jl                       # regenerate src/LibMLX.jl
deps/build.sh                                                          # rebuild mlx-c into deps/usr
julia --project=docs docs/make.jl                                      # Documenter.jl docs (docs/src)
```

Set `HF_HUB_OFFLINE=1` when the checkpoints are already cached, so nothing is downloaded by
accident.

CI (`.github/workflows/CI.yml`) runs only the `aqua` and `smoke` groups on Linux, macOS and
Windows. The comparisons against the Python reference (all other groups, Metal and LayaMLX)
must be run locally before pushing changes to the model, tokenizer or prompts.

## Code conventions

- **Array layout**: Julia arrays are column-major with axes reversed relative to NumPy/MLX.
  `(b, L, d)` in Python is `(d, L, b)` in Julia, in the same memory order. Token ids and
  marker positions stay 0-based, as in Python, wherever they cross the reference boundary.
- **`src/LibMLX.jl` is generated** by Clang.jl. Do not edit it by hand; change `LayaMLX/gen/`
  and regenerate.
- **mlx-c version**: `deps/mlx-c` is pinned at `ebc88f1`, which supports MLX v0.32.2. It must
  match the MLX in the `extern/laya-mlx` venv that `libmlxc.dylib` links against. If you move
  it, rebuild, regenerate the bindings and rerun the LayaMLX tests (they expect 0.0 error).
- **Aqua.jl**: `Laya` and `LayaMLX` pass `Aqua.test_all` (ambiguities, piracy, compat bounds,
  stale deps, …); keep it that way, e.g. give every new dependency a `[compat]` entry.
- **Precompilation**: `src/precompile.jl` runs `load` and `predict` on a tiny random checkpoint
  (byte-level and Metaspace BPE tokenizers) at precompile time. Extend it when adding code paths
  that every user hits, and re-measure time to first `predict`.
- **Style**: match the surrounding code. Keep comments short and only where they explain why,
  and keep docstrings on the public API.

## Repository hygiene

- `extern/laya-mlx` (upstream, pinned at `0a85951`) and `deps/mlx-c` are submodules. Move a
  pin only on purpose: the port and its tests follow that upstream version.
- `deps/usr/`, `Manifest.toml` and `LocalPreferences.toml` are gitignored.
  `LocalPreferences.toml` holds absolute paths for this machine.
- Do not commit absolute local paths. Benchmark JSON records only `basename`s for libraries.
- `extern/laya-mlx/uv.lock` has local changes (a PyPI index instead of a mirror). Leave them
  there and do not commit them.
