# Setup

What each part of the repository needs:

| to … | needs |
|---|---|
| use `Laya` (CPU inference) | Julia ≥ 1.10 only. Checkpoints are downloaded on first use. |
| run `Laya`'s tests or `reference/` | `extern/laya-mlx` with its venv, plus `LocalPreferences.toml` |
| use or test `LayaMLX` | the above, plus mlx-c built into `deps/usr` (Apple silicon only) |
| run `benchmark/` | everything above, plus the checkpoints in the Hugging Face cache |

`extern/` and `deps/usr/` are gitignored. `deps/mlx-c` is a git submodule.

## 1. Clone

```bash
git clone --recursive <this repo> Laya.jl && cd Laya.jl
# or, in an existing clone:
git submodule update --init deps/mlx-c
```

`deps/mlx-c` is pinned at `ebc88f1`, the mlx-c `main` commit that supports MLX v0.32.2.
No mlx-c release supports it yet: v0.6.0 targets MLX v0.31.1.

## 2. extern/laya-mlx (the Python reference)

The upstream Python implementation. It is the ground truth for the tests and a backend for the
benchmarks. Its MLX is also the `libmlx.dylib` that LayaMLX links against. Requires
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mizorewww/laya-mlx.git extern/laya-mlx
git -C extern/laya-mlx checkout 0a859518634112655cb97c745dbf04f5191aaf13   # version used for the port
cd extern/laya-mlx
uv sync --extra dev --extra reference --default-index https://pypi.org/simple
cd ../..
```

- The `reference` extra brings torch and transformers. `tiny_checkpoint` and the reference
  trace use them.
- Upstream's `uv.lock` points at a PyPI mirror (tuna.tsinghua). With `--default-index`, uv
  resolves against PyPI instead, so `uv.lock` shows local changes. Leave them uncommitted.
- Check that MLX is 0.32.x, because mlx-c is built for it:

  ```bash
  extern/laya-mlx/.venv/bin/python -c 'import mlx.core as mx; print(mx.__version__)'
  ```

## 3. Point PythonCall at the venv

The test environments use PythonCall with the venv above instead of CondaPkg. The
`LocalPreferences.toml` files are machine-specific and gitignored, so create them:

```bash
py="$PWD/extern/laya-mlx/.venv/bin/python"
for d in test reference reference/test LayaMLX/test; do
  printf '# Machine-specific: PythonCall uses the extern/laya-mlx venv (bypasses CondaPkg).\n[PythonCall]\nexe = "%s"\n' "$py" > "$d/LocalPreferences.toml"
done
```

## 4. deps/usr (mlx-c, for LayaMLX)

```bash
deps/build.sh          # CMake → deps/usr/{lib/libmlxc.dylib,include}
```

- mlx-c is built against the venv's MLX (`MLX_C_USE_SYSTEM_MLX=ON`).
- `libmlxc.dylib` finds `libmlx.dylib` through an absolute rpath into the venv. Re-run the
  script whenever the venv moves or its MLX version changes.
- After an mlx-c update, regenerate the bindings:
  `cd LayaMLX/gen && julia --project=. generator.jl`.

## 5. Julia environments

```bash
for p in . test reference reference/test LayaMLX LayaMLX/test LayaMLX/bench benchmark; do
  julia --project=$p -e 'using Pkg; Pkg.instantiate()'
done
```

## 6. Checkpoints (for tests and benchmarks)

`Laya.load` downloads missing repositories into its Scratch.jl space. The Python side,
`benchmark/run.sh` (which sets `HF_HUB_OFFLINE=1`) and the checkpoint tests read the Hugging
Face cache instead, so download there:

```bash
extern/laya-mlx/.venv/bin/hf download aac6fef/laya-mlx                 # 421M, English
extern/laya-mlx/.venv/bin/hf download aac6fef/laya-multilingual-mlx    # 322M
```

## Checks

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                  # Laya vs reference
julia --project=LayaMLX/test LayaMLX/test/runtests.jl         # LayaMLX ops + tiny
ITERATIONS=3 WARMUP=1 WORKLOADS=short:1,short:10 benchmark/run.sh
```

- **Memory**: load one checkpoint per process and run the jobs one after another (see
  `LayaMLX/README.md` for the RSS of each run).
- **GPU timings**: they depend on the machine's thermal state. Compare GPU backends from a
  cool machine, or alternate their order (see `benchmark/README.md`).
