# Development

The repository holds more than the `Laya` package:

| path | contents |
|---|---|
| `src/`, `ext/` | the `Laya` package and its extensions (`LayaAppleAccelerateExt`, `LayaMetalExt`) |
| `test/` | tests; most compare against the Python reference |
| `LayaMLX/` | the MLX backend package |
| `reference/` | `LayaMLXReference`, a PythonCall bridge to the Python implementation, for tests |
| `benchmark/` | Python, CPU, Metal and MLX benchmarks |
| `extern/laya-mlx`, `deps/mlx-c` | submodules: upstream laya-mlx and mlx-c |

`SETUP.md` in the repository describes the full setup (the Python reference, the mlx-c build
and the checkpoints), and `AGENTS.md` the conventions.

## Tests

```bash
LAYA_TEST_GROUPS=aqua,smoke julia --project=. -e 'using Pkg; Pkg.test()'   # no Python, no checkpoints (CI)
julia --project=. -e 'using Pkg; Pkg.test()'                               # everything, against the reference
LAYA_TEST_METAL=1 LAYA_TEST_GROUPS=backends julia --project=. -e 'using Pkg; Pkg.test()'
```

CI runs the `aqua` (Aqua.jl) and `smoke` groups on Linux, macOS and Windows.

## Documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```
