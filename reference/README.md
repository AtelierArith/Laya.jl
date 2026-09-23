# LayaMLXReference

Test-only bridge to the Python implementation in `../extern/laya-mlx`, used as ground
truth for the pure-Julia port in `../src`. Not a runtime dependency of Laya.

## Setup

```bash
cd ../extern/laya-mlx && uv sync --extra dev --extra reference
```

`LocalPreferences.toml` (and `test/LocalPreferences.toml`) point PythonCall's `exe` at
`../extern/laya-mlx/.venv/bin/python`; this is machine-specific and bypasses CondaPkg.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Conventions

- NumPy arrays of shape `(b, L, d)` come back as Julia arrays of size `(d, L, b)`
  (axes reversed, same memory order).
- Token ids and marker positions are Python's 0-based values.
- States and questions are sent as JSON (JSON.jl) and parsed on the Python side.
