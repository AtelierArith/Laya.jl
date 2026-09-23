# Accuracy and speed

Every backend is checked against the Python implementation (`test/runtests.jl` and
`LayaMLX/test/runtests.jl` in the repository):

- **CPU**: every answer matches, and the relative error is below the noise MLX shows between
  its own float32 GPU and CPU results.
- **Metal**: the selected answers match. The relative error of the logits is about 1e-6 in
  float32 and about 2e-3 in float16.
- **LayaMLX**: bit-identical (0.0 error) in float32 and float16, on both checkpoints.

Timings (all backends measured under the same conditions) are in the repository's `README.md`; the method and raw results are in `benchmark/`.

Time to first `predict` is small because `load` and `predict` are exercised at precompile
time on a tiny random checkpoint (PrecompileTools): on the CPU, the first `predict` of the
421M checkpoint takes about 0.5 s against 0.4 s for later calls.
