# CUDA backend notes (`LayaCUDAExt`)

Measured on 2x RTX 3060 (sm_86, 12 GB), CUDA.jl 6.4, CUDA runtime 13.4, Julia 1.13.

## Rules

- `release!` may free: CUDA.jl's `unsafe_free!` is stream-ordered, so a released buffer is
  only reused by work queued after the work that reads it. Pool buffers (`pooled`, keyed by
  stream and size) go back to the pool, others to CUDA's allocator.
- Never upload with `CuArray(x)` / `copyto!` in the forward pass: a copy from pageable host
  memory waits for the whole stream (measured: an upload queued behind 10 ms of work
  returned after 10 ms). `Laya.on_device_of` copies into pinned staging memory first;
  `Laya.to_host` synchronizes and recycles the staging space.
- cuBLAS runs in device pointer mode in CUDA.jl: a host `alpha`/`beta` becomes a fresh device
  scalar plus an upload per call, so `gemm!` passes pointers into a constant `[1, 0]` array.
- cuBLAS with `beta = 0` does not read `C`; pool buffers need no zeroing (tested with NaN
  garbage).
- Kernel closures: a closure that captures a variable reassigned in a loop boxes it and
  fails to compile (`unsupported dynamic function invocation`); wrap in `let x = x`. Mark
  `ntuple` closures `@inline` (`x -> (@inline; ...)`), otherwise they become real calls with
  their arguments in local memory.
- Model code: every temporary must be `release!`d (including the parents of `reshape`
  views, see `Laya.reshaped`), or the pool misses buffers until the next GC.
- After touching the extension: `LAYA_TEST_GROUPS=cuda` and the determinism stress (50+
  identical forwards of `short:1/3/10`, `long:1/10`, compared bitwise; 0 differences and 0
  pool misses as of this writing).

## Performance (what helped, what did not)

- Float32 linear layers are cuBLAS SGEMM at ~7.5 TFLOP/s, about 75% of the forward. NN
  weights (stored transposed) are 9-14% faster than TN at 93 and >4000 tokens: kept.
  TF32 would give ~1.5x on the GEMMs but breaks the 1e-5 agreement: opt-in only through
  `CUDA.math_mode!(CUDA.FAST_MATH)`.
- Attention: strided-batched cuBLAS + softmax kernel costs ~117 ms of a 610 ms `long:10`
  forward (S matrices, softmax passes, split/merge copies). The fused FP32 kernel won only
  after its shared memory went from 34.8 KB to 32 KB (XOR-swizzled float4 groups instead of
  padding), which fits three blocks per SM: 7.6 -> 5.1 ms per (global + local) layer pair at
  `long:10`. Manual unrolling of its inner loops did not help (ptxas unrolls). Kept: -8%
  on `long:10`, -2..3% on `short:10/50`.
- Elementwise/LayerNorm kernels run at 235-320 GB/s (of 360); together ~6% of the forward.
