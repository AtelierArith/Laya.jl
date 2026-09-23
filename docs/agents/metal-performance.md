# How the Metal backend got to MLX speed

What made `MetalBackend()` (`ext/LayaMetalExt.jl`) as fast as Python laya-mlx, in the order
it was found, with the numbers that justified each step. The pitfalls met on the way are in
`workarounds.md`; this file is about what to do and how to measure. Machine: Apple M2 Max
(38 GPU cores, 96 GB), Metal.jl 1.11.1, `aac6fef/laya-mlx`, float32.

## Where it ended

End-to-end p50 (`benchmark/results/gpu-fair-2026-09-23-m2max-2/`, fair ABBA order on an idle
machine):

| | short:1 | short:10 | short:50 | long:1 | long:10 |
|---|---:|---:|---:|---:|---:|
| Python MLX | 16.9 ms | 93.7 ms | 410 ms | 55.5 ms | 481 ms |
| Metal, start (M4, before this work) | 62 | 339 | 1937 | 245 | 4890 |
| Metal, after PR #2 | 23 | 97 | 402 | 66 | 501 |
| Metal, now | 16.7 | 89.1 | 386 | 55.7 | 475 |

## How to measure

1. **Full forward first, kernels second.** `scratchpad`-style scripts that time
   `agent.model(batch)` (see `benchmark/bench_julia.jl` and the `prof.jl` pattern: 5 warm-ups,
   20 iterations, `Metal.synchronize()` around the timed region) decide whether a change
   stays. Microbenchmarks of one kernel move 2-3x with the GPU's clock state.
2. **Split the forward into stages** with `Metal.synchronize()` between them (embedding,
   the 28 layers, each head layer, gathers, downloads). Each synchronized stage carries about
   0.25 ms of wait, so the stage sum exceeds the real forward; the ranking is what matters.
   That split is what showed the tail (head layers, gathers, uploads, downloads) costing
   5 ms of a 23 ms single-question forward.
3. **Per-op microbenchmarks against MLX** (`ops4.jl` vs `mlxops.py` patterns: 100 launches
   back to back) only to compare the same op on both sides. They showed the matmuls were
   already faster than MLX's at B=1 (364 vs 443 µs per layer) and the attention kernel 10x
   slower (125 vs 12 µs), which redirected the work.
4. **`Metal.@profile`** gives per-kernel GPU time and the number of command buffers, but not
   MPSGraph work, and it disables command batching, so use it for counts and ratios only.
5. **Count allocations and command buffers.** `Metal.alloc_stats` before and after a forward
   (`allocs/fwd`), and the command-buffer count from `@profile`. Both went from hundreds to a
   dozen; each hundred cost milliseconds.
6. **Idle machine, one GPU job at a time, cool-down between backends** (`AGENTS.md`). Other
   CPU load hits the Metal backend far harder than MLX because enqueuing is Julia CPU work.
7. **After every kernel change:** the `backends` test group and the determinism stress
   (50 identical forwards per workload, bitwise comparison). All the NaN and ordering bugs
   were transient and invisible to one comparison against the reference.

## What worked, in order of payoff

### 1. Buffer pool instead of allocating per op (long inputs: 2-5x)

Metal.jl has no caching allocator; a long:10 forward allocated 22 GiB in 526 buffers, and
repeated forwards in one process got slower and erratic. The pool (`pooled`, keyed by queue,
element type and rounded size) hands released buffers to the next allocation of the same
size. Two rules keep it correct: `release!` never frees (kernel arguments are GPU addresses,
so a freed buffer that queued work still reads is a use-after-free), and pool buffers are
zeroed once and typed, because MPSGraph's matmul reads its output operand even with
`beta = 0`. A warm forward pass now allocates no device memory for intermediates.

### 2. Fused attention on the simdgroup matrix units (long inputs: attention 19 -> 6 ms/layer)

One kernel per (32 queries, head, sequence): Q staged once and kept as fragments, K and V
tiles of 32 keys staged in 8 KB of threadgroup memory (RoPE and the scale applied on the
way in), `S = KᵀQ` and `O += V P` as 8x8 `simdgroup_multiply_accumulate`, an online softmax
over S in a 4 KB threadgroup buffer, output written directly in the layout of the output
projection. No S or P in device memory (they were 168 MB per layer at long:10), and tiles
whose mask is all false are skipped, which makes the local-attention layers (2/3 of the
model) 3x cheaper. A scalar version of the same idea was 2-5x slower than batched MPS
matmuls; the matrix units are what make it pay.

### 3. Simdgroup reductions for LayerNorm and softmax (short inputs)

One simdgroup per column with `simd_shuffle_xor` instead of a 256-thread threadgroup tree
with barriers: a single question has only 93 columns, so what matters is latency per column,
not parallelism across them. The residual add is fused into the following LayerNorm
(`residual_norm`), across layers, and the LayerNorm reads its column once into registers.

### 4. Matmuls encoded into the batched command buffer (single question: 23 -> 21 ms)

`Metal.MPSGraphs.graph_matmul!` commits a command buffer per product and flushes Metal.jl's
kernel batch with it: 112 products became about 170 command buffers per forward, each
with submission latency and a GPU idle gap. `graph_matmul_batched!` encodes the same cached
graph into the batched queue's open command buffer, so a forward pass is a handful of command
buffers. Hand-encoded `MPSMatrixMultiplication` would have been cheaper still but produced
NaN once in about 100 forwards, so it is not used.

### 5. Uploads without waiting for the GPU (single question: 21 -> 17 ms)

Every host-to-device copy in Metal.jl calls `synchronize()` and stages through a fresh
buffer; a forward pass did nine of them (token ids, three masks, indices for two gathers,
pooled features), each draining the queue. `Laya.on_device_of(::MtlArray, x)` now memcpys
into a pooled shared buffer; released upload buffers wait until a download (which waits for
the queue anyway) recycles them. This was the single largest item of the second round.

### 6. Kernels for the tail, downloads reordered (single question, with 5: 21 -> 17 ms)

The decision head's small steps (type-embedding add, gathers of marker columns and column
1, ReLU, GELU, residual adds) were GPUArrays broadcasts and `getindex`, each allocating a
fresh buffer. They are now kernels into pooled memory behind generic hooks
(`elementwise`, `add_columns`, `columns_at`, `gather_columns`, `residual`), dtype
conversions happen on the host after the download, and the column-1 gather is issued
before the first download so the second download finds the queue idle. Allocations per
forward went from 44 to 14.

### 7. Fast exp (attention kernel 16% at long:10)

`air.fast_exp.f32` is 1.7x faster than `air.exp.f32` at 1e-6 relative error, and the softmax
kernels spend a sixth of their time in exp. MLX compiles everything with fast math.

## What did not work

- **Register-resident softmax (MLX style).** Reading and writing the two elements each lane
  holds of a fragment works (layout in `workarounds.md` item 17) and gives a correct kernel
  with no S/P round trip and two barriers per tile instead of nine. It was 25-35% slower at
  L=512, B=10: the extra live values dropped `maxTotalThreadsPerThreadgroup` from 512 to
  384-448, i.e. fewer threadgroups per core, and this kernel is occupancy-bound. It was
  faster only at L=93, B=1 (latency-bound). The same happened to every variant that held a
  little more in registers (P fragments across a barrier, a diagonal rescale matrix, the
  `valid`/`window` mask arithmetic). Check the register budget before the barrier count.
- **Loading V fragments straight from device memory** (no staging): slower, same reason.
- **`simdgroup_barrier` around simdgroup matrix stores**: wrong results (item 12).
- **Dropping the loop-end barrier** of the attention kernel: no gain.
- **Larger threadgroup memory** (20 KB): 1.7x slower, one threadgroup per core.
- **Hand-encoded MPS matmuls**: transient NaN (item 3); **MPSGraph alpha**: reads garbage
  from the output (item 1b).
- **Raising `JULIA_METAL_COMMAND_BATCHING_OPS`**: no effect (the flushes came from the
  matmuls' own command buffers, see 4).

## What is left

- The attention kernel is still about 3x MLX's `scaled_dot_product_attention` at L=512,
  B=10 (6.2 vs 2.0 ms per layer), hidden behind the matmuls in the totals. The remaining
  ideas all need more registers or memory per threadgroup, so they need a different tiling
  (e.g. 64 queries per threadgroup with the O rescale done in halves) to keep occupancy.
- Two downloads per forward remain; the action head needs host-side features between them.
- MPSGraph enqueue is still the CPU-side cost per product; a custom skinny GEMM for N <= 128
  would be the next step for single-question latency, but the matmuls are already ahead of
  MLX there.
