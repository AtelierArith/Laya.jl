# Metal.jl / MPS pitfalls and workarounds

Bugs and surprising behaviour met while making `LayaMetalExt` as fast as MLX (issue #1),
with minimal examples. Environment: Metal.jl 1.11.1, GPUArrays 11.5, Julia 1.13.0,
macOS 26.5, Apple M2 Max. Run the examples with `julia --project=benchmark` (it has Metal).

## 1. `MPSMatrixMultiplication` reads its output operand when `alpha != 1`

With `beta = 0` one expects `C = alpha * A * B` regardless of what `C` holds. For
`alpha == 1` that is what happens; for any other `alpha` the kernel computes
`alpha * A * B + 0 * C` literally, so NaN (or Inf) garbage in `C` ends up in the result.
This bit us because intermediates come from a buffer pool and carry old contents.

```julia
using Metal
using Metal.MPS: MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication, encode!

# C = alpha * Aᵀ * B for column-major A (k, m), B (k, n), C (m, n), NaN in C beforehand.
function mps_product(alpha; m=3072, n=93, k=1024)
    A, B = MtlArray(randn(Float32, k, m)), MtlArray(randn(Float32, k, n))
    C = MtlArray(fill(NaN32, m, n))
    mat(x, rows, cols) = MPSMatrix(x, MPSMatrixDescriptor(cols, rows, 1, 4rows, 4rows * cols, Float32), 0)
    # MPS is row-major: Cᵀ = Bᵀ * A, i.e. left = B (transposed), right = A (transposed)
    kernel = MPSMatrixMultiplication(Metal.device(), false, true, n, m, k, alpha, 0)
    cmdbuf = Metal.MTLCommandBuffer(Metal.global_queue(Metal.device()))
    encode!(cmdbuf, kernel, mat(B, k, n), mat(A, k, m), mat(C, m, n))
    Metal.commit!(cmdbuf)
    count(isnan, Array(C))
end
mps_product(1.0)     # 0
mps_product(0.125)   # 285696 (= m * n): every element is NaN
```

Workaround: never pass `alpha != 1`; fold scale factors into an input (the attention scale
is applied to `q` while it is staged) or into a following kernel.

## 1b. MPSGraph's matmul reads its output operand too (any `alpha`, `beta = 0`)

Metal.jl's `graph_matmul!` (what `mul!` and `*` use) builds `alpha * A * B + beta * C` as a
graph and MPSGraph evaluates it literally, so garbage in `C` leaks into the result for every
`alpha`, in `Float32` and `Float16`:

```julia
using Metal
using Metal.MPSGraphs: graph_matmul!
A, B = MtlArray(randn(Float16, 1024, 3072)), MtlArray(randn(Float16, 1024, 93))
C = MtlArray(fill(Float16(NaN), 3072, 93))
graph_matmul!(C, A, B, true, false, 'T', 'N')      # C = Aᵀ B, beta = 0
count(isnan, Array(C))                              # 285696: every element
```

`*` gets away with it because `similar` returns a fresh buffer, whose contents happened to
be finite in our runs. With a buffer pool it bit at once in `Float16` (a pooled buffer that
was last used as `Float32` reads as `Float16` NaN in about 1 of 32 elements). Workaround in
`LayaMetalExt`: the pool key includes the element type, so a buffer is only ever reused by
arrays of the type that wrote it, and a buffer fresh from `Metal.alloc` is zero-filled once.

## 2. Kernel arguments are passed by GPU address; `unsafe_free!` while queued is a use-after-free

`@metal` encodes `MtlArray` arguments with `set_argument!` (their GPU address as bytes), not
`set_buffer!`, so the command buffer does not retain them. Metal.jl only keeps the *Julia
objects* alive (`BatchedCommandQueue.roots`) until the command buffer completes. An explicit
`Metal.unsafe_free!` releases the `MTLBuffer` immediately, and Metal may hand the memory to
the next allocation while the queued kernel has not run yet. With command batching (up to 32
operations per command buffer, 3 in flight) the window is tens of milliseconds.

The symptom in Laya was a transient NaN about once in 50 forwards: a freed attention mask
was overwritten by a host upload, every key of some query became masked and the softmax
computed `exp(-Inf - -Inf)`. The pattern:

```julia
using Metal
function read_kernel(y, x, n)
    i = Int32(Metal.thread_position_in_grid_1d())
    i <= n && (@inbounds y[i] = x[i])
    return
end
x = MtlArray(ones(Float32, 1 << 20))
y = MtlArray(zeros(Float32, 1 << 20))
@metal threads=256 groups=cld(1 << 20, 256) read_kernel(y, x, Int32(1 << 20))
Metal.unsafe_free!(x)                      # queued, not run: the buffer is released
z = MtlArray(fill(2.0f0, 1 << 20))         # may reuse x's memory, written from the host
Metal.synchronize()
count(!=(1.0f0), Array(y))                 # 0 when lucky; the copy of `z` may be visible
```

Whether the memory is reused depends on the driver, so this is not deterministic; in a full
model it was. Workaround (`LayaMetalExt`): `release!` never frees. Released buffers go to a
pool keyed by size and are reused by later operations on the same queue, which run in
submission order; the pool frees only when it exceeds its limit, after `Metal.synchronize()`.
The generic `MtlArray` finalizer path is safe: roots keep the object alive until completion.

## 3. Hand-encoded MPS kernels gave transient NaN (unresolved)

Encoding `MPSMatrixMultiplication` ourselves (with `alpha = 1`, the arrays and the
`MPSMatrix` wrappers recorded as roots, either in Metal.jl's batched command buffer or in a
command buffer of our own) still produced NaN in about one of 100 forwards, at iterations
that coincide with Julia GC runs. The same model with Metal.jl's MPSGraph path
(`Metal.MPSGraphs.graph_matmul!`) ran 250 forwards without a difference. The cause was not
found (suspects: lifetime of the ObjectiveC wrappers, autorelease pools). Workaround: use
`graph_matmul!`, whose graphs Metal.jl caches by shape, so its enqueue cost is acceptable.

## 4. Simdgroup and lane indices are 1-based

Like `thread_position_in_grid_1d()`, `thread_index_in_simdgroup()` and
`simdgroup_index_in_threadgroup()` return 1-based values (the Metal builtins are 0-based).
Code ported from MSL that adds 1 gets lane 33 and silently wrong reductions.

```julia
using Metal
function lanes(out)
    i = Int32(Metal.thread_position_in_grid_1d())
    @inbounds out[i] = Int32(Metal.thread_index_in_simdgroup())
    return
end
out = MtlArray(zeros(Int32, 32))
@metal threads=32 lanes(out)
Array(out)[1:3]    # [1, 2, 3], not [0, 1, 2]
```

`simd_shuffle_xor(v, Int16(mask))` itself behaves as documented (lane `i` gets lane
`i ⊻ mask`).

## 5. `simdgroup_load` transpose flag and column-major arrays

`simdgroup_load(A, (row, col), Val(true))` loads the 8x8 block of a column-major array at that
1-based origin as a regular matrix; `Val(false)` loads its transpose. With that,
`simdgroup_multiply_accumulate(a, b, c)` computes `a * b + c` and `simdgroup_store(m, C,
origin, Val(true))` writes it back in Julia's matrix semantics, also from 2-D
`MtlThreadGroupArray`s. (Checked numerically against `A[r:r+7, c:c+7] * B[...]`.)

## 6. Accumulator tuples and closures in kernels

Updating a tuple of simdgroup matrices with `ntuple(i -> mma(load(...), b, acc[i]), Val(4))`
inside a loop fails with `unsupported dynamic function invocation (call to getindex)`: the
closure captures `acc`, which is reassigned in the loop, so Julia boxes it. Pass the tuple
through a helper function instead (`acc = mma_k(acc, Kt, hb, q)` where the `ntuple` closure
captures only the helper's arguments).

## 7. Kernels defined at the top level of a script

A kernel defined in `Main` of a script (not inside a module) crashed LLVM's inliner during
GPU compilation (`CallAnalyzer::analyze` in `libLLVM.dylib`). Defining the same kernel inside
a `module` compiled fine. Not reduced further.

## 8. `Base.reshape` may return the array itself

Not a Metal.jl bug, but it cost model weights: `reshape(A, size(A)...)` returns `A`, so
"free the temporary view after use" must check `view === A` first. `LayaMetalExt.matmul!`
does that.

## 9. CPU load from other processes hits the Metal backend much harder than MLX

With a load average around 30 from unrelated jobs (a `cargo nextest` run, indexing), the
Python MLX numbers rose by about 40% while the Metal backend's forward times became erratic
(70 ms to 2.5 s for the same `long:1` batch, the same with the old and the fused attention).
Enqueuing a forward pass is CPU work in Julia (about 500 kernel and matmul launches), and a
starved main thread stalls the GPU queue. Benchmark on an idle machine: the fair-comparison
scripts wait for the load to drop and cool the GPU down between backends.

## 10. Microbenchmarks of small kernels are unreliable

Timings of a 20-100 µs kernel varied 2-3x between runs depending on the GPU's performance
state (an idle GPU clocks down; a run after heavy matmuls clocks up). Compare variants inside
a full forward pass, or after a warm-up that keeps the GPU busy.

## 11. Every host-to-device copy waits for the whole queue

`copyto!(::MtlArray, ::Array)` (so `MtlArray(x)`, `on_device_of`, `similar` + `copyto!`)
calls `Metal.synchronize()` first and then stages the data through a freshly allocated
shared buffer and a blit. Each upload therefore drains the GPU queue: a 93-element index
vector took 170 µs, a `(93, 93)` mask 460 µs, a `(512, 512, 10)` mask 2 ms, and a forward
pass did nine of them. `LayaMetalExt` uploads with a `memcpy` into pooled shared buffers
instead (`Laya.on_device_of(::MtlArray, x)`), which are reused only after a download has
waited for the queue (`Laya.to_host`), or after an explicit wait once 64 MB are pending.

```julia
using Metal
x = rand(Int32, 93)
@time MtlArray(x)            # ~170 µs, and the GPU queue is empty afterwards
```

## 12. `simdgroup_barrier` does not order a `simdgroup_store` before lane reads

In the attention kernel, a `simdgroup_store` of the O accumulator into this simdgroup's own
columns of threadgroup memory, followed by `simdgroup_barrier(MemoryFlagThreadGroup)`, then
per-lane reads and writes of those columns, then `simdgroup_load`, gave wrong results
(relative error 0.1-0.6, varying from run to run) although only the one simdgroup touches
that memory. The same sequence with `threadgroup_barrier` is correct. A small kernel that
does nothing else passes either way (and even without a barrier), so the failure needs the
surrounding load. Use `threadgroup_barrier` around simdgroup matrix stores and loads.

## 13. Kernels on a 1-D grid crash LLVM's inliner when defined in the extension

`function relu_kernel(y, x, n); i = Int32(thread_position_in_grid_1d()); i <= n && (@inbounds
y[i] = Laya.relu(x[i])); return; end` defined in `LayaMetalExt` crashes the GPU compilation
(`CallAnalyzer::analyze` in `libLLVM.dylib`, signal 11 or 10) at the first launch, also when
the body is `ifelse(v > 0, v, 0)` or `Laya.gelu(x[i])`, and also for a kernel that takes the
function as an argument. The same kernel in a module of a script compiles and runs. The
extension's other kernels, which read `thread_position_in_grid_2d/3d`, are fine, so the
elementwise kernels use a 2-D grid `(n, 1)`. Not reduced further (see also item 7).

## 14. `exp` is 1.7x slower than Metal's fast exp

Metal.jl lowers `exp(::Float32)` to `air.exp.f32`. `air.fast_exp.f32` (relative error about
1e-6, `fast_exp(-Inf) == 0`, `fast_exp(-88) == 0`, `fast_exp(NaN)` is NaN) runs 1.7x faster;
the fused attention kernel spends a sixth of its time in exp, so the softmax kernels call it
through `ccall("extern air.fast_exp.f32", llvmcall, Float32, (Float32,), x)`. MLX compiles
its kernels with fast math throughout.

## 15. MPSGraph matmuls commit a command buffer per call

`Metal.MPSGraphs.graph_matmul!` (behind `mul!` and `*`) creates and commits an
`MPSCommandBuffer` of its own for every product, and because a command buffer derived from
the queue flushes Metal.jl's kernel batch, a forward pass with 112 products ran in about 170
command buffers. `LayaMetalExt.graph_matmul_batched!` encodes the same cached graph into the
batched queue's open command buffer (`MPSCommandBuffer(Metal.ensure_cmdbuf!(bq))` after
`Metal.end_encoder!(bq)`, then `record_operation!`/`maybe_autoflush!`), which cut a
single-question forward pass from 23 to 21 ms. It uses Metal.jl internals
(`MatmulGraphKey`, `CachedMatmulGraph`, `_matmul_graph_cache`), pinned by the `[compat]`
bound on Metal.

## 16. Occupancy: registers and threadgroup memory

`pipeline.maxTotalThreadsPerThreadgroup` of a compiled kernel (`@metal launch=false`) tells
how many threads the register use allows per core: 512 for the attention kernel, and
variants that held a few more values (P fragments across a barrier, a diagonal rescale
matrix, `valid`/`window` mask logic) dropped to 448 or 384 and ran 25-35% slower at L=512,
B=10 although they did less work per tile. Threadgroup memory limits the same way (32 KB per
core: 12 KB allows 2 threadgroups, 20 KB one). Check both before judging a kernel change.

## 17. Reading and writing simdgroup matrix elements

Lane `l` (0-based) of a simdgroup holds elements `(r, c)` and `(r, c + 1)` of every 8x8
fragment in slots 1 and 2 of the `NTuple{64, VecElement}` (`m[1].value`, `m[2].value`), with
`r = ((l >> 1) & 3) + 4 ((l >> 4) & 1)` and `c = 2 (l & 1) + 4 ((l >> 3) & 1)` (1-based rows
and columns after `+ 1`); the other slots repeat them. Writing a fragment as
`ntuple(i -> i == 1 ? VecElement(a) : i == 2 ? VecElement(b) : m[i], Val(64))` and storing
it works (the layout MLX's `thread_elements()` relies on). A row reduction of a column is a
shuffle-xor over lane bits 1, 2 and 4. An attention kernel that ran the whole softmax on
fragments this way was correct but slower than the shipped one (item 16).
