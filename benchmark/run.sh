#!/usr/bin/env bash
# Run the benchmarks sequentially, one checkpoint per process, so that at most one model is
# resident at a time.
#
#   benchmark/run.sh [model ...]            (default: aac6fef/laya-mlx)
#   ITERATIONS=10 WARMUP=2 DTYPE=float32 WORKLOADS=short:1,short:10 benchmark/run.sh
#   BACKENDS="python julia mlxc"            (default: all three)
#   BLAS="accelerate openblas"              (Julia CPU BLAS backends; default: accelerate)
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=../extern/laya-mlx/.venv/bin/python
ITERATIONS=${ITERATIONS:-10}
WARMUP=${WARMUP:-2}
DTYPE=${DTYPE:-float32}
WORKLOADS=${WORKLOADS:-}
BACKENDS=${BACKENDS:-python julia mlxc}
export HF_HUB_OFFLINE=1
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(aac6fef/laya-mlx)
wl=(); [ -n "$WORKLOADS" ] && wl=(--workloads "$WORKLOADS")
common=(--dtype "$DTYPE" --iterations "$ITERATIONS" --warmup "$WARMUP" ${wl[@]+"${wl[@]}"})
for model in "${models[@]}"; do
  tag=$(echo "$model" | tr '/' '_')
  for backend in $BACKENDS; do
    case $backend in
      python)
        echo "== python-mlx-gpu $model $DTYPE"
        "$PYTHON" bench_python.py --model "$model" "${common[@]}" --output "results/${tag}-${DTYPE}-python-mlx-gpu.json" ;;
      julia)
        for blas in ${BLAS:-accelerate}; do
          echo "== julia-cpu-$blas $model $DTYPE"
          julia --startup-file=no -t "${JULIA_THREADS:-auto}" --project=. bench_julia.jl --model "$model" "${common[@]}" \
            --blas "$blas" --output "results/${tag}-${DTYPE}-julia-cpu-${blas}.json"
        done ;;
      mlxc)
        echo "== julia-mlxc-gpu $model $DTYPE"
        julia --startup-file=no --project=../LayaMLX/bench ../LayaMLX/bench.jl --model "$model" "${common[@]}" \
          --output "results/${tag}-${DTYPE}-julia-mlxc-gpu.json" ;;
      *) echo "unknown backend $backend" >&2; exit 1 ;;
    esac
  done
done
julia --startup-file=no --project=. compare.jl
