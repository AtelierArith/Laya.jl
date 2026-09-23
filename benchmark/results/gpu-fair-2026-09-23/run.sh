#!/usr/bin/env bash
# Fair GPU comparison: cool-down, then ABCCBA order, one checkpoint per process.
set -euo pipefail
OUT=$(cd "$(dirname "$0")" && pwd)
cd "$OUT/../../.."
export HF_HUB_OFFLINE=1
C=(--model aac6fef/laya-mlx --dtype float32 --iterations 10 --warmup 2 --workloads short:1,short:10,short:50,long:1,long:10)
run() {
  local name=$1 idx=$2
  echo "== $(date +%T) $name run $idx"
  case $name in
    metal)  julia --startup-file=no --project=benchmark benchmark/bench_julia.jl "${C[@]}" --backend metal --output "$OUT/run$idx-metal.json" ;;
    mlxc)   julia --startup-file=no --project=LayaMLX/bench LayaMLX/bench.jl "${C[@]}" --output "$OUT/run$idx-mlxc.json" ;;
    python) extern/laya-mlx/.venv/bin/python benchmark/bench_python.py "${C[@]}" --output "$OUT/run$idx-python.json" ;;
  esac 2>&1 | grep -E "p50|Error|ERROR" || true
}
echo "== $(date +%T) initial cool-down 180 s"; sleep 180
i=0
for name in metal mlxc python python mlxc metal; do
  i=$((i+1))
  run $name $i
  echo "== $(date +%T) cool-down 90 s"; sleep 90
done
echo "== $(date +%T) done"
