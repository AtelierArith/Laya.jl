#!/usr/bin/env bash
# Fair GPU comparison on an Apple M2 Max (96 GB): cool-down, then ABBA order, one checkpoint
# per process. LayaMLX is left out (no mlx-c build on this machine).
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
    python) extern/laya-mlx/.venv/bin/python benchmark/bench_python.py "${C[@]}" --output "$OUT/run$idx-python.json" ;;
  esac 2>&1 | grep -E "p50|Error|ERROR" || true
}
echo "== $(date +%T) initial cool-down 120 s"; sleep 120
i=0
for name in metal python python metal; do
  i=$((i+1))
  run $name $i
  echo "== $(date +%T) cool-down 60 s"; sleep 60
done
echo "== $(date +%T) done"
