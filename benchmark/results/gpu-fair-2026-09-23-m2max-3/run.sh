#!/usr/bin/env bash
# Fair GPU comparison on an Apple M2 Max (96 GB), third round (attention: 64 queries per
# threadgroup, O rescale in registers, window tile range): this branch ("new") against main
# (a git worktree at $MAIN) and Python laya-mlx, cool-down, order new main python python
# main new, one checkpoint per process.
set -euo pipefail
OUT=$(cd "$(dirname "$0")" && pwd)
NEW=$(cd "$OUT/../../.." && pwd)
MAIN=${MAIN:?set MAIN to a worktree of main with benchmark/ instantiated}
export HF_HUB_OFFLINE=1
C=(--model aac6fef/laya-mlx --dtype float32 --iterations 10 --warmup 2 --workloads short:1,short:10,short:50,long:1,long:10)
run() {
  local name=$1 idx=$2
  echo "== $(date +%T) $name run $idx"
  case $name in
    new)    (cd "$NEW" && julia --startup-file=no --project=benchmark benchmark/bench_julia.jl "${C[@]}" --backend metal --output "$OUT/run$idx-new.json") ;;
    main)   (cd "$MAIN" && julia --startup-file=no --project=benchmark benchmark/bench_julia.jl "${C[@]}" --backend metal --output "$OUT/run$idx-main.json") ;;
    python) (cd "$NEW" && extern/laya-mlx/.venv/bin/python benchmark/bench_python.py "${C[@]}" --output "$OUT/run$idx-python.json") ;;
  esac 2>&1 | grep -E "p50|Error|ERROR" || true
}
echo "== $(date +%T) initial cool-down 120 s"; sleep 120
i=0
for name in new main python python main new; do
  i=$((i+1))
  run $name $i
  echo "== $(date +%T) cool-down 60 s"; sleep 60
done
echo "== $(date +%T) done"
