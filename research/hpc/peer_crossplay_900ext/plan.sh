#!/bin/bash
# Dry-run print of the 100-task mapping (task_id -> focal, cell, mode, opponent).
# Pure bash mirror of the run_array.sbatch mapping; run anywhere, no env needed.
set -euo pipefail
labels=(ippo_s0 ippo_s1 ippo_s2 ippo_s3 ippo_s4 mappo_s0 mappo_s1 mappo_s2 mappo_s3 mappo_s4)
for id in $(seq 0 99); do
  focal="${labels[$((id / 10))]}"
  cell=$((id % 10))
  opps=()
  for l in "${labels[@]}"; do
    [[ "$l" == "$focal" ]] || opps+=("$l")
  done
  if (( cell == 0 )); then
    echo "task ${id}: focal=${focal} mode=mixed selfplay n=900 base=900100"
  else
    echo "task ${id}: focal=${focal} mode=1v3 opp=${opps[$((cell - 1))]} n=900 base=900100"
  fi
done
