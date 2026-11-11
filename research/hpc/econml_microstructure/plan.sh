#!/bin/bash
# Dry-run print of the 32-task mapping. Pure bash mirror of
# run_array.sbatch; run anywhere, no env needed.
set -euo pipefail
labels=(ippo_s0 ippo_s1 ippo_s2 ippo_s3 ippo_s4 mappo_s0 mappo_s1 mappo_s2 mappo_s3 mappo_s4)
pairs=(
  "ippo_s2 ippo_s4"  "ippo_s4 ippo_s2"
  "mappo_s2 mappo_s1" "mappo_s1 mappo_s2"
  "ippo_s0 mappo_s4"  "mappo_s4 ippo_s0"
  "ippo_s3 mappo_s0"  "mappo_s0 ippo_s3"
  "ippo_s0 ippo_s1"   "ippo_s1 ippo_s0"
  "mappo_s2 mappo_s3" "mappo_s3 mappo_s2"
  "ippo_s2 mappo_s2"  "mappo_s2 ippo_s2"
  "ippo_s1 mappo_s3"  "mappo_s3 ippo_s1"
)
mixpairs=(
  "ippo_s2 ippo_s4"
  "mappo_s2 mappo_s1"
  "ippo_s0 mappo_s4"
  "ippo_s3 mappo_s0"
  "ippo_s2 mappo_s2"
  "ippo_s1 mappo_s3"
)
for id in $(seq 0 31); do
  if (( id <= 9 )); then
    echo "task ${id}: selfplay ${labels[$id]} n=100 base=900000 ledger=1"
  elif (( id <= 25 )); then
    read -r focal opp <<< "${pairs[$((id - 10))]}"
    echo "task ${id}: 1v3 focal=${focal} opp=${opp} n=100 base=900000 ledger=1"
  else
    read -r a b <<< "${mixpairs[$((id - 26))]}"
    echo "task ${id}: mixed22 ${a}+${b} n=100 base=900000 ledger=1"
  fi
done
