#!/bin/bash
# One model's policy lane: 4 cells x 100 games vs the selected small_90
# best-config seeds (ippo s0/s2, mappo s1/s3, all @1200).
# Waits for the model's ladder lane to finish first so each provider
# serves one game stream at a time; each cell retries with cache-resume.
# Usage: run_policy_lane.sh <label> <provider/model>
set -u
LABEL="$1"; SPEC="$2"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="$ROOT/research/.venv-mappo/bin/python"
OUT="$ROOT/workshops/results/llm_arena/campaign_90s"
CK="$ROOT/research/eval/results/best_config_small90_mlp"
LADDER_LOG="$OUT/logs/lane_${LABEL}.log"

until grep -q "lane $LABEL complete" "$LADDER_LOG" 2>/dev/null; do sleep 300; done
echo "=== $(date '+%F %T') ladder lane done; starting policy lane $LABEL ==="

run_cell() {
  local OPP_LABEL="$1" OPP_PATH="$2"
  for attempt in $(seq 1 10); do
    echo "=== $(date '+%F %T') $LABEL vs $OPP_LABEL (attempt $attempt) ==="
    "$PY" "$ROOT/workshops/llm_arena/run_llm_arena.py" \
      --mode 1v3 \
      --focal "llm:$LABEL=$SPEC" \
      --opponent "policy:$OPP_LABEL=$OPP_PATH" \
      --environment-variant small_90 \
      --n-games 100 \
      --game-seed-base 910000 \
      --http-attempts 6 \
      --output "$OUT/${LABEL}_vs_${OPP_LABEL}" \
      --quiet && return 0
    echo "=== cell failed; resuming from cache in 180s ==="
    sleep 180
  done
  echo "cell $LABEL vs $OPP_LABEL FAILED after 10 attempts; moving on"
}

run_cell ippo_s0  "$CK/ippo_seed0_headmlp/checkpoints/iter_1200"
run_cell ippo_s2  "$CK/ippo_seed2_headmlp/checkpoints/iter_1200"
run_cell mappo_s1 "$CK/mappo_seed1_headmlp/checkpoints/iter_1200"
run_cell mappo_s3 "$CK/mappo_seed3_headmlp/checkpoints/iter_1200"
echo "=== $(date '+%F %T') policy lane $LABEL complete ==="
