#!/bin/bash
# One model's ladder lane: 4 cells x 100 games vs scripted v2-v5 at 90s.
# Usage: run_ladder_lane.sh <label> <provider/model>
# Cells run sequentially (per-game caching makes reruns resume).
set -u
LABEL="$1"; SPEC="$2"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="$ROOT/research/.venv-mappo/bin/python"
OUT="$ROOT/workshops/results/llm_arena/campaign_90s"
for BOT in v2 v3 v4 v5; do
  echo "=== $(date '+%F %T') $LABEL vs $BOT ==="
  "$PY" "$ROOT/workshops/llm_arena/run_llm_arena.py" \
    --mode 1v3 \
    --focal "llm:$LABEL=$SPEC" \
    --opponent "scripted:$BOT" \
    --environment-variant small_90 \
    --n-games 100 \
    --game-seed-base 910000 \
    --output "$OUT/${LABEL}_vs_${BOT}" \
    --quiet
  STATUS=$?
  echo "=== $LABEL vs $BOT exit=$STATUS ==="
  if [ $STATUS -ne 0 ]; then
    echo "cell failed; retrying once after 120s (cache resumes)"
    sleep 120
    "$PY" "$ROOT/workshops/llm_arena/run_llm_arena.py" \
      --mode 1v3 \
      --focal "llm:$LABEL=$SPEC" \
      --opponent "scripted:$BOT" \
      --environment-variant small_90 \
      --n-games 100 \
      --game-seed-base 910000 \
      --output "$OUT/${LABEL}_vs_${BOT}" \
      --quiet || echo "cell $LABEL vs $BOT FAILED TWICE; moving on"
  fi
done
echo "=== $(date '+%F %T') lane $LABEL complete ==="
