#!/bin/bash
# Gemini catch-up worker: grind a priority list of cells to n=100 each,
# with unbounded patience against 503 weather (cache-resume per game).
# Usage: gemini_worker.sh <cell:opp> [<cell:opp> ...]
#   where <opp> is "scripted:vN" or a policy label resolved below.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="$ROOT/research/.venv-mappo/bin/python"
OUT="$ROOT/workshops/results/llm_arena/campaign_90s"
CK="$ROOT/research/eval/results/best_config_small90_mlp"
SPEC="gemini_api/gemini-3.5-flash-lite"

opp_spec() {
  case "$1" in
    v2|v3|v4|v5) echo "scripted:$1" ;;
    ippo_s0)  echo "policy:ippo_s0=$CK/ippo_seed0_headmlp/checkpoints/iter_1200" ;;
    ippo_s2)  echo "policy:ippo_s2=$CK/ippo_seed2_headmlp/checkpoints/iter_1200" ;;
    mappo_s1) echo "policy:mappo_s1=$CK/mappo_seed1_headmlp/checkpoints/iter_1200" ;;
    mappo_s3) echo "policy:mappo_s3=$CK/mappo_seed3_headmlp/checkpoints/iter_1200" ;;
  esac
}

for OPP in "$@"; do
  NAME="gemini_vs_$OPP"
  for attempt in $(seq 1 200); do
    n=$(ls "$OUT/$NAME/games" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -ge 100 ]; then echo "=== $(date '+%F %T') $NAME COMPLETE ($n) ==="; break; fi
    echo "=== $(date '+%F %T') $NAME $n/100 attempt $attempt ==="
    "$PY" "$ROOT/workshops/llm_arena/run_llm_arena.py" \
      --mode 1v3 --focal "llm:gemini=$SPEC" --opponent "$(opp_spec "$OPP")" \
      --environment-variant small_90 --n-games 100 --game-seed-base 910000 \
      --http-attempts 6 --output "$OUT/$NAME" --quiet && { echo "=== $NAME COMPLETE ==="; break; }
    sleep 90
  done
done
echo "=== $(date '+%F %T') worker done: $* ==="
