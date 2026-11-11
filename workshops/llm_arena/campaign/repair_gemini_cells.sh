#!/bin/bash
# Fill in incomplete gemini cells after its policy lane finishes.
# Loops each cell until 100 games are cached (503 spells just delay it).
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="$ROOT/research/.venv-mappo/bin/python"
OUT="$ROOT/workshops/results/llm_arena/campaign_90s"
CK="$ROOT/research/eval/results/best_config_small90_mlp"
SPEC="gemini_api/gemini-3.5-flash-lite"

until grep -q "policy lane gemini complete" "$OUT/logs/policylane_gemini.log" 2>/dev/null; do sleep 600; done
echo "=== $(date '+%F %T') starting gemini repair sweep ==="

repair() {
  local NAME="$1"; shift
  for attempt in $(seq 1 30); do
    n=$(ls "$OUT/$NAME/games" 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -ge 100 ] && { echo "cell $NAME complete ($n games)"; return 0; }
    echo "=== $(date '+%F %T') repair $NAME ($n/100, attempt $attempt) ==="
    "$PY" "$ROOT/workshops/llm_arena/run_llm_arena.py" \
      --mode 1v3 --focal "llm:gemini=$SPEC" "$@" \
      --environment-variant small_90 --n-games 100 --game-seed-base 910000 \
      --http-attempts 6 --output "$OUT/$NAME" --quiet && return 0
    sleep 300
  done
  echo "cell $NAME STILL INCOMPLETE after 30 repair attempts"
}

repair gemini_vs_v2 --opponent scripted:v2
repair gemini_vs_v3 --opponent scripted:v3
repair gemini_vs_v4 --opponent scripted:v4
repair gemini_vs_v5 --opponent scripted:v5
repair gemini_vs_ippo_s0  --opponent "policy:ippo_s0=$CK/ippo_seed0_headmlp/checkpoints/iter_1200"
repair gemini_vs_ippo_s2  --opponent "policy:ippo_s2=$CK/ippo_seed2_headmlp/checkpoints/iter_1200"
repair gemini_vs_mappo_s1 --opponent "policy:mappo_s1=$CK/mappo_seed1_headmlp/checkpoints/iter_1200"
repair gemini_vs_mappo_s3 --opponent "policy:mappo_s3=$CK/mappo_seed3_headmlp/checkpoints/iter_1200"
echo "=== $(date '+%F %T') gemini repair sweep complete ==="
