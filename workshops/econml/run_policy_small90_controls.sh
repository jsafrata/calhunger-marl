#!/bin/bash
# E26 control: policy-only small_90 economies with full ledger (local).
# Same env variant + seed base as the LLM arena (small_90, 910000) so books
# and prices are directly comparable with what the LLM seats faced.
set -euo pipefail
cd "$(dirname "$0")"
# Paths are relative to this repository. Checkpoints come from the public
# checkpoint archive (see data/README.md): the small_90_mlp arm, iter_1200.
R=$(cd ../../research && pwd)
H=$R/hpc/econml_microstructure/checkpoint_policy_crossplay.py
CK=${CALHUNGER_CHECKPOINTS:-$R/../checkpoints}/best_config_small90_mlp
OUT=results/policy_small90_ledger
export HG_LEDGER=1 PYTHONPATH=$R HG_MAX_QTY=5 HG_MAX_PRICE=50 PYTHONHASHSEED=0
for p in ippo_seed0 ippo_seed2 mappo_seed1 mappo_seed3; do
  label=$(echo "$p" | sed 's/_seed/_s/')
  python3 "$H" --mode mixed \
    --seat "policy:${label}=$CK/${p}_headmlp/checkpoints/iter_1200" \
    --seat "policy:${label}=$CK/${p}_headmlp/checkpoints/iter_1200" \
    --seat "policy:${label}=$CK/${p}_headmlp/checkpoints/iter_1200" \
    --seat "policy:${label}=$CK/${p}_headmlp/checkpoints/iter_1200" \
    --environment-variant small_90 --n-games 100 --game-seed-base 910000 \
    --output "$OUT"
done
for p in ippo_seed0 ippo_seed2; do
  label=$(echo "$p" | sed 's/_seed/_s/')
  python3 "$H" --mode 1v3 \
    --focal "policy:${label}=$CK/${p}_headmlp/checkpoints/iter_1200" \
    --opponent scripted:v2 \
    --environment-variant small_90 --n-games 100 --game-seed-base 910000 \
    --output "$OUT"
done
echo ALL_CONTROLS_DONE
