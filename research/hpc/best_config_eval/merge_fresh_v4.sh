#!/bin/bash
# Rebuild the two summary CSVs of the fresh-seed v4 re-evaluation
# (run_array_scripted_fresh_v4.sbatch) from the per-checkpoint raw cache:
# one harness pass per campaign root over the full checkpoint range. Every
# checkpoint is cached, so no games are played. Run after every array task
# has COMPLETED (check with sacct); a missing raw file would make the pass
# play that checkpoint's games on the login node, so check first.
set -euo pipefail

: "${HUNGER_RESEARCH_ROOT:?source hpc/best_config_2seed_cpu/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/best_config_2seed_cpu/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/best_config_2seed_cpu/campaign.env first}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

root_seeds01="${HUNGER_CAMPAIGN_ROOT}"
root_seeds234="${HUNGER_ROOT_SEEDS234:-/resnick/home/${USER}/hunger_runs/best_config_mlp_seeds234_ippo_mappo_1200}"
fresh_out="${HUNGER_FRESH_OUT:-/resnick/home/${USER}/hunger_runs/best_config_fresh_eval_v4_500}"
n_games="${HUNGER_FRESH_N_GAMES:-500}"
seed_base="${HUNGER_FRESH_SEED_BASE:-950000}"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export HUNGER_ENVIRONMENT_VARIANT="default"

cd "${HUNGER_RESEARCH_ROOT}"

merge_root() {
  local arm_root="$1" label="$2" seeds="$3"
  "${HUNGER_HPC_VENV}/bin/python" \
    -m eval.scaled_checkpoint_scripted_diagnostics \
    --campaign-root "${arm_root}/default_180_mlp" \
    --output-dir "${fresh_out}/_scripted_eval_fresh${n_games}/default_180_mlp_${label}/v4" \
    --algorithms ippo,mappo \
    --seeds "${seeds}" \
    --checkpoint-start 100 \
    --checkpoint-stop 1200 \
    --checkpoint-every 100 \
    --n-games "${n_games}" \
    --game-seed-base "${seed_base}" \
    --opponent v4
}

merge_root "${root_seeds01}" seeds01 "0,1"
merge_root "${root_seeds234}" seeds234 "2,3,4"
