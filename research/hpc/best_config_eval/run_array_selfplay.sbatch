#!/bin/bash
#SBATCH --job-name=hunger-bestcfg-selfplay
#SBATCH --partition=expansion
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/best_config_eval/slurm/%x-%A_%a.out
#SBATCH --error=hpc/best_config_eval/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

# Self-play survival + per-head diagnostics for the best-config campaign:
# one task per arm, 50 games per checkpoint.

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source hpc/best_config_2seed_cpu/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/best_config_2seed_cpu/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/best_config_2seed_cpu/campaign.env first}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

if [[ ! -x "${HUNGER_HPC_VENV}/bin/python" ]]; then
  echo "Missing HPC Python environment: ${HUNGER_HPC_VENV}" >&2
  exit 2
fi

arms=(small_90_mlp small_90_lstm default_180_mlp default_180_lstm)
variants=(small_90 small_90 default default)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
environment_variant="${variants[$SLURM_ARRAY_TASK_ID]}"

campaign_arm_root="${HUNGER_CAMPAIGN_ROOT}/${arm}"
output_dir="${HUNGER_CAMPAIGN_ROOT}/_selfplay_survival_50/${arm}"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-bestcfg-selfplay-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

cd "${HUNGER_RESEARCH_ROOT}"

"${HUNGER_HPC_VENV}/bin/python" \
  -m eval.checkpoint_selfplay_survival_diagnostics \
  --campaign-root "${campaign_arm_root}" \
  --output-dir "${output_dir}" \
  --environment-variant "${environment_variant}" \
  --algorithms ippo,mappo \
  --seeds "${HUNGER_EVAL_SEEDS:-0,1}" \
  --checkpoint-start 100 \
  --checkpoint-stop 1200 \
  --checkpoint-every 100 \
  --n-games 50 \
  --game-seed-base 970000
