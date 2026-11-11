#!/bin/bash
#SBATCH --job-name=hunger-headdiag-eval
#SBATCH --partition=expansion
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/episode_matched_scripted_eval/slurm/%x-%A_%a.out
#SBATCH --error=hpc/episode_matched_scripted_eval/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

# Focused re-run of the vs-scripted diagnostics with the updated eval code:
# late checkpoints (1000-1200) at the 90s and 120s horizons only, 50 games,
# to a fresh output dir. Purpose: populate `head_diagnostics` (notably the
# conditional price-head entropy) which the original sweep discarded because
# it ran with pre-update code. Survival/win numbers here are NOT meant to
# replace the main `_scripted_eval_100` sweep (different n-games).

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source hpc/episode_matched_5env_2seed_cpu/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/episode_matched_5env_2seed_cpu/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/episode_matched_5env_2seed_cpu/campaign.env first}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

if [[ ! -x "${HUNGER_HPC_VENV}/bin/python" ]]; then
  echo "Missing HPC Python environment: ${HUNGER_HPC_VENV}" >&2
  exit 2
fi

variants=(small_90 small_120)
opponents=(v2 v3 v4 v5)
environment_variant="${variants[$((SLURM_ARRAY_TASK_ID / 4))]}"
opponent="${opponents[$((SLURM_ARRAY_TASK_ID % 4))]}"

campaign_horizon_root="${HUNGER_CAMPAIGN_ROOT}/${environment_variant}"
output_dir="${HUNGER_CAMPAIGN_ROOT}/_scripted_eval_headdiag_1000_1200/${environment_variant}/${opponent}"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-headdiag-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export HUNGER_ENVIRONMENT_VARIANT="${environment_variant}"

cd "${HUNGER_RESEARCH_ROOT}"

"${HUNGER_HPC_VENV}/bin/python" \
  -m eval.scaled_checkpoint_scripted_diagnostics \
  --campaign-root "${campaign_horizon_root}" \
  --output-dir "${output_dir}" \
  --algorithms ippo,mappo \
  --seeds 0,1 \
  --checkpoint-start 1000 \
  --checkpoint-stop 1200 \
  --checkpoint-every 100 \
  --n-games 50 \
  --game-seed-base 900000 \
  --opponent "${opponent}"
