#!/bin/bash
#SBATCH --job-name=hunger-selfplay-survival
#SBATCH --partition=expansion
#SBATCH --array=0-4%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/selfplay_survival_5env/slurm/%x-%A_%a.out
#SBATCH --error=hpc/selfplay_survival_5env/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?set HUNGER_RESEARCH_ROOT before sbatch}"
: "${HUNGER_HPC_VENV:?set HUNGER_HPC_VENV before sbatch}"
: "${HUNGER_SMALL_CAMPAIGN_ROOT:?set HUNGER_SMALL_CAMPAIGN_ROOT before sbatch}"
: "${HUNGER_SCALED_CAMPAIGN_ROOT:?set HUNGER_SCALED_CAMPAIGN_ROOT before sbatch}"
: "${HUNGER_SELFPLAY_SURVIVAL_ROOT:?set HUNGER_SELFPLAY_SURVIVAL_ROOT before sbatch}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi
if [[ ! -x "${HUNGER_HPC_VENV}/bin/python" ]]; then
  echo "Missing HPC Python environment: ${HUNGER_HPC_VENV}" >&2
  exit 2
fi

variants=(small small_60 small_90 small_120 small_150)
environment_variant="${variants[$SLURM_ARRAY_TASK_ID]}"
if [[ "${environment_variant}" == "small" ]]; then
  campaign_root="${HUNGER_SMALL_CAMPAIGN_ROOT}"
else
  campaign_root="${HUNGER_SCALED_CAMPAIGN_ROOT}/${environment_variant}"
fi
output_dir="${HUNGER_SELFPLAY_SURVIVAL_ROOT}/${environment_variant}"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

cd "${HUNGER_RESEARCH_ROOT}"
"${HUNGER_HPC_VENV}/bin/python" \
  -m eval.checkpoint_selfplay_survival_diagnostics \
  --campaign-root "${campaign_root}" \
  --output-dir "${output_dir}" \
  --environment-variant "${environment_variant}" \
  --algorithms ippo,mappo \
  --seeds 0,1 \
  --checkpoint-start 600 \
  --checkpoint-stop 1200 \
  --checkpoint-every 100 \
  --n-games 50 \
  --game-seed-base 970000
