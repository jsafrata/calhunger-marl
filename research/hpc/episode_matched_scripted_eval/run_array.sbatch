#!/bin/bash
#SBATCH --job-name=hunger-scripted-eval
#SBATCH --partition=expansion
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/episode_matched_scripted_eval/slurm/%x-%A_%a.out
#SBATCH --error=hpc/episode_matched_scripted_eval/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source hpc/episode_matched_5env_2seed_cpu/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/episode_matched_5env_2seed_cpu/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/episode_matched_5env_2seed_cpu/campaign.env first}"
: "${HUNGER_ENVIRONMENT_VARIANT:?submit with hpc/episode_matched_scripted_eval/submit_all.sh}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

if [[ ! -x "${HUNGER_HPC_VENV}/bin/python" ]]; then
  echo "Missing HPC Python environment: ${HUNGER_HPC_VENV}" >&2
  exit 2
fi

opponents=(v2 v3 v4 v5)
opponent="${opponents[$SLURM_ARRAY_TASK_ID]}"

case "${HUNGER_ENVIRONMENT_VARIANT}" in
  small_60|small_90|small_120|small_150)
    horizon_dir="${HUNGER_ENVIRONMENT_VARIANT}"
    ;;
  default)
    horizon_dir="default_180"
    ;;
  *)
    echo "Unknown environment variant: ${HUNGER_ENVIRONMENT_VARIANT}" >&2
    exit 2
    ;;
esac

campaign_horizon_root="${HUNGER_CAMPAIGN_ROOT}/${horizon_dir}"
output_dir="${HUNGER_CAMPAIGN_ROOT}/_scripted_eval_100/${horizon_dir}/${opponent}"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-scripted-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

cd "${HUNGER_RESEARCH_ROOT}"

"${HUNGER_HPC_VENV}/bin/python" \
  -m eval.scaled_checkpoint_scripted_diagnostics \
  --campaign-root "${campaign_horizon_root}" \
  --output-dir "${output_dir}" \
  --algorithms ippo,mappo \
  --seeds "${HUNGER_EVAL_SEEDS:-0,1}" \
  --checkpoint-start 100 \
  --checkpoint-stop 1200 \
  --checkpoint-every 100 \
  --n-games 100 \
  --game-seed-base 900000 \
  --opponent "${opponent}"
