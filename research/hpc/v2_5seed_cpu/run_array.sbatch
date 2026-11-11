#!/bin/bash
#SBATCH --job-name=hunger-v2-5seed
#SBATCH --partition=expansion
#SBATCH --array=0-9%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/v2_5seed_cpu/slurm/%x-%A_%a.out
#SBATCH --error=hpc/v2_5seed_cpu/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source campaign.env before sbatch}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env before sbatch}"
: "${HUNGER_HPC_VENV:?source campaign.env before sbatch}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a SLURM array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

if [[ ! -x "${HUNGER_HPC_VENV}/bin/python" ]]; then
  echo "Missing HPC Python environment: ${HUNGER_HPC_VENV}" >&2
  exit 2
fi

if (( SLURM_ARRAY_TASK_ID < 5 )); then
  campaign_seed="${SLURM_ARRAY_TASK_ID}"
else
  campaign_seed="$((SLURM_ARRAY_TASK_ID - 5))"
fi

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED="${campaign_seed}"
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

cd "${HUNGER_RESEARCH_ROOT}"

preflight_args=(
  -m hpc.v2_5seed_cpu.preflight
  --research-root "${HUNGER_RESEARCH_ROOT}"
  --campaign-root "${HUNGER_CAMPAIGN_ROOT}"
)
if [[ "${HUNGER_ALLOW_DIRTY:-0}" != "1" ]]; then
  preflight_args+=(--strict)
fi
"${HUNGER_HPC_VENV}/bin/python" "${preflight_args[@]}"

run_args=(
  -m hpc.v2_5seed_cpu.run_task
  --task-id "${SLURM_ARRAY_TASK_ID}"
  --research-root "${HUNGER_RESEARCH_ROOT}"
  --campaign-root "${HUNGER_CAMPAIGN_ROOT}"
  --execute
)
if [[ "${HUNGER_RESUME:-0}" == "1" ]]; then
  run_args+=(--resume)
fi
"${HUNGER_HPC_VENV}/bin/python" "${run_args[@]}"
