#!/bin/bash
#SBATCH --job-name=hunger-cr-heldout-scripted
#SBATCH --partition=expansion
#SBATCH --array=0-15%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/camera_ready_heldout/slurm/%x-%A_%a.out
#SBATCH --error=hpc/camera_ready_heldout/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

# Part A — held-out game-seed block for the longitudinal scripted evaluation
# (the Figure 1 / Table 2 inputs): all 10 best-config default_180_mlp runs,
# checkpoints 100..1200, 100 games/cell vs 3x each ladder archetype, on a
# FRESH seed base. Identical harness and flags to hpc/best_config_eval except
# the seed base and the output directory.
# Task id = root_index * 8 + opponent_index * 2 + algorithm_index.

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_CKPT_ROOT_2SEED:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_CKPT_ROOT_S234:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_HELDOUT_SEED_BASE:?source hpc/camera_ready_heldout/campaign.env first}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

if [[ ! -x "${HUNGER_HPC_VENV}/bin/python" ]]; then
  echo "Missing HPC Python environment: ${HUNGER_HPC_VENV}" >&2
  exit 2
fi

root_names=(2seed s234)
root_paths=("${HUNGER_CKPT_ROOT_2SEED}" "${HUNGER_CKPT_ROOT_S234}")
root_seeds=("0,1" "2,3,4")
opponents=(v2 v3 v4 v5)
algorithms=(ippo mappo)

root_index="$((SLURM_ARRAY_TASK_ID / 8))"
opponent_index="$(((SLURM_ARRAY_TASK_ID % 8) / 2))"
algorithm_index="$((SLURM_ARRAY_TASK_ID % 2))"
root_name="${root_names[$root_index]}"
campaign_arm_root="${root_paths[$root_index]}"
seeds="${root_seeds[$root_index]}"
opponent="${opponents[$opponent_index]}"
algorithm="${algorithms[$algorithm_index]}"

output_dir="${HUNGER_CAMPAIGN_ROOT}/_scripted_eval_100_heldout/${root_name}/${opponent}/${algorithm}"
mkdir -p "${output_dir}"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-cr-heldout-scripted-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export HUNGER_ENVIRONMENT_VARIANT="default"

cd "${HUNGER_RESEARCH_ROOT}"

echo "task ${SLURM_ARRAY_TASK_ID}: root=${root_name} opponent=${opponent} algorithm=${algorithm} seeds=${seeds} seed_base=${HUNGER_HELDOUT_SEED_BASE}"

"${HUNGER_HPC_VENV}/bin/python" \
  -m eval.scaled_checkpoint_scripted_diagnostics \
  --campaign-root "${campaign_arm_root}" \
  --output-dir "${output_dir}" \
  --algorithms "${algorithm}" \
  --seeds "${seeds}" \
  --checkpoint-start 100 \
  --checkpoint-stop 1200 \
  --checkpoint-every 100 \
  --n-games 100 \
  --game-seed-base "${HUNGER_HELDOUT_SEED_BASE}" \
  --opponent "${opponent}"

echo "task ${SLURM_ARRAY_TASK_ID} done"
