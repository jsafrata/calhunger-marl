#!/bin/bash
#SBATCH --job-name=hunger-finetune-ladder
#SBATCH --partition=expansion
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/finetune_vs_ladder/slurm/%x-%A_%a.out
#SBATCH --error=hpc/finetune_vs_ladder/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source hpc/finetune_vs_ladder/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/finetune_vs_ladder/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/finetune_vs_ladder/campaign.env first}"
: "${HUNGER_CKPT_ROOT_2SEED:?source hpc/finetune_vs_ladder/campaign.env first}"
: "${HUNGER_CKPT_ROOT_S234:?source hpc/finetune_vs_ladder/campaign.env first}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

if [[ ! -x "${HUNGER_HPC_VENV}/bin/python" ]]; then
  echo "Missing HPC Python environment: ${HUNGER_HPC_VENV}" >&2
  exit 2
fi

# 8 warm starts (all default_180_mlp best-config arms, iter_1200):
# tasks 0-4 = ippo seeds 0-4, tasks 5-7 = mappo seeds 0,2,3.
# Must match hpc/finetune_vs_ladder/campaign.py TASKS.
algos=(ippo ippo ippo ippo ippo mappo mappo mappo)
seeds=(0 1 2 3 4 0 2 3)
algo="${algos[$SLURM_ARRAY_TASK_ID]}"
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"

# Seeds 0-1 come from the 2seed campaign, seeds 2-4 from the seeds234
# campaign (same convention as best_config_peer_crossplay).
if (( seed <= 1 )); then
  ckpt_root="${HUNGER_CKPT_ROOT_2SEED}"
else
  ckpt_root="${HUNGER_CKPT_ROOT_S234}"
fi
ckpt="${ckpt_root}/${algo}_seed${seed}_headmlp/checkpoints/iter_1200"
if [[ ! -f "${ckpt}/rllib_checkpoint.json" ]]; then
  echo "Missing warm-start checkpoint: ${ckpt}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED="${seed}"
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-ftladder-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${HUNGER_RESEARCH_ROOT}"

cd "${HUNGER_RESEARCH_ROOT}"
mkdir -p "${HUNGER_CAMPAIGN_ROOT}"

"${HUNGER_HPC_VENV}/bin/python" -m hpc.finetune_vs_ladder.run_finetune \
  --checkpoint "${ckpt}" \
  --algo "${algo}" \
  --seed "${seed}" \
  --iterations 300 \
  --checkpoint-every 50 \
  --output-root "${HUNGER_CAMPAIGN_ROOT}" \
  --num-env-runners 2 \
  --torso mlp \
  --head-kind mlp \
  --include-survival-features \
  --include-trade-history

echo "task ${SLURM_ARRAY_TASK_ID} (${algo} seed${seed}) done"
