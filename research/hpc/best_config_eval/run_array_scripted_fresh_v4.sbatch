#!/bin/bash
#SBATCH --job-name=hunger-bestcfg-fresh-v4
#SBATCH --partition=expansion
#SBATCH --array=0-119%60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/best_config_eval/slurm/%x-%A_%a.out
#SBATCH --error=hpc/best_config_eval/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

# Fresh-seed re-evaluation of the longitudinal default_180_mlp checkpoints
# (100..1200) against 3x scripted v4, the deployment archetype of the TAE
# paper's checkpoint-selection regret analysis (EXPERIMENTS.md E25).
#
# Purpose: an evaluation stage DISJOINT from the 100 selection games (seed
# base 900000, seeds 900000-900099) so that selection regret can be estimated
# out of sample: every selector (oracle included) picks its checkpoint on the
# original 100 games; the picked checkpoint is then scored on these fresh
# games (seed base 950000, 500 games). Same harness, same protocol
# (stochastic sampling, learned seat 0, 3x v4) as the original evaluation.
#
# Task id = run_index * 12 + checkpoint_index, checkpoint = 100 * (index + 1).
# run_index enumerates the ten best-config default_180_mlp runs across the
# two campaign roots (seeds 0-1 root and seeds 2-4 root):
#   0-1: ippo seeds 0,1    2-4: ippo seeds 2,3,4
#   5-6: mappo seeds 0,1   7-9: mappo seeds 2,3,4
# Tasks of one campaign root share an output dir; per-checkpoint raw JSONs
# have distinct names, so the only shared write is the summary CSV, which
# merge_fresh_v4.sh rebuilds from the raw cache once every task is done.

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

root_seeds01="${HUNGER_CAMPAIGN_ROOT}"
root_seeds234="${HUNGER_ROOT_SEEDS234:-/resnick/home/${USER}/hunger_runs/best_config_mlp_seeds234_ippo_mappo_1200}"
fresh_out="${HUNGER_FRESH_OUT:-/resnick/home/${USER}/hunger_runs/best_config_fresh_eval_v4_500}"
n_games="${HUNGER_FRESH_N_GAMES:-500}"
seed_base="${HUNGER_FRESH_SEED_BASE:-950000}"

algos=(ippo ippo ippo ippo ippo mappo mappo mappo mappo mappo)
seeds=(0 1 2 3 4 0 1 2 3 4)

run_index="$((SLURM_ARRAY_TASK_ID / 12))"
ckpt_index="$((SLURM_ARRAY_TASK_ID % 12))"
checkpoint="$((100 * (ckpt_index + 1)))"
algo="${algos[$run_index]}"
seed="${seeds[$run_index]}"

if (( seed <= 1 )); then
  campaign_arm_root="${root_seeds01}/default_180_mlp"
  root_label="seeds01"
else
  campaign_arm_root="${root_seeds234}/default_180_mlp"
  root_label="seeds234"
fi
output_dir="${fresh_out}/_scripted_eval_fresh${n_games}/default_180_mlp_${root_label}/v4"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-bestcfg-fresh-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export HUNGER_ENVIRONMENT_VARIANT="default"

cd "${HUNGER_RESEARCH_ROOT}"

echo "task ${SLURM_ARRAY_TASK_ID}: ${algo} seed${seed} iter${checkpoint} vs v4, ${n_games} games, seed base ${seed_base}"
echo "campaign arm root: ${campaign_arm_root}"
echo "output dir: ${output_dir}"

"${HUNGER_HPC_VENV}/bin/python" \
  -m eval.scaled_checkpoint_scripted_diagnostics \
  --campaign-root "${campaign_arm_root}" \
  --output-dir "${output_dir}" \
  --algorithms "${algo}" \
  --seeds "${seed}" \
  --checkpoint-start "${checkpoint}" \
  --checkpoint-stop "${checkpoint}" \
  --checkpoint-every 100 \
  --n-games "${n_games}" \
  --game-seed-base "${seed_base}" \
  --opponent v4
