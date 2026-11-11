#!/bin/bash
#SBATCH --job-name=hunger-v2-smoke
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/v2_5seed_cpu/slurm/%x-%j.out
#SBATCH --error=hpc/v2_5seed_cpu/slurm/%x-%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source campaign.env before sbatch}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env before sbatch}"
: "${HUNGER_HPC_VENV:?source campaign.env before sbatch}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=999
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-smoke-${SLURM_JOB_ID}"

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

"${HUNGER_HPC_VENV}/bin/python" -m eval.long_baseline \
  --iters 100 \
  --checkpoint_every 100 \
  --checkpoint_start 100 \
  --seed 999 \
  --algos mappo \
  --train_batch_size 4096 \
  --minibatch_size 1024 \
  --num_epochs 4 \
  --num_env_runners 2 \
  --out_root "${HUNGER_CAMPAIGN_ROOT}/_smoke" \
  --variant_suffix seed999_headmlp \
  --head_kind mlp \
  --gamma 1.0 \
  --lambda_ 1.0 \
  --reward_kind terminal_only \
  --reward_variant baseline \
  --entropy_coeff 0.01 \
  --fail_if_output_exists
