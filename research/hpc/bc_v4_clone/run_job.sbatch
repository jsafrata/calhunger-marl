#!/bin/bash
#SBATCH --job-name=hunger-bc-v4-clone
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/bc_v4_clone/slurm/%x-%j.out
#SBATCH --error=hpc/bc_v4_clone/slurm/%x-%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source a campaign.env first}"
: "${HUNGER_HPC_VENV:?source a campaign.env first}"
: "${HUNGER_BCV4_ROOT:?export HUNGER_BCV4_ROOT=<workdir>}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=4

cd "${HUNGER_RESEARCH_ROOT}"
"${HUNGER_HPC_VENV}/bin/python" hpc/bc_v4_clone/run_bc_v4_clone.py \
  --root "${HUNGER_BCV4_ROOT}" \
  --n-train 3000 --n-val 300 --n-test 300 \
  --epochs 8 --eval-games 100
