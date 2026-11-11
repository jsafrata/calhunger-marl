#!/bin/bash
#SBATCH --job-name=hunger-bc-vs-ladder
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/bc_vs_ladder/slurm/%x-%j.out
#SBATCH --error=hpc/bc_vs_ladder/slurm/%x-%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source a campaign.env first}"
: "${HUNGER_HPC_VENV:?source a campaign.env first}"
: "${HUNGER_BC_ACTOR:?export HUNGER_BC_ACTOR=<path to bc_actor.pt>}"
: "${HUNGER_OUTPUT_DIR:?export HUNGER_OUTPUT_DIR=<output dir>}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1

cd "${HUNGER_RESEARCH_ROOT}"
"${HUNGER_HPC_VENV}/bin/python" hpc/bc_vs_ladder/run_bc_vs_ladder.py \
  --bc-actor "${HUNGER_BC_ACTOR}" \
  --output-dir "${HUNGER_OUTPUT_DIR}" \
  --n-games 100
