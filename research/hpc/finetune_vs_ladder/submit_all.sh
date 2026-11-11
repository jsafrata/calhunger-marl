#!/bin/bash
set -euo pipefail
: "${HUNGER_SLURM_ACCOUNT:?source hpc/finetune_vs_ladder/campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source hpc/finetune_vs_ladder/campaign.env first}"
cd "${HUNGER_RESEARCH_ROOT}"
echo "Submitting finetune_vs_ladder (8 warm starts x 300 fine-tune iters vs scripted ladder)"
sbatch --account="${HUNGER_SLURM_ACCOUNT}" hpc/finetune_vs_ladder/run_array.sbatch
