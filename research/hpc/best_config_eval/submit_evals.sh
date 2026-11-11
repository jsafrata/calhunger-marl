#!/bin/bash
set -euo pipefail

: "${HUNGER_SLURM_ACCOUNT:?source hpc/best_config_2seed_cpu/campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source hpc/best_config_2seed_cpu/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/best_config_2seed_cpu/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/best_config_2seed_cpu/campaign.env first}"

cd "${HUNGER_RESEARCH_ROOT}"

echo "Submitting best-config scripted diagnostics (4 arms x 4 bots, 100 games)"
sbatch \
  --account="${HUNGER_SLURM_ACCOUNT}" \
  hpc/best_config_eval/run_array_scripted.sbatch

echo "Submitting best-config self-play survival diagnostics (4 arms, 50 games)"
sbatch \
  --account="${HUNGER_SLURM_ACCOUNT}" \
  hpc/best_config_eval/run_array_selfplay.sbatch
