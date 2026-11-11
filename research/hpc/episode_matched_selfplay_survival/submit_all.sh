#!/bin/bash
set -euo pipefail

: "${HUNGER_SLURM_ACCOUNT:?source hpc/episode_matched_5env_2seed_cpu/campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source hpc/episode_matched_5env_2seed_cpu/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/episode_matched_5env_2seed_cpu/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/episode_matched_5env_2seed_cpu/campaign.env first}"

cd "${HUNGER_RESEARCH_ROOT}"

echo "Submitting self-play survival diagnostics (5 horizons, 50 games each)"
sbatch \
  --account="${HUNGER_SLURM_ACCOUNT}" \
  hpc/episode_matched_selfplay_survival/run_array.sbatch
