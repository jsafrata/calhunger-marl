#!/bin/bash
set -euo pipefail

: "${HUNGER_SLURM_ACCOUNT:?source campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env first}"
: "${HUNGER_HPC_VENV:?source campaign.env first}"

cd "${HUNGER_RESEARCH_ROOT}"

environments=(small_60 small_90 small_120 small_150 default)
walltimes=(08:00:00 12:00:00 16:00:00 20:00:00 24:00:00)

for index in "${!environments[@]}"; do
  environment="${environments[$index]}"
  walltime="${walltimes[$index]}"
  echo "Submitting ${environment} with wall time ${walltime}"
  sbatch \
    --account="${HUNGER_SLURM_ACCOUNT}" \
    --time="${walltime}" \
    --export="ALL,HUNGER_ENVIRONMENT_VARIANT=${environment}" \
    hpc/episode_matched_5env_2seed_cpu/run_array.sbatch
done
