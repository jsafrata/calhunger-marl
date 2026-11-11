#!/bin/bash
set -euo pipefail

: "${HUNGER_SLURM_ACCOUNT:?source campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env first}"
: "${HUNGER_HPC_VENV:?source campaign.env first}"

cd "${HUNGER_RESEARCH_ROOT}"

environments=(small_150)
walltimes=(20:00:00)

for index in "${!environments[@]}"; do
  environment="${environments[$index]}"
  walltime="${walltimes[$index]}"
  echo "Submitting ${environment} with wall time ${walltime}"
  sbatch \
    --account="${HUNGER_SLURM_ACCOUNT}" \
    --time="${walltime}" \
    --export="ALL,HUNGER_ENVIRONMENT_VARIANT=${environment}" \
    hpc/small150_cliff_8seed_cpu/run_array.sbatch
done
