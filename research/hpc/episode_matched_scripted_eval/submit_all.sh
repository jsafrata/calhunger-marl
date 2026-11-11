#!/bin/bash
set -euo pipefail

: "${HUNGER_SLURM_ACCOUNT:?source campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env first}"
: "${HUNGER_HPC_VENV:?source campaign.env first}"

cd "${HUNGER_RESEARCH_ROOT}"

# Submit only horizons whose training runs are currently complete. The
# 180-second/default horizon can be launched later with submit_180.sh.
environments=(small_60 small_90 small_120 small_150)
walltimes=(04:00:00 06:00:00 08:00:00 10:00:00)

for index in "${!environments[@]}"; do
  environment="${environments[$index]}"
  walltime="${walltimes[$index]}"
  echo "Submitting ${environment} diagnostics with wall time ${walltime}"
  sbatch \
    --account="${HUNGER_SLURM_ACCOUNT}" \
    --time="${walltime}" \
    --export="ALL,HUNGER_ENVIRONMENT_VARIANT=${environment}" \
    hpc/episode_matched_scripted_eval/run_array.sbatch
done
