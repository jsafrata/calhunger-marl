#!/bin/bash
set -euo pipefail

: "${HUNGER_SLURM_ACCOUNT:?source campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env first}"
: "${HUNGER_HPC_VENV:?source campaign.env first}"

cd "${HUNGER_RESEARCH_ROOT}"

echo "Submitting default/180-second diagnostics with wall time 05:00:00"
sbatch \
  --account="${HUNGER_SLURM_ACCOUNT}" \
  --time=05:00:00 \
  --export=ALL,HUNGER_ENVIRONMENT_VARIANT=default \
  hpc/episode_matched_scripted_eval/run_array.sbatch
