#!/bin/bash
set -euo pipefail

: "${HUNGER_SLURM_ACCOUNT:?source campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env first}"
: "${HUNGER_HPC_VENV:?source campaign.env first}"

cd "${HUNGER_RESEARCH_ROOT}"

echo "Submitting head-diagnostics re-run (90s+120s, checkpoints 1000-1200)"
sbatch \
  --account="${HUNGER_SLURM_ACCOUNT}" \
  hpc/episode_matched_scripted_eval/run_array_headdiag.sbatch
