#!/bin/bash
set -euo pipefail

: "${HUNGER_SLURM_ACCOUNT:?source campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env first}"
: "${HUNGER_HPC_VENV:?source campaign.env first}"

cd "${HUNGER_RESEARCH_ROOT}"

# One six-task array (ippo/mappo x seeds 2-4) per arm, each with the
# arm's own wall time from campaign.json.
arms=(small_90_mlp default_180_mlp)
walltimes=(14:00:00 30:00:00)

for index in "${!arms[@]}"; do
  arm="${arms[$index]}"
  walltime="${walltimes[$index]}"
  echo "Submitting ${arm} with wall time ${walltime}"
  sbatch \
    --account="${HUNGER_SLURM_ACCOUNT}" \
    --time="${walltime}" \
    --export="ALL,HUNGER_ARM=${arm}" \
    hpc/best_config_mlp_seeds234_cpu/run_array.sbatch
done
