#!/bin/bash
# Print the planned command for every task of every arm without executing.
set -euo pipefail

: "${HUNGER_RESEARCH_ROOT:?source campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env first}"
: "${HUNGER_HPC_VENV:?source campaign.env first}"

cd "${HUNGER_RESEARCH_ROOT}"

for arm in small_90_mlp default_180_mlp; do
  for task_id in 0 1 2 3 4 5; do
    "${HUNGER_HPC_VENV}/bin/python" \
      -m hpc.best_config_mlp_seeds234_cpu.run_task \
      --arm "${arm}" \
      --task-id "${task_id}" \
      --research-root "${HUNGER_RESEARCH_ROOT}" \
      --campaign-root "${HUNGER_CAMPAIGN_ROOT}"
  done
done
