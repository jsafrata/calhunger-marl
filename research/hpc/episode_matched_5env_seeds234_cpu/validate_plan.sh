#!/bin/bash
set -euo pipefail

: "${HUNGER_RESEARCH_ROOT:?source campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env first}"
: "${HUNGER_HPC_VENV:?source campaign.env first}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

cd "${HUNGER_RESEARCH_ROOT}"
for environment in small_60 small_90 small_120 small_150 default; do
  for task_id in $(seq 0 5); do
    "${HUNGER_HPC_VENV}/bin/python" \
      -m hpc.episode_matched_5env_seeds234_cpu.run_task \
      --environment-variant "${environment}" \
      --task-id "${task_id}" \
      --research-root "${HUNGER_RESEARCH_ROOT}" \
      --campaign-root "${HUNGER_CAMPAIGN_ROOT}"
  done
done
