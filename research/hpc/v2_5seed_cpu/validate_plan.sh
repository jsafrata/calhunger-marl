#!/bin/bash
set -euo pipefail

: "${HUNGER_RESEARCH_ROOT:?source campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env first}"
: "${HUNGER_HPC_VENV:?source campaign.env first}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

cd "${HUNGER_RESEARCH_ROOT}"
for task_id in {0..9}; do
  "${HUNGER_HPC_VENV}/bin/python" -m hpc.v2_5seed_cpu.run_task \
    --task-id "${task_id}" \
    --research-root "${HUNGER_RESEARCH_ROOT}" \
    --campaign-root "${HUNGER_CAMPAIGN_ROOT}"
done
