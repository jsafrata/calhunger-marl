#!/bin/bash
set -euo pipefail
: "${HUNGER_SLURM_ACCOUNT:?source hpc/best_config_peer_crossplay/campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source hpc/best_config_peer_crossplay/campaign.env first}"
cd "${HUNGER_RESEARCH_ROOT}"
echo "Submitting best-config peer cross-play (10 focal tasks x 10 jobs, 100 games each)"
sbatch --account="${HUNGER_SLURM_ACCOUNT}" hpc/best_config_peer_crossplay/run_array.sbatch
