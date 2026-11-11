#!/bin/bash
set -euo pipefail
: "${HUNGER_SLURM_ACCOUNT:?source hpc/peer_crossplay_900ext/campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source hpc/peer_crossplay_900ext/campaign.env first}"
cd "${HUNGER_RESEARCH_ROOT}"
echo "Submitting peer cross-play 900-game seed extension (100 tasks, 1 cell x 900 games each, seed base 900100)"
sbatch --account="${HUNGER_SLURM_ACCOUNT}" hpc/peer_crossplay_900ext/run_array.sbatch
