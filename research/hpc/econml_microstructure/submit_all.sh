#!/bin/bash
set -euo pipefail
: "${HUNGER_SLURM_ACCOUNT:?source hpc/econml_microstructure/campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source hpc/econml_microstructure/campaign.env first}"
cd "${HUNGER_RESEARCH_ROOT}"
echo "Submitting E31 econml microstructure (32 tasks x 100 games, seed base 900000, HG_LEDGER=1)"
sbatch --account="${HUNGER_SLURM_ACCOUNT}" hpc/econml_microstructure/run_array.sbatch
