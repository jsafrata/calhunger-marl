#!/bin/bash
set -euo pipefail
: "${HUNGER_SLURM_ACCOUNT:?source hpc/ladder_validity_dose_response/campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source hpc/ladder_validity_dose_response/campaign.env first}"
cd "${HUNGER_RESEARCH_ROOT}"
echo "Submitting ladder-validity (50 jobs) + peer 2+2 dose-response (45 jobs), 100 games each"
sbatch --account="${HUNGER_SLURM_ACCOUNT}" hpc/ladder_validity_dose_response/run_array.sbatch
