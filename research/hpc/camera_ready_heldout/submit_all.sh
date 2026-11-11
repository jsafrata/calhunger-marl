#!/bin/bash
set -euo pipefail

: "${HUNGER_SLURM_ACCOUNT:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_RESEARCH_ROOT:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/camera_ready_heldout/campaign.env first}"

cd "${HUNGER_RESEARCH_ROOT}"
mkdir -p "${HUNGER_CAMPAIGN_ROOT}" hpc/camera_ready_heldout/slurm

echo "Submitting Part A: held-out seed block, scripted ladder (16 tasks)"
sbatch --account="${HUNGER_SLURM_ACCOUNT}" hpc/camera_ready_heldout/run_array_scripted.sbatch

echo "Submitting Part B: held-out learned population trajectory (30 tasks)"
sbatch --account="${HUNGER_SLURM_ACCOUNT}" hpc/camera_ready_heldout/run_array_peers.sbatch
