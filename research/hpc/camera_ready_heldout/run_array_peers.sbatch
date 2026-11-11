#!/bin/bash
#SBATCH --job-name=hunger-cr-heldout-peers
#SBATCH --partition=expansion
#SBATCH --array=0-29%30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/camera_ready_heldout/slurm/%x-%A_%a.out
#SBATCH --error=hpc/camera_ready_heldout/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

# Part B — the checkpoint trajectory against a HELD-OUT LEARNED population.
# Focal: each of the 10 best-config default_180_mlp runs at every checkpoint
# 100..1200. Populations (episode-matched default_180 policies @ iter_1200,
# healthy in self-play, never used in any audit or checkpoint selection):
#   p=0: 3x ep_ippo_s2        (homogeneous held-out peer)
#   p=1: 3x ep_mappo_s2       (homogeneous held-out peer, other critic)
#   p=2: mixed table {ep_ippo_s0, ep_mappo_s1, ep_ippo_s4} (heterogeneous)
# 100 seat-balanced games per cell on the fresh seed block.
# Task id = population_index * 10 + run_index.

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_CKPT_ROOT_2SEED:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_CKPT_ROOT_S234:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_EPMATCH_ROOT_2SEED:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_EPMATCH_ROOT_S234:?source hpc/camera_ready_heldout/campaign.env first}"
: "${HUNGER_HELDOUT_SEED_BASE:?source hpc/camera_ready_heldout/campaign.env first}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

runs=(ippo_s0 ippo_s1 ippo_s2 ippo_s3 ippo_s4 mappo_s0 mappo_s1 mappo_s2 mappo_s3 mappo_s4)
checkpoints=(0100 0200 0300 0400 0500 0600 0700 0800 0900 1000 1100 1200)

# best-config checkpoint for run label + 4-digit iteration
bestcfg_ckpt() {
  local label="$1" iter="$2" algo seed root
  algo="${label%_s*}"; seed="${label#*_s}"
  if (( seed <= 1 )); then root="${HUNGER_CKPT_ROOT_2SEED}"; else root="${HUNGER_CKPT_ROOT_S234}"; fi
  echo "${root}/${algo}_seed${seed}_headmlp/checkpoints/iter_${iter}"
}
# episode-matched iter_1200 checkpoint for label ep_<algo>_s<seed>
epmatch_ckpt() {
  local label="$1" body algo seed root
  body="${label#ep_}"; algo="${body%_s*}"; seed="${body#*_s}"
  if (( seed <= 1 )); then root="${HUNGER_EPMATCH_ROOT_2SEED}"; else root="${HUNGER_EPMATCH_ROOT_S234}"; fi
  echo "${root}/${algo}_seed${seed}_headmlp/checkpoints/iter_1200"
}

population_index="$((SLURM_ARRAY_TASK_ID / 10))"
run_index="$((SLURM_ARRAY_TASK_ID % 10))"
run="${runs[$run_index]}"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-cr-heldout-peers-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${HUNGER_RESEARCH_ROOT}"

cd "${HUNGER_RESEARCH_ROOT}"
HARNESS="hpc/camera_ready_heldout/checkpoint_policy_crossplay.py"
PY="${HUNGER_HPC_VENV}/bin/python"

echo "task ${SLURM_ARRAY_TASK_ID}: population=${population_index} run=${run} seed_base=${HUNGER_HELDOUT_SEED_BASE}"

for iter in "${checkpoints[@]}"; do
  focal_label="${run}_c${iter}"
  focal_ckpt="$(bestcfg_ckpt "${run}" "${iter}")"
  case "${population_index}" in
    0|1)
      if (( population_index == 0 )); then peer="ep_ippo_s2"; else peer="ep_mappo_s2"; fi
      out_dir="${HUNGER_CAMPAIGN_ROOT}/_heldout_peers_100/3x_${peer}"
      mkdir -p "${out_dir}"
      "${PY}" "${HARNESS}" --mode 1v3 \
        --focal "policy:${focal_label}=${focal_ckpt}" \
        --opponent "policy:${peer}=$(epmatch_ckpt "${peer}")" \
        --environment-variant default --n-games 100 \
        --game-seed-base "${HUNGER_HELDOUT_SEED_BASE}" \
        --output "${out_dir}"
      ;;
    2)
      out_dir="${HUNGER_CAMPAIGN_ROOT}/_heldout_peers_100/mixed_ep3"
      mkdir -p "${out_dir}"
      "${PY}" "${HARNESS}" --mode mixed \
        --seat "policy:${focal_label}=${focal_ckpt}" \
        --seat "policy:ep_ippo_s0=$(epmatch_ckpt ep_ippo_s0)" \
        --seat "policy:ep_mappo_s1=$(epmatch_ckpt ep_mappo_s1)" \
        --seat "policy:ep_ippo_s4=$(epmatch_ckpt ep_ippo_s4)" \
        --environment-variant default --n-games 100 \
        --game-seed-base "${HUNGER_HELDOUT_SEED_BASE}" \
        --output "${out_dir}"
      ;;
    *)
      echo "bad population index ${population_index}" >&2; exit 2 ;;
  esac
done

echo "task ${SLURM_ARRAY_TASK_ID} (${run}, population ${population_index}) done"
