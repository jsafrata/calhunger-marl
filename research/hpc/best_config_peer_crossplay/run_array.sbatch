#!/bin/bash
#SBATCH --job-name=hunger-peer-crossplay
#SBATCH --partition=expansion
#SBATCH --array=0-9%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/best_config_peer_crossplay/slurm/%x-%A_%a.out
#SBATCH --error=hpc/best_config_peer_crossplay/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source hpc/best_config_peer_crossplay/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/best_config_peer_crossplay/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/best_config_peer_crossplay/campaign.env first}"
: "${HUNGER_CKPT_ROOT_2SEED:?source hpc/best_config_peer_crossplay/campaign.env first}"
: "${HUNGER_CKPT_ROOT_S234:?source hpc/best_config_peer_crossplay/campaign.env first}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

# 10 policies: ippo/mappo x seeds 0-4, iter_1200, default_180_mlp best-config arms.
labels=(ippo_s0 ippo_s1 ippo_s2 ippo_s3 ippo_s4 mappo_s0 mappo_s1 mappo_s2 mappo_s3 mappo_s4)
ckpt_for() {
  local label="$1" algo seed root
  algo="${label%_s*}"; seed="${label#*_s}"
  if (( seed <= 1 )); then root="${HUNGER_CKPT_ROOT_2SEED}"; else root="${HUNGER_CKPT_ROOT_S234}"; fi
  echo "${root}/${algo}_seed${seed}_headmlp/checkpoints/iter_1200"
}

focal="${labels[$SLURM_ARRAY_TASK_ID]}"
focal_ckpt="$(ckpt_for "$focal")"
output_dir="${HUNGER_CAMPAIGN_ROOT}/_peer_crossplay_100"
mkdir -p "$output_dir"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-peerxp-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${HUNGER_RESEARCH_ROOT}"

cd "${HUNGER_RESEARCH_ROOT}"
HARNESS="hpc/best_config_peer_crossplay/checkpoint_policy_crossplay.py"

# Self-play diagonal (mandatory), then all 9 ordered 1v3 pairs with this focal.
"${HUNGER_HPC_VENV}/bin/python" "$HARNESS" --mode mixed \
  --seat "policy:${focal}=${focal_ckpt}" --seat "policy:${focal}=${focal_ckpt}" \
  --seat "policy:${focal}=${focal_ckpt}" --seat "policy:${focal}=${focal_ckpt}" \
  --environment-variant default --n-games 100 --game-seed-base 900000 \
  --output "$output_dir"

for opp in "${labels[@]}"; do
  [[ "$opp" == "$focal" ]] && continue
  "${HUNGER_HPC_VENV}/bin/python" "$HARNESS" --mode 1v3 \
    --focal "policy:${focal}=${focal_ckpt}" \
    --opponent "policy:${opp}=$(ckpt_for "$opp")" \
    --environment-variant default --n-games 100 --game-seed-base 900000 \
    --output "$output_dir"
done

echo "task ${SLURM_ARRAY_TASK_ID} (${focal}) done"
