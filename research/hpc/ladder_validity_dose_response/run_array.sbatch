#!/bin/bash
#SBATCH --job-name=hunger-ladder-dose
#SBATCH --partition=expansion
#SBATCH --array=0-9%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/ladder_validity_dose_response/slurm/%x-%A_%a.out
#SBATCH --error=hpc/ladder_validity_dose_response/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source hpc/ladder_validity_dose_response/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/ladder_validity_dose_response/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/ladder_validity_dose_response/campaign.env first}"
: "${HUNGER_CKPT_ROOT_2SEED:?source hpc/ladder_validity_dose_response/campaign.env first}"
: "${HUNGER_CKPT_ROOT_S234:?source hpc/ladder_validity_dose_response/campaign.env first}"
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
ladder_dir="${HUNGER_CAMPAIGN_ROOT}/_ladder_validity_100"
mix_dir="${HUNGER_CAMPAIGN_ROOT}/_peer_mix22_100"
mkdir -p "$ladder_dir" "$mix_dir"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-ladderdose-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${HUNGER_RESEARCH_ROOT}"

cd "${HUNGER_RESEARCH_ROOT}"
HARNESS="hpc/ladder_validity_dose_response/checkpoint_policy_crossplay.py"

# Experiment A: this focal vs 3x each scripted ladder rung, 100 games each.
for v in v1 v2 v3 v4 v5; do
  "${HUNGER_HPC_VENV}/bin/python" "$HARNESS" --mode 1v3 \
    --focal "policy:${focal}=${focal_ckpt}" \
    --opponent "scripted:${v}" \
    --environment-variant default --n-games 100 --game-seed-base 900000 \
    --output "$ladder_dir"
done

# Experiment B: k=2 dose point. All 45 unordered pairs enumerated in fixed
# lexicographic order; this task runs the pairs with index % 10 == task id.
pair_index=0
for (( i=0; i<10; i++ )); do
  for (( j=i+1; j<10; j++ )); do
    if (( pair_index % 10 == SLURM_ARRAY_TASK_ID )); then
      a="${labels[$i]}"; b="${labels[$j]}"
      a_ckpt="$(ckpt_for "$a")"; b_ckpt="$(ckpt_for "$b")"
      "${HUNGER_HPC_VENV}/bin/python" "$HARNESS" --mode mixed \
        --seat "policy:${a}=${a_ckpt}" --seat "policy:${a}=${a_ckpt}" \
        --seat "policy:${b}=${b_ckpt}" --seat "policy:${b}=${b_ckpt}" \
        --environment-variant default --n-games 100 --game-seed-base 900000 \
        --output "$mix_dir"
    fi
    pair_index=$((pair_index + 1))
  done
done

echo "task ${SLURM_ARRAY_TASK_ID} (${focal}) done"
