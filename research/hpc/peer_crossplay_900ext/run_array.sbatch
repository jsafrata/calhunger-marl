#!/bin/bash
#SBATCH --job-name=hunger-peerxp-900ext
#SBATCH --partition=expansion
#SBATCH --array=0-99%50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=02:30:00
#SBATCH --output=hpc/peer_crossplay_900ext/slurm/%x-%A_%a.out
#SBATCH --error=hpc/peer_crossplay_900ext/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source hpc/peer_crossplay_900ext/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/peer_crossplay_900ext/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/peer_crossplay_900ext/campaign.env first}"
: "${HUNGER_CKPT_ROOT_2SEED:?source hpc/peer_crossplay_900ext/campaign.env first}"
: "${HUNGER_CKPT_ROOT_S234:?source hpc/peer_crossplay_900ext/campaign.env first}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

# Same 10 policies as best_config_peer_crossplay: ippo/mappo x seeds 0-4,
# iter_1200, default_180_mlp best-config arms.
labels=(ippo_s0 ippo_s1 ippo_s2 ippo_s3 ippo_s4 mappo_s0 mappo_s1 mappo_s2 mappo_s3 mappo_s4)
ckpt_for() {
  local label="$1" algo seed root
  algo="${label%_s*}"; seed="${label#*_s}"
  if (( seed <= 1 )); then root="${HUNGER_CKPT_ROOT_2SEED}"; else root="${HUNGER_CKPT_ROOT_S234}"; fi
  echo "${root}/${algo}_seed${seed}_headmlp/checkpoints/iter_1200"
}

# One cell per task: focal = id/10; cell 0 = self-play diagonal, cells 1-9 =
# ordered 1v3 vs the other nine labels in label order.
focal="${labels[$((SLURM_ARRAY_TASK_ID / 10))]}"
cell=$((SLURM_ARRAY_TASK_ID % 10))
focal_ckpt="$(ckpt_for "$focal")"
output_dir="${HUNGER_CAMPAIGN_ROOT}/_peer_crossplay_900ext"
mkdir -p "$output_dir"

opps=()
for l in "${labels[@]}"; do
  [[ "$l" == "$focal" ]] || opps+=("$l")
done

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-peerxp900-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${HUNGER_RESEARCH_ROOT}"

cd "${HUNGER_RESEARCH_ROOT}"
HARNESS="hpc/peer_crossplay_900ext/checkpoint_policy_crossplay.py"

if (( cell == 0 )); then
  "${HUNGER_HPC_VENV}/bin/python" "$HARNESS" --mode mixed \
    --seat "policy:${focal}=${focal_ckpt}" --seat "policy:${focal}=${focal_ckpt}" \
    --seat "policy:${focal}=${focal_ckpt}" --seat "policy:${focal}=${focal_ckpt}" \
    --environment-variant default --n-games 900 --game-seed-base 900100 \
    --output "$output_dir"
else
  opp="${opps[$((cell - 1))]}"
  "${HUNGER_HPC_VENV}/bin/python" "$HARNESS" --mode 1v3 \
    --focal "policy:${focal}=${focal_ckpt}" \
    --opponent "policy:${opp}=$(ckpt_for "$opp")" \
    --environment-variant default --n-games 900 --game-seed-base 900100 \
    --output "$output_dir"
fi

echo "task ${SLURM_ARRAY_TASK_ID} (${focal}, cell ${cell}) done"
