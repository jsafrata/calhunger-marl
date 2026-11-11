#!/bin/bash
#SBATCH --job-name=hunger-econml-micro
#SBATCH --partition=expansion
#SBATCH --array=0-31%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:30:00
#SBATCH --output=hpc/econml_microstructure/slurm/%x-%A_%a.out
#SBATCH --error=hpc/econml_microstructure/slurm/%x-%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source hpc/econml_microstructure/campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source hpc/econml_microstructure/campaign.env first}"
: "${HUNGER_HPC_VENV:?source hpc/econml_microstructure/campaign.env first}"
: "${HUNGER_CKPT_ROOT_2SEED:?source hpc/econml_microstructure/campaign.env first}"
: "${HUNGER_CKPT_ROOT_S234:?source hpc/econml_microstructure/campaign.env first}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  module load "${HUNGER_PYTHON_MODULE}"
fi

# Same 10 policies / checkpoint mapping as best_config_peer_crossplay and
# peer_crossplay_900ext: ippo/mappo x seeds 0-4, iter_1200, default_180_mlp.
labels=(ippo_s0 ippo_s1 ippo_s2 ippo_s3 ippo_s4 mappo_s0 mappo_s1 mappo_s2 mappo_s3 mappo_s4)
ckpt_for() {
  local label="$1" algo seed root
  algo="${label%_s*}"; seed="${label#*_s}"
  if (( seed <= 1 )); then root="${HUNGER_CKPT_ROOT_2SEED}"; else root="${HUNGER_CKPT_ROOT_S234}"; fi
  echo "${root}/${algo}_seed${seed}_headmlp/checkpoints/iter_1200"
}

# E31 cell plan (32 tasks x 100 games, seed base 900000 = the base
# campaign's paired seeds, so every cell replays the SAME games the
# published matrices scored — the count-level fields double as the
# validation target):
#   tasks 0-9   : self-play diagonals (k=0 monocultures), labels[id]
#   tasks 10-25 : 8 ordered 1v3 pairs x both directions (k=3 extremes:
#                 the canonical ippo_s2<->ippo_s4 dominance pair, the
#                 catastrophic mappo_s1/mappo_s4 columns, cross-algo and
#                 same-algo healthy pairs)
#   tasks 26-31 : 6 mixed 2+2 cells (k=2) over the same six pairs
pairs=(
  "ippo_s2 ippo_s4"  "ippo_s4 ippo_s2"
  "mappo_s2 mappo_s1" "mappo_s1 mappo_s2"
  "ippo_s0 mappo_s4"  "mappo_s4 ippo_s0"
  "ippo_s3 mappo_s0"  "mappo_s0 ippo_s3"
  "ippo_s0 ippo_s1"   "ippo_s1 ippo_s0"
  "mappo_s2 mappo_s3" "mappo_s3 mappo_s2"
  "ippo_s2 mappo_s2"  "mappo_s2 ippo_s2"
  "ippo_s1 mappo_s3"  "mappo_s3 ippo_s1"
)
mixpairs=(
  "ippo_s2 ippo_s4"
  "mappo_s2 mappo_s1"
  "ippo_s0 mappo_s4"
  "ippo_s3 mappo_s0"
  "ippo_s2 mappo_s2"
  "ippo_s1 mappo_s3"
)

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
export HG_LEDGER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray-econmicro-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${HUNGER_RESEARCH_ROOT}"

cd "${HUNGER_RESEARCH_ROOT}"
HARNESS="hpc/econml_microstructure/checkpoint_policy_crossplay.py"
output_dir="${HUNGER_CAMPAIGN_ROOT}/_econml_microstructure_100"
mkdir -p "$output_dir"

id="${SLURM_ARRAY_TASK_ID}"
if (( id <= 9 )); then
  focal="${labels[$id]}"
  ckpt="$(ckpt_for "$focal")"
  "${HUNGER_HPC_VENV}/bin/python" "$HARNESS" --mode mixed \
    --seat "policy:${focal}=${ckpt}" --seat "policy:${focal}=${ckpt}" \
    --seat "policy:${focal}=${ckpt}" --seat "policy:${focal}=${ckpt}" \
    --environment-variant default --n-games 100 --game-seed-base 900000 \
    --output "$output_dir"
elif (( id <= 25 )); then
  read -r focal opp <<< "${pairs[$((id - 10))]}"
  "${HUNGER_HPC_VENV}/bin/python" "$HARNESS" --mode 1v3 \
    --focal "policy:${focal}=$(ckpt_for "$focal")" \
    --opponent "policy:${opp}=$(ckpt_for "$opp")" \
    --environment-variant default --n-games 100 --game-seed-base 900000 \
    --output "$output_dir"
else
  read -r a b <<< "${mixpairs[$((id - 26))]}"
  "${HUNGER_HPC_VENV}/bin/python" "$HARNESS" --mode mixed \
    --seat "policy:${a}=$(ckpt_for "$a")" --seat "policy:${a}=$(ckpt_for "$a")" \
    --seat "policy:${b}=$(ckpt_for "$b")" --seat "policy:${b}=$(ckpt_for "$b")" \
    --environment-variant default --n-games 100 --game-seed-base 900000 \
    --output "$output_dir"
fi

echo "task ${SLURM_ARRAY_TASK_ID} done"
