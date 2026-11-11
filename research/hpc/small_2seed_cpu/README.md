# Resnick campaign: small environment, two seeds

This campaign runs four independent CPU tasks:

- task 0: IPPO seed 0
- task 1: IPPO seed 1
- task 2: MAPPO seed 0
- task 3: MAPPO seed 1

Each task trains for 1,500 iterations in `SmallFoodTradingEnv` and saves full
RLlib checkpoints at iterations 100, 200, ..., 1500. Training otherwise uses
the entropy-0.01 v2 baseline configuration. Outputs use a separate campaign
root and cannot overwrite either prior full-game campaign.

The Slurm job runs a strict preflight that verifies the code is committed,
the locked Python dependencies match, and RLlib actually constructs the
30-second small environment with its constrained required-food dealer.

## Configure on Resnick

```bash
cd /resnick/home/$USER/Hunger_games_2/research
cp hpc/small_2seed_cpu/campaign.env.example hpc/small_2seed_cpu/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' hpc/small_2seed_cpu/campaign.env
source hpc/small_2seed_cpu/campaign.env
bash hpc/small_2seed_cpu/validate_plan.sh
```

Every printed command must contain `--env_variant small`, `--iters 1500`, and
the checkpoint schedule starting at 100.

## Submit production

```bash
sbatch --account="$HUNGER_SLURM_ACCOUNT" --export=ALL \
  hpc/small_2seed_cpu/run_array.sbatch
```

## Verify or resume

```bash
"$HUNGER_HPC_VENV/bin/python" -m hpc.small_2seed_cpu.verify_campaign \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT"

sbatch --account="$HUNGER_SLURM_ACCOUNT" --array=<task-id> \
  --export=ALL,HUNGER_RESUME=1 hpc/small_2seed_cpu/run_array.sbatch
```

Resume is accepted only when `iters.jsonl` ends exactly at the latest saved
checkpoint, preventing duplicate iteration rows.
