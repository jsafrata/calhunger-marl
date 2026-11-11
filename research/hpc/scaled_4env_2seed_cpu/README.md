# Resnick campaign: four scaled environments, two seeds

This campaign runs 16 independent CPU tasks:

| Task IDs | Environment | IPPO | MAPPO |
|---|---|---|---|
| 0–3 | `small_60` | seeds 0, 1 | seeds 0, 1 |
| 4–7 | `small_90` | seeds 0, 1 | seeds 0, 1 |
| 8–11 | `small_120` | seeds 0, 1 | seeds 0, 1 |
| 12–15 | `small_150` | seeds 0, 1 | seeds 0, 1 |

Every task trains for 1,200 iterations and saves complete RLlib checkpoints
at iterations 100, 200, ..., 1200.  Training uses the entropy-0.01 baseline
configuration from the previous CPU campaigns.  There is intentionally no
smoke-test job.

Outputs are separated by environment:

```text
$HUNGER_CAMPAIGN_ROOT/
├── small_60/
│   ├── ippo_seed0_headmlp/
│   └── ...
├── small_90/
├── small_120/
└── small_150/
```

## Configure on Resnick

```bash
cd /resnick/home/$USER/Hunger_games_2/research
cp hpc/scaled_4env_2seed_cpu/campaign.env.example \
  hpc/scaled_4env_2seed_cpu/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' \
  hpc/scaled_4env_2seed_cpu/campaign.env
source hpc/scaled_4env_2seed_cpu/campaign.env
```

Run the preflight and print all 16 commands before submission:

```bash
"$HUNGER_HPC_VENV/bin/python" \
  -m hpc.scaled_4env_2seed_cpu.preflight \
  --research-root "$HUNGER_RESEARCH_ROOT" \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT" \
  --strict

bash hpc/scaled_4env_2seed_cpu/validate_plan.sh
```

The plan must contain four tasks for each environment, `--iters 1200`, and
`--checkpoint_every 100 --checkpoint_start 100`.

## Submit production

```bash
sbatch --account="$HUNGER_SLURM_ACCOUNT" --export=ALL \
  hpc/scaled_4env_2seed_cpu/run_array.sbatch
```

## Monitor

Replace `JOB_ID` with the ID returned by `sbatch`:

```bash
watch -n 15 'squeue -r -j JOB_ID -o "%.18i %.2t %.10M %.10l %.4C %R"'
```

## Verify or resume

```bash
"$HUNGER_HPC_VENV/bin/python" \
  -m hpc.scaled_4env_2seed_cpu.verify_campaign \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT"

sbatch --account="$HUNGER_SLURM_ACCOUNT" --array=<task-id> \
  --export=ALL,HUNGER_RESUME=1 \
  hpc/scaled_4env_2seed_cpu/run_array.sbatch
```

Resume is accepted only when `iters.jsonl` ends exactly at the latest saved
checkpoint, preventing duplicated iteration rows.
