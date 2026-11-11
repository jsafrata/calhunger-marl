# Resnick campaign: episode-matched batches across five horizons

This campaign runs 20 CPU training runs: IPPO and MAPPO, seeds 0 and 1,
on the 60, 90, 120, 150, and 180-second environments. Each run trains for
1,200 iterations and saves full RLlib checkpoints at iterations 100, 200,
..., 1200. There is no smoke-test job.

The PPO batch holds 45 complete games per update, so the sampled environment
steps are 2,700, 4,050, 5,400, 6,750, and 8,100 as the horizon increases.
The 90-second reference therefore remains effectively the prior 4,096-step
batch (4,050 is 1.1% smaller). Wall limits scale in the same proportions:
8, 12, 16, 20, and 24 hours.

Five independent four-task arrays are submitted, one per horizon. Within
each array, IDs 0 and 1 are IPPO seeds 0 and 1; IDs 2 and 3 are MAPPO seeds
0 and 1. Outputs are separated into `small_60`, `small_90`, `small_120`,
`small_150`, and `default_180` directories under `HUNGER_CAMPAIGN_ROOT`.

## Configure and validate on Resnick

```bash
cd /resnick/home/$USER/Hunger_games_2/research
cp hpc/episode_matched_5env_2seed_cpu/campaign.env.example \
  hpc/episode_matched_5env_2seed_cpu/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' \
  hpc/episode_matched_5env_2seed_cpu/campaign.env
source hpc/episode_matched_5env_2seed_cpu/campaign.env

"$HUNGER_HPC_VENV/bin/python" \
  -m hpc.episode_matched_5env_2seed_cpu.preflight \
  --research-root "$HUNGER_RESEARCH_ROOT" \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT" \
  --strict

bash hpc/episode_matched_5env_2seed_cpu/validate_plan.sh
```

## Submit and monitor

```bash
bash hpc/episode_matched_5env_2seed_cpu/submit_all.sh

watch -n 15 'squeue -r -u "$USER" -n hunger-epmatch \
  -o "%.18i %.2t %.10M %.10l %.4C %R"'
```

## Verify all 20 completed runs

```bash
"$HUNGER_HPC_VENV/bin/python" \
  -m hpc.episode_matched_5env_2seed_cpu.verify_campaign \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT"
```
