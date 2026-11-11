# Resnick campaign: episode-matched five horizons, seed extension (seeds 2-4)

Purpose: extend the headline `episode_matched_5env_2seed_cpu` campaign from
n=2 to n=5 seeds per (algorithm, horizon) cell for the workshop and ICLR
papers. This campaign adds seeds 2, 3, and 4; combined with the existing
seeds 0-1 runs, every cell reaches five seeds.

This campaign runs 30 CPU training runs: IPPO and MAPPO, seeds 2, 3, and 4,
on the 60, 90, 120, 150, and 180-second environments. Each run trains for
1,200 iterations and saves full RLlib checkpoints at iterations 100, 200,
..., 1200. There is no smoke-test job. Every training flag is identical to
the seeds 0-1 campaign; only the seed set (and hence the array size) changed.

The PPO batch holds 45 complete games per update, so the sampled environment
steps are 2,700, 4,050, 5,400, 6,750, and 8,100 as the horizon increases.
Wall limits scale in the same proportions: 8, 12, 16, 20, and 24 hours.

Five independent six-task arrays are submitted, one per horizon. Within
each array, IDs 0-2 are IPPO seeds 2, 3, 4; IDs 3-5 are MAPPO seeds 2, 3, 4.
Outputs are separated into `small_60`, `small_90`, `small_120`, `small_150`,
and `default_180` directories under `HUNGER_CAMPAIGN_ROOT` (a fresh
campaign root distinct from the seeds 0-1 root; run directories are named
`{algo}_seed{seed}_headmlp`, so seeds 2-4 cannot collide with seeds 0-1
even under a shared root, but keep the roots separate for clean rsyncs).

## Deploy-time checks (cannot be completed on the laptop)

Staged and plan-tested locally; the following require the cluster and are
run per `hpc/DEPLOY.md` after this directory is added under `research/hpc/`:

- `preflight.py --strict` against the locked HPC venv
  (`hunger-v2-ray253-cpu`): Python 3.11 + locked ray/torch/numpy versions
  differ from any laptop venv, so version checks only pass on Resnick.
- `validate_plan.sh` via `python -m hpc.episode_matched_5env_seeds234_cpu.run_task`
  (the `hpc.`-qualified module path exists only once the directory is in
  place under `research/hpc/`); expect 30 planned commands.
- Storage-path check for `HUNGER_CAMPAIGN_ROOT` (only enforced under
  `SLURM_JOB_ID`).

## Configure and validate on Resnick

```bash
cd /resnick/home/$USER/Hunger_games_2/research
cp hpc/episode_matched_5env_seeds234_cpu/campaign.env.example \
  hpc/episode_matched_5env_seeds234_cpu/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' \
  hpc/episode_matched_5env_seeds234_cpu/campaign.env
source hpc/episode_matched_5env_seeds234_cpu/campaign.env

"$HUNGER_HPC_VENV/bin/python" \
  -m hpc.episode_matched_5env_seeds234_cpu.preflight \
  --research-root "$HUNGER_RESEARCH_ROOT" \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT" \
  --strict

bash hpc/episode_matched_5env_seeds234_cpu/validate_plan.sh
```

## Submit and monitor

```bash
bash hpc/episode_matched_5env_seeds234_cpu/submit_all.sh

watch -n 15 'squeue -r -u "$USER" -n hunger-epmatch-s234 \
  -o "%.18i %.2t %.10M %.10l %.4C %R"'
```

## Verify all 30 completed runs

```bash
"$HUNGER_HPC_VENV/bin/python" \
  -m hpc.episode_matched_5env_seeds234_cpu.verify_campaign \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT"
```
