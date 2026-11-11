# Resnick campaign: small_150 discovery cliff, seeds 2-9

Purpose: characterize the bimodal 0%-survival discovery cliff at T=150
with a defensible n. In the existing episode-matched runs, 3 of the 4
small_150 runs (IPPO/MAPPO x seeds 0-1) sit at 0% survival — the horizon
where policies either discover the surviving trade equilibrium or never
do. This campaign adds 8 fresh seeds (2-9) for both IPPO and MAPPO,
n=16 new runs (n=20 total at T=150 combined with seeds 0-1), enough to
estimate the discovery probability per algorithm instead of anecdote.

Configuration is identical to the `episode_matched_5env_2seed_cpu`
campaign's small_150 arm — same episode-matched batch (45 games per
update = 6,750 env steps, minibatch 1,687), 1,200 iterations, checkpoints
at 100, 200, ..., 1200, 20-hour walltime, entropy 0.01, terminal-only
reward, MLP heads. Only the seed set and array size differ.

One sixteen-task array is submitted. IDs 0-7 are IPPO seeds 2-9;
IDs 8-15 are MAPPO seeds 2-9. Outputs land under
`HUNGER_CAMPAIGN_ROOT/small_150/` (fresh campaign root, distinct from the
episode-matched roots).

## Deploy-time checks (cannot be completed on the laptop)

Staged and plan-tested locally; run per `hpc/DEPLOY.md` after this
directory is added under `research/hpc/`:

- `preflight.py --strict` against the locked HPC venv
  (`hunger-v2-ray253-cpu`): Python 3.11 + locked package versions only
  match on Resnick.
- `validate_plan.sh` via `python -m hpc.small150_cliff_8seed_cpu.run_task`
  (the `hpc.`-qualified module path exists only once the directory is in
  place); expect 16 planned commands.
- Check partition queue depth: this array requests 16 concurrent
  20-hour tasks (%16 throttle); reduce the throttle at submit time if
  the expansion queue is contended.

## Configure and validate on Resnick

```bash
cd /resnick/home/$USER/Hunger_games_2/research
cp hpc/small150_cliff_8seed_cpu/campaign.env.example \
  hpc/small150_cliff_8seed_cpu/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' \
  hpc/small150_cliff_8seed_cpu/campaign.env
source hpc/small150_cliff_8seed_cpu/campaign.env

"$HUNGER_HPC_VENV/bin/python" \
  -m hpc.small150_cliff_8seed_cpu.preflight \
  --research-root "$HUNGER_RESEARCH_ROOT" \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT" \
  --strict

bash hpc/small150_cliff_8seed_cpu/validate_plan.sh
```

## Submit and monitor

```bash
bash hpc/small150_cliff_8seed_cpu/submit_all.sh

watch -n 15 'squeue -r -u "$USER" -n hunger-cliff150 \
  -o "%.18i %.2t %.10M %.10l %.4C %R"'
```

## Verify all 16 completed runs

```bash
"$HUNGER_HPC_VENV/bin/python" \
  -m hpc.small150_cliff_8seed_cpu.verify_campaign \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT"
```
