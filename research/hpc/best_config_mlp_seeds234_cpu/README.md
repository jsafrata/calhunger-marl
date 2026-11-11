# Resnick campaign: best-config MLP arms, seed extension (seeds 2-4)

Purpose: seed extension of the MLP best-config arms from
`best_config_2seed_cpu` (seeds 0-1) to n=5 seeds per (algorithm, arm) cell.
The LSTM arms (`small_90_lstm`, `default_180_lstm`) are deliberately
EXCLUDED pending the LSTM post-mortem; only the two MLP arms run here.

12 CPU training runs: IPPO and MAPPO, seeds 2, 3, and 4, across two arms:

| arm | environment | torso | batch (env steps) | walltime |
|---|---|---|---|---|
| `small_90_mlp`    | 90 s  | MLP  | 4050 | 14 h |
| `default_180_mlp` | 180 s | MLP  | 8100 | 30 h |

Each arm is a six-task array (IPPO/MAPPO x seeds 2-4), 1,200 iterations,
checkpoints every 100. The best-config bundle is identical to the seeds 0-1
campaign — flag emission is unchanged:

- episode-matched batching (45 complete games per PPO update),
- survival features in the local observation (per-food starvation cover +
  consumption-cycle phase),
- previous-tick trade matrices in the centralized critic observation,
- entropy coefficient annealed 0.01 -> 0.001 over the run,
- learning rate annealed 3e-4 -> 1e-4 over the run,
- terminal-only reward, gamma = lambda = 1, head_kind = mlp, qty 5 / price 50.

Within each array, IDs 0-2 are IPPO seeds 2, 3, 4; IDs 3-5 are MAPPO seeds
2, 3, 4. Outputs land under `HUNGER_CAMPAIGN_ROOT/<arm>/` (a fresh campaign
root distinct from the seeds 0-1 root).

## Deploy-time checks (cannot be completed on the laptop)

Staged and plan-tested locally; run per `hpc/DEPLOY.md` after this
directory is added under `research/hpc/`:

- `preflight.py --strict` against the locked HPC venv
  (`hunger-v2-ray253-cpu`): Python 3.11 + locked package versions only
  match on Resnick.
- `validate_plan.sh` via `python -m hpc.best_config_mlp_seeds234_cpu.run_task`
  (the `hpc.`-qualified module path exists only once the directory is in
  place); expect 12 planned commands, 12 x `--include_survival_features`,
  0 x `--torso lstm`.
- Decide whether the post-training eval sweeps (`best_config_eval`) should
  also cover these seeds; that harness currently targets the seeds 0-1
  campaign root and would need its own extension.

## Configure, validate, submit (on Resnick)

```bash
cd /resnick/home/$USER/Hunger_games_2/research
cp hpc/best_config_mlp_seeds234_cpu/campaign.env.example \
  hpc/best_config_mlp_seeds234_cpu/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' hpc/best_config_mlp_seeds234_cpu/campaign.env
source hpc/best_config_mlp_seeds234_cpu/campaign.env

"$HUNGER_HPC_VENV/bin/python" -m hpc.best_config_mlp_seeds234_cpu.preflight \
  --research-root "$HUNGER_RESEARCH_ROOT" \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT" --strict

bash hpc/best_config_mlp_seeds234_cpu/validate_plan.sh   # 12 planned commands

bash hpc/best_config_mlp_seeds234_cpu/submit_all.sh
```

If a task hits its wall, resubmit that arm with
`HUNGER_RESUME=1 sbatch --account=... --time=... \
  --export=ALL,HUNGER_ARM=<arm>,HUNGER_RESUME=1 \
  hpc/best_config_mlp_seeds234_cpu/run_array.sbatch` — training resumes
from the latest checkpoint.

## Verify all 12 completed runs

```bash
"$HUNGER_HPC_VENV/bin/python" -m hpc.best_config_mlp_seeds234_cpu.verify_campaign \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT"
```
