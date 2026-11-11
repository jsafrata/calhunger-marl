# Resnick campaign: the best-config bundle (16 runs)

The "no more baselines" campaign: every currently-believed improvement is ON
in a single configuration, across four arms:

| arm | environment | torso | batch (env steps) | walltime |
|---|---|---|---|---|
| `small_90_mlp`    | 90 s  | MLP  | 4050 | 14 h |
| `small_90_lstm`   | 90 s  | LSTM | 4050 | 28 h |
| `default_180_mlp` | 180 s | MLP  | 8100 | 30 h |
| `default_180_lstm`| 180 s | LSTM | 8100 | 56 h |

Each arm is a four-task array (IPPO/MAPPO x seeds 0-1), 1,200 iterations,
checkpoints every 100. The bundle, identical in every arm:

- episode-matched batching (45 complete games per PPO update),
- survival features in the local observation (per-food starvation cover +
  consumption-cycle phase),
- previous-tick trade matrices in the centralized critic observation,
- entropy coefficient annealed 0.01 -> 0.001 over the run,
- learning rate annealed 3e-4 -> 1e-4 over the run,
- terminal-only reward, gamma = lambda = 1, head_kind = mlp, qty 5 / price 50.

The control for every comparison is the `episode_matched_5env_2seed` campaign
(same batching, none of the other updates). The MLP arms isolate the
features+annealing bundle; the LSTM arms add memory on top.

LSTM note: Ray 2.53 minibatches recurrent batches along the sequence axis;
with campaign-sized batches this is safe, but do not shrink
`train_batch_size`/`minibatch_size` below a few dozen sequences or minibatch
slices round down to zero sequences (empty-batch error).

## Configure, validate, submit (on Resnick)

```bash
cd /resnick/home/$USER/Hunger_games_2/research
cp hpc/best_config_2seed_cpu/campaign.env.example \
  hpc/best_config_2seed_cpu/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' hpc/best_config_2seed_cpu/campaign.env
source hpc/best_config_2seed_cpu/campaign.env

"$HUNGER_HPC_VENV/bin/python" -m hpc.best_config_2seed_cpu.preflight \
  --research-root "$HUNGER_RESEARCH_ROOT" \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT" --strict

bash hpc/best_config_2seed_cpu/validate_plan.sh   # 16 planned commands

bash hpc/best_config_2seed_cpu/submit_all.sh
```

If an LSTM task hits its wall, resubmit that arm with
`HUNGER_RESUME=1 sbatch --account=... --time=... \
  --export=ALL,HUNGER_ARM=<arm>,HUNGER_RESUME=1 \
  hpc/best_config_2seed_cpu/run_array.sbatch` — training resumes from the
latest checkpoint.

## After training

`bash hpc/best_config_eval/submit_evals.sh` runs the vs-scripted sweep
(4 arms x 4 bots x 100 games per checkpoint) and the self-play survival +
per-head diagnostics sweep (4 arms x 50 games per checkpoint). Outputs land
under `$HUNGER_CAMPAIGN_ROOT/_scripted_eval_100/<arm>/<bot>/` and
`$HUNGER_CAMPAIGN_ROOT/_selfplay_survival_50/<arm>/`.
