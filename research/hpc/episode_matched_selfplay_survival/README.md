# Self-play survival diagnostics for the episode-matched campaign

One Slurm array (5 tasks, one per trained horizon: `small_60`, `small_90`,
`small_120`, `small_150`, `default_180`). Each task sweeps every checkpoint
(100, 200, ..., 1200) of all four runs (IPPO/MAPPO x seeds 0-1) in that
horizon's output directory and plays 50 four-seat self-play games per
checkpoint at `--game-seed-base 970000` (matching the earlier
`selfplay_survival_5env` sweep, so numbers are directly comparable).

Beyond survival, every summary row and raw report now carries per-head
policy diagnostics (`type_prob_post_mean`, `type_entropy_mean`,
`price_given_post_entropy`) computed on each checkpoint's own self-play
observations — the trajectory that decides whether entropy annealing alone
tames price noise or per-head normalization is needed.

Outputs land under
`HUNGER_CAMPAIGN_ROOT/_selfplay_survival_50/<horizon>/`:
`selfplay_survival_diagnostics.{csv,json}` plus `raw/` per-checkpoint
reports. Reruns resume from the raw cache; pass `--overwrite` in the sbatch
to force fresh rollouts.

## Run

```bash
cd "$HUNGER_RESEARCH_ROOT"
source hpc/episode_matched_5env_2seed_cpu/campaign.env
bash hpc/episode_matched_selfplay_survival/submit_all.sh
```

Requires the training campaign's checkpoints to be complete for all five
horizons; the job validates every expected checkpoint path up front and
fails fast if any are missing.
