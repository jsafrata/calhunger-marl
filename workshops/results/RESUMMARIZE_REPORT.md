# Resummarize-from-raw validation report

Results root: `/Users/honzasafrata/Desktop/CALTECH/Research/Hunger_games_2/research/eval/results`

Extended summaries regenerated from cached raw JSONs only — no games were re-run and Ray was never initialised.

## Metric computability from cache

Game-derived extended metrics, computable from every raw cache generation (all needed per-game fields are present in `raw_result`):

- `win_given_survival_percentage`
- `mean_trades_as_buyer`
- `mean_trades_as_seller`
- `mean_posted_bids`
- `mean_posted_asks`
- `mean_takes`
- `frac_wait`
- `frac_post`
- `survivor_frac_min_required_0_to_1`

Head-derived extended metrics, computable from cache ONLY where the raw JSON stored a `head_diagnostics` block (episode-matched and best-config campaigns). Everywhere else these columns are empty and would need re-rollouts (the observation sample `_head_diag_sample` is never persisted, so recomputing them requires restoring the checkpoint and replaying games):

- `type_prob_post_mean`
- `type_entropy_mean`
- `price_given_post_entropy`

## Directories processed

| Source dir (relative) | Rows | Head diag rows | Validation | Shared cols checked |
|---|---|---|---|---|
| `best_config_scripted_eval_100/small_90_lstm/v2` | 48 | 48 | PASS | 15 |
| `best_config_scripted_eval_100/small_90_lstm/v3` | 48 | 48 | PASS | 15 |
| `best_config_scripted_eval_100/small_90_lstm/v4` | 48 | 48 | PASS | 15 |
| `best_config_scripted_eval_100/small_90_lstm/v5` | 48 | 48 | PASS | 15 |
| `best_config_scripted_eval_100/small_90_mlp/v2` | 48 | 48 | PASS | 15 |
| `best_config_scripted_eval_100/small_90_mlp/v3` | 48 | 48 | PASS | 15 |
| `best_config_scripted_eval_100/small_90_mlp/v4` | 48 | 48 | PASS | 15 |
| `best_config_scripted_eval_100/small_90_mlp/v5` | 48 | 48 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/default_180/v2` | 48 | 48 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/default_180/v3` | 48 | 48 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/default_180/v4` | 48 | 48 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/default_180/v5` | 48 | 48 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_120/v2` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_120/v3` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_120/v4` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_120/v5` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_150/v2` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_150/v3` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_150/v4` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_150/v5` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_60/v2` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_60/v3` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_60/v4` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_60/v5` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_90/v2` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_90/v3` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_90/v4` | 48 | 0 | PASS | 15 |
| `episode_matched_5env_scripted_eval_100/small_90/v5` | 48 | 0 | PASS | 15 |
| `hpc_v2_5seed_ippo_mappo_1500/scripted_checkpoint_diagnostics` | 110 | 0 | PASS | 11 |
| `hpc_v2_5seed_ippo_mappo_1500/scripted_diag_v2` | 100 | 0 | PASS | 15 |
| `hpc_v2_5seed_ippo_mappo_1500/scripted_diag_v3` | 100 | 0 | PASS | 15 |
| `hpc_v2_entropy005_seed0_ippo_mappo_1500/scripted_checkpoint_diagnostics` | 22 | 0 | PASS | 11 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_120/scripted_diag_v2` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_120/scripted_diag_v3` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_120/scripted_diag_v4` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_120/scripted_diag_v5` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_150/scripted_diag_v2` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_150/scripted_diag_v3` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_150/scripted_diag_v4` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_150/scripted_diag_v5` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_60/scripted_diag_v2` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_60/scripted_diag_v3` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_60/scripted_diag_v4` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_60/scripted_diag_v5` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_90/scripted_diag_v2` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_90/scripted_diag_v3` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_90/scripted_diag_v4` | 28 | 0 | PASS | 15 |
| `scaled_4env_2seed_ippo_mappo_1200/_diagnostics/small_90/scripted_diag_v5` | 28 | 0 | PASS | 15 |
| `small_2seed_partial/scripted_diag_v1` | 20 | 0 | PASS | 15 |
| `small_2seed_partial/scripted_diag_v2` | 20 | 0 | PASS | 15 |
| `small_2seed_partial/scripted_diag_v3` | 20 | 0 | PASS | 15 |
| `small_2seed_partial_diagnostics_600_1000` | 20 | 0 | PASS | 14 |

## Raw-cache field gaps

None — every raw file in every processed directory contained all fields needed for the game-derived extended metrics.

## Directories skipped

- `best_config_selfplay_survival_50/small_90_lstm`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `best_config_selfplay_survival_50/small_90_mlp`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `episode_matched_5env_selfplay_survival_50/default_180`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `episode_matched_5env_selfplay_survival_50/small_120`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `episode_matched_5env_selfplay_survival_50/small_150`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `episode_matched_5env_selfplay_survival_50/small_60`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `episode_matched_5env_selfplay_survival_50/small_90`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `selfplay_survival_5env_600_1200_50games/small`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `selfplay_survival_5env_600_1200_50games/small_120`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `selfplay_survival_5env_600_1200_50games/small_150`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `selfplay_survival_5env_600_1200_50games/small_60`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `selfplay_survival_5env_600_1200_50games/small_90`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `small_2seed_partial_selfplay_diagnostics_600_1000`: raw cache is from the self-play survival diagnostics (no *_1learned_3scripted_* files)
- `archive`: no scripted-diagnostics raw/ cache anywhere under it
- `required_food_time_series`: no scripted-diagnostics raw/ cache anywhere under it
- `scripted_v1_v4_tournament_small_30s`: no scripted-diagnostics raw/ cache anywhere under it
- `scripted_v2_v5_tournament_small_30s`: no scripted-diagnostics raw/ cache anywhere under it
- `v2_full_horizon_terminal_only_qty5_price50_deep_heads_gamma1_lambda1`: no scripted-diagnostics raw/ cache anywhere under it
- `v3_shared_vs_independent`: no scripted-diagnostics raw/ cache anywhere under it
