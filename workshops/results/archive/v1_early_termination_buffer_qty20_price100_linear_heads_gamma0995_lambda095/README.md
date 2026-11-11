# v1 archive — early-termination + buffer reward + qty20 + price100 + linear heads + gamma 0.995 + lambda 0.95

**Frozen on:** 2026-07-19
**Status:** immutable. Do not modify or add files. All future results go under
`research/eval/results/v2_full_horizon_terminal_only_qty5_price50_deep_heads_gamma1_lambda1/`
(or the next versioned directory).

**These are legacy diagnostic results, NOT final baselines for the intended CalHunger game.**
See §"Explicit note on early single-survivor termination" below.

## Environment / code state

| item | value |
|---|---|
| repo commit hash | `0a57e1a1df757da22a16ffed0082fb1529deb841` |
| `env.env.FoodTradingEnv` termination | ends at **single_survivor OR zero_alive OR time_limit** — early termination when only one player remains |
| `env.rl.rewards.ShapedRewardTracker` | dense buffer shaping every consumption tick (`SURVIVAL_REWARD_SCALE=0.01 × min(20, min_required_food_buffer)`) + terminal (defaults: win=+5 split k-way, survive=+1, dead=0) |
| survival-variant terminal (A/B) | win=+5 no split, survive=+2, dead=-5 |
| `env.actions.MAX_PRICE` | 100 (module-level constant, not configurable in v1) |
| `env.actions.MAX_QTY` | 20 (module-level constant, not configurable in v1) |
| `env.actions.NUM_ACTIONS` | 16169 (= 1 + 8 + 160 + 16000) |
| PPO discount `gamma` | 0.995 |
| GAE `lambda` | 0.95 |
| actor architecture | `MLPTorso` (256 → LN → SiLU → 256 → LN → SiLU) → `FactoredActor` with **linear** per-head projections (type/food/side/price/qty heads = single `nn.Linear`) + 16-dim conditioning embeddings |
| critic | IPPO = local OBS_DIM=171; MAPPO = GLOBAL_OBS_DIM=305; both `MLPCritic(512→512→256→1)` |
| entropy_coeff (baseline) | 0.01 |
| entropy_coeff (ablation) | 0.05 |
| algorithms | IPPO, MAPPO (only critic input differs) |
| seed used for all training | 0 (single seed per variant) |
| PPO hyperparams | gamma=0.995, lambda=0.95, lr=3e-4, clip=0.2, grad_clip=0.5, train_batch=4096, minibatch=1024, num_epochs=4, kl_coeff=0.0 |

## Contents of this archive

### Scripted-v1 characterization (7 cadence configs × 1000 games)

- `main/` — full run with per-tick logs, audit report, cadence + producer splits
- `validation/` — validation duplicate (fresh seeds)
- `smoke/` — reduced-scale smoke run
- `tiny/` — very small smoke

Cadence configs: 4_fast, 4_medium, 4_slow, 2fast_2medium, 2fast_2slow, 1fast_3slow, 1fast_1medium_2slow.

Files include `summary.csv`, `starvation_causes.csv`, `market_by_food.csv`,
`price_distribution.csv`, `ci_audit.csv`, `producer_pair_by_cadence.csv`,
`audit_report.md`, `primary_questions.md`, and per-config JSONL game logs.

### RL training (long_baseline/)

Per algo/variant subdirectories: each with `iters.jsonl` (per-iter metrics)
and `checkpoints/iter_0025..iter_0500`.

- `ippo/`, `mappo/` — 500-iter unchanged baselines
- `ippo_survival/`, `mappo_survival/` — 200-iter survival-reward A/B (win=+5, survive=+2, dead=-5, no split)
- `ippo_ent005/`, `mappo_ent005/` — 200-iter entropy-coefficient ablation (0.01 → 0.05, baseline reward)

### Deep evaluation (deep_eval/)

500-game deterministic + stochastic head-diagnostics per checkpoint:

- `ippo_long500.json`, `mappo_long500.json` — 500-iter baseline eval
- `ippo_survival_200.json`, `mappo_survival_200.json` — survival variant eval
- `ippo_ent005_200.json`, `mappo_ent005_200.json` — entropy variant eval
- `entropy_variant_comparison.json` — cross-variant summary
- `ippo_pilot60.json`, `mappo_pilot60.json` — early pilot eval
- `ippo_ent005_qty_hist.json`, `mappo_ent005_qty_hist.json` — POST-qty histograms (100 games self-play, both modes)
- `ent005_qty_histograms.tex` — pgfplots figure of the above

### Pilot (pilot/)

- `pilot_report.json` — earlier smoke + head-collapse + overfit + checkpoint round-trip

### Behavioral-cloning diagnostic (bc_diagnostic/)

- `train.npz`, `val.npz`, `test.npz` — 5000/500/500-game supervised dataset (Scripted-v1 Fast teacher, disjoint seeds)
- `dataset_stats.json` — type distribution (99.4% POST, 0.5% WAIT, 0% CANCEL, 0% TAKE)
- `bc_actor.pt` — cloned weights (6 epochs, batch 2048, lr 3e-4)
- `bc_train_report.json` — per-epoch curves + test metrics
- `bc_rollout_report.json` — 1000-game rollout (BC self-play, 1BC+3Fast, Scripted-v1 Fast reference)

Also: `bc_diagnostic_smoke/` (tiny end-to-end smoke).

### Logs

- `long_baseline.log`, `survival_variant.log`, `bc_diagnostic.log` — training/generation stdout for the runs above.

## Which conclusions from these runs remain useful

**Conclusions that survive the environment / reward / arch change:**

1. **The MLP torso + factored autoregressive actor + parameter sharing across seats is a viable representation.** Behavioral-cloning of Scripted-v1 Fast reaches ~102 trades/game (96% of Scripted-v1) and 24% self-play survival — several-fold better than any PPO variant to date. This lower-bounds the representation capacity independent of the reward.
2. **Simultaneous-clearing races produce ~all of the "invalid_action" signal**, which is why the per-code rejection breakdown was added (`action_rejections/no_order_to_cancel_per_seat`). The rejection code semantics are v2-portable.
3. **Ray 2.53 callback quirks:** `episode.get_infos()` returns a dict-of-lists; single-agent step-info list is truncated before terminal step so terminal-only metrics must be aggregated at learner-side or from Ray's own `episode_return_*`. Same fix applies in v2.
4. **Scripted-v1 behavioral fingerprint:** it never CANCELs or TAKEs; it only POSTs (99.4% of decisions) and cadence-WAITs (0.5%). This is a property of the scripted policy itself, not the env, and will be the same v2 baseline behavior modulo the qty change.
5. **Head diagnostic infrastructure** (per-head entropy, argmax freq, POST-conditional side/price/qty) in `eval/deep_eval.py::sample_head_diagnostics` is v2-portable — only the qty-head dimension shrinks.

**Conclusions that are INVALIDATED by v2 and must be re-derived:**

1. **All absolute survival / reach-180 numbers** — v1 games often ended at single_survivor before 180s, so per-seat survival rates conflate "alive at 180" with "alive when the game happened to end." v2 measures the intended CalHunger objective.
2. **All reward-shape conclusions** (buffer + survival + winner terminal). v2 has terminal-only rewards.
3. **The IPPO / MAPPO baseline plateau numbers** — a different reward and termination will train a different policy. The failure mode (WAIT-collapse / no TAKE / degenerate deterministic policy) may or may not reappear.
4. **POST-price and POST-qty distributions** — the qty action-space cardinality shrinks from 20 to 5 in the new default, so all qty-conditional metrics need to be recomputed.
5. **Entropy-coefficient ablation numbers.** With terminal-only rewards the entropy trade-off shifts.
6. **Absolute BC accuracy numbers.** Cardinality of qty head changes; class balance of type head under the new termination may shift (unclear how many WAIT ticks the cadence bot emits when there is no early-death exit).

## Explicit note on early single-survivor termination

**All runs in this archive used early-termination games.** Whenever three players
died and a single survivor remained, the game ended at that moment (typically
well before t=180). Under that rule:

- "survival rate" measured **conditional on the game ending**, not conditional on reaching 180s
- winners were determined at the early-termination tick, not at 180s
- players who might have died between the early-termination moment and 180s got a "survived" credit they wouldn't have earned in the corrected v2 game

**These runs are therefore NOT baselines for the intended CalHunger game.** The
v2 branch corrects this by ending only at zero-alive or at t=180.

## Do not modify

This directory is frozen. Add nothing. If a v1 artifact turns out to be needed
for cross-reference, read it in place — do not overwrite. Any v2 output goes
under `research/eval/results/v2_full_horizon_terminal_only_qty5_deep_heads/`.
