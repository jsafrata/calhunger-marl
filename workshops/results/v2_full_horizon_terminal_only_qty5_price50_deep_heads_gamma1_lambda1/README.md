# v2 environment — full-horizon, terminal-only reward, qty=5, price=50, deep heads, gamma=1, lambda=1

**Frozen v1 archive:** [`research/eval/results/archive/v1_early_termination_buffer_qty20_price100_linear_heads_gamma0995_lambda095/`](../archive/v1_early_termination_buffer_qty20_price100_linear_heads_gamma0995_lambda095/README.md).

**All new outputs (this file's parent directory) use the v2 configuration described below.**

## Configuration snapshot

| item | value |
|---|---|
| repo commit hash (implementation) | see `git log` for changes after `49cd059cb3fc6278f43eb29870bc94e4b78cd404` |
| `env.env.FoodTradingEnv` termination | ends at **t=180 (time_limit) OR zero alive (no_survivors)**. Sole-survivor states never end the game early. |
| reward tracker | `env.rl.rewards.TerminalOnlyRewardTracker`. Every nonterminal reward = 0. On termination: dead=-1, alive-nonwinner=0, winner=+1 (each of a k-way tie gets +1). |
| winner rule | (1) highest cash, (2) highest total inventory, (3) shared win only on exact tie. |
| `env.actions.MAX_PRICE` default | **50** (configurable via `HG_MAX_PRICE` env var or `set_action_space` at import). Legacy 100 supported for ablation. |
| `env.actions.MAX_QTY` default | **5** (configurable via `HG_MAX_QTY` env var). Supports 1, 5, 20. |
| `env.actions.NUM_ACTIONS` default | **2049** (= 1 + 8 + 40 + 2000) |
| actor architecture | `MLPTorso` (256 → LN → SiLU → 256 → LN → SiLU) → `FactoredActor` with **MLP** per-head projections `Linear→LayerNorm→SiLU→Linear` (hidden 128). Linear heads retained for ablation via `head_kind="linear"`. |
| critic | IPPO = local OBS_DIM=171; MAPPO = GLOBAL_OBS_DIM=305; both `MLPCritic(512→512→256→1)`. UNCHANGED from v1. |
| discount `gamma` | **1.0** (v2 default). |
| GAE `lambda` | **1.0** (v2 default). |
| algorithms | IPPO, MAPPO (only critic input differs). |
| PPO hyperparams (unchanged) | lr=3e-4, clip=0.2, grad_clip=0.5, kl_coeff=0.0, num_epochs=4. `entropy_coeff=0.01` default. |
| per-episode telemetry | `game/n_survived_to_180`, `game/n_died_before_180`, `game/terminal_reward_per_seat`, plus per-seat `seat_{i}/{win,survived_to_180,died_before_180,terminal_rank,terminal_reward,final_total_inventory}`. |

## Contents of this directory (populated as runs land)

- `scripted_v1_fast_v2/` — Scripted-v1 Fast under the corrected v2 env (see `4_fast.jsonl`).
- `bc_diagnostic/` — BC linear-vs-MLP comparison (`shared_dataset/`, `linear/`, `mlp/`).
- `long_baseline/` — v2 IPPO/MAPPO RL baselines (**not yet run** — awaits sign-off).

## What is INTENTIONALLY absent

- IPPO/MAPPO RL training runs. The user explicitly requested the archive + test + Scripted + BC report first, before starting the RL baseline campaign.

## Reproducing the v2 defaults

```bash
# The action-space defaults (MAX_QTY=5, MAX_PRICE=50) are baked into env/actions.py.
# Override with env vars for ablations:
HG_MAX_QTY=20 HG_MAX_PRICE=100 python -m eval.long_baseline --algos ippo --iters 200
```
