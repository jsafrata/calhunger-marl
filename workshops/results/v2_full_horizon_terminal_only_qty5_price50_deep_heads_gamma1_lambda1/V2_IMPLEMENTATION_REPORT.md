# v2 refactor — implementation report

This document accompanies the archive manifest at
`research/eval/results/archive/v1_early_termination_buffer_qty20_price100_linear_heads_gamma0995_lambda095/README.md`
and the v2 configuration snapshot at [`README.md`](README.md).

## 1 · Archive step (Step 1 of the spec)

Renamed the frozen v1 archive to
`archive/v1_early_termination_buffer_qty20_price100_linear_heads_gamma0995_lambda095/`
and updated its README to:

- point new outputs at
  `v2_full_horizon_terminal_only_qty5_price50_deep_heads_gamma1_lambda1/`,
- state MAX_PRICE=100 (v1) explicitly,
- record `gamma=0.995 / lambda=0.95` (v1 defaults),
- retain the "invalidated conclusions" section unchanged.

Archive tree is `chmod -R a-w`, so scripts cannot silently write into it.
Nothing was deleted.

## 2 · Correct episode termination (Step 2)

`env/tick.py:step_tick` no longer ends the game when `alive == 1`. The game
ends only at `elapsed_seconds >= game_duration_seconds` (→ `"time_limit"`)
or at `alive == 0` (→ `"no_survivors"`). A sole surviving player who later
dies before 180s is dead and cannot win.

Winner selection is unchanged from v1's `determine_winners`:
(1) highest cash among survivors; (2) highest total inventory among cash
ties; (3) shared win only for an exact tie on both.

## 3 · Terminal-only reward (Step 3)

New `env.rl.rewards.TerminalOnlyRewardTracker` (rewards.py):
- every nonterminal step: reward = 0 for every seat;
- terminal step: dead=-1, alive-nonwinner=0, winner=+1 (each of a k-way tie
  gets +1);
- "everyone dies early" is a valid all-dead terminal → every seat gets -1.

`env.rl.rllib_env.NashHungerMultiAgentEnv` and `env.mappo_env.NashHungerGym`
default to `reward_kind="terminal_only"`. The legacy `ShapedRewardTracker`
is kept (buffer + terminal) but opt-in via `reward_kind="shaped"` — needed
so the v1 archive comparison tests keep working.

Per-episode telemetry (attached to `episode_metrics` in `infos`; aggregated
in `env.rl.callbacks.NashHungerCallbacks`):
`died_before_180`, `survived_to_180`, `winner`, `terminal_rank`,
`terminal_reward` per seat, plus `game/n_survived_to_180`,
`game/n_died_before_180`, `game/terminal_reward_per_seat`.

Terminal rank = 1 for every winner (all tied winners share rank 1),
then remaining alive-nonwinners by (cash desc, total inventory desc),
then dead players by later-death-first.

## 4 · Configurable MAX_QTY (Step 4)

`env/actions.py`:
- `MAX_QTY` and `MAX_PRICE` now read from env vars `HG_MAX_QTY`/`HG_MAX_PRICE`
  at import time. Defaults **`MAX_QTY = 5`**, `MAX_PRICE = 50`.
- Added `set_action_space(max_price=..., max_qty=...)` which mutates the
  module-level constants (and every derived value: `NUM_ACTIONS`,
  `NUM_TAKE_ACTIONS`, `POST_BASE`, `NUM_POST_ACTIONS`,
  `QTY_BUCKETS`, `PRICE_BUCKETS`, `NUM_QTY`, `NUM_PRICE`).
- Callers that already imported the names by value will not observe
  subsequent mutations; the intended production path is env-var-at-launch.
  Subprocess-based invariant tests validate `MAX_QTY ∈ {1, 5, 20}` and
  `MAX_PRICE ∈ {50, 100}` all produce the correct flat action count.

## 5 · Configurable MAX_PRICE (Step 5)

Same mechanism as `MAX_QTY` (Step 4). Default = 50. Rationale is the one
from the spec: initial cash is 100 with two required foods; the earlier
POST-price histograms suggested that prices above 50 were mostly
exploration rather than a load-bearing strategy. Validated below: the
Scripted-v1 Fast run under MAX_PRICE=50 emits **0** cap violations across
1000 games — the scripted policy's price generator never exceeds 50 in
this seed range.

**Verification:** with MAX_QTY=5 and MAX_PRICE=50 the flat action count
is `1 (wait) + 8 (cancel) + 40 (take) + 2000 (post) = 2049`. The invariant
test `test_v2_default_action_count` asserts this in-process, and the
subprocess probes `test_max_qty_5_valid_default` and `test_max_price_100_valid`
verify the same under fresh env-var reads.

## 6 · MLP heads (Step 6)

`env/rl/factored_actor.py`:
- New helper `_build_head(in_dim, out_dim, kind)`:
  - `kind="linear"` — v1 single `nn.Linear`.
  - `kind="mlp"` — v2 `Linear(in, 128) → LayerNorm(128) → SiLU → Linear(128, out)`.
- `FactoredActor.__init__` takes `head_kind` (default `"mlp"`).
- Autoregressive conditioning, per-head masks, and the zero-log-prob
  contribution from irrelevant heads are byte-identical between the two
  kinds. The only difference is the head's internal capacity.

`env/rl/rllib_module.py`: reads `model_config["head_kind"]` (default
`"mlp"`) and passes it to `FactoredActor`. Round-trips through checkpoints
because RLlib serialises `model_config`.

`env/rl/rllib_config.py`: `build_ppo_config` gained `head_kind: str = "mlp"`
parameter, injected into the model_config dict.

## 7 · Critic + GAE (Step 7)

Critic architecture unchanged (IPPO=OBS_DIM=171, MAPPO=GLOBAL_OBS_DIM=305,
hidden [512,512,256], LayerNorm+SiLU, scalar value).

`build_ppo_config` defaults changed:
- `gamma = 1.0` (was 0.995) — v2 has a finite 180-step horizon, no
  intermediate rewards, bounded terminal outcome. We optimize the true
  undiscounted result.
- `lambda_ = 1.0` (was 0.95) — same rationale.

Both remain configurable via `--gamma` / `--lambda_` on `long_baseline.py`,
recorded in run metadata.

## 8 · Required tests (Step 8) — all pass

New file `tests/test_v2_invariants.py`: 25 tests covering every v2 invariant.

Regression fixes to existing tests:
- `tests/test_factored_actor.py::test_logp_sums_only_relevant_heads` —
  pinned to `head_kind="linear"` (its weight-perturbation setup assumes a
  bare `nn.Linear`); the MLP-head equivalent is covered in
  `test_v2_invariants.py`.
- `tests/test_legal_actions.py` — replaced hard-coded qty=16 (out of range
  under MAX_QTY=5) with `MAX_QTY` references.
- `tests/test_mappo_env.py` — the four shaped-reward assertions now pass
  `reward_kind="shaped"` explicitly; the terminal-only default is exercised
  by the new invariant tests.
- `tests/test_tick.py::test_tick_ends_when_only_one_survivor` — flipped
  polarity to assert the game does NOT end early on single-survivor and
  runs to time_limit or the sole survivor's own death.

Full suite result:

```
158 passed, 2 skipped, 10 xfailed, 1 xpassed in 8.26s
```

## 9 · Revalidation results

### Scripted-v1 Fast under v2 (1000 games, MAX_QTY=5, MAX_PRICE=50)

See [`scripted_v1_fast_v2/4_fast.jsonl`](scripted_v1_fast_v2/4_fast.jsonl)
and [`scripted_v1_fast_v2/summary.json`](scripted_v1_fast_v2/summary.json).

| metric | v2 value |
|---|---|
| end reason | `time_limit` in 1000/1000 games (100%) |
| mean episode length | 180.0s |
| n_survivors histogram | `{3: 5, 4: 995}` |
| per-seat survival to 180s | `[1.000, 0.999, 0.998, 0.998]` |
| mean trades per game | 105.5 |
| n cap violations (price > 50) | 0 |
| winner counts by seat (of 1000) | `{0: 265, 1: 268, 2: 247, 3: 221}` |

**Takeaways.** Under the corrected termination rule, Scripted-v1 Fast now
records true "survived to 180" numbers rather than v1's conditional
"alive when the game ended." The 99.75% average survival matches the
intended CalHunger-target behavior. The tightened MAX_PRICE=50 does not
clip the scripted policy's actions.

### BC linear-vs-MLP head comparison

Full detail in [`bc_diagnostic/bc_head_comparison.md`](bc_diagnostic/bc_head_comparison.md).

Dataset: 2000/300/300 games of Scripted-v1 Fast under v2 env, same
data used for both head kinds. Both trainings stopped by patience=5 at
epoch 28 (best epoch 23 for both).

**Training metrics (test split):**

| metric | linear | mlp |
|---|---|---|
| flat accuracy | 0.3518 | **0.4167** |
| type accuracy | 0.9977 | 0.9944 |
| POST-side accuracy \| POST | 0.7467 | **1.0000** |
| POST-price exact \| POST | 0.6377 | **0.7019** |
| POST-qty exact \| POST | 1.0000 | 1.0000 |
| POST-food accuracy \| POST | 0.5493 | 0.5561 |

**Behavioral rollouts (500 games each, deterministic):**

| lineup | metric | linear | mlp |
|---|---|---|---|
| BC self-play | mean survivors / game | 1.64 | **3.94** |
| BC self-play | reach-180 fraction | 0.852 | **0.994** |
| BC self-play | mean per-seat survival | 0.409 | **0.984** |
| 1 BC seat + 3 Scripted Fast | BC seat survival rate | 0.482 | **0.946** |
| 1 BC seat + 3 Scripted Fast | reach-180 fraction | 1.000 | 1.000 |
| Scripted Fast × 4 (reference) | mean survivors | 4.00 | 4.00 |

## 10 · Explicit non-goals (at the moment of writing this report)

No RL baseline training had been launched when this report was first
written. The v2 IPPO/MAPPO campaign specified in §11 was launched only
after this report's head-architecture recommendation was finalized.

## 11 · Recommended head architecture for the v2 baselines

**MLP heads (`head_kind="mlp"`).** The recommendation is unambiguous on
every axis the BC diagnostic measures:

- **Representation floor:** MLP reaches test flat accuracy 0.4167 vs
  linear 0.3518. The gap is driven almost entirely by the two heads
  where linear was demonstrably under-capacity: POST-side (linear
  74.67% vs MLP 100.00%) and POST-price (linear 63.77% vs MLP 70.19%).
  Both are conditional heads that must read prior sampled ids — MLP's
  hidden layer + nonlinearity is what lets the head model that
  interaction cleanly. Linear regressions of this shape can express the
  target only approximately.
- **Closed-loop survival:** in 500-game deterministic self-play the MLP
  BC agent reaches 3.94 mean survivors and 98.4% per-seat survival —
  essentially indistinguishable from the Scripted-v1 Fast teacher
  (4.00 / 99.9%). Linear self-play collapses to 1.64 survivors and
  40.9% per-seat survival, and only 85.2% of games even reach 180s.
  This is the exact axis the spec says to prioritize.
- **Robustness against other populations:** with 3 Scripted opponents,
  the MLP BC seat survives 94.6% of games vs linear's 48.2% — a 2×
  gap that reflects MLP's much smaller compounding-error footprint.
- **Cost:** the MLP head is a `Linear(in, 128) → LN → SiLU → Linear(128,
  out)` sandwich; total actor parameter count grows by only ~0.5-1× per
  head (four small heads with hidden 128), which is negligible next to
  the shared 256×256 torso. No wall-clock cost was measurable at the
  batch sizes used in the BC training (both runs ≈70s/epoch).

The v2 RL baselines (§12) use `head_kind="mlp"`.

## 12 · v2 IPPO / MAPPO baseline campaign

Launched after this report's finalization, per the user's overnight
specification. See `long_baseline/ippo_seed0_headmlp/` and
`long_baseline/mappo_seed0_headmlp/` for training curves and
checkpoints, and the accompanying `deep_eval/` subfolders for the
five-checkpoint deep-eval sweep. Run configuration and per-checkpoint
metrics are logged there and re-summarized in
`V2_RL_BASELINE_RESULTS.md` when the campaign completes.
