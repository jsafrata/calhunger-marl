# Checkpoint-vs-checkpoint cross-play harness

`workshops/eval/checkpoint_policy_crossplay.py` plays learned policies
against *independently trained* learned policies (different seeds, or
IPPO vs MAPPO), the paper's missing evidence that self-play policies fail
against peers and not merely against scripted bots.

Requires `ray[rllib]==2.53.0` (the version the checkpoints were written
with) in the interpreter, and `PYTHONPATH` pointing at `research/` so
Ray's worker actors can import the env packages.

## Usage

```bash
# 1v3: focal policy in 1 seat vs 3 seats of an opponent policy
PYTHONPATH=research python workshops/eval/checkpoint_policy_crossplay.py \
    --mode 1v3 \
    --focal    "policy:ippo_s0=<ckpt-dir>" \
    --opponent "policy:ippo_s1=<ckpt-dir>" \
    --environment-variant small --n-games 100 \
    --output workshops/results/policy_crossplay/small30

# mixed: four arbitrary seats (learned and/or scripted)
PYTHONPATH=research python workshops/eval/checkpoint_policy_crossplay.py \
    --mode mixed \
    --seat "policy:ippo_s0=<ckpt>" --seat "policy:mappo_s1=<ckpt>" \
    --seat "scripted:v4" --seat "scripted:v5" \
    --environment-variant default --n-games 100 \
    --output workshops/results/policy_crossplay/mixed180
```

Spec grammar:

- `policy:LABEL=PATH` — RLlib checkpoint dir (contains
  `rllib_checkpoint.json`), shared policy.
- `policy:LABEL=PATH@pi_0` — explicit module id (v3 independent runs).
- `scripted:v1` .. `scripted:v5` — scripted bot, fast cadence.

Flags: `--n-games` (multiple of 4), `--game-seed-base` (default 900000,
same convention as `checkpoint_scripted_diagnostics`), `--deterministic`
(argmax instead of stochastic sampling), `--overwrite`,
`--environment-variant` in `default` (180 s), `small` (30 s),
`small_60/90/120/150`.

A self-play baseline for policy X is a `mixed` run with the same
`policy:` spec in all four seats (the row reports `opponent=self`).

## Guarantees

- **Seat balance**: cyclic rotation puts every spec in every physical
  seat exactly `n_games / 4` times (the 1v3 focal seat included).
- **Seed identity**: game `g` always plays with seed
  `game_seed_base + g` and the same rotation, so every pair sees
  identical initial states; `seed_everything(game_seed_base)` is called
  per job so the stochastic sampling stream is also identical.
- **Cache/resume**: each job writes `raw/<focal>_vs_<opponent>_...json`
  with a full identity block (specs with resolved paths, mode, seed
  base, env). Re-running with the same identity reuses the cache;
  a mismatched cache file raises unless `--overwrite`.
- **Accumulating matrix**: `crossplay_matrix.csv` (+ `.json`) is rebuilt
  from *all* raw files in the output dir after every run, so the full
  matrix accumulates across single-pair invocations into one CSV.
- **Invariants**: every action is checked against `env.legal_actions`
  per step (same invariant as `tests/scripted_harness.py`).

Metrics per row (one row per distinct spec per job):
`survival_percentage`, `win_percentage`,
`win_given_survival_percentage`, `mean_terminal_cash`,
`mean_terminal_cash_given_survival`, `mean_survival_time_ticks`,
`mean_trades_as_buyer`, `mean_trades_as_seller`.

Tests: `PYTHONPATH=research pytest workshops/eval/test_policy_crossplay.py`
(18 tests: lineup balance, spec parsing, seed determinism, CSV schema,
cache identity).

## Planned full matrix

Per horizon, all ordered 1v3 pairs (A vs 3xB is a different measurement
from B vs 3xA) in both sampling modes, 100+ games each, plus uniform
self-play baselines:

| Campaign (local checkpoints) | Env | Policies | 1v3 jobs/mode | Modes | Total jobs |
|---|---|---|---|---|---|
| `hpc_v2_5seed_ippo_mappo_1500` @ iter_1500 | default 180 s | ippo/mappo x seeds 0-4 = 10 | 90 + 10 self | 2 | 200 |
| `small_2seed_partial` @ iter_1000 | small 30 s | ippo/mappo x seeds 0-1 = 4 | 12 + 4 self | 2 | 32 |
| `v3_shared_vs_independent` @ iter_1100 | default 180 s | shared s0-2, shared_id s0-2, independent s0 (pi_0..pi_3 via `@pi_i`) | cross-condition pairs (~42) + self | 2 | ~90 |

Optional extensions: `hpc_v2_entropy005_seed0` and the
`v2_full_horizon...` long-baseline seed-0 checkpoints (both 180 s) as
extra columns; scripted `v1`-`v5` opponents as calibration rows.

**Blocker**: no local checkpoints exist for the scaled 60/90/120/150-second
campaigns (`scaled_4env_2seed...` and `episode_matched_5env...` result
dirs hold only eval outputs) — the per-horizon sweep beyond 30 s and
180 s needs checkpoints pulled from the HPC or the harness run there.

## Runtime (measured on the smoke runs, Apple Silicon, 1 process)

- Checkpoint load: ~4-5 s per distinct checkpoint (once per job).
- Small 30 s env: ~1.0-1.4 s/game (8 games incl. two loads: 15.5 s).
- Default 180 s env: ~1.1-2.3 s/game (8 games incl. two loads: 18.2 s).

At 100 games a 1v3 job is roughly 2-3 minutes, so:
180 s full matrix (200 jobs) ≈ 7-10 h serial; small 30 s matrix
(32 jobs) ≈ 1-1.5 h; v3 matrix ≈ 3-4 h. All resumable via the raw
cache; shard across processes by giving each shard its own pair list
(same `--output` dir is safe as long as shards run disjoint pairs, then
re-run any one job to rebuild the merged CSV).

## Smoke results (8 games, seed base 900000)

`workshops/results/policy_crossplay/smoke/` — small 30 s env,
`small_2seed_partial` iter_1000:

| Matchup | Mode | Survival | Win | Mean cash |
|---|---|---|---|---|
| ippo_s0 self-play | stoch | 75.0% | 25.0% | 100.00 |
| ippo_s1 self-play | stoch | 93.8% | 25.0% | 100.00 |
| ippo_s0 vs 3x ippo_s1 | stoch | 75.0% | 0.0% | 59.00 |
| ippo_s1 (3 seats) vs ippo_s0 | stoch | 87.5% | 33.3% | 113.67 |
| ippo_s0 vs 3x ippo_s1 | det | 87.5% | 12.5% | 49.38 |

`workshops/results/policy_crossplay/smoke_default180/` — default 180 s
env, `hpc_v2_5seed` iter_1500, stochastic:

| Matchup | Survival | Win | Mean cash |
|---|---|---|---|
| ippo_s0 self-play | 65.6% | 25.0% | 100.00 |
| ippo_s1 self-play | 0.0% | 0.0% | 100.00 |
| ippo_s0 vs 3x ippo_s1 | 75.0% | 75.0% | 201.88 |
| ippo_s1 (3 seats) vs ippo_s0 | 0.0% | 0.0% | 66.04 |

Two lessons for the full matrix. (1) The diagonal (self-play baselines)
is mandatory: ippo_s1's final checkpoint dies in its *own* self-play at
these seeds, so its 0% cross-play survival is checkpoint weakness, not
cross-play incompatibility — without the diagonal the two are
indistinguishable. (2) The small-env pair shows the genuine cross-play
effect: s0 and s1 both survive their own self-play (75% / 93.8%) with
25% wins and cash 100, but s0 facing s1 wins 0% of games with mean cash
59 — surviving the horizon while being out-traded by an
independently trained peer.
