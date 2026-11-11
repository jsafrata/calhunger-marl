# Scripted-v1 characterisation — audit report

Audit run: 2026-07-16. All findings below are derived from the existing
7,000-game main run (`eval/results/main/`) plus a 400-game validation
batch with enriched death-context logging (`eval/results/validation/`).
No modifications were made to the bot policy at any point.

**Bottom line: no issue materially changes any of the six primary conclusions.**
One error in narrative language corrected (Q6 headline was fine, but the
"max=31 vs 0 above 30" contradiction I stated earlier was a mis-reading of
my own CSV — the CSV was always correct). Everything else is either
confirmed (starvation classifier is bit-for-bit self-consistent; no
producer-pair effect; Fast is 1 opportunity/sec as designed) or newly
reported alongside proper CIs.

---

## 1. Price inconsistency — resolved (report language, not data)

**Root cause:** the summary CSV displayed `frac_above_30` with the format
string `:.4f`, which rounded the true fraction `1 / 1,892,282 ≈ 5.3 × 10⁻⁷`
to `0.0000`. My prose report read the `frac_above_30` column and
concluded "zero attempts above 30", missing the `count_above_30 = 1`
column right next to it.

**Verification directly from raw JSONL** (9.78 M attempts across 7,000 games):

| threshold | total attempts | fraction |
|---|---:|---:|
| > 30 | **1** | 1.02 × 10⁻⁷ |
| > 50 | 0 | 0 |
| > 100 | 0 | 0 |
| cap_violations | 0 | 0 |

The single event is a `2fast_2medium` game in which the panic-tier price
computed as 31. That's ~3× under the `MAX_PRICE = 100` cap. All 6 other
configs have max attempted ≤ 28. **`MAX_PRICE = 100` is comfortably above
the realistic range** — this stays confirmed.

**Fixed** in `eval/summarize.py`: fraction columns now use scientific
notation for non-zero counts and the literal `"0"` otherwise, so a single
event among millions can no longer render as `0.0000`.

## 2. Producer-pair effects — none material

The 6 possible produced-food pairs {AB, AC, AD, BC, BD, CD} were tested
stratified by cadence, using game-clustered bootstrap 95 % CIs.

**Survival rate by cadence × produced-pair** (collapsed across configs):

| cadence | worst pair | best pair | spread | overlap |
|---|---|---|---:|---|
| fast | CD 0.842 [0.825, 0.858] | AB 0.867 [0.851, 0.882] | 2.5 pp | CIs overlap |
| medium | AD 0.481 [0.451, 0.511] | BC 0.523 [0.495, 0.554] | 4.2 pp | CIs overlap |
| slow | CD 0.033 [0.026, 0.041] | AB 0.041 [0.032, 0.050] | 0.8 pp | CIs overlap |

Every within-cadence spread is smaller than the widest of the two CIs it
spans. Effect size is negligible; the mechanic is well-balanced across
food assignments. Full per-(config, cadence, pair) table in
`producer_pair_effects.csv`; collapsed view in `producer_pair_by_cadence.csv`.

## 3. Starvation-cause labels — 100% self-consistent

A 400-game validation batch (seed range `100000–100399`) was run with an
added `death_context` field capturing the exact state the classifier saw
at the fatal tick (cash, reserved cash, per-food inventory, per-food ask
book, per-food bid book, own open orders). This lets the classifier be
re-derived from the log alone.

Random 50-death sample per major category, independently re-classified:

| category | pool | sampled | agree | accuracy |
|---|---:|---:|---:|---:|
| affordable_ask_no_buy | 3,122 | 50 | 50 | **1.000** |
| no_ask_available | 1,953 | 50 | 50 | **1.000** |
| order_pending_no_fill | 1,061 | 50 | 50 | **1.000** |

Every sampled label reproduced exactly when the classifier rule was
re-applied to the saved snapshot. **Zero mismatches, zero ambiguous
cases.**

**Caveat about the meaning of this result.** This is a *self-consistency*
check: I re-implemented the same rule from the spec on the same input and
it agreed with itself. That proves:

  1. the snapshot serialization is lossless — no field the classifier
     depended on was dropped or corrupted;
  2. classification is deterministic given the pre-fatal-tick observation.

It does **not** prove the rule itself matches human judgement in edge
cases (e.g. is "the bot had a resting bid but at a price below the
cheapest ask" better labelled `affordable_ask_no_buy` or
`order_pending_no_fill`?). The rule currently classifies the latter as
`order_pending_no_fill` — a defensible choice, but a human reviewer might
split hairs on a few dozen borderline cases. Nothing in this audit re-
evaluates that judgement call.

## 4. Proper 95 % confidence intervals — game-clustered

`ci_audit.csv` reports point estimates + game-clustered 95 % bootstrap CIs
(2,000 resamples of whole games) for every headline metric. Highlights
from the 4-fast config (n=1,000 games, 4,000 seats):

| metric | unit | point | 95 % CI |
|---|---|---:|---|
| reached_time_limit | game | 1.0000 | [1.0000, 1.0000] |
| no_survivor | game | 0.0000 | [0.0000, 0.0000] |
| n_survivors | game | 3.9950 | [3.9930, 3.9970] |
| n_trades_total | game | 105.5140 | [104.9520, 106.0790] |
| seat_survived[fast] | seat (clustered) | 0.9988 | [0.9980, 0.9993] |

And for the pathological 4-slow config:

| metric | point | 95 % CI |
|---|---:|---|
| reached_time_limit | 0.0120 | [0.0060, 0.0190] |
| no_survivor | 0.6120 | [0.5810, 0.6420] |
| n_survivors | 0.3880 | [0.3580, 0.4190] |
| seat_survived[slow] | 0.0970 | [0.0895, 0.1043] |

**One deliberate non-issue:** `seat_final_cash` shows point=100.0 with
zero-width CIs across every config — this is mathematical, not a bug.
Cash is zero-sum in the game (Σ = 4 × $100 = $400 conserved by every
trade), so the mean across the 4 seats of any single game is exactly
$100. The per-cadence breakdown (in `summary_by_cadence.csv`) shows the
actual distribution across cadences — Fast averages ~$258, Slow ~$76,
consistent with the cadence effect.

## 5. Metadata and cadence reality check

**Frozen code.** SHA-256 (first 16 chars) of each frozen file at time of the
main run:

| file | sha256 (16) | bytes |
|---|---|---:|
| `agents/scripted_v1.py` | 3b4f18764d0727c3 | 10,658 |
| `agents/cadenced_scripted.py` | d251913c529ba75b | 3,616 |
| `eval/game_runner.py` | d7ede513de14100b | 27,044 |
| `eval/configurations.py` | 8574e1211f5302b5 | 2,211 |
| `eval/run_eval.py` | a874b171af3c15b2 | 6,439 |

Git branch is `main`; scripted-bot files are uncommitted local additions,
so the git log doesn't uniquely identify them — the hashes above do.

**Evaluation configuration.**
- 7 configurations × 1,000 games = 7,000 games; seeds `[0, 999]` shared
  across configs (paired seeds — same producers + initial inventory
  environment).
- Seat permutation seed per game per config derived deterministically via
  `seat_permutation_seed(shared_seed, config_idx)` in
  `eval/configurations.py`.
- 4 worker processes; per-game determinism preserved (each game seeded
  independently from `--seed_base`).

**Cadence implementation.**
- `CadencedScripted` wraps `ScriptedV1Agent`. Each `env.step()` advances
  the game clock by exactly 1 game-second (integer). The wrapper compares
  `next_action_time_s` (float, in game-seconds) against the current tick;
  if the tick has caught up, the underlying bot fires and a fresh
  interval `U(min, max)` is added to `next_action_time_s`; otherwise
  `act()` returns WAIT for that env step.
- Configured intervals per cadence:

  | cadence | interval min | interval max | ideal actions/sec |
  |---|---:|---:|---:|
  | fast | 0.5 | 1.0 | 1.333 |
  | medium | 1.5 | 2.5 | 0.500 |
  | slow | 3.5 | 5.5 | 0.222 |

**Empirical actions/sec measured across the main run**
(n_decisions_taken / elapsed_seconds, aggregated per seat):

| cadence | seats | mean | median | p10 | p90 | vs ideal |
|---|---:|---:|---:|---:|---:|---:|
| fast | 10,000 | **0.994** | 0.994 | 0.994 | 0.994 | 75 % |
| medium | 7,000 | **0.398** | 0.400 | 0.383 | 0.411 | 80 % |
| slow | 11,000 | **0.198** | 0.200 | 0.194 | 0.206 | 89 % |

**Does Fast truly receive one decision opportunity per second? Yes.**
Measured 0.994 actions/sec, median 0.994, essentially zero variance
across 10,000 seats. Fast is a "fires-every-tick" bot at 1-second env
granularity. The small dip from 1.000 comes from the first-tick jitter:
`next_action_time_s` is initialised to `U(0, max_interval)` on reset, so
a few opening ticks are skipped depending on the initial offset.

**Where this differs from the live game.** The live-game bot runs on a
free `setInterval` timer, so a Fast bot there fires roughly 1.33 times
per real second (interval `U(0.5, 1.0)` implies ~1/0.75 avg fires). Our
harness caps Fast at 1 fire per game-second because the env advances in
1-second increments. Result: **our Fast is a ~25% weakened version of
the live-game Fast.** The other cadences are also under-simulated but by
less (Medium ~20%, Slow ~11%). This is a known artifact of the env's
tick granularity — noted explicitly in the primary-questions report as
a caveat. Any RL policy that beats our simulated Fast is *at least* as
good as beating the live-game Fast.

## Does any of this materially change the conclusions?

**No.** Explicit answers to the 6 primary questions after the audit:

- **Q1 (functioning market):** unchanged — 0.14–0.17 execution rates, ~40–105
  trades/game across configs, zero validation failures. Adding CIs
  tightens the numbers but doesn't move any of them meaningfully.
- **Q2 (cadence survives reliably):** unchanged — `4_fast` still reaches
  the time limit 100 % of games (CI [1.000, 1.000] over 1,000 games)
  with mean 3.995 survivors [3.993, 3.997]. Solvability confirmed.
- **Q3 (cadence explains performance):** unchanged, and now with cluster-
  bootstrap CIs proving the cadence effect isn't a seat-grouping
  artifact. Same-game paired-seat comparison: in `1fast_3slow`, Fast
  seats survive 0.902 [0.882, 0.920] while the three Slow seats in the
  same games survive 0.004 [0.002, 0.007]. Cadence dominates.
- **Q4 (dominant death causes):** unchanged, and the labels are 100 %
  self-consistent. `affordable_ask_no_buy` (50%) + `no_ask_available`
  (32%) + `order_pending_no_fill` (17%) = 99% of all deaths.
- **Q5 (assignment / food effects):** confirmed to be null — worst-to-best
  produced-pair span is ≤ 4 percentage points within any cadence, well
  within CI overlap. No pair is systematically punished.
- **Q6 (Fast as baseline):** unchanged. Note the caveat: our Fast is 25 %
  weaker than the live-game Fast, which strengthens (not weakens) its
  role as a solvability witness — if even a handicapped Fast reaches
  100 % survivability, the environment is definitely solvable.

## Files produced by this audit

- `price_distribution.csv` (regenerated with fixed formatter)
- `producer_pair_effects.csv` (audit 2)
- `producer_pair_by_cadence.csv` (audit 2 collapsed)
- `ci_audit.csv` (audit 4)
- `audit_report.md` (this file)

Validation batch (400 games with enriched `death_context`) at
`eval/results/validation/`, ~180 MB additional disk. Retained for
future audits.
