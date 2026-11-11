# Scripted-v1 evaluation — primary questions

Based on 100 games per configuration across 7 configurations (total games: 700).

## Q1: Does Scripted-v1 reliably sustain a functioning market?

| config | trades/game (mean) | posts/game (mean) | validation fails/seat |
|---|---:|---:|---:|
| 1fast_1medium_2slow | 48.4 | 295.7 | 0.0 |
| 1fast_3slow | 39.5 | 249.2 | 0.0 |
| 2fast_2medium | 78.3 | 495.2 | 0.0 |
| 2fast_2slow | 57.0 | 399.5 | 0.0 |
| 4_fast | 106.4 | 716.0 | 0.0 |
| 4_medium | 70.3 | 282.1 | 0.0 |
| 4_slow | 38.2 | 124.6 | 0.0 |

A functioning market means: nonzero trade counts, orders that actually cross, and validation failures at a low fraction of attempts. Judge by whether the above rows have trades > 0 across all foods (see the market_by_food.csv for the per-food breakdown).

## Q2: Does at least one cadence reliably survive in the full native action space?

| config | reaches 180s (frac) | no-survivor games (frac) | mean survivors |
|---|---:|---:|---:|
| 1fast_1medium_2slow | 0.420 | 0.150 | 1.190 |
| 1fast_3slow | 0.020 | 0.040 | 0.960 |
| 2fast_2medium | 0.960 | 0.030 | 2.310 |
| 2fast_2slow | 0.450 | 0.300 | 1.150 |
| 4_fast | 1.000 | 0.000 | 3.990 |
| 4_medium | 0.950 | 0.060 | 2.190 |
| 4_slow | 0.010 | 0.600 | 0.400 |

## Q3: How much of performance is explained purely by decision frequency?

Per-seat survival by cadence (all seats across all configs):
| cadence | n seat-games | survival rate |
|---|---:|---:|
| fast | 1000 | 0.848 |
| medium | 700 | 0.469 |
| slow | 1100 | 0.039 |

Compare with survival rates *within each mixed-cadence config* (by_cadence.csv) — if Fast seats survive more often than Slow seats in the SAME game (same producers, same initial inventory), cadence explains a lot; if survival tracks the config composition rather than the seat cadence, it's mostly a market-density effect.

## Q4: What are the dominant causes of death?

Total deaths across all configs: **1581**.

| cause | count | share |
|---|---:|---:|
| affordable_ask_no_buy | 798 | 0.505 |
| no_ask_available | 496 | 0.314 |
| order_pending_no_fill | 280 | 0.177 |
| insufficient_cash | 4 | 0.003 |
| cash_tied_in_reservations | 3 | 0.002 |

## Q5: Are particular assignments or foods systematically causing failures?

Per-food across all configs: fraction of games with 0 trades (dead market for that food).
| food | fraction of games with 0 trades |
|---|---:|
| A | 0.000 |
| B | 0.000 |
| C | 0.000 |
| D | 0.000 |

For a fuller producer-assignment breakdown, join the per-game JSONL against seat-cadence and per-food death classification.

## Q6: Is Fast suitable as the primary frozen heuristic baseline and solvability witness?

- 4_fast: mean survivors = **3.990**, no-survivor games = **0.000**
- 4_slow: mean survivors = **0.400**, no-survivor games = **0.600**

**Verdict for Fast as baseline:** yes if 4_fast reaches 180s with a positive mean survivor count *and* a small no-survivor fraction. If the market density (posts/game, trades/game) also looks healthy in Q1, Fast is a solvability witness.
