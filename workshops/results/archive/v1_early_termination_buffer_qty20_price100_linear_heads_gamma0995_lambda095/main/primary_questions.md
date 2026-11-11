# Scripted-v1 evaluation — primary questions

Based on 1000 games per configuration across 7 configurations (total games: 7000).

## Q1: Does Scripted-v1 reliably sustain a functioning market?

| config | trades/game (mean) | posts/game (mean) | validation fails/seat |
|---|---:|---:|---:|
| 1fast_1medium_2slow | 48.0 | 295.4 | 0.0 |
| 1fast_3slow | 39.5 | 250.5 | 0.0 |
| 2fast_2medium | 78.8 | 496.3 | 0.0 |
| 2fast_2slow | 57.1 | 400.9 | 0.0 |
| 4_fast | 105.5 | 716.0 | 0.0 |
| 4_medium | 70.7 | 282.7 | 0.0 |
| 4_slow | 38.7 | 125.2 | 0.0 |

A functioning market means: nonzero trade counts, orders that actually cross, and validation failures at a low fraction of attempts. Judge by whether the above rows have trades > 0 across all foods (see the market_by_food.csv for the per-food breakdown).

## Q2: Does at least one cadence reliably survive in the full native action space?

| config | reaches 180s (frac) | no-survivor games (frac) | mean survivors |
|---|---:|---:|---:|
| 1fast_1medium_2slow | 0.401 | 0.116 | 1.204 |
| 1fast_3slow | 0.026 | 0.085 | 0.915 |
| 2fast_2medium | 0.983 | 0.022 | 2.493 |
| 2fast_2slow | 0.470 | 0.248 | 1.211 |
| 4_fast | 1.000 | 0.000 | 3.995 |
| 4_medium | 0.954 | 0.079 | 2.309 |
| 4_slow | 0.012 | 0.612 | 0.388 |

## Q3: How much of performance is explained purely by decision frequency?

Per-seat survival by cadence (all seats across all configs):
| cadence | n seat-games | survival rate |
|---|---:|---:|
| fast | 10000 | 0.855 |
| medium | 7000 | 0.505 |
| slow | 11000 | 0.039 |

Compare with survival rates *within each mixed-cadence config* (by_cadence.csv) — if Fast seats survive more often than Slow seats in the SAME game (same producers, same initial inventory), cadence explains a lot; if survival tracks the config composition rather than the seat cadence, it's mostly a market-density effect.

## Q4: What are the dominant causes of death?

Total deaths across all configs: **15485**.

| cause | count | share |
|---|---:|---:|
| affordable_ask_no_buy | 7797 | 0.504 |
| no_ask_available | 4894 | 0.316 |
| order_pending_no_fill | 2679 | 0.173 |
| cash_tied_in_reservations | 80 | 0.005 |
| insufficient_cash | 35 | 0.002 |

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

- 4_fast: mean survivors = **3.995**, no-survivor games = **0.000**
- 4_slow: mean survivors = **0.388**, no-survivor games = **0.612**

**Verdict for Fast as baseline:** yes if 4_fast reaches 180s with a positive mean survivor count *and* a small no-survivor fraction. If the market density (posts/game, trades/game) also looks healthy in Q1, Fast is a solvability witness.
