# Scripted-v1 evaluation — primary questions

Based on 2 games per configuration across 7 configurations (total games: 14).

## Q1: Does Scripted-v1 reliably sustain a functioning market?

| config | trades/game (mean) | posts/game (mean) | validation fails/seat |
|---|---:|---:|---:|
| 1fast_1medium_2slow | 46.0 | 297.5 | 0.0 |
| 1fast_3slow | 34.0 | 246.5 | 0.0 |
| 2fast_2medium | 73.0 | 487.0 | 0.0 |
| 2fast_2slow | 62.5 | 398.5 | 0.0 |
| 4_fast | 114.5 | 716.0 | 0.0 |
| 4_medium | 64.5 | 276.0 | 0.0 |
| 4_slow | 36.5 | 124.0 | 0.0 |

A functioning market means: nonzero trade counts, orders that actually cross, and validation failures at a low fraction of attempts. Judge by whether the above rows have trades > 0 across all foods (see the market_by_food.csv for the per-food breakdown).

## Q2: Does at least one cadence reliably survive in the full native action space?

| config | reaches 180s (frac) | no-survivor games (frac) | mean survivors |
|---|---:|---:|---:|
| 1fast_1medium_2slow | 0.500 | 0.500 | 1.000 |
| 1fast_3slow | 0.000 | 0.000 | 1.000 |
| 2fast_2medium | 0.500 | 0.000 | 2.000 |
| 2fast_2slow | 0.500 | 0.500 | 1.000 |
| 4_fast | 1.000 | 0.000 | 4.000 |
| 4_medium | 1.000 | 0.000 | 2.500 |
| 4_slow | 0.000 | 1.000 | 0.000 |

## Q3: How much of performance is explained purely by decision frequency?

Per-seat survival by cadence (all seats across all configs):
| cadence | n seat-games | survival rate |
|---|---:|---:|
| fast | 20 | 0.800 |
| medium | 14 | 0.500 |
| slow | 22 | 0.000 |

Compare with survival rates *within each mixed-cadence config* (by_cadence.csv) — if Fast seats survive more often than Slow seats in the SAME game (same producers, same initial inventory), cadence explains a lot; if survival tracks the config composition rather than the seat cadence, it's mostly a market-density effect.

## Q4: What are the dominant causes of death?

Total deaths across all configs: **33**.

| cause | count | share |
|---|---:|---:|
| no_ask_available | 12 | 0.364 |
| affordable_ask_no_buy | 11 | 0.333 |
| order_pending_no_fill | 10 | 0.303 |

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

- 4_fast: mean survivors = **4.000**, no-survivor games = **0.000**
- 4_slow: mean survivors = **0.000**, no-survivor games = **1.000**

**Verdict for Fast as baseline:** yes if 4_fast reaches 180s with a positive mean survivor count *and* a small no-survivor fraction. If the market density (posts/game, trades/game) also looks healthy in Q1, Fast is a solvability witness.
