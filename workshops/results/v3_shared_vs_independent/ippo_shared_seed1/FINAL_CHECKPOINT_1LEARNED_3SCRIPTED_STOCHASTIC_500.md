# IPPO shared seed 1 final-checkpoint evaluation

- Checkpoint: `iter_1100`
- Lineup: 1 learned + 3 Scripted-v1
- Learned policy mode: stochastic
- Evaluation games: 500
- Evaluation seed base: 200000

## Outcome metrics for the learned seat

| Metric | Result |
|---|---:|
| Survival rate | 49.6% (248/500) |
| Winner rate | 14.0% (70/500) |
| Alive non-winner rate | 35.6% (178/500) |
| Death rate | 50.4% (252/500) |
| Mean recorded death/end tick | 169.30 |
| Mean final cash | $118.19 |

## Terminal cash by outcome

| Outcome | n | Mean | Median | Std. dev. |
|---|---:|---:|---:|---:|
| All | 500 | $118.19 | $107.50 | $65.66 |
| Dead | 252 | $121.49 | $110.00 | $66.51 |
| Alive non-winner | 178 | $82.54 | $80.00 | $39.21 |
| Winner | 70 | $196.94 | $196.00 | $38.55 |

## Required-food inventory at termination

| Outcome | Mean required-food total | Mean required-food minimum | Mean total inventory |
|---|---:|---:|---:|
| All | 13.72 | 1.89 | 92.67 |
| Dead | 13.76 | 0.81 | 93.06 |
| Alive non-winner | 15.12 | 3.28 | 95.33 |
| Winner | 10.00 | 2.23 | 84.51 |

Among the 248 surviving learned seats, the minimum required-food inventory was 0–1 in 39.9%, 2–3 in 26.6%, 4–5 in 14.5%, and greater than 5 in 19.0%.

## Market and action metrics

| Metric | Mean per game |
|---|---:|
| Lineup survivors | 3.468 |
| Lineup trades | 174.272 |
| Learned-seat completed trades as buyer | 59.104 |
| Learned-seat completed trades as seller | 68.934 |
| Learned-seat posted bids | 26.124 |
| Learned-seat posted asks | 28.870 |
| Learned-seat take actions | 86.982 |

All 500 games reached the time limit. Learned-seat action frequencies were: wait 11.37%, cancel 9.75%, take 48.32%, and post 30.55%.

## Policy-head diagnostics

- Type entropy: 0.7033
- Food entropy given post: 1.0265
- Side entropy given post: 0.3491
- Price entropy given post: 2.4139
- Quantity entropy given post: 0.9065
- Median price-head argmax given post: 8
- Median quantity-head argmax given post: 2

The complete machine-readable output, including all means, medians, standard deviations, counts, inventory fields, action probabilities, and head diagnostics, is in `deep_eval_1learned_3scripted_stochastic_500.json`.
