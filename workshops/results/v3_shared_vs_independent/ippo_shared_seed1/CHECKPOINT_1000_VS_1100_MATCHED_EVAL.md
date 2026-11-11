# IPPO shared seed 1: checkpoint 1000 vs. 1100

Protocol: one stochastic learned seat plus three Scripted-v1 seats, 200 evaluation games per checkpoint, using the same evaluation seed base (`200000`). These are evaluation games, not additional training iterations.

Checkpoint 900 could not be evaluated because this run saved checkpoints at iterations 800, 1000, and 1100; no iteration-900 checkpoint exists.

| Metric | Iteration 1000 | Iteration 1100 | Change |
|---|---:|---:|---:|
| Learned-seat survival | 52.5% | 51.5% | -1.0 pp |
| Learned-seat winner rate | 4.5% | 15.0% | +10.5 pp |
| Alive non-winner rate | 48.0% | 36.5% | -11.5 pp |
| Mean final cash | $87.69 | $123.71 | +$36.02 |
| Median final cash | $81.00 | $112.00 | +$31.00 |
| Mean recorded death/end tick | 166.65 | 169.75 | +3.10 |
| Mean lineup survivors | 3.435 | 3.500 | +0.065 |
| Mean lineup trades | 146.55 | 174.77 | +28.22 |
| Learned buyer trades | 46.12 | 59.14 | +13.02 |
| Learned seller trades | 49.61 | 70.25 | +20.65 |
| Mean minimum required-food inventory | 2.37 | 1.98 | -0.39 |

Both evaluations reached the time limit in all 200 games.

## Learned-seat action frequencies

| Action | Iteration 1000 | Iteration 1100 | Change |
|---|---:|---:|---:|
| Wait | 12.81% | 10.61% | -2.20 pp |
| Cancel | 10.67% | 9.96% | -0.71 pp |
| Take | 28.60% | 49.05% | +20.45 pp |
| Post | 47.92% | 30.38% | -17.54 pp |

## Interpretation

The policy was still changing materially between iterations 1000 and 1100. It did not improve survival—the one-percentage-point difference is negligible at 200 games—but it improved the competitive/wealth outcomes substantially. The later policy traded more, shifted strongly from posting to taking, produced more buyer and seller completions, increased mean cash by $36.02, and more than tripled its observed winner rate.

Therefore, iteration 1000 was not yet a plateau for terminal wealth or winning behavior. The evidence is not that continued training improved everything uniformly; it indicates a transition from mostly surviving without winning toward more aggressive trading and substantially better terminal cash/win performance, while survival remained flat.

Raw files:

- `deep_eval_iter1000_1learned_3scripted_stochastic_200.json`
- `deep_eval_iter1100_1learned_3scripted_stochastic_200.json`
