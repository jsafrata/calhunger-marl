# IPPO shared seed 1: matched checkpoint evaluation

Protocol: one stochastic learned seat plus three Scripted-v1 seats, 200 evaluation games per checkpoint, using the same evaluation seed base (`200000`). All games reached the 180-tick time limit.

| Checkpoint | Survival | Winner | Alive non-winner | Dead | Implied terminal reward | Mean cash | Median cash | Mean death/end tick |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 52.5% | 4.5% | 48.0% | 47.5% | -0.430 | $87.69 | $81.00 | 166.65 |
| 1100 | 51.5% | 15.0% | 36.5% | 48.5% | -0.335 | $123.71 | $112.00 | 169.75 |
| 1200 | 33.5% | 19.0% | 14.5% | 66.5% | -0.475 | $179.06 | $179.00 | 160.45 |
| 1300 | 16.0% | 12.5% | 3.5% | 84.0% | -0.715 | $211.66 | $225.50 | 153.95 |
| 1400 | 9.5% | 9.0% | 0.5% | 90.5% | -0.815 | $239.39 | $252.50 | 145.05 |

The implied terminal reward is `P(winner) - P(dead)`, matching the configured rewards of +1 for winners, 0 for surviving non-winners, and -1 for deaths.

## Market behavior

| Checkpoint | Lineup trades | Learned buyer trades | Learned seller trades | Wait | Cancel | Take | Post |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 146.55 | 46.12 | 49.61 | 12.81% | 10.67% | 28.60% | 47.92% |
| 1100 | 174.77 | 59.14 | 70.25 | 10.61% | 9.96% | 49.05% | 30.38% |
| 1200 | 165.07 | 49.70 | 67.98 | 15.16% | 7.78% | 43.88% | 33.19% |
| 1300 | 149.53 | 38.65 | 60.61 | 17.00% | 7.98% | 36.16% | 38.86% |
| 1400 | 137.25 | 30.02 | 56.27 | 23.70% | 5.38% | 32.36% | 38.57% |

## Interpretation

Checkpoint 1100 is the best of these checkpoints under the actual terminal reward in this fixed cross-play setting. Checkpoint 1200 has a higher winner rate, but its much larger death rate makes its expected terminal reward worse. Checkpoints 1300 and 1400 degrade sharply on survival, winning, death tick, trading activity, and implied terminal reward.

Mean cash rises monotonically while competent behavior collapses. This is not evidence of improving task performance: most of the additional cash is held by agents that subsequently die. Mean cash among dead learned seats rises from $134.53 at checkpoint 1100 to $187.51, $215.77, and $238.90 at checkpoints 1200, 1300, and 1400. Cash must therefore be interpreted conditional on survival/outcome rather than alone.

The extension shows a strong cross-play robustness collapse after checkpoint 1100. Because contemporaneous self-play survival remained near 3.5 agents while performance against fixed Scripted-v1 opponents collapsed, the result is consistent with adaptation to a co-evolving self-play regime rather than general improvement. It supports—but by itself does not uniquely prove—self-play overfitting or policy co-adaptation as a mechanism.

Raw files:

- `deep_eval_iter1000_1learned_3scripted_stochastic_200.json`
- `deep_eval_iter1100_1learned_3scripted_stochastic_200.json`
- `deep_eval_iter1200_1learned_3scripted_stochastic_200.json`
- `deep_eval_iter1300_1learned_3scripted_stochastic_200.json`
- `deep_eval_iter1400_1learned_3scripted_stochastic_200.json`
