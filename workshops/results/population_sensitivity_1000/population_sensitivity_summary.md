# E12 rebase: population sensitivity, best-config peer matrix (1000 games/cell)

## Same policy, different population: survival / win spread

| policy | selfplay surv | peer surv min-max | peer win min-max |
|---|---|---|---|
| ippo_s0 | 88.6 | 4.4 - 81.2 | 4.2 - 30.7 |
| ippo_s1 | 91.8 | 5.5 - 89.8 | 5.2 - 35.5 |
| ippo_s2 | 90.3 | 2.7 - 88.7 | 2.3 - 45.1 |
| ippo_s3 | 94.3 | 8.1 - 86.3 | 4.9 - 25.3 |
| ippo_s4 | 92.1 | 16.3 - 82.8 | 4.6 - 22.0 |
| mappo_s0 | 87.6 | 15.8 - 81.3 | 2.9 - 23.6 |
| mappo_s1 | 54.2 | 14.4 - 32.5 | 0.6 - 18.9 |
| mappo_s2 | 92.1 | 0.5 - 77.7 | 0.5 - 37.1 |
| mappo_s3 | 94.5 | 2.2 - 84.6 | 2.1 - 33.7 |
| mappo_s4 | 0.0 | 0.0 - 0.4 | 0.0 - 0.0 |

## Rank a policy receives depending on which single population evaluates it

| policy | best rank | worst rank |
|---|---|---|
| ippo_s0 | 5 | 7 |
| ippo_s1 | 1 | 6 |
| ippo_s2 | 1 | 6 |
| ippo_s3 | 2 | 8 |
| ippo_s4 | 1 | 7 |
| mappo_s0 | 2 | 7 |
| mappo_s1 | 6 | 8 |
| mappo_s2 | 1 | 9 |
| mappo_s3 | 1 | 7 |
| mappo_s4 | 9 | 9 |

## Do two defensible single-population evaluations agree? (healthy strangers)

Spearman between single-context leaderboards over shared healthy focals: mean 0.35, min -0.77, max 0.90
