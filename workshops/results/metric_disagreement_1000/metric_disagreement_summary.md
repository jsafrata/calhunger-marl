# E20: metric-portfolio disagreement audit (peer cross-play, n=1000 paired games/cell)

B=2000 paired-seed bootstrap replicates; 'decided' = 95% CI of
the pairwise difference excludes 0; strict inversion = decided under both metrics
with opposite signs.

## Leaderboards (full data)

| policy | survival | win | cash | cash_given_survival | survival_time | win_given_survival | selfplay_survival | selfplay_cash_given_survival |
|---|---|---|---|---|---|---|---|---|
| ippo_s0 | 51.1 | 16.8 | 166.4 | 163.3 | 158.2 | 45.7 | 88.6 | 99.1 |
| ippo_s1 | 64.2 | 16.1 | 138.7 | 138.6 | 166.4 | 38.4 | 91.8 | 95.9 |
| ippo_s2 | 65.1 | 26.9 | 157.7 | 147.8 | 165.0 | 48.6 | 90.3 | 93.3 |
| ippo_s3 | 62.2 | 10.5 | 112.7 | 112.3 | 164.5 | 31.9 | 94.3 | 97.4 |
| ippo_s4 | 63.2 | 9.8 | 107.9 | 99.7 | 166.6 | 27.4 | 92.1 | 102.1 |
| mappo_s0 | 58.5 | 9.9 | 126.3 | 109.8 | 163.3 | 28.8 | 87.6 | 99.1 |
| mappo_s1 | 23.1 | 3.5 | 17.0 | 23.9 | 153.7 | 17.6 | 54.2 | 109.5 |
| mappo_s2 | 54.7 | 22.5 | 177.8 | 180.7 | 159.7 | 52.9 | 92.1 | 98.5 |
| mappo_s3 | 63.7 | 17.7 | 155.7 | 161.3 | 167.6 | 39.6 | 94.5 | 99.0 |
| mappo_s4 | 0.0 | 0.0 | 21.5 | 5.5 | 105.2 | 0.0 | 0.0 | nan |

## Metric x metric leaderboard Spearman (all10)

- survival~win: rho=+0.59 (p=0.074)
- survival~cash: rho=+0.35 (p=0.328)
- survival~cash_given_survival: rho=+0.35 (p=0.328)
- survival~survival_time: rho=+0.89 (p=0.001)
- survival~win_given_survival: rho=+0.45 (p=0.187)
- survival~selfplay_survival: rho=+0.61 (p=0.062)
- survival~selfplay_cash_given_survival: rho=+nan (p=nan)
- win~cash: rho=+0.92 (p=0.000)
- win~cash_given_survival: rho=+0.92 (p=0.000)
- win~survival_time: rho=+0.42 (p=0.229)
- win~win_given_survival: rho=+0.98 (p=0.000)
- win~selfplay_survival: rho=+0.53 (p=0.116)
- win~selfplay_cash_given_survival: rho=+nan (p=nan)
- cash~cash_given_survival: rho=+0.96 (p=0.000)
- cash~survival_time: rho=+0.22 (p=0.533)
- cash~win_given_survival: rho=+0.96 (p=0.000)
- cash~selfplay_survival: rho=+0.39 (p=0.266)
- cash~selfplay_cash_given_survival: rho=+nan (p=nan)
- cash_given_survival~survival_time: rho=+0.28 (p=0.425)
- cash_given_survival~win_given_survival: rho=+0.96 (p=0.000)
- cash_given_survival~selfplay_survival: rho=+0.53 (p=0.111)
- cash_given_survival~selfplay_cash_given_survival: rho=+nan (p=nan)
- survival_time~win_given_survival: rho=+0.30 (p=0.405)
- survival_time~selfplay_survival: rho=+0.78 (p=0.008)
- survival_time~selfplay_cash_given_survival: rho=+nan (p=nan)
- win_given_survival~selfplay_survival: rho=+0.49 (p=0.154)
- win_given_survival~selfplay_cash_given_survival: rho=+nan (p=nan)
- selfplay_survival~selfplay_cash_given_survival: rho=+nan (p=nan)

## Metric x metric leaderboard Spearman (healthy8)

- survival~win: rho=+0.19 (p=0.651)
- survival~cash: rho=-0.26 (p=0.531)
- survival~cash_given_survival: rho=-0.29 (p=0.493)
- survival~survival_time: rho=+0.79 (p=0.021)
- survival~win_given_survival: rho=-0.07 (p=0.867)
- survival~selfplay_survival: rho=+0.23 (p=0.588)
- survival~selfplay_cash_given_survival: rho=-0.60 (p=0.120)
- win~cash: rho=+0.86 (p=0.007)
- win~cash_given_survival: rho=+0.83 (p=0.010)
- win~survival_time: rho=-0.14 (p=0.736)
- win~win_given_survival: rho=+0.95 (p=0.000)
- win~selfplay_survival: rho=+0.07 (p=0.866)
- win~selfplay_cash_given_survival: rho=-0.57 (p=0.139)
- cash~cash_given_survival: rho=+0.95 (p=0.000)
- cash~survival_time: rho=-0.50 (p=0.207)
- cash~win_given_survival: rho=+0.95 (p=0.000)
- cash~selfplay_survival: rho=-0.18 (p=0.670)
- cash~selfplay_cash_given_survival: rho=-0.24 (p=0.570)
- cash_given_survival~survival_time: rho=-0.40 (p=0.320)
- cash_given_survival~win_given_survival: rho=+0.93 (p=0.001)
- cash_given_survival~selfplay_survival: rho=+0.08 (p=0.844)
- cash_given_survival~selfplay_cash_given_survival: rho=-0.21 (p=0.610)
- survival_time~win_given_survival: rho=-0.38 (p=0.352)
- survival_time~selfplay_survival: rho=+0.56 (p=0.146)
- survival_time~selfplay_cash_given_survival: rho=-0.10 (p=0.823)
- win_given_survival~selfplay_survival: rho=-0.01 (p=0.978)
- win_given_survival~selfplay_cash_given_survival: rho=-0.45 (p=0.260)
- selfplay_survival~selfplay_cash_given_survival: rho=-0.10 (p=0.821)

## Sign flips and strict inversions per metric pair

| metric A | metric B | group | sign flips | strict @95 | strict @99 |
|---|---|---|---|---|---|
| survival | win | all10 | 12/45 | 8 | 8 |
| survival | win | healthy8 | 12/28 | 8 | 8 |
| survival | cash | all10 | 17/45 | 15 | 15 |
| survival | cash | healthy8 | 16/28 | 14 | 14 |
| survival | cash_given_survival | all10 | 16/45 | 12 | 12 |
| survival | cash_given_survival | healthy8 | 16/28 | 12 | 12 |
| survival | survival_time | all10 | 5/45 | 1 | 0 |
| survival | survival_time | healthy8 | 5/28 | 1 | 0 |
| survival | win_given_survival | all10 | 14/45 | 11 | 10 |
| survival | win_given_survival | healthy8 | 14/28 | 11 | 10 |
| survival | selfplay_survival | all10 | 13/45 | 5 | 3 |
| survival | selfplay_survival | healthy8 | 13/28 | 5 | 3 |
| survival | selfplay_cash_given_survival | all10 | 38/45 | 20 | 18 |
| survival | selfplay_cash_given_survival | healthy8 | 21/28 | 12 | 10 |
| win | cash | all10 | 5/45 | 3 | 3 |
| win | cash | healthy8 | 4/28 | 2 | 2 |
| win | cash_given_survival | all10 | 4/45 | 3 | 3 |
| win | cash_given_survival | healthy8 | 4/28 | 3 | 3 |
| win | survival_time | all10 | 15/45 | 11 | 11 |
| win | survival_time | healthy8 | 15/28 | 11 | 11 |
| win | win_given_survival | all10 | 2/45 | 1 | 0 |
| win | win_given_survival | healthy8 | 2/28 | 1 | 0 |
| win | selfplay_survival | all10 | 14/45 | 10 | 8 |
| win | selfplay_survival | healthy8 | 14/28 | 10 | 8 |
| win | selfplay_cash_given_survival | all10 | 38/45 | 21 | 21 |
| win | selfplay_cash_given_survival | healthy8 | 21/28 | 13 | 13 |
| cash | cash_given_survival | all10 | 3/45 | 1 | 1 |
| cash | cash_given_survival | healthy8 | 2/28 | 0 | 0 |
| cash | survival_time | all10 | 20/45 | 18 | 17 |
| cash | survival_time | healthy8 | 19/28 | 17 | 16 |
| cash | win_given_survival | all10 | 3/45 | 3 | 2 |
| cash | win_given_survival | healthy8 | 2/28 | 2 | 1 |
| cash | selfplay_survival | all10 | 17/45 | 14 | 12 |
| cash | selfplay_survival | healthy8 | 16/28 | 13 | 11 |
| cash | selfplay_cash_given_survival | all10 | 34/45 | 20 | 19 |
| cash | selfplay_cash_given_survival | healthy8 | 17/28 | 12 | 11 |
| cash_given_survival | survival_time | all10 | 17/45 | 15 | 15 |
| cash_given_survival | survival_time | healthy8 | 17/28 | 15 | 15 |
| cash_given_survival | win_given_survival | all10 | 2/45 | 2 | 1 |
| cash_given_survival | win_given_survival | healthy8 | 2/28 | 2 | 1 |
| cash_given_survival | selfplay_survival | all10 | 14/45 | 11 | 9 |
| cash_given_survival | selfplay_survival | healthy8 | 14/28 | 11 | 9 |
| cash_given_survival | selfplay_cash_given_survival | all10 | 34/45 | 20 | 20 |
| cash_given_survival | selfplay_cash_given_survival | healthy8 | 17/28 | 12 | 12 |
| survival_time | win_given_survival | all10 | 17/45 | 15 | 14 |
| survival_time | win_given_survival | healthy8 | 17/28 | 15 | 14 |
| survival_time | selfplay_survival | all10 | 8/45 | 4 | 3 |
| survival_time | selfplay_survival | healthy8 | 8/28 | 4 | 3 |
| survival_time | selfplay_cash_given_survival | all10 | 33/45 | 19 | 18 |
| survival_time | selfplay_cash_given_survival | healthy8 | 16/28 | 11 | 10 |
| win_given_survival | selfplay_survival | all10 | 14/45 | 11 | 10 |
| win_given_survival | selfplay_survival | healthy8 | 14/28 | 11 | 10 |
| win_given_survival | selfplay_cash_given_survival | all10 | 36/45 | 22 | 21 |
| win_given_survival | selfplay_cash_given_survival | healthy8 | 19/28 | 14 | 13 |
| selfplay_survival | selfplay_cash_given_survival | all10 | 31/45 | 17 | 15 |
| selfplay_survival | selfplay_cash_given_survival | healthy8 | 14/28 | 9 | 7 |

## Metric-family collapse (cross-family strict inversions)

| family A | family B | group | strict @95 | strict @99 |
|---|---|---|---|---|
| survival_family | win_family | all10 | 15/45 | 14/45 |
| survival_family | win_family | healthy8 | 15/28 | 14/28 |
| survival_family | cash_family | all10 | 18/45 | 18/45 |
| survival_family | cash_family | healthy8 | 17/28 | 17/28 |
| win_family | cash_family | all10 | 5/45 | 5/45 |
| win_family | cash_family | healthy8 | 4/28 | 4/28 |

## Headline (vs the survival verdict)

- all10: 21/45 pairwise verdicts flip under >=1 alternative peer metric; 16 are STRICT inversions at 95% (15 survive at 99%) (ippo_s0 vs ippo_s1, ippo_s0 vs ippo_s2, ippo_s0 vs ippo_s3, ippo_s0 vs ippo_s4, ippo_s0 vs mappo_s0, ippo_s0 vs mappo_s3, ippo_s1 vs mappo_s2, ippo_s2 vs ippo_s4, ippo_s2 vs mappo_s2, ippo_s3 vs mappo_s0, ippo_s3 vs mappo_s2, ippo_s4 vs mappo_s0, ippo_s4 vs mappo_s2, mappo_s0 vs mappo_s2, mappo_s1 vs mappo_s4, mappo_s2 vs mappo_s3)
- healthy8: 20/28 pairwise verdicts flip under >=1 alternative peer metric; 15 are STRICT inversions at 95% (14 survive at 99%) (ippo_s0 vs ippo_s1, ippo_s0 vs ippo_s2, ippo_s0 vs ippo_s3, ippo_s0 vs ippo_s4, ippo_s0 vs mappo_s0, ippo_s0 vs mappo_s3, ippo_s1 vs mappo_s2, ippo_s2 vs ippo_s4, ippo_s2 vs mappo_s2, ippo_s3 vs mappo_s0, ippo_s3 vs mappo_s2, ippo_s4 vs mappo_s0, ippo_s4 vs mappo_s2, mappo_s0 vs mappo_s2, mappo_s2 vs mappo_s3)
