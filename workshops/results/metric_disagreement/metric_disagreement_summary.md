# E20: metric-portfolio disagreement audit (peer cross-play, n=100 paired games/cell)

B=2000 paired-seed bootstrap replicates; 'decided' = 95% CI of
the pairwise difference excludes 0; strict inversion = decided under both metrics
with opposite signs.

## Leaderboards (full data)

| policy | survival | win | cash | cash_given_survival | survival_time | win_given_survival | selfplay_survival | selfplay_cash_given_survival |
|---|---|---|---|---|---|---|---|---|
| ippo_s0 | 52.9 | 16.4 | 157.7 | 162.0 | 158.6 | 44.2 | 88.2 | 96.6 |
| ippo_s1 | 62.4 | 17.2 | 144.3 | 147.3 | 166.0 | 40.1 | 93.2 | 97.0 |
| ippo_s2 | 65.6 | 26.3 | 153.6 | 155.1 | 165.4 | 49.8 | 92.5 | 94.1 |
| ippo_s3 | 64.1 | 9.4 | 111.0 | 107.0 | 165.7 | 28.5 | 95.2 | 96.4 |
| ippo_s4 | 61.0 | 9.9 | 110.1 | 101.5 | 165.8 | 28.5 | 92.2 | 103.3 |
| mappo_s0 | 62.4 | 9.3 | 123.3 | 108.9 | 165.0 | 27.1 | 89.2 | 99.3 |
| mappo_s1 | 24.9 | 3.8 | 15.4 | 20.8 | 156.0 | 17.6 | 56.8 | 103.9 |
| mappo_s2 | 57.2 | 25.4 | 174.3 | 161.6 | 161.1 | 50.5 | 93.5 | 98.5 |
| mappo_s3 | 64.0 | 18.2 | 153.7 | 158.4 | 167.6 | 40.0 | 93.2 | 99.0 |
| mappo_s4 | 0.1 | 0.0 | 22.4 | 2.0 | 105.1 | 0.0 | 0.0 | nan |

## Metric x metric leaderboard Spearman (all10)

- survival~win: rho=+0.59 (p=0.074)
- survival~cash: rho=+0.31 (p=0.385)
- survival~cash_given_survival: rho=+0.31 (p=0.385)
- survival~survival_time: rho=+0.77 (p=0.009)
- survival~win_given_survival: rho=+0.44 (p=0.200)
- survival~selfplay_survival: rho=+0.72 (p=0.020)
- survival~selfplay_cash_given_survival: rho=+nan (p=nan)
- win~cash: rho=+0.82 (p=0.004)
- win~cash_given_survival: rho=+0.79 (p=0.006)
- win~survival_time: rho=+0.52 (p=0.128)
- win~win_given_survival: rho=+0.93 (p=0.000)
- win~selfplay_survival: rho=+0.62 (p=0.056)
- win~selfplay_cash_given_survival: rho=+nan (p=nan)
- cash~cash_given_survival: rho=+0.98 (p=0.000)
- cash~survival_time: rho=+0.26 (p=0.467)
- cash~win_given_survival: rho=+0.89 (p=0.001)
- cash~selfplay_survival: rho=+0.49 (p=0.148)
- cash~selfplay_cash_given_survival: rho=+nan (p=nan)
- cash_given_survival~survival_time: rho=+0.26 (p=0.467)
- cash_given_survival~win_given_survival: rho=+0.88 (p=0.001)
- cash_given_survival~selfplay_survival: rho=+0.43 (p=0.213)
- cash_given_survival~selfplay_cash_given_survival: rho=+nan (p=nan)
- survival_time~win_given_survival: rho=+0.31 (p=0.385)
- survival_time~selfplay_survival: rho=+0.68 (p=0.030)
- survival_time~selfplay_cash_given_survival: rho=+nan (p=nan)
- win_given_survival~selfplay_survival: rho=+0.61 (p=0.062)
- win_given_survival~selfplay_cash_given_survival: rho=+nan (p=nan)
- selfplay_survival~selfplay_cash_given_survival: rho=+nan (p=nan)

## Metric x metric leaderboard Spearman (healthy8)

- survival~win: rho=+0.19 (p=0.651)
- survival~cash: rho=-0.33 (p=0.420)
- survival~cash_given_survival: rho=-0.36 (p=0.385)
- survival~survival_time: rho=+0.55 (p=0.160)
- survival~win_given_survival: rho=-0.10 (p=0.823)
- survival~selfplay_survival: rho=+0.44 (p=0.272)
- survival~selfplay_cash_given_survival: rho=-0.45 (p=0.260)
- win~cash: rho=+0.67 (p=0.071)
- win~cash_given_survival: rho=+0.60 (p=0.120)
- win~survival_time: rho=+0.05 (p=0.911)
- win~win_given_survival: rho=+0.86 (p=0.007)
- win~selfplay_survival: rho=+0.25 (p=0.548)
- win~selfplay_cash_given_survival: rho=-0.38 (p=0.352)
- cash~cash_given_survival: rho=+0.98 (p=0.000)
- cash~survival_time: rho=-0.43 (p=0.289)
- cash~win_given_survival: rho=+0.81 (p=0.015)
- cash~selfplay_survival: rho=+0.02 (p=0.955)
- cash~selfplay_cash_given_survival: rho=-0.26 (p=0.531)
- cash_given_survival~survival_time: rho=-0.45 (p=0.260)
- cash_given_survival~win_given_survival: rho=+0.76 (p=0.028)
- cash_given_survival~selfplay_survival: rho=-0.12 (p=0.778)
- cash_given_survival~selfplay_cash_given_survival: rho=-0.31 (p=0.456)
- survival_time~win_given_survival: rho=-0.36 (p=0.385)
- survival_time~selfplay_survival: rho=+0.37 (p=0.365)
- survival_time~selfplay_cash_given_survival: rho=+0.24 (p=0.570)
- win_given_survival~selfplay_survival: rho=+0.23 (p=0.588)
- win_given_survival~selfplay_cash_given_survival: rho=-0.57 (p=0.139)
- selfplay_survival~selfplay_cash_given_survival: rho=-0.26 (p=0.528)

## Sign flips and strict inversions per metric pair

| metric A | metric B | group | sign flips | strict inversions |
|---|---|---|---|---|
| survival | win | all10 | 12/45 | 7 |
| survival | win | healthy8 | 12/28 | 7 |
| survival | cash | all10 | 17/45 | 10 |
| survival | cash | healthy8 | 16/28 | 9 |
| survival | cash_given_survival | all10 | 17/45 | 6 |
| survival | cash_given_survival | healthy8 | 17/28 | 6 |
| survival | survival_time | all10 | 8/45 | 0 |
| survival | survival_time | healthy8 | 8/28 | 0 |
| survival | win_given_survival | all10 | 15/45 | 7 |
| survival | win_given_survival | healthy8 | 15/28 | 7 |
| survival | selfplay_survival | all10 | 10/45 | 1 |
| survival | selfplay_survival | healthy8 | 10/28 | 1 |
| survival | selfplay_cash_given_survival | all10 | 36/45 | 6 |
| survival | selfplay_cash_given_survival | healthy8 | 19/28 | 1 |
| win | cash | all10 | 9/45 | 1 |
| win | cash | healthy8 | 8/28 | 0 |
| win | cash_given_survival | all10 | 9/45 | 0 |
| win | cash_given_survival | healthy8 | 9/28 | 0 |
| win | survival_time | all10 | 12/45 | 9 |
| win | survival_time | healthy8 | 12/28 | 9 |
| win | win_given_survival | all10 | 5/45 | 0 |
| win | win_given_survival | healthy8 | 5/28 | 0 |
| win | selfplay_survival | all10 | 12/45 | 1 |
| win | selfplay_survival | healthy8 | 12/28 | 1 |
| win | selfplay_cash_given_survival | all10 | 34/45 | 12 |
| win | selfplay_cash_given_survival | healthy8 | 17/28 | 7 |
| cash | cash_given_survival | all10 | 2/45 | 1 |
| cash | cash_given_survival | healthy8 | 1/28 | 0 |
| cash | survival_time | all10 | 19/45 | 11 |
| cash | survival_time | healthy8 | 18/28 | 10 |
| cash | win_given_survival | all10 | 6/45 | 1 |
| cash | win_given_survival | healthy8 | 5/28 | 0 |
| cash | selfplay_survival | all10 | 15/45 | 4 |
| cash | selfplay_survival | healthy8 | 14/28 | 3 |
| cash | selfplay_cash_given_survival | all10 | 34/45 | 11 |
| cash | selfplay_cash_given_survival | healthy8 | 17/28 | 6 |
| cash_given_survival | survival_time | all10 | 19/45 | 7 |
| cash_given_survival | survival_time | healthy8 | 19/28 | 7 |
| cash_given_survival | win_given_survival | all10 | 6/45 | 0 |
| cash_given_survival | win_given_survival | healthy8 | 6/28 | 0 |
| cash_given_survival | selfplay_survival | all10 | 15/45 | 1 |
| cash_given_survival | selfplay_survival | healthy8 | 15/28 | 1 |
| cash_given_survival | selfplay_cash_given_survival | all10 | 35/45 | 11 |
| cash_given_survival | selfplay_cash_given_survival | healthy8 | 18/28 | 6 |
| survival_time | win_given_survival | all10 | 17/45 | 9 |
| survival_time | win_given_survival | healthy8 | 17/28 | 9 |
| survival_time | selfplay_survival | all10 | 10/45 | 1 |
| survival_time | selfplay_survival | healthy8 | 10/28 | 1 |
| survival_time | selfplay_cash_given_survival | all10 | 28/45 | 5 |
| survival_time | selfplay_cash_given_survival | healthy8 | 11/28 | 1 |
| win_given_survival | selfplay_survival | all10 | 12/45 | 1 |
| win_given_survival | selfplay_survival | healthy8 | 12/28 | 1 |
| win_given_survival | selfplay_cash_given_survival | all10 | 37/45 | 12 |
| win_given_survival | selfplay_cash_given_survival | healthy8 | 20/28 | 7 |
| selfplay_survival | selfplay_cash_given_survival | all10 | 34/45 | 5 |
| selfplay_survival | selfplay_cash_given_survival | healthy8 | 17/28 | 0 |

## Headline (vs the survival verdict)

- all10: 24/45 pairwise verdicts flip under >=1 alternative peer metric; 10 are STRICT inversions (ippo_s0 vs ippo_s1, ippo_s0 vs ippo_s3, ippo_s0 vs ippo_s4, ippo_s0 vs mappo_s0, ippo_s1 vs mappo_s2, ippo_s2 vs mappo_s2, ippo_s3 vs mappo_s2, mappo_s0 vs mappo_s2, mappo_s1 vs mappo_s4, mappo_s2 vs mappo_s3)
- healthy8: 23/28 pairwise verdicts flip under >=1 alternative peer metric; 9 are STRICT inversions (ippo_s0 vs ippo_s1, ippo_s0 vs ippo_s3, ippo_s0 vs ippo_s4, ippo_s0 vs mappo_s0, ippo_s1 vs mappo_s2, ippo_s2 vs mappo_s2, ippo_s3 vs mappo_s2, mappo_s0 vs mappo_s2, mappo_s2 vs mappo_s3)
