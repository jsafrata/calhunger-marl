# E19: bootstrap stability of the peer cross-play leaderboard

B=2000 replicates; seed 20260827; score = macro-mean over the 9 focal 1v3 cells; full data = 100 paired games/cell.

## Leaderboard stability vs evaluation budget (survival metric)

| n/cell | scheme | group | mean rho | p5 rho | top-1 stable | mean flip | pairs ≥5% flip | adj. flip |
|---|---|---|---|---|---|---|---|---|
| 5 | paired | all10 | 0.704 | 0.445 | 26% | 23.0% | 28/45 | 36.0% |
| 5 | paired | healthy8 | 0.416 | -0.086 | 26% | 36.8% | 28/28 | 46.2% |
| 5 | independent | all10 | 0.751 | 0.497 | 27% | 20.7% | 28/45 | 34.8% |
| 5 | independent | healthy8 | 0.509 | 0.000 | 27% | 33.2% | 28/28 | 44.8% |
| 10 | paired | all10 | 0.764 | 0.564 | 29% | 19.2% | 27/45 | 33.2% |
| 10 | paired | healthy8 | 0.535 | 0.143 | 29% | 30.8% | 27/28 | 42.7% |
| 10 | independent | all10 | 0.812 | 0.614 | 33% | 16.7% | 25/45 | 31.6% |
| 10 | independent | healthy8 | 0.629 | 0.239 | 33% | 26.8% | 25/28 | 40.6% |
| 20 | paired | all10 | 0.821 | 0.657 | 33% | 15.7% | 23/45 | 30.8% |
| 20 | paired | healthy8 | 0.648 | 0.323 | 33% | 25.3% | 23/28 | 39.6% |
| 20 | independent | all10 | 0.864 | 0.707 | 42% | 13.1% | 19/45 | 28.3% |
| 20 | independent | healthy8 | 0.732 | 0.422 | 42% | 21.0% | 19/28 | 36.4% |
| 50 | paired | all10 | 0.886 | 0.758 | 47% | 11.0% | 19/45 | 25.2% |
| 50 | paired | healthy8 | 0.776 | 0.524 | 47% | 17.7% | 19/28 | 32.4% |
| 50 | independent | all10 | 0.918 | 0.802 | 55% | 8.8% | 16/45 | 22.4% |
| 50 | independent | healthy8 | 0.840 | 0.611 | 55% | 14.1% | 16/28 | 28.8% |
| 100 | paired | all10 | 0.924 | 0.818 | 59% | 8.1% | 15/45 | 21.2% |
| 100 | paired | healthy8 | 0.850 | 0.643 | 59% | 13.1% | 15/28 | 27.2% |
| 100 | independent | all10 | 0.947 | 0.867 | 67% | 6.3% | 11/45 | 18.4% |
| 100 | independent | healthy8 | 0.897 | 0.738 | 67% | 10.1% | 11/28 | 23.7% |

## Same, win-rate metric

| n/cell | scheme | group | mean rho | p5 rho | top-1 stable | mean flip | pairs ≥5% flip | adj. flip |
|---|---|---|---|---|---|---|---|---|
| 5 | paired | all10 | 0.848 | 0.640 | 45% | 16.7% | 29/45 | 38.0% |
| 5 | paired | healthy8 | 0.726 | 0.333 | 45% | 23.9% | 25/28 | 43.4% |
| 5 | independent | all10 | 0.871 | 0.716 | 49% | 15.3% | 26/45 | 36.3% |
| 5 | independent | healthy8 | 0.766 | 0.479 | 49% | 22.0% | 22/28 | 42.4% |
| 10 | paired | all10 | 0.909 | 0.786 | 52% | 11.2% | 22/45 | 31.6% |
| 10 | paired | healthy8 | 0.828 | 0.595 | 52% | 17.1% | 20/28 | 38.9% |
| 10 | independent | all10 | 0.924 | 0.833 | 55% | 10.0% | 21/45 | 29.9% |
| 10 | independent | healthy8 | 0.857 | 0.683 | 55% | 15.3% | 18/28 | 37.2% |
| 20 | paired | all10 | 0.945 | 0.875 | 60% | 7.8% | 9/45 | 26.8% |
| 20 | paired | healthy8 | 0.892 | 0.755 | 60% | 12.4% | 9/28 | 34.1% |
| 20 | independent | all10 | 0.949 | 0.900 | 61% | 7.4% | 7/45 | 26.0% |
| 20 | independent | healthy8 | 0.902 | 0.810 | 61% | 11.8% | 7/28 | 33.3% |
| 50 | paired | all10 | 0.961 | 0.915 | 64% | 6.0% | 7/45 | 23.0% |
| 50 | paired | healthy8 | 0.923 | 0.833 | 64% | 9.7% | 7/28 | 29.5% |
| 50 | independent | all10 | 0.961 | 0.915 | 64% | 5.9% | 7/45 | 22.4% |
| 50 | independent | healthy8 | 0.924 | 0.833 | 64% | 9.5% | 7/28 | 28.8% |
| 100 | paired | all10 | 0.967 | 0.927 | 72% | 5.1% | 7/45 | 20.1% |
| 100 | paired | healthy8 | 0.935 | 0.857 | 72% | 8.3% | 7/28 | 25.9% |
| 100 | independent | all10 | 0.967 | 0.927 | 69% | 5.1% | 7/45 | 19.9% |
| 100 | independent | healthy8 | 0.935 | 0.857 | 69% | 8.2% | 7/28 | 25.6% |

## E17 conclusion under resampling (healthy 8, ladder mean v2-v5 vs peer survival)

- point estimate rho = 0.262 (p = 0.531)
- bootstrap 95% CI: [-0.024, 0.647] (n=100 games/cell, paired)
- P(rho <= 0) = 3.9%
- P(rho >= 0.74, the n=8 significance threshold) = 0.5%
