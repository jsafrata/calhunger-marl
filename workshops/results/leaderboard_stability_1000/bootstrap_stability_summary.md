# E19: bootstrap stability of the peer cross-play leaderboard

B=2000 replicates; seed 20260827; score = macro-mean over the 9 focal 1v3 cells; full data = 1000 paired games/cell (merged 100+900 seed extension).

## Leaderboard stability vs evaluation budget (survival metric)

| n/cell | scheme | group | mean rho | p5 rho | top-1 stable | mean flip | pairs ≥5% flip | adj. flip |
|---|---|---|---|---|---|---|---|---|
| 5 | paired | all10 | 0.731 | 0.442 | 21% | 21.6% | 28/45 | 36.2% |
| 5 | paired | healthy8 | 0.471 | -0.095 | 21% | 34.6% | 28/28 | 46.3% |
| 5 | independent | all10 | 0.787 | 0.543 | 27% | 18.9% | 28/45 | 34.5% |
| 5 | independent | healthy8 | 0.580 | 0.096 | 27% | 30.3% | 28/28 | 44.4% |
| 10 | paired | all10 | 0.795 | 0.547 | 25% | 17.4% | 26/45 | 32.6% |
| 10 | paired | healthy8 | 0.597 | 0.108 | 25% | 28.0% | 26/28 | 41.9% |
| 10 | independent | all10 | 0.847 | 0.673 | 32% | 14.6% | 22/45 | 30.9% |
| 10 | independent | healthy8 | 0.699 | 0.357 | 32% | 23.5% | 22/28 | 39.7% |
| 20 | paired | all10 | 0.852 | 0.675 | 32% | 13.8% | 21/45 | 29.6% |
| 20 | paired | healthy8 | 0.710 | 0.359 | 32% | 22.2% | 21/28 | 38.0% |
| 20 | independent | all10 | 0.893 | 0.760 | 37% | 11.1% | 17/45 | 26.2% |
| 20 | independent | healthy8 | 0.789 | 0.524 | 37% | 17.9% | 17/28 | 33.7% |
| 50 | paired | all10 | 0.905 | 0.770 | 41% | 9.7% | 16/45 | 23.5% |
| 50 | paired | healthy8 | 0.813 | 0.548 | 41% | 15.6% | 16/28 | 30.2% |
| 50 | independent | all10 | 0.932 | 0.842 | 45% | 7.7% | 14/45 | 20.6% |
| 50 | independent | healthy8 | 0.867 | 0.690 | 45% | 12.3% | 14/28 | 26.4% |
| 100 | paired | all10 | 0.933 | 0.842 | 50% | 7.4% | 14/45 | 19.5% |
| 100 | paired | healthy8 | 0.868 | 0.690 | 50% | 11.9% | 14/28 | 25.0% |
| 100 | independent | all10 | 0.952 | 0.879 | 57% | 5.7% | 9/45 | 16.4% |
| 100 | independent | healthy8 | 0.906 | 0.762 | 57% | 9.2% | 9/28 | 21.0% |
| 200 | paired | all10 | 0.956 | 0.891 | 60% | 5.3% | 9/45 | 15.4% |
| 200 | paired | healthy8 | 0.914 | 0.786 | 60% | 8.5% | 9/28 | 19.8% |
| 200 | independent | all10 | 0.970 | 0.915 | 69% | 4.0% | 9/45 | 12.9% |
| 200 | independent | healthy8 | 0.941 | 0.833 | 69% | 6.4% | 9/28 | 16.6% |
| 500 | paired | all10 | 0.978 | 0.939 | 79% | 3.1% | 7/45 | 11.0% |
| 500 | paired | healthy8 | 0.957 | 0.881 | 79% | 4.9% | 7/28 | 14.1% |
| 500 | independent | all10 | 0.985 | 0.952 | 85% | 2.2% | 5/45 | 9.0% |
| 500 | independent | healthy8 | 0.971 | 0.905 | 85% | 3.6% | 5/28 | 11.5% |
| 1000 | paired | all10 | 0.989 | 0.964 | 91% | 1.8% | 5/45 | 7.4% |
| 1000 | paired | healthy8 | 0.978 | 0.929 | 91% | 2.8% | 5/28 | 9.6% |
| 1000 | independent | all10 | 0.993 | 0.964 | 95% | 1.2% | 2/45 | 5.5% |
| 1000 | independent | healthy8 | 0.986 | 0.929 | 95% | 2.0% | 2/28 | 7.1% |

## Same, win-rate metric

| n/cell | scheme | group | mean rho | p5 rho | top-1 stable | mean flip | pairs ≥5% flip | adj. flip |
|---|---|---|---|---|---|---|---|---|
| 5 | paired | all10 | 0.835 | 0.616 | 58% | 17.4% | 30/45 | 37.3% |
| 5 | paired | healthy8 | 0.697 | 0.263 | 58% | 25.3% | 25/28 | 43.1% |
| 5 | independent | all10 | 0.862 | 0.679 | 63% | 15.8% | 28/45 | 36.2% |
| 5 | independent | healthy8 | 0.743 | 0.386 | 63% | 23.2% | 24/28 | 41.9% |
| 10 | paired | all10 | 0.897 | 0.754 | 71% | 11.6% | 24/45 | 29.7% |
| 10 | paired | healthy8 | 0.802 | 0.524 | 71% | 18.0% | 22/28 | 36.9% |
| 10 | independent | all10 | 0.912 | 0.796 | 75% | 10.7% | 21/45 | 29.2% |
| 10 | independent | healthy8 | 0.830 | 0.619 | 75% | 16.7% | 20/28 | 36.2% |
| 20 | paired | all10 | 0.939 | 0.855 | 84% | 8.0% | 13/45 | 24.7% |
| 20 | paired | healthy8 | 0.881 | 0.714 | 84% | 12.7% | 13/28 | 31.6% |
| 20 | independent | all10 | 0.948 | 0.879 | 87% | 7.2% | 11/45 | 23.4% |
| 20 | independent | healthy8 | 0.898 | 0.778 | 87% | 11.5% | 11/28 | 30.0% |
| 50 | paired | all10 | 0.962 | 0.915 | 96% | 5.6% | 6/45 | 19.6% |
| 50 | paired | healthy8 | 0.924 | 0.833 | 96% | 9.0% | 6/28 | 25.2% |
| 50 | independent | all10 | 0.964 | 0.915 | 97% | 5.3% | 6/45 | 18.9% |
| 50 | independent | healthy8 | 0.930 | 0.833 | 97% | 8.5% | 6/28 | 24.3% |
| 100 | paired | all10 | 0.969 | 0.915 | 99% | 4.6% | 6/45 | 17.0% |
| 100 | paired | healthy8 | 0.938 | 0.833 | 99% | 7.5% | 6/28 | 21.9% |
| 100 | independent | all10 | 0.971 | 0.939 | 100% | 4.4% | 6/45 | 16.5% |
| 100 | independent | healthy8 | 0.943 | 0.881 | 100% | 7.0% | 6/28 | 21.2% |
| 200 | paired | all10 | 0.976 | 0.927 | 100% | 3.7% | 6/45 | 14.3% |
| 200 | paired | healthy8 | 0.952 | 0.857 | 100% | 5.9% | 6/28 | 18.4% |
| 200 | independent | all10 | 0.977 | 0.939 | 100% | 3.5% | 6/45 | 13.6% |
| 200 | independent | healthy8 | 0.956 | 0.881 | 100% | 5.6% | 6/28 | 17.5% |
| 500 | paired | all10 | 0.983 | 0.952 | 100% | 2.7% | 5/45 | 11.1% |
| 500 | paired | healthy8 | 0.967 | 0.905 | 100% | 4.3% | 5/28 | 14.2% |
| 500 | independent | all10 | 0.986 | 0.952 | 100% | 2.3% | 5/45 | 10.1% |
| 500 | independent | healthy8 | 0.972 | 0.905 | 100% | 3.8% | 5/28 | 13.0% |
| 1000 | paired | all10 | 0.990 | 0.964 | 100% | 1.8% | 5/45 | 8.1% |
| 1000 | paired | healthy8 | 0.980 | 0.929 | 100% | 2.9% | 5/28 | 10.5% |
| 1000 | independent | all10 | 0.991 | 0.964 | 100% | 1.6% | 4/45 | 7.4% |
| 1000 | independent | healthy8 | 0.982 | 0.929 | 100% | 2.6% | 4/28 | 9.5% |

## E17 conclusion under resampling (healthy 8, ladder mean v2-v5 vs peer survival)

- point estimate rho = 0.000 (p = 1.000)
- bootstrap 95% CI: [-0.286, 0.262] (n=100 games/cell, paired)
- P(rho <= 0) = 61.6%
- P(rho >= 0.74, the n=8 significance threshold) = 0.0%
