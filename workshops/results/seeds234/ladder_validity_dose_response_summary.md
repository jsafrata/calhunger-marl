# Ladder validity + population dose-response (campaign 1724783)

## Experiment A: does the scripted ladder proxy real peers?

| policy | vs v1 | vs v2 | vs v3 | vs v4 | vs v5 | mean v2-v5 | peer off-diag surv | peer off-diag win |
|---|---|---|---|---|---|---|---|---|
| ippo_s0 | 0 | 19 | 87 | 21 | 14 | 35.2 | 52.9 | 16.4 |
| ippo_s1 | 0 | 28 | 96 | 11 | 8 | 35.8 | 62.4 | 17.2 |
| ippo_s2 | 2 | 19 | 94 | 3 | 11 | 31.8 | 65.6 | 26.3 |
| ippo_s3 | 0 | 51 | 99 | 48 | 13 | 52.8 | 64.1 | 9.4 |
| ippo_s4 | 2 | 29 | 94 | 26 | 20 | 42.2 | 61.0 | 9.9 |
| mappo_s0 | 0 | 34 | 99 | 26 | 25 | 46.0 | 62.4 | 9.3 |
| mappo_s1 | 3 | 12 | 40 | 45 | 33 | 32.5 | 24.9 | 3.8 |
| mappo_s2 | 0 | 5 | 98 | 6 | 12 | 30.2 | 57.2 | 25.4 |
| mappo_s3 | 0 | 31 | 93 | 18 | 25 | 41.8 | 64.0 | 18.2 |
| mappo_s4 | 0 | 1 | 0 | 0 | 0 | 0.2 | 0.1 | 0.0 |

Spearman (all 10 policies), vs peer off-diagonal survival:

- vs v1 survival: rho=-0.05 (p=0.902)
- vs v2 survival: rho=+0.70 (p=0.024)
- vs v3 survival: rho=+0.62 (p=0.055)
- vs v4 survival: rho=+0.11 (p=0.769)
- vs v5 survival: rho=-0.02 (p=0.967)
- mean v2-v5 survival: rho=+0.52 (p=0.121)
- mean v2-v5 WIN vs peer off-diag WIN: rho=+0.58 (p=0.077)

Spearman (healthy 8 policies), vs peer off-diagonal survival:

- vs v1 survival: rho=+0.25 (p=0.545)
- vs v2 survival: rho=+0.46 (p=0.254)
- vs v3 survival: rho=+0.25 (p=0.553)
- vs v4 survival: rho=-0.07 (p=0.876)
- vs v5 survival: rho=-0.14 (p=0.744)
- mean v2-v5 survival: rho=+0.30 (p=0.471)
- mean v2-v5 WIN vs peer off-diag WIN: rho=+0.18 (p=0.670)

## Experiment B: survival vs number of strangers k at the table

Ordered pairs with complete k=0..3 curves: 90 / 90

Mean survival, all ordered pairs (n=90):

- k=0: 79.4% (median 92.4, min 0.0, max 95.2)
- k=1: 68.2% (median 82.8, min 0.0, max 95.0)
- k=2: 59.0% (median 73.5, min 0.0, max 94.5)
- k=3: 51.5% (median 65.0, min 0.0, max 93.0)

Mean survival, healthy focal AND healthy stranger (n=56):

- k=0: 92.2% (median 92.9, min 88.2, max 95.2)
- k=1: 87.3% (median 88.3, min 71.3, max 95.0)
- k=2: 80.7% (median 82.0, min 55.0, max 94.5)
- k=3: 74.0% (median 75.0, min 45.0, max 93.0)

Per-pair curves (survival % at k=0,1,2,3):

| focal | stranger | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|---|
| ippo_s0 | ippo_s1 | 88.2 | 76.7 | 62.0 | 45.0 |
| ippo_s0 | ippo_s2 | 88.2 | 89.3 | 82.5 | 70.0 |
| ippo_s0 | ippo_s3 | 88.2 | 79.0 | 69.5 | 67.0 |
| ippo_s0 | ippo_s4 | 88.2 | 82.3 | 71.0 | 64.0 |
| ippo_s0 | mappo_s0 | 88.2 | 83.0 | 68.5 | 55.0 |
| ippo_s0 | mappo_s1 | 88.2 | 58.7 | 33.5 | 3.0 |
| ippo_s0 | mappo_s2 | 88.2 | 91.3 | 89.5 | 81.0 |
| ippo_s0 | mappo_s3 | 88.2 | 87.7 | 84.5 | 70.0 |
| ippo_s0 | mappo_s4 | 88.2 | 54.7 | 34.0 | 21.0 |
| ippo_s1 | ippo_s0 | 93.2 | 87.3 | 82.0 | 63.0 |
| ippo_s1 | ippo_s2 | 93.2 | 90.7 | 86.5 | 83.0 |
| ippo_s1 | ippo_s3 | 93.2 | 88.0 | 78.5 | 66.0 |
| ippo_s1 | ippo_s4 | 93.2 | 94.0 | 87.5 | 89.0 |
| ippo_s1 | mappo_s0 | 93.2 | 83.0 | 71.5 | 65.0 |
| ippo_s1 | mappo_s1 | 93.2 | 73.3 | 37.0 | 10.0 |
| ippo_s1 | mappo_s2 | 93.2 | 88.0 | 89.0 | 91.0 |
| ippo_s1 | mappo_s3 | 93.2 | 93.0 | 85.0 | 72.0 |
| ippo_s1 | mappo_s4 | 93.2 | 60.0 | 52.0 | 23.0 |
| ippo_s2 | ippo_s0 | 92.5 | 93.7 | 86.5 | 81.0 |
| ippo_s2 | ippo_s1 | 92.5 | 83.7 | 74.0 | 65.0 |
| ippo_s2 | ippo_s3 | 92.5 | 88.7 | 81.5 | 76.0 |
| ippo_s2 | ippo_s4 | 92.5 | 82.7 | 81.0 | 74.0 |
| ippo_s2 | mappo_s0 | 92.5 | 91.3 | 77.5 | 71.0 |
| ippo_s2 | mappo_s1 | 92.5 | 62.3 | 23.5 | 5.0 |
| ippo_s2 | mappo_s2 | 92.5 | 92.0 | 92.0 | 91.0 |
| ippo_s2 | mappo_s3 | 92.5 | 95.0 | 90.5 | 87.0 |
| ippo_s2 | mappo_s4 | 92.5 | 63.0 | 46.0 | 40.0 |
| ippo_s3 | ippo_s0 | 95.2 | 88.7 | 83.5 | 67.0 |
| ippo_s3 | ippo_s1 | 95.2 | 86.0 | 80.5 | 75.0 |
| ippo_s3 | ippo_s2 | 95.2 | 90.3 | 89.0 | 80.0 |
| ippo_s3 | ippo_s4 | 95.2 | 85.7 | 81.0 | 78.0 |
| ippo_s3 | mappo_s0 | 95.2 | 91.3 | 81.0 | 71.0 |
| ippo_s3 | mappo_s1 | 95.2 | 70.0 | 32.5 | 13.0 |
| ippo_s3 | mappo_s2 | 95.2 | 95.0 | 89.0 | 93.0 |
| ippo_s3 | mappo_s3 | 95.2 | 95.0 | 85.5 | 80.0 |
| ippo_s3 | mappo_s4 | 95.2 | 68.7 | 40.5 | 20.0 |
| ippo_s4 | ippo_s0 | 92.2 | 92.0 | 74.5 | 65.0 |
| ippo_s4 | ippo_s1 | 92.2 | 92.7 | 83.5 | 75.0 |
| ippo_s4 | ippo_s2 | 92.2 | 89.0 | 85.5 | 77.0 |
| ippo_s4 | ippo_s3 | 92.2 | 85.3 | 83.0 | 69.0 |
| ippo_s4 | mappo_s0 | 92.2 | 89.7 | 82.0 | 78.0 |
| ippo_s4 | mappo_s1 | 92.2 | 76.3 | 41.0 | 10.0 |
| ippo_s4 | mappo_s2 | 92.2 | 88.0 | 82.0 | 80.0 |
| ippo_s4 | mappo_s3 | 92.2 | 91.3 | 85.0 | 76.0 |
| ippo_s4 | mappo_s4 | 92.2 | 73.3 | 50.0 | 19.0 |
| mappo_s0 | ippo_s0 | 89.2 | 74.7 | 78.0 | 77.0 |
| mappo_s0 | ippo_s1 | 89.2 | 71.3 | 55.0 | 55.0 |
| mappo_s0 | ippo_s2 | 89.2 | 77.0 | 70.5 | 72.0 |
| mappo_s0 | ippo_s3 | 89.2 | 81.3 | 72.5 | 75.0 |
| mappo_s0 | ippo_s4 | 89.2 | 75.7 | 66.0 | 73.0 |
| mappo_s0 | mappo_s1 | 89.2 | 50.3 | 27.0 | 17.0 |
| mappo_s0 | mappo_s2 | 89.2 | 78.0 | 80.0 | 86.0 |
| mappo_s0 | mappo_s3 | 89.2 | 81.0 | 78.5 | 81.0 |
| mappo_s0 | mappo_s4 | 89.2 | 56.3 | 42.5 | 26.0 |
| mappo_s1 | ippo_s0 | 56.8 | 28.7 | 18.5 | 24.0 |
| mappo_s1 | ippo_s1 | 56.8 | 26.0 | 20.0 | 28.0 |
| mappo_s1 | ippo_s2 | 56.8 | 28.3 | 19.5 | 20.0 |
| mappo_s1 | ippo_s3 | 56.8 | 35.0 | 24.0 | 28.0 |
| mappo_s1 | ippo_s4 | 56.8 | 29.0 | 33.5 | 40.0 |
| mappo_s1 | mappo_s0 | 56.8 | 25.7 | 27.0 | 23.0 |
| mappo_s1 | mappo_s2 | 56.8 | 22.7 | 24.5 | 27.0 |
| mappo_s1 | mappo_s3 | 56.8 | 17.3 | 13.5 | 12.0 |
| mappo_s1 | mappo_s4 | 56.8 | 45.7 | 39.5 | 22.0 |
| mappo_s2 | ippo_s0 | 93.5 | 89.7 | 92.5 | 75.0 |
| mappo_s2 | ippo_s1 | 93.5 | 86.3 | 71.5 | 60.0 |
| mappo_s2 | ippo_s2 | 93.5 | 91.7 | 85.0 | 84.0 |
| mappo_s2 | ippo_s3 | 93.5 | 87.7 | 78.5 | 76.0 |
| mappo_s2 | ippo_s4 | 93.5 | 86.3 | 75.5 | 64.0 |
| mappo_s2 | mappo_s0 | 93.5 | 86.3 | 74.5 | 62.0 |
| mappo_s2 | mappo_s1 | 93.5 | 67.3 | 26.5 | 0.0 |
| mappo_s2 | mappo_s3 | 93.5 | 89.7 | 88.5 | 77.0 |
| mappo_s2 | mappo_s4 | 93.5 | 60.0 | 39.0 | 17.0 |
| mappo_s3 | ippo_s0 | 93.2 | 91.3 | 86.0 | 71.0 |
| mappo_s3 | ippo_s1 | 93.2 | 84.0 | 73.0 | 63.0 |
| mappo_s3 | ippo_s2 | 93.2 | 94.7 | 94.5 | 85.0 |
| mappo_s3 | ippo_s3 | 93.2 | 89.0 | 78.5 | 78.0 |
| mappo_s3 | ippo_s4 | 93.2 | 87.0 | 87.0 | 80.0 |
| mappo_s3 | mappo_s0 | 93.2 | 92.3 | 89.5 | 77.0 |
| mappo_s3 | mappo_s1 | 93.2 | 50.3 | 14.5 | 4.0 |
| mappo_s3 | mappo_s2 | 93.2 | 91.7 | 88.5 | 83.0 |
| mappo_s3 | mappo_s4 | 93.2 | 48.0 | 34.5 | 35.0 |
| mappo_s4 | ippo_s0 | 0.0 | 0.0 | 0.0 | 0.0 |
| mappo_s4 | ippo_s1 | 0.0 | 0.0 | 0.0 | 0.0 |
| mappo_s4 | ippo_s2 | 0.0 | 0.0 | 0.0 | 0.0 |
| mappo_s4 | ippo_s3 | 0.0 | 0.0 | 0.0 | 1.0 |
| mappo_s4 | ippo_s4 | 0.0 | 0.0 | 0.0 | 0.0 |
| mappo_s4 | mappo_s0 | 0.0 | 0.0 | 0.0 | 0.0 |
| mappo_s4 | mappo_s1 | 0.0 | 0.0 | 0.0 | 0.0 |
| mappo_s4 | mappo_s2 | 0.0 | 0.0 | 0.0 | 0.0 |
| mappo_s4 | mappo_s3 | 0.0 | 0.0 | 0.0 | 0.0 |
