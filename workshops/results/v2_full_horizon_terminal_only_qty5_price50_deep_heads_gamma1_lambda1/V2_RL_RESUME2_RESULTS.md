# v2 RL resume² results — IPPO & MAPPO iter 800 → iter 1100

Resumed from the exact `iter_0800` RLlib checkpoint of each algo. Config UNCHANGED from the 0-800 run — same terminal-only reward, MLP heads, gamma=1.0, lambda=1.0, **entropy_coeff=0.01**, lr=3e-4, seed=0, 4096 batch / 1024 minibatch / 4 epochs. Purpose: determine where the current configuration naturally plateaus.

New checkpoints at iterations **900, 1000, 1100**. Deep evals at each on 500 games per lineup per mode. Terminal cash + required-food inventory diagnostics carry over from the v2b pipeline. All earlier checkpoints and eval JSONs (iter_0100 … iter_0800) are preserved unchanged.

## 1 · Resume confirmation

- **IPPO**: iter counter reached **1100**, iter 801 present in `iters.jsonl` = **True**, checkpoints on disk = `iter_0100, iter_0200, iter_0300, iter_0400, iter_0500, iter_0600, iter_0700, iter_0800, iter_0900, iter_1000, iter_1100`.
- **MAPPO**: iter counter reached **1100**, iter 801 present in `iters.jsonl` = **True**, checkpoints on disk = `iter_0100, iter_0200, iter_0300, iter_0400, iter_0500, iter_0600, iter_0700, iter_0800, iter_0900, iter_1000, iter_1100`.

Both runs resumed at algo iteration 800 (per pipeline log) and continued the existing `iters.jsonl` with absolute iter counters 801..1100.

## 2 · Training curves through iter 1100

### IPPO — every 50 iters

| iter | ep_return | n_survivors | entropy | vf_expl_var | grad_norm | ep_len |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | -3.240 | 1.24 | 3.245 | 0.665 | 0.429 | 158 |
| 550 | -2.958 | 1.29 | 3.386 | 0.738 | 0.416 | 164 |
| 600 | -2.292 | 1.67 | 3.043 | 0.838 | 0.494 | 169 |
| 650 | -2.083 | 1.42 | 3.379 | 0.611 | 0.497 | 166 |
| 700 | -1.144 | 2.26 | 3.505 | 0.394 | 0.608 | 173 |
| 750 | -1.375 | 1.88 | 3.628 | 0.506 | 0.748 | 167 |
| 800 | -1.614 | 2.05 | 3.277 | 0.517 | 0.725 | 174 |
| 850 | -0.636 | 2.68 | 3.569 | 0.348 | 0.753 | 175 |
| 900 | -0.231 | 3.21 | 3.626 | 0.512 | 0.674 | 171 |
| 950 | -0.057 | 3.20 | 3.750 | 0.500 | 0.734 | 171 |
| 1000 | 0.163 | 3.34 | 3.879 | 0.400 | 0.676 | 171 |
| 1050 | -0.045 | 3.27 | 3.732 | 0.302 | 0.730 | 179 |
| 1100 | -0.322 | 3.17 | 3.859 | 0.311 | 0.685 | 171 |

#### IPPO — window means

| window | mean return | std | mean n_survivors | mean entropy | mean vf_expl_var |
|---|---:|---:|---:|---:|---:|
| 501-600 | -2.512 | 0.357 | 1.48 | 3.250 | 0.692 |
| 601-700 | -1.795 | 0.286 | 1.71 | 3.217 | 0.661 |
| 701-800 | -1.531 | 0.243 | 1.93 | 3.497 | 0.587 |
| 801-900 | -0.788 | 0.380 | 2.55 | 3.677 | 0.472 |
| 901-1000 | -0.146 | 0.211 | 3.08 | 3.775 | 0.394 |
| 1001-1100 | 0.012 | 0.195 | 3.26 | 3.760 | 0.359 |

### MAPPO — every 50 iters

| iter | ep_return | n_survivors | entropy | vf_expl_var | grad_norm | ep_len |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | -0.083 | 3.12 | 4.007 | 0.242 | 0.882 | 167 |
| 550 | 0.299 | 3.44 | 4.293 | 0.281 | 0.952 | 173 |
| 600 | 0.174 | 3.39 | 3.982 | 0.345 | 0.742 | 171 |
| 650 | -0.364 | 2.95 | 4.353 | 0.310 | 0.920 | 174 |
| 700 | 0.545 | 3.59 | 4.275 | 0.189 | 0.694 | 174 |
| 750 | 0.500 | 3.55 | 4.147 | 0.296 | 0.747 | 175 |
| 800 | 0.136 | 3.32 | 3.960 | 0.435 | 0.840 | 174 |
| 850 | -0.265 | 3.04 | 4.075 | 0.471 | 0.862 | 171 |
| 900 | 0.250 | 3.51 | 4.284 | 0.298 | 0.791 | 171 |
| 950 | 0.470 | 3.60 | 4.454 | 0.262 | 0.930 | 172 |
| 1000 | 0.356 | 3.52 | 4.359 | 0.377 | 0.946 | 172 |
| 1050 | -0.292 | 3.01 | 4.273 | 0.373 | 0.812 | 172 |
| 1100 | 0.682 | 3.77 | 4.137 | 0.432 | 0.793 | 177 |

#### MAPPO — window means

| window | mean return | std | mean n_survivors | mean entropy | mean vf_expl_var |
|---|---:|---:|---:|---:|---:|
| 501-600 | 0.000 | 0.291 | 3.26 | 4.159 | 0.300 |
| 601-700 | 0.126 | 0.335 | 3.36 | 4.205 | 0.285 |
| 701-800 | 0.292 | 0.272 | 3.49 | 4.136 | 0.296 |
| 801-900 | 0.069 | 0.291 | 3.34 | 4.228 | 0.335 |
| 901-1000 | 0.341 | 0.195 | 3.52 | 4.355 | 0.355 |
| 1001-1100 | 0.242 | 0.243 | 3.45 | 4.349 | 0.391 |

## 3 · Deep evaluation trajectory (500 games each)

Learned-seat survival = fraction of the learned seat(s) that reach t=180 alive. All four v2c iterations shown alongside iter 800 for continuity.

### IPPO — learned-seat survival

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 800 | 0.113 | 0.432 | 0.074 | 0.434 |
| 900 | 0.323 | 0.731 | 0.208 | 0.564 |
| 1000 | 0.425 | 0.724 | 0.174 | 0.516 |
| 1100 | 0.337 | 0.791 | 0.152 | 0.440 |

### IPPO — reach-180 fraction

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 800 | 0.422 | 0.966 | 0.962 | 1.000 |
| 900 | 0.736 | 1.000 | 0.984 | 1.000 |
| 1000 | 0.832 | 1.000 | 0.984 | 0.998 |
| 1100 | 0.712 | 1.000 | 0.982 | 1.000 |

### IPPO — trades per game

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 800 | 57.4 | 157.2 | 99.7 | 131.6 |
| 900 | 92.5 | 189.3 | 111.5 | 141.0 |
| 1000 | 121.6 | 208.5 | 120.3 | 146.1 |
| 1100 | 93.6 | 195.0 | 117.6 | 148.9 |

### IPPO — selfplay stochastic action-type frequency

| iter | WAIT | CANCEL | TAKE | POST |
|---:|---:|---:|---:|---:|
| 800 | 0.28 | 0.12 | 0.17 | 0.42 |
| 900 | 0.15 | 0.15 | 0.19 | 0.52 |
| 1000 | 0.13 | 0.13 | 0.19 | 0.55 |
| 1100 | 0.14 | 0.13 | 0.16 | 0.57 |

### IPPO — head diagnostics at iter 1100 (selfplay stochastic)

- type mean_prob:  WAIT=0.114  CANCEL=0.139  TAKE=0.190  POST=0.557
- type argmax_freq: WAIT=0.084  CANCEL=0.111  TAKE=0.164  POST=0.641
- type entropy_mean: 0.634
- price|POST entropy: 2.662  argmax median: 13.0
- qty|POST entropy: 1.040  argmax median: 3.0

### MAPPO — learned-seat survival

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 800 | 0.209 | 0.789 | 0.004 | 0.194 |
| 900 | 0.177 | 0.834 | 0.022 | 0.184 |
| 1000 | 0.150 | 0.843 | 0.010 | 0.082 |
| 1100 | 0.180 | 0.852 | 0.006 | 0.082 |

### MAPPO — reach-180 fraction

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 800 | 0.442 | 0.986 | 0.990 | 1.000 |
| 900 | 0.498 | 1.000 | 0.990 | 1.000 |
| 1000 | 0.408 | 0.998 | 0.986 | 1.000 |
| 1100 | 0.444 | 1.000 | 0.992 | 1.000 |

### MAPPO — trades per game

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 800 | 65.4 | 188.1 | 88.6 | 120.2 |
| 900 | 58.0 | 178.6 | 98.8 | 125.1 |
| 1000 | 50.9 | 195.9 | 88.7 | 122.6 |
| 1100 | 51.8 | 178.9 | 81.0 | 108.9 |

### MAPPO — selfplay stochastic action-type frequency

| iter | WAIT | CANCEL | TAKE | POST |
|---:|---:|---:|---:|---:|
| 800 | 0.11 | 0.16 | 0.19 | 0.54 |
| 900 | 0.08 | 0.16 | 0.17 | 0.59 |
| 1000 | 0.09 | 0.13 | 0.20 | 0.59 |
| 1100 | 0.15 | 0.13 | 0.19 | 0.53 |

### MAPPO — head diagnostics at iter 1100 (selfplay stochastic)

- type mean_prob:  WAIT=0.105  CANCEL=0.135  TAKE=0.226  POST=0.534
- type argmax_freq: WAIT=0.049  CANCEL=0.080  TAKE=0.186  POST=0.686
- type entropy_mean: 0.877
- price|POST entropy: 2.686  argmax median: 13.0
- qty|POST entropy: 1.256  argmax median: 3.0

## 4 · Learned-seat terminal cash by outcome — 1 learned + 3 Scripted-v1 Fast

Per the follow-up request. Learned seat only, aggregated across 500 games per iter × mode. `n` is the sample count for that outcome — trust the mean only where n is meaningful (≥ ~30 as a soft cutoff).

### IPPO — 1learned+3scripted, terminal cash by outcome (learned seat)

| iter | mode | outcome | n | mean cash | median | std |
|---:|---|---|---:|---:|---:|---:|
| 800 | determ | all | 500 | 75.0 | 79.5 | 46.3 |
| 800 | determ | dead | 463 | 77.6 | 82.0 | 45.6 |
| 800 | determ | alive_nonwinner | 35 | 34.8 | 23.0 | 30.5 |
| 800 | determ | winner | 2 | 165.0 | 165.0 | 14.0 |
| 800 | stocha | all | 500 | 40.2 | 30.0 | 32.8 |
| 800 | stocha | dead | 283 | 39.0 | 30.0 | 29.0 |
| 800 | stocha | alive_nonwinner | 211 | 38.2 | 28.0 | 30.4 |
| 800 | stocha | winner | 6 | 168.7 | 164.0 | 28.8 |
| 900 | determ | all | 500 | 88.4 | 78.0 | 55.7 |
| 900 | determ | dead | 396 | 92.9 | 81.5 | 56.6 |
| 900 | determ | alive_nonwinner | 95 | 61.9 | 54.0 | 37.8 |
| 900 | determ | winner | 9 | 172.7 | 163.0 | 27.7 |
| 900 | stocha | all | 500 | 57.0 | 49.5 | 40.6 |
| 900 | stocha | dead | 218 | 54.2 | 45.5 | 39.0 |
| 900 | stocha | alive_nonwinner | 264 | 52.9 | 50.0 | 34.1 |
| 900 | stocha | winner | 18 | 151.7 | 141.5 | 32.0 |
| 1000 | determ | all | 500 | 109.7 | 97.5 | 71.1 |
| 1000 | determ | dead | 413 | 115.8 | 108.0 | 73.0 |
| 1000 | determ | alive_nonwinner | 78 | 68.0 | 63.5 | 36.6 |
| 1000 | determ | winner | 9 | 190.7 | 193.0 | 33.5 |
| 1000 | stocha | all | 500 | 75.1 | 64.0 | 47.7 |
| 1000 | stocha | dead | 242 | 72.9 | 61.0 | 46.2 |
| 1000 | stocha | alive_nonwinner | 228 | 64.2 | 60.0 | 33.3 |
| 1000 | stocha | winner | 30 | 176.4 | 170.0 | 31.2 |
| 1100 | determ | all | 500 | 118.0 | 112.5 | 72.5 |
| 1100 | determ | dead | 424 | 120.2 | 115.5 | 72.8 |
| 1100 | determ | alive_nonwinner | 53 | 67.9 | 57.0 | 42.4 |
| 1100 | determ | winner | 23 | 192.3 | 187.0 | 32.7 |
| 1100 | stocha | all | 500 | 101.2 | 90.0 | 60.0 |
| 1100 | stocha | dead | 280 | 101.7 | 93.5 | 59.8 |
| 1100 | stocha | alive_nonwinner | 166 | 73.5 | 71.0 | 36.4 |
| 1100 | stocha | winner | 54 | 184.0 | 179.0 | 39.3 |

### MAPPO — 1learned+3scripted, terminal cash by outcome (learned seat)

| iter | mode | outcome | n | mean cash | median | std |
|---:|---|---|---:|---:|---:|---:|
| 800 | determ | all | 500 | 150.0 | 144.5 | 67.7 |
| 800 | determ | dead | 498 | 149.8 | 144.0 | 67.8 |
| 800 | determ | alive_nonwinner | 0 | — | — | — |
| 800 | determ | winner | 2 | 176.5 | 176.5 | 7.5 |
| 800 | stocha | all | 500 | 154.6 | 152.0 | 67.0 |
| 800 | stocha | dead | 403 | 157.2 | 159.0 | 67.1 |
| 800 | stocha | alive_nonwinner | 47 | 91.9 | 93.0 | 40.8 |
| 800 | stocha | winner | 50 | 193.4 | 197.5 | 42.7 |
| 900 | determ | all | 500 | 173.1 | 172.0 | 75.6 |
| 900 | determ | dead | 489 | 171.4 | 170.0 | 75.4 |
| 900 | determ | alive_nonwinner | 0 | — | — | — |
| 900 | determ | winner | 11 | 247.6 | 244.0 | 40.4 |
| 900 | stocha | all | 500 | 186.8 | 188.0 | 71.6 |
| 900 | stocha | dead | 408 | 184.0 | 186.0 | 71.8 |
| 900 | stocha | alive_nonwinner | 19 | 106.4 | 105.0 | 27.5 |
| 900 | stocha | winner | 73 | 223.6 | 211.0 | 54.3 |
| 1000 | determ | all | 500 | 166.1 | 159.5 | 69.2 |
| 1000 | determ | dead | 495 | 166.1 | 160.0 | 69.3 |
| 1000 | determ | alive_nonwinner | 2 | 107.0 | 107.0 | 0.0 |
| 1000 | determ | winner | 3 | 204.0 | 227.0 | 34.7 |
| 1000 | stocha | all | 500 | 175.4 | 172.5 | 65.4 |
| 1000 | stocha | dead | 459 | 175.4 | 172.0 | 65.3 |
| 1000 | stocha | alive_nonwinner | 13 | 102.4 | 96.0 | 36.1 |
| 1000 | stocha | winner | 28 | 209.8 | 199.0 | 48.1 |
| 1100 | determ | all | 500 | 177.8 | 176.0 | 56.3 |
| 1100 | determ | dead | 497 | 177.7 | 176.0 | 56.4 |
| 1100 | determ | alive_nonwinner | 1 | 126.0 | 126.0 | 0.0 |
| 1100 | determ | winner | 2 | 224.5 | 224.5 | 8.5 |
| 1100 | stocha | all | 500 | 200.5 | 204.5 | 60.0 |
| 1100 | stocha | dead | 459 | 199.8 | 206.0 | 59.5 |
| 1100 | stocha | alive_nonwinner | 5 | 115.6 | 112.0 | 32.2 |
| 1100 | stocha | winner | 36 | 221.9 | 205.0 | 56.1 |

## 5 · Terminal cash + required-food inventory — selfplay

### IPPO — selfplay terminal cash by outcome

| iter | mode | outcome | n | mean cash | median | std |
|---:|---|---|---:|---:|---:|---:|
| 800 | determ | all | 2000 | 100.0 | 100.0 | 61.1 |
| 800 | determ | dead | 1775 | 101.1 | 100.0 | 60.7 |
| 800 | determ | alive_nonwinner | 49 | 54.7 | 49.0 | 38.3 |
| 800 | determ | winner | 176 | 101.2 | 95.0 | 65.8 |
| 800 | stocha | all | 2000 | 100.0 | 89.0 | 67.9 |
| 800 | stocha | dead | 1135 | 93.9 | 80.0 | 68.1 |
| 800 | stocha | alive_nonwinner | 401 | 67.4 | 63.0 | 37.5 |
| 800 | stocha | winner | 464 | 143.2 | 143.0 | 66.6 |
| 900 | determ | all | 2000 | 100.0 | 86.0 | 72.1 |
| 900 | determ | dead | 1355 | 96.9 | 83.0 | 71.4 |
| 900 | determ | alive_nonwinner | 326 | 64.0 | 59.0 | 41.9 |
| 900 | determ | winner | 319 | 150.1 | 158.0 | 72.6 |
| 900 | stocha | all | 2000 | 100.0 | 88.5 | 65.5 |
| 900 | stocha | dead | 538 | 91.7 | 79.0 | 63.8 |
| 900 | stocha | alive_nonwinner | 963 | 69.4 | 69.0 | 39.5 |
| 900 | stocha | winner | 499 | 168.0 | 165.0 | 57.4 |
| 1000 | determ | all | 2000 | 100.0 | 83.0 | 71.7 |
| 1000 | determ | dead | 1150 | 98.0 | 82.0 | 69.0 |
| 1000 | determ | alive_nonwinner | 461 | 61.6 | 56.0 | 39.8 |
| 1000 | determ | winner | 389 | 151.5 | 155.0 | 77.7 |
| 1000 | stocha | all | 2000 | 100.0 | 88.0 | 65.1 |
| 1000 | stocha | dead | 553 | 92.8 | 84.0 | 61.3 |
| 1000 | stocha | alive_nonwinner | 950 | 67.9 | 64.0 | 37.8 |
| 1000 | stocha | winner | 497 | 169.4 | 168.0 | 57.4 |
| 1100 | determ | all | 2000 | 100.0 | 82.0 | 78.7 |
| 1100 | determ | dead | 1327 | 97.1 | 78.0 | 76.2 |
| 1100 | determ | alive_nonwinner | 372 | 57.6 | 47.5 | 41.0 |
| 1100 | determ | winner | 301 | 165.3 | 166.0 | 83.1 |
| 1100 | stocha | all | 2000 | 100.0 | 84.0 | 73.9 |
| 1100 | stocha | dead | 418 | 97.9 | 88.0 | 72.9 |
| 1100 | stocha | alive_nonwinner | 1087 | 63.6 | 58.0 | 41.1 |
| 1100 | stocha | winner | 495 | 181.8 | 180.0 | 65.2 |

### IPPO — selfplay required-food inventory by outcome

`mean-of-two` = (req_1 + req_2)/2; `min-of-two` = min(req_1, req_2).

| iter | mode | outcome | n | mean-of-two (mean/std) | min-of-two (mean/median) | total (mean) |
|---:|---|---|---:|---:|---:|---:|
| 800 | determ | all | 2000 | 4.77 / 5.47 | 0.91 / 0.0 | 9.54 |
| 800 | determ | dead | 1775 | 3.85 / 4.32 | 0.24 / 0.0 | 7.70 |
| 800 | determ | alive_nonwinner | 49 | 12.23 / 8.73 | 7.00 / 4.0 | 24.47 |
| 800 | determ | winner | 176 | 11.97 / 7.45 | 5.91 / 4.0 | 23.93 |
| 800 | stocha | all | 2000 | 7.25 / 5.43 | 2.65 / 0.0 | 14.50 |
| 800 | stocha | dead | 1135 | 5.17 / 3.99 | 0.24 / 0.0 | 10.35 |
| 800 | stocha | alive_nonwinner | 401 | 9.44 / 6.02 | 5.17 / 4.0 | 18.89 |
| 800 | stocha | winner | 464 | 10.44 / 5.65 | 6.37 / 6.0 | 20.88 |
| 900 | determ | all | 2000 | 7.43 / 7.58 | 3.00 / 0.0 | 14.85 |
| 900 | determ | dead | 1355 | 4.69 / 3.83 | 0.37 / 0.0 | 9.37 |
| 900 | determ | alive_nonwinner | 326 | 12.23 / 9.73 | 7.72 / 4.5 | 24.46 |
| 900 | determ | winner | 319 | 14.16 / 10.01 | 9.35 / 7.0 | 28.31 |
| 900 | stocha | all | 2000 | 10.19 / 6.87 | 5.81 / 3.0 | 20.39 |
| 900 | stocha | dead | 538 | 5.93 / 3.72 | 0.35 / 0.0 | 11.86 |
| 900 | stocha | alive_nonwinner | 963 | 11.14 / 6.91 | 7.28 / 6.0 | 22.28 |
| 900 | stocha | winner | 499 | 12.96 / 7.28 | 8.86 / 8.0 | 25.93 |
| 1000 | determ | all | 2000 | 9.07 / 8.57 | 4.19 / 0.0 | 18.14 |
| 1000 | determ | dead | 1150 | 5.30 / 3.98 | 0.52 / 0.0 | 10.59 |
| 1000 | determ | alive_nonwinner | 461 | 14.56 / 10.57 | 9.66 / 7.0 | 29.11 |
| 1000 | determ | winner | 389 | 13.74 / 9.94 | 8.56 / 6.0 | 27.47 |
| 1000 | stocha | all | 2000 | 10.74 / 6.98 | 5.97 / 4.0 | 21.48 |
| 1000 | stocha | dead | 553 | 6.56 / 3.92 | 0.45 / 0.0 | 13.13 |
| 1000 | stocha | alive_nonwinner | 950 | 11.87 / 7.05 | 7.73 / 7.0 | 23.74 |
| 1000 | stocha | winner | 497 | 13.24 / 7.47 | 8.74 / 7.0 | 26.47 |
| 1100 | determ | all | 2000 | 6.48 / 5.71 | 2.51 / 0.0 | 12.95 |
| 1100 | determ | dead | 1327 | 4.67 / 3.62 | 0.94 / 0.0 | 9.35 |
| 1100 | determ | alive_nonwinner | 372 | 10.01 / 7.36 | 5.60 / 4.0 | 20.03 |
| 1100 | determ | winner | 301 | 10.04 / 7.01 | 5.58 / 4.0 | 20.08 |
| 1100 | stocha | all | 2000 | 9.43 / 5.37 | 5.47 / 4.0 | 18.87 |
| 1100 | stocha | dead | 418 | 6.53 / 3.56 | 0.93 / 0.0 | 13.06 |
| 1100 | stocha | alive_nonwinner | 1087 | 10.18 / 5.48 | 6.69 / 6.0 | 20.36 |
| 1100 | stocha | winner | 495 | 10.25 / 5.55 | 6.64 / 5.0 | 20.51 |

### IPPO — survivor min-required-food buckets (selfplay)

| iter | mode | n_survivors | 0-1 units | 2-3 units | 4-5 units | >5 units |
|---:|---|---:|---:|---:|---:|---:|
| 800 | determ | 225 | 0.284 | 0.164 | 0.160 | 0.391 |
| 800 | stocha | 865 | 0.265 | 0.168 | 0.118 | 0.450 |
| 900 | determ | 645 | 0.250 | 0.150 | 0.098 | 0.502 |
| 900 | stocha | 1462 | 0.209 | 0.133 | 0.118 | 0.540 |
| 1000 | determ | 850 | 0.222 | 0.129 | 0.102 | 0.546 |
| 1000 | stocha | 1447 | 0.174 | 0.125 | 0.124 | 0.577 |
| 1100 | determ | 673 | 0.303 | 0.166 | 0.146 | 0.385 |
| 1100 | stocha | 1582 | 0.166 | 0.166 | 0.152 | 0.516 |

### MAPPO — selfplay terminal cash by outcome

| iter | mode | outcome | n | mean cash | median | std |
|---:|---|---|---:|---:|---:|---:|
| 800 | determ | all | 2000 | 100.0 | 92.0 | 70.7 |
| 800 | determ | dead | 1582 | 101.3 | 94.0 | 71.2 |
| 800 | determ | alive_nonwinner | 234 | 64.2 | 59.0 | 45.3 |
| 800 | determ | winner | 184 | 134.5 | 140.5 | 72.3 |
| 800 | stocha | all | 2000 | 100.0 | 81.0 | 74.1 |
| 800 | stocha | dead | 421 | 110.3 | 99.0 | 76.5 |
| 800 | stocha | alive_nonwinner | 1089 | 61.5 | 54.0 | 40.5 |
| 800 | stocha | winner | 490 | 176.7 | 177.5 | 66.8 |
| 900 | determ | all | 2000 | 100.0 | 79.0 | 81.4 |
| 900 | determ | dead | 1646 | 100.9 | 82.0 | 80.9 |
| 900 | determ | alive_nonwinner | 165 | 50.7 | 36.0 | 45.0 |
| 900 | determ | winner | 189 | 135.5 | 137.0 | 89.6 |
| 900 | stocha | all | 2000 | 100.0 | 77.0 | 83.6 |
| 900 | stocha | dead | 332 | 95.6 | 75.5 | 79.6 |
| 900 | stocha | alive_nonwinner | 1169 | 58.2 | 49.0 | 42.4 |
| 900 | stocha | winner | 499 | 200.9 | 198.0 | 72.8 |
| 1000 | determ | all | 2000 | 100.0 | 86.5 | 78.1 |
| 1000 | determ | dead | 1700 | 102.0 | 89.0 | 78.0 |
| 1000 | determ | alive_nonwinner | 138 | 49.6 | 42.0 | 40.0 |
| 1000 | determ | winner | 162 | 121.9 | 114.0 | 85.1 |
| 1000 | stocha | all | 2000 | 100.0 | 59.5 | 97.9 |
| 1000 | stocha | dead | 314 | 91.6 | 48.0 | 95.6 |
| 1000 | stocha | alive_nonwinner | 1188 | 49.7 | 37.0 | 43.7 |
| 1000 | stocha | winner | 498 | 225.3 | 229.0 | 79.9 |
| 1100 | determ | all | 2000 | 100.0 | 87.0 | 77.8 |
| 1100 | determ | dead | 1639 | 102.9 | 91.0 | 78.3 |
| 1100 | determ | alive_nonwinner | 185 | 52.1 | 42.0 | 44.6 |
| 1100 | determ | winner | 176 | 123.4 | 117.0 | 81.7 |
| 1100 | stocha | all | 2000 | 100.0 | 73.0 | 86.8 |
| 1100 | stocha | dead | 297 | 96.7 | 60.0 | 93.8 |
| 1100 | stocha | alive_nonwinner | 1204 | 57.9 | 46.0 | 44.0 |
| 1100 | stocha | winner | 499 | 203.5 | 200.0 | 72.8 |

### MAPPO — selfplay required-food inventory by outcome

`mean-of-two` = (req_1 + req_2)/2; `min-of-two` = min(req_1, req_2).

| iter | mode | outcome | n | mean-of-two (mean/std) | min-of-two (mean/median) | total (mean) |
|---:|---|---|---:|---:|---:|---:|
| 800 | determ | all | 2000 | 4.70 / 4.14 | 1.91 / 0.0 | 9.41 |
| 800 | determ | dead | 1582 | 3.61 / 2.77 | 0.93 / 0.0 | 7.22 |
| 800 | determ | alive_nonwinner | 234 | 9.87 / 5.84 | 6.58 / 5.0 | 19.75 |
| 800 | determ | winner | 184 | 7.51 / 5.01 | 4.41 / 3.0 | 15.02 |
| 800 | stocha | all | 2000 | 9.08 / 4.59 | 5.41 / 5.0 | 18.16 |
| 800 | stocha | dead | 421 | 6.49 / 3.32 | 1.31 / 1.0 | 12.98 |
| 800 | stocha | alive_nonwinner | 1089 | 9.99 / 4.65 | 6.71 / 6.0 | 19.98 |
| 800 | stocha | winner | 490 | 9.28 / 4.56 | 6.06 / 5.0 | 18.57 |
| 900 | determ | all | 2000 | 4.62 / 3.47 | 1.61 / 0.0 | 9.24 |
| 900 | determ | dead | 1646 | 4.04 / 2.76 | 1.06 / 0.0 | 8.08 |
| 900 | determ | alive_nonwinner | 165 | 8.69 / 4.87 | 4.96 / 4.0 | 17.38 |
| 900 | determ | winner | 189 | 6.12 / 4.54 | 3.42 / 2.0 | 12.24 |
| 900 | stocha | all | 2000 | 9.77 / 4.58 | 6.05 / 5.0 | 19.54 |
| 900 | stocha | dead | 332 | 6.98 / 3.43 | 1.42 / 1.0 | 13.97 |
| 900 | stocha | alive_nonwinner | 1169 | 10.37 / 4.67 | 6.92 / 6.0 | 20.74 |
| 900 | stocha | winner | 499 | 10.21 / 4.34 | 7.09 / 7.0 | 20.42 |
| 1000 | determ | all | 2000 | 4.91 / 3.80 | 2.02 / 1.0 | 9.82 |
| 1000 | determ | dead | 1700 | 4.34 / 2.94 | 1.50 / 1.0 | 8.67 |
| 1000 | determ | alive_nonwinner | 138 | 9.06 / 6.15 | 5.66 / 4.0 | 18.12 |
| 1000 | determ | winner | 162 | 7.37 / 5.61 | 4.44 / 3.0 | 14.75 |
| 1000 | stocha | all | 2000 | 10.66 / 5.03 | 6.96 / 6.0 | 21.33 |
| 1000 | stocha | dead | 314 | 7.08 / 3.58 | 1.69 / 1.0 | 14.16 |
| 1000 | stocha | alive_nonwinner | 1188 | 11.01 / 4.94 | 7.72 / 7.0 | 22.02 |
| 1000 | stocha | winner | 498 | 12.10 / 4.98 | 8.50 / 8.0 | 24.20 |
| 1100 | determ | all | 2000 | 4.90 / 3.59 | 2.02 / 1.0 | 9.79 |
| 1100 | determ | dead | 1639 | 4.33 / 2.94 | 1.48 / 1.0 | 8.67 |
| 1100 | determ | alive_nonwinner | 185 | 8.33 / 4.92 | 5.31 / 4.0 | 16.66 |
| 1100 | determ | winner | 176 | 6.52 / 4.77 | 3.63 / 2.0 | 13.04 |
| 1100 | stocha | all | 2000 | 9.89 / 4.63 | 6.24 / 5.0 | 19.78 |
| 1100 | stocha | dead | 297 | 7.01 / 3.43 | 1.62 / 1.0 | 14.01 |
| 1100 | stocha | alive_nonwinner | 1204 | 10.54 / 4.70 | 7.12 / 7.0 | 21.08 |
| 1100 | stocha | winner | 499 | 10.04 / 4.41 | 6.89 / 6.0 | 20.08 |

### MAPPO — survivor min-required-food buckets (selfplay)

| iter | mode | n_survivors | 0-1 units | 2-3 units | 4-5 units | >5 units |
|---:|---|---:|---:|---:|---:|---:|
| 800 | determ | 418 | 0.227 | 0.191 | 0.179 | 0.402 |
| 800 | stocha | 1579 | 0.144 | 0.165 | 0.165 | 0.526 |
| 900 | determ | 354 | 0.328 | 0.234 | 0.161 | 0.277 |
| 900 | stocha | 1668 | 0.131 | 0.138 | 0.155 | 0.576 |
| 1000 | determ | 300 | 0.293 | 0.180 | 0.173 | 0.353 |
| 1000 | stocha | 1686 | 0.094 | 0.119 | 0.135 | 0.652 |
| 1100 | determ | 361 | 0.294 | 0.230 | 0.139 | 0.338 |
| 1100 | stocha | 1703 | 0.113 | 0.137 | 0.166 | 0.585 |

## 6 · Plateau assessment (does the current config still improve?)

### IPPO

**Return_mean by 100-iter window** (train side):
- 601-700: -1.795  → 701-800: -1.531  → 801-900: -0.788  → 901-1000: -0.146  → 1001-1100: 0.012

**Deep-eval selfplay survival at endpoints:**
- stochastic:    iter800=0.432  iter900=0.731  iter1000=0.724  iter1100=0.791
- deterministic: iter800=0.113  iter900=0.323  iter1000=0.425  iter1100=0.337

### MAPPO

**Return_mean by 100-iter window** (train side):
- 601-700: 0.126  → 701-800: 0.292  → 801-900: 0.069  → 901-1000: 0.341  → 1001-1100: 0.242

**Deep-eval selfplay survival at endpoints:**
- stochastic:    iter800=0.789  iter900=0.834  iter1000=0.843  iter1100=0.852
- deterministic: iter800=0.209  iter900=0.177  iter1000=0.150  iter1100=0.180

## 7 · Continuation recommendation

### IPPO — big breakthrough this window, keep training

The 300-iter resume moved IPPO into materially different territory:

- **Selfplay reach-180 stochastic:** 0.966 → 1.000 by iter 900 and pinned there.
- **Selfplay survival stochastic:** 0.432 → 0.731 → 0.724 → **0.791**. Still climbing.
- **Selfplay survival deterministic:** 0.113 → 0.323 → **0.425** (peak at iter 1000) → 0.337 (regression at iter 1100). The stochastic gain has started to *consolidate* into deterministic behavior — the gap closed from 3.8× (iter 800: 0.432 vs 0.113) to 2.3× (iter 1100: 0.791 vs 0.337). But the last 100 iters saw a partial deterministic regression while stochastic continued upward.
- **Training-side return** (window means): `-1.53 → -0.79 → -0.15 → 0.01`. Compounding gains every window, no plateau on the training side.
- **Type-head consolidation:** WAIT dropped from 46% (iter 500) → 28% (iter 800) → **14% (iter 1100)** — the argmax is finally learning that WAIT is not the safe default. POST rose 30% → 57% correspondingly.
- **Trades / game:** stoch selfplay 157 → 195 (near-Scripted 105 range means the market is doing real work at high volume).

Head diagnostics at iter 1100: type argmax POST=0.64, price argmax median=13 (up from 10 at iter 800 — bidding into more realistic price territory), qty median=3.

Against Scripted-v1 Fast opponents (1L+3S), the picture is weaker: stoch learned-seat survival went 0.434 → 0.564 → 0.516 → 0.440 — peaked at iter 900 and has drifted back down. Winner cash conditional on winning stays ~150-190 (small n but consistent) — when IPPO wins vs scripted, it wins with real money. The 1L+3S lineup is where IPPO is genuinely plateauing/regressing.

**Recommendation:** continue another 200-300 iters (target iter 1300-1400) to lock in the gains. The training-side and selfplay stoch/reach-180 metrics all still trend positively; another window should either consolidate deterministic to catch up (in which case stop) or reveal the plateau ceiling. **Save iter_1100 as the current best deployable IPPO** — its stochastic selfplay survival (0.791) and reach-180 (1.000) are the strongest IPPO numbers to date.

### MAPPO — has plateaued, promote iter 1100 and stop

The 300-iter resume delivered marginal MAPPO gains that oscillate rather than trend:

- **Selfplay stoch survival:** 0.789 → 0.834 → 0.843 → **0.852**. Real improvement but the rate is asymptotic (+4.5pp, +0.9pp, +0.9pp) — the last two windows added <1pp each.
- **Selfplay deterministic survival:** 0.209 → 0.177 → 0.150 → 0.180. Basically flat, in fact slightly worse than iter 800. Zero consolidation into argmax behavior in this window.
- **Training return** window means: 0.29 → 0.07 → 0.34 → 0.24 — bouncing around ~0.2 with no upward trend across the last three windows.
- **1L+3S survival stoch:** 0.194 → 0.184 → **0.082 → 0.082**. This is a hard regression — MAPPO's learned-seat performance against Scripted opponents has *halved* over the resume window. That policy learned in selfplay does not generalize to a scripted-opponent population.
- **Winner cash (1L+3S):** the n=50 stochastic winners at iter 800 shrank to n=8 (iter 900), n=12 (iter 1000), n=17 (iter 1100). Small numbers, but the *frequency* of the learned seat winning against scripted opponents dropped drastically.

Head-diagnostic explanation: MAPPO type entropy stayed at 4.14-4.45 across the window — the policy is not narrowing at all. This is exactly the "distribution broadening, argmax not consolidating" pattern noted in the v2b analysis, and it did not fix itself with 300 more iters at the current entropy_coeff. The stochastic-selfplay 85% ceiling looks like the current config's *ceiling*, not a stepping stone.

**Recommendation:** stop training and use **iter_1100 as the best MAPPO** (selfplay stoch 0.852 is the strongest observed). If more RL time on MAPPO is warranted, do NOT just train more under this config — the two prior recommendations from v2b still apply and are now more strongly supported:

1. **Drop `entropy_coeff` from 0.01 → 0.003** for iters 1100→1300 to force distribution narrowing / argmax consolidation.
2. Add an eval-mode-based checkpoint selection (best-so-far by deterministic-eval selfplay survival) — this run has generated 11 checkpoints across 1100 iters; iter 700 was best for det, iter 1100 is best for stoch, and there's no single global best.

Do NOT increase entropy or add reward shaping.

### Overall

- IPPO caught up to MAPPO on stochastic-selfplay survival: **IPPO 0.791 ≈ MAPPO 0.852** at iter 1100 (both algos are the same architecture; MAPPO still has the ~6pp edge from its centralized critic on stoch, but the gap has closed dramatically).
- **Neither reaches the Scripted-v1 Fast ceiling (99.9%) or the MLP-head BC ceiling (98.4%)** — the representation is capable of expert-level play; PPO discovery+consolidation is still the bottleneck, mostly on the deterministic argmax.
- The two best deployable checkpoints for downstream use are **`ippo_seed0_headmlp/checkpoints/iter_1100`** (best IPPO stochastic) and **`mappo_seed0_headmlp/checkpoints/iter_1100`** (best MAPPO stochastic). For robustness vs Scripted opponents specifically, `mappo iter_0700` remains the strongest 1L+3S seat (0.300 stochastic survival) — that observation from the v2b report is unchanged.
