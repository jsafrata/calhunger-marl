# v2 RL resume results — IPPO & MAPPO iter 500 → iter 800

Resumed from the exact `iter_0500` RLlib checkpoint of each algo (full restore: actor+critic weights, optimizer state, iteration/timestep counters, RNG state where supported). Config unchanged from the 0-500 run:

- terminal-only reward: dead=-1, alive-nonwinner=0, winner=+1 (each tied winner +1)
- full 180-second horizon unless zero-alive
- MAX_QTY=5, MAX_PRICE=50, MLP actor heads, unchanged critic
- gamma=1.0, lambda=1.0, entropy_coeff=0.01, lr=3e-4, seed=0
- train_batch=4096, minibatch=1024, num_epochs=4

New checkpoints at iterations **600, 700, 800**. Deep evals at each on 500 games per lineup per mode. Existing 100-500 checkpoints and eval JSONs untouched.

## 1 · Resume confirmation

- **IPPO**: iter counter reached **800**, iter 501 present in `iters.jsonl` = **True**, checkpoints on disk = `iter_0100, iter_0200, iter_0300, iter_0400, iter_0500, iter_0600, iter_0700, iter_0800`.
- **MAPPO**: iter counter reached **800**, iter 501 present in `iters.jsonl` = **True**, checkpoints on disk = `iter_0100, iter_0200, iter_0300, iter_0400, iter_0500, iter_0600, iter_0700, iter_0800`.

Both runs resumed cleanly at algo iteration 500 (per pipeline log: `resumed at algo iteration 500` at start of each resume run) and continued the existing `iters.jsonl` with absolute iter counters 501..800.

## 2 · Training curves through iter 800

### IPPO — every 50 iters

| iter | ep_return | n_survivors | entropy | vf_expl_var | grad_norm | ep_len |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | -3.933 | 1.30 | 4.333 | 0.030 | 0.293 | 136 |
| 100 | -3.929 | 1.25 | 3.589 | 0.106 | 0.365 | 139 |
| 150 | -3.929 | 1.15 | 2.844 | 0.010 | 0.350 | 146 |
| 200 | -4.000 | 1.36 | 3.064 | 0.003 | 0.262 | 142 |
| 250 | -4.000 | 1.15 | 2.788 | 0.006 | 0.314 | 148 |
| 300 | -3.615 | 1.15 | 2.479 | 0.524 | 0.479 | 157 |
| 350 | -3.763 | 1.12 | 2.536 | 0.502 | 0.346 | 154 |
| 400 | -2.689 | 1.08 | 2.953 | 0.839 | 0.354 | 168 |
| 450 | -2.667 | 1.04 | 2.924 | 0.684 | 0.732 | 164 |
| 500 | -3.240 | 1.24 | 3.245 | 0.665 | 0.429 | 158 |
| 550 | -2.958 | 1.29 | 3.386 | 0.738 | 0.416 | 164 |
| 600 | -2.292 | 1.67 | 3.043 | 0.838 | 0.494 | 169 |
| 650 | -2.083 | 1.42 | 3.379 | 0.611 | 0.497 | 166 |
| 700 | -1.144 | 2.26 | 3.505 | 0.394 | 0.608 | 173 |
| 750 | -1.375 | 1.88 | 3.628 | 0.506 | 0.748 | 167 |
| 800 | -1.614 | 2.05 | 3.277 | 0.517 | 0.725 | 174 |

#### IPPO — window means (return_mean)

| window | mean return | std | mean n_survivors | mean entropy | mean vf_expl_var |
|---|---:|---:|---:|---:|---:|
| 401-500 | -3.122 | 0.305 | 1.23 | 2.967 | 0.662 |
| 501-600 | -2.512 | 0.357 | 1.48 | 3.250 | 0.692 |
| 601-700 | -1.795 | 0.286 | 1.71 | 3.217 | 0.661 |
| 701-800 | -1.531 | 0.243 | 1.93 | 3.497 | 0.587 |

### MAPPO — every 50 iters

| iter | ep_return | n_survivors | entropy | vf_expl_var | grad_norm | ep_len |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | -4.000 | 1.20 | 4.904 | 0.005 | 0.232 | 135 |
| 100 | -3.929 | 1.21 | 3.512 | 0.004 | 0.254 | 140 |
| 150 | -4.000 | 1.25 | 3.326 | -0.001 | 0.265 | 144 |
| 200 | -4.000 | 1.18 | 3.174 | -0.000 | 0.265 | 141 |
| 250 | -2.693 | 1.52 | 2.747 | 0.461 | 0.523 | 168 |
| 300 | -1.375 | 2.29 | 3.591 | 0.505 | 0.648 | 165 |
| 350 | -0.655 | 2.47 | 4.000 | 0.341 | 0.794 | 169 |
| 400 | -1.045 | 2.41 | 3.805 | 0.520 | 0.732 | 179 |
| 450 | 0.102 | 3.36 | 4.119 | 0.179 | 0.883 | 172 |
| 500 | -0.083 | 3.12 | 4.007 | 0.242 | 0.882 | 167 |
| 550 | 0.299 | 3.44 | 4.293 | 0.281 | 0.952 | 173 |
| 600 | 0.174 | 3.39 | 3.982 | 0.345 | 0.742 | 171 |
| 650 | -0.364 | 2.95 | 4.353 | 0.310 | 0.920 | 174 |
| 700 | 0.545 | 3.59 | 4.275 | 0.189 | 0.694 | 174 |
| 750 | 0.500 | 3.55 | 4.147 | 0.296 | 0.747 | 175 |
| 800 | 0.136 | 3.32 | 3.960 | 0.435 | 0.840 | 174 |

#### MAPPO — window means (return_mean)

| window | mean return | std | mean n_survivors | mean entropy | mean vf_expl_var |
|---|---:|---:|---:|---:|---:|
| 401-500 | -0.236 | 0.284 | 3.05 | 3.981 | 0.318 |
| 501-600 | 0.000 | 0.291 | 3.26 | 4.159 | 0.300 |
| 601-700 | 0.126 | 0.335 | 3.36 | 4.205 | 0.285 |
| 701-800 | 0.292 | 0.272 | 3.49 | 4.136 | 0.296 |

## 3 · Deep evaluation across checkpoints (500 games each)

Learned-seat survival = fraction of the learned seat(s) that reach t=180 alive.

### IPPO — learned-seat survival

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 500 | 0.009 | 0.133 | 0.008 | 0.232 |
| 600 | 0.237 | 0.300 | 0.100 | 0.350 |
| 700 | 0.211 | 0.364 | 0.112 | 0.456 |
| 800 | 0.113 | 0.432 | 0.074 | 0.434 |

### IPPO — reach-180 fraction

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 500 | 0.060 | 0.536 | 0.988 | 1.000 |
| 600 | 0.710 | 0.820 | 0.986 | 0.998 |
| 700 | 0.692 | 0.884 | 0.982 | 1.000 |
| 800 | 0.422 | 0.966 | 0.962 | 1.000 |

### IPPO — trades per game

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 500 | 19.6 | 93.1 | 90.1 | 119.0 |
| 600 | 90.7 | 133.0 | 110.3 | 130.4 |
| 700 | 112.9 | 185.2 | 117.8 | 143.9 |
| 800 | 57.4 | 157.2 | 99.7 | 131.6 |

### IPPO — selfplay stochastic action-type frequency

| iter | WAIT | CANCEL | TAKE | POST |
|---:|---:|---:|---:|---:|
| 500 | 0.46 | 0.11 | 0.13 | 0.30 |
| 600 | 0.39 | 0.13 | 0.20 | 0.29 |
| 700 | 0.28 | 0.10 | 0.23 | 0.39 |
| 800 | 0.28 | 0.12 | 0.17 | 0.42 |

### IPPO — head diagnostics at iter 800 (selfplay stochastic)

- type mean_prob:  WAIT=0.193  CANCEL=0.168  TAKE=0.207  POST=0.432
- type argmax_freq: WAIT=0.219  CANCEL=0.156  TAKE=0.170  POST=0.455
- type entropy_mean: 0.635
- price|POST entropy: 2.932  argmax median: 10.0
- qty|POST entropy: 0.935  argmax median: 3.0

### MAPPO — learned-seat survival

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 500 | 0.032 | 0.702 | 0.010 | 0.300 |
| 600 | 0.255 | 0.809 | 0.048 | 0.248 |
| 700 | 0.341 | 0.896 | 0.010 | 0.240 |
| 800 | 0.209 | 0.789 | 0.004 | 0.194 |

### MAPPO — reach-180 fraction

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 500 | 0.114 | 0.992 | 0.988 | 1.000 |
| 600 | 0.548 | 1.000 | 0.998 | 0.998 |
| 700 | 0.686 | 1.000 | 0.996 | 1.000 |
| 800 | 0.442 | 0.986 | 0.990 | 1.000 |

### MAPPO — trades per game

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 500 | 24.8 | 173.8 | 88.6 | 122.7 |
| 600 | 65.1 | 174.8 | 100.8 | 127.6 |
| 700 | 86.6 | 207.3 | 96.5 | 130.1 |
| 800 | 65.4 | 188.1 | 88.6 | 120.2 |

### MAPPO — selfplay stochastic action-type frequency

| iter | WAIT | CANCEL | TAKE | POST |
|---:|---:|---:|---:|---:|
| 500 | 0.19 | 0.13 | 0.15 | 0.53 |
| 600 | 0.17 | 0.15 | 0.16 | 0.52 |
| 700 | 0.11 | 0.14 | 0.20 | 0.55 |
| 800 | 0.11 | 0.16 | 0.19 | 0.54 |

### MAPPO — head diagnostics at iter 800 (selfplay stochastic)

- type mean_prob:  WAIT=0.153  CANCEL=0.146  TAKE=0.217  POST=0.484
- type argmax_freq: WAIT=0.125  CANCEL=0.133  TAKE=0.184  POST=0.559
- type entropy_mean: 0.720
- price|POST entropy: 2.702  argmax median: 13.0
- qty|POST entropy: 1.088  argmax median: 2.0

## 4 · Terminal cash + required-food inventory (new v2b diagnostics)

Per learned seat, aggregated across the 500 games of each eval. Required foods are the two the seat consumes (i.e. the two it does *not* produce). For dead agents the values are read at the point of death; for alive agents they are the post-final-consumption terminal state.

### IPPO — terminal cash by outcome (selfplay)

| iter | mode | outcome | n | mean cash | median | std |
|---:|---|---|---:|---:|---:|---:|
| 600 | determ | all | 2000 | 100.0 | 73.5 | 90.8 |
| 600 | determ | dead | 1526 | 88.3 | 68.0 | 79.5 |
| 600 | determ | alive_nonwinner | 145 | 51.3 | 34.0 | 47.3 |
| 600 | determ | winner | 329 | 175.8 | 196.0 | 111.2 |
| 600 | stocha | all | 2000 | 100.0 | 75.0 | 88.1 |
| 600 | stocha | dead | 1399 | 90.2 | 65.0 | 80.8 |
| 600 | stocha | alive_nonwinner | 228 | 47.7 | 36.0 | 39.7 |
| 600 | stocha | winner | 373 | 168.7 | 173.0 | 96.8 |
| 700 | determ | all | 2000 | 100.0 | 80.0 | 80.7 |
| 700 | determ | dead | 1578 | 94.8 | 75.0 | 77.0 |
| 700 | determ | alive_nonwinner | 116 | 55.0 | 43.5 | 43.3 |
| 700 | determ | winner | 306 | 143.9 | 134.5 | 91.9 |
| 700 | stocha | all | 2000 | 100.0 | 82.0 | 76.2 |
| 700 | stocha | dead | 1271 | 89.7 | 72.0 | 69.7 |
| 700 | stocha | alive_nonwinner | 319 | 60.5 | 53.0 | 41.5 |
| 700 | stocha | winner | 410 | 162.5 | 161.0 | 80.2 |
| 800 | determ | all | 2000 | 100.0 | 100.0 | 61.1 |
| 800 | determ | dead | 1775 | 101.1 | 100.0 | 60.7 |
| 800 | determ | alive_nonwinner | 49 | 54.7 | 49.0 | 38.3 |
| 800 | determ | winner | 176 | 101.2 | 95.0 | 65.8 |
| 800 | stocha | all | 2000 | 100.0 | 89.0 | 67.9 |
| 800 | stocha | dead | 1135 | 93.9 | 80.0 | 68.1 |
| 800 | stocha | alive_nonwinner | 401 | 67.4 | 63.0 | 37.5 |
| 800 | stocha | winner | 464 | 143.2 | 143.0 | 66.6 |

### IPPO — required-food inventory by outcome (selfplay)

Fields: `mean` = (req_1 + req_2) / 2; `min` = min(req_1, req_2); `total` = req_1 + req_2. Median values are per-metric medians across seat-games in that bucket.

| iter | mode | outcome | n | mean-of-two (mean/std) | min-of-two (mean/median) | total (mean) |
|---:|---|---|---:|---:|---:|---:|
| 600 | determ | all | 2000 | 7.92 / 9.64 | 3.28 / 0.0 | 15.83 |
| 600 | determ | dead | 1526 | 4.40 / 4.29 | 0.08 / 0.0 | 8.81 |
| 600 | determ | alive_nonwinner | 145 | 14.59 / 13.16 | 9.74 / 5.0 | 29.17 |
| 600 | determ | winner | 329 | 21.26 / 12.17 | 15.30 / 14.0 | 42.52 |
| 600 | stocha | all | 2000 | 6.78 / 6.34 | 2.28 / 0.0 | 13.57 |
| 600 | stocha | dead | 1399 | 4.52 / 3.59 | 0.17 / 0.0 | 9.05 |
| 600 | stocha | alive_nonwinner | 228 | 8.39 / 6.90 | 4.51 / 2.0 | 16.78 |
| 600 | stocha | winner | 373 | 14.28 / 7.83 | 8.85 / 8.0 | 28.57 |
| 700 | determ | all | 2000 | 7.85 / 9.24 | 2.86 / 0.0 | 15.70 |
| 700 | determ | dead | 1578 | 5.00 / 5.00 | 0.33 / 0.0 | 10.01 |
| 700 | determ | alive_nonwinner | 116 | 13.40 / 11.12 | 7.97 / 5.0 | 26.79 |
| 700 | determ | winner | 306 | 20.42 / 13.09 | 13.95 / 11.0 | 40.84 |
| 700 | stocha | all | 2000 | 8.54 / 7.00 | 3.06 / 0.0 | 17.08 |
| 700 | stocha | dead | 1271 | 6.04 / 4.80 | 0.32 / 0.0 | 12.08 |
| 700 | stocha | alive_nonwinner | 319 | 10.34 / 7.24 | 5.81 / 4.0 | 20.67 |
| 700 | stocha | winner | 410 | 14.89 / 8.04 | 9.40 / 8.0 | 29.78 |
| 800 | determ | all | 2000 | 4.77 / 5.47 | 0.91 / 0.0 | 9.54 |
| 800 | determ | dead | 1775 | 3.85 / 4.32 | 0.24 / 0.0 | 7.70 |
| 800 | determ | alive_nonwinner | 49 | 12.23 / 8.73 | 7.00 / 4.0 | 24.47 |
| 800 | determ | winner | 176 | 11.97 / 7.45 | 5.91 / 4.0 | 23.93 |
| 800 | stocha | all | 2000 | 7.25 / 5.43 | 2.65 / 0.0 | 14.50 |
| 800 | stocha | dead | 1135 | 5.17 / 3.99 | 0.24 / 0.0 | 10.35 |
| 800 | stocha | alive_nonwinner | 401 | 9.44 / 6.02 | 5.17 / 4.0 | 18.89 |
| 800 | stocha | winner | 464 | 10.44 / 5.65 | 6.37 / 6.0 | 20.88 |

### IPPO — survivor min-required-food buckets (selfplay)

Fraction of surviving learned seats whose min required-food inventory falls in each bucket.

| iter | mode | n_survivors | 0-1 units | 2-3 units | 4-5 units | >5 units |
|---:|---|---:|---:|---:|---:|---:|
| 600 | determ | 474 | 0.190 | 0.076 | 0.072 | 0.662 |
| 600 | stocha | 601 | 0.265 | 0.135 | 0.103 | 0.498 |
| 700 | determ | 422 | 0.206 | 0.085 | 0.100 | 0.609 |
| 700 | stocha | 729 | 0.240 | 0.126 | 0.104 | 0.529 |
| 800 | determ | 225 | 0.284 | 0.164 | 0.160 | 0.391 |
| 800 | stocha | 865 | 0.265 | 0.168 | 0.118 | 0.450 |

### IPPO — terminal cash + required-food, 1learned + 3scripted at iter 800

Learned seat only.

**determ** — outcomes: dead=463, alive_nonwinner=35, winner=2

- terminal cash (all): mean=75.0  median=79.5  std=46.3
- dead: n=463  req_mean=3.77±4.75  req_min mean=0.29  median=0.0
- alive_nonwinner: n=35  req_mean=10.53±6.17  req_min mean=3.83  median=3.0
- winner: n=2  req_mean=11.00±1.50  req_min mean=7.00  median=7.0

**stocha** — outcomes: dead=283, alive_nonwinner=211, winner=6

- terminal cash (all): mean=40.2  median=30.0  std=32.8
- dead: n=283  req_mean=4.83±3.44  req_min mean=1.00  median=0.0
- alive_nonwinner: n=211  req_mean=5.92±4.45  req_min mean=2.72  median=2.0
- winner: n=6  req_mean=3.42±1.48  req_min mean=2.17  median=2.0

### MAPPO — terminal cash by outcome (selfplay)

| iter | mode | outcome | n | mean cash | median | std |
|---:|---|---|---:|---:|---:|---:|
| 600 | determ | all | 2000 | 100.0 | 90.5 | 75.6 |
| 600 | determ | dead | 1491 | 101.4 | 93.0 | 74.9 |
| 600 | determ | alive_nonwinner | 282 | 57.0 | 49.0 | 43.6 |
| 600 | determ | winner | 227 | 144.1 | 139.0 | 82.9 |
| 600 | stocha | all | 2000 | 100.0 | 81.0 | 78.1 |
| 600 | stocha | dead | 381 | 106.9 | 88.0 | 78.2 |
| 600 | stocha | alive_nonwinner | 1121 | 59.5 | 54.0 | 40.4 |
| 600 | stocha | winner | 498 | 185.9 | 182.5 | 70.9 |
| 700 | determ | all | 2000 | 100.0 | 88.0 | 75.3 |
| 700 | determ | dead | 1319 | 104.3 | 94.0 | 74.2 |
| 700 | determ | alive_nonwinner | 381 | 54.4 | 43.0 | 44.9 |
| 700 | determ | winner | 300 | 139.2 | 137.5 | 82.3 |
| 700 | stocha | all | 2000 | 100.0 | 85.0 | 74.3 |
| 700 | stocha | dead | 208 | 103.0 | 89.0 | 71.7 |
| 700 | stocha | alive_nonwinner | 1292 | 64.0 | 60.0 | 41.7 |
| 700 | stocha | winner | 500 | 191.9 | 183.0 | 60.3 |
| 800 | determ | all | 2000 | 100.0 | 92.0 | 70.7 |
| 800 | determ | dead | 1582 | 101.3 | 94.0 | 71.2 |
| 800 | determ | alive_nonwinner | 234 | 64.2 | 59.0 | 45.3 |
| 800 | determ | winner | 184 | 134.5 | 140.5 | 72.3 |
| 800 | stocha | all | 2000 | 100.0 | 81.0 | 74.1 |
| 800 | stocha | dead | 421 | 110.3 | 99.0 | 76.5 |
| 800 | stocha | alive_nonwinner | 1089 | 61.5 | 54.0 | 40.5 |
| 800 | stocha | winner | 490 | 176.7 | 177.5 | 66.8 |

### MAPPO — required-food inventory by outcome (selfplay)

Fields: `mean` = (req_1 + req_2) / 2; `min` = min(req_1, req_2); `total` = req_1 + req_2. Median values are per-metric medians across seat-games in that bucket.

| iter | mode | outcome | n | mean-of-two (mean/std) | min-of-two (mean/median) | total (mean) |
|---:|---|---|---:|---:|---:|---:|
| 600 | determ | all | 2000 | 5.30 / 4.88 | 1.95 / 0.0 | 10.59 |
| 600 | determ | dead | 1491 | 3.75 / 3.14 | 0.53 / 0.0 | 7.51 |
| 600 | determ | alive_nonwinner | 282 | 10.41 / 5.94 | 6.68 / 6.0 | 20.82 |
| 600 | determ | winner | 227 | 9.07 / 6.24 | 5.37 / 4.0 | 18.15 |
| 600 | stocha | all | 2000 | 9.30 / 4.63 | 5.37 / 4.0 | 18.61 |
| 600 | stocha | dead | 381 | 6.89 / 3.41 | 1.04 / 0.0 | 13.78 |
| 600 | stocha | alive_nonwinner | 1121 | 9.96 / 4.70 | 6.38 / 6.0 | 19.92 |
| 600 | stocha | winner | 498 | 9.68 / 4.68 | 6.39 / 5.0 | 19.36 |
| 700 | determ | all | 2000 | 6.19 / 5.22 | 2.64 / 0.0 | 12.38 |
| 700 | determ | dead | 1319 | 4.26 / 3.27 | 0.78 / 0.0 | 8.52 |
| 700 | determ | alive_nonwinner | 381 | 10.77 / 5.95 | 6.96 / 6.0 | 21.54 |
| 700 | determ | winner | 300 | 8.85 / 6.33 | 5.33 / 4.0 | 17.69 |
| 700 | stocha | all | 2000 | 11.19 / 5.13 | 7.47 / 7.0 | 22.38 |
| 700 | stocha | dead | 208 | 7.13 / 3.43 | 1.34 / 1.0 | 14.26 |
| 700 | stocha | alive_nonwinner | 1292 | 11.75 / 5.16 | 8.20 / 8.0 | 23.51 |
| 700 | stocha | winner | 500 | 11.43 / 4.87 | 8.14 / 8.0 | 22.86 |
| 800 | determ | all | 2000 | 4.70 / 4.14 | 1.91 / 0.0 | 9.41 |
| 800 | determ | dead | 1582 | 3.61 / 2.77 | 0.93 / 0.0 | 7.22 |
| 800 | determ | alive_nonwinner | 234 | 9.87 / 5.84 | 6.58 / 5.0 | 19.75 |
| 800 | determ | winner | 184 | 7.51 / 5.01 | 4.41 / 3.0 | 15.02 |
| 800 | stocha | all | 2000 | 9.08 / 4.59 | 5.41 / 5.0 | 18.16 |
| 800 | stocha | dead | 421 | 6.49 / 3.32 | 1.31 / 1.0 | 12.98 |
| 800 | stocha | alive_nonwinner | 1089 | 9.99 / 4.65 | 6.71 / 6.0 | 19.98 |
| 800 | stocha | winner | 490 | 9.28 / 4.56 | 6.06 / 5.0 | 18.57 |

### MAPPO — survivor min-required-food buckets (selfplay)

Fraction of surviving learned seats whose min required-food inventory falls in each bucket.

| iter | mode | n_survivors | 0-1 units | 2-3 units | 4-5 units | >5 units |
|---:|---|---:|---:|---:|---:|---:|
| 600 | determ | 509 | 0.234 | 0.165 | 0.155 | 0.446 |
| 600 | stocha | 1619 | 0.163 | 0.161 | 0.174 | 0.502 |
| 700 | determ | 681 | 0.235 | 0.154 | 0.162 | 0.449 |
| 700 | stocha | 1792 | 0.091 | 0.105 | 0.143 | 0.661 |
| 800 | determ | 418 | 0.227 | 0.191 | 0.179 | 0.402 |
| 800 | stocha | 1579 | 0.144 | 0.165 | 0.165 | 0.526 |

### MAPPO — terminal cash + required-food, 1learned + 3scripted at iter 800

Learned seat only.

**determ** — outcomes: dead=498, alive_nonwinner=0, winner=2

- terminal cash (all): mean=150.0  median=144.5  std=67.7
- dead: n=498  req_mean=2.55±1.98  req_min mean=0.52  median=0.0
- winner: n=2  req_mean=3.25±1.75  req_min mean=1.50  median=1.5

**stocha** — outcomes: dead=403, alive_nonwinner=47, winner=50

- terminal cash (all): mean=154.6  median=152.0  std=67.0
- dead: n=403  req_mean=3.66±2.25  req_min mean=1.01  median=0.0
- alive_nonwinner: n=47  req_mean=3.93±2.15  req_min mean=2.04  median=2.0
- winner: n=50  req_mean=2.53±2.18  req_min mean=0.82  median=0.0

## 5 · Plateau assessment

### IPPO

**Return_mean by window** (train side):
- 400-500: -3.122  → 500-600: -2.512  → 600-700: -1.795  → 700-800: -1.531

**Env-runner n_survivors by window** (train side):
- 400-500: 1.23  → 500-600: 1.48  → 600-700: 1.71  → 700-800: 1.93

**Deep-eval selfplay survival at endpoints of each window:**
- deterministic: iter500=0.009  iter600=0.237  iter700=0.211  iter800=0.113
- stochastic:    iter500=0.133  iter600=0.300  iter700=0.364  iter800=0.432

### MAPPO

**Return_mean by window** (train side):
- 400-500: -0.236  → 500-600: 0.000  → 600-700: 0.126  → 700-800: 0.292

**Env-runner n_survivors by window** (train side):
- 400-500: 3.05  → 500-600: 3.26  → 600-700: 3.36  → 700-800: 3.49

**Deep-eval selfplay survival at endpoints of each window:**
- deterministic: iter500=0.032  iter600=0.255  iter700=0.341  iter800=0.209
- stochastic:    iter500=0.702  iter600=0.809  iter700=0.896  iter800=0.789

## 6 · Continuation recommendation

### IPPO — keep training

Both training-side metrics and closed-loop stochastic evals are still monotonically
climbing across all three new windows. The training-side return has walked
`-3.12 → -2.51 → -1.80 → -1.53` across the four 100-iter windows ending at 800 —
a compounding **50% error-reduction from iter 500 to iter 800**, no sign of a knee.
Env-runner n_survivors is on the same trajectory (1.23 → 1.48 → 1.71 → 1.93).
Stochastic deep-eval selfplay survival tracks it: `0.133 → 0.300 → 0.364 → 0.432`,
tripling from iter 500 to iter 800. Selfplay reach-180 (stoch) is 0.966 at iter 800
— the *shape* of the game is coming together; what remains is the *outcome* inside it.

The catch is that stochastic gains have NOT consolidated into deterministic behavior.
Deterministic selfplay survival went 0.009 → 0.237 → 0.211 → 0.113 — it climbed
into iter 600, plateaued at 700, then regressed by iter 800. The argmax type-head
distribution is smoothing (WAIT 46% → 28%, POST 30% → 42%) but is still not stable
around the strategy the stochastic tail has found. Terminal inventory diagnostics
confirm the same story: iter 800 stochastic winners hold 6.4 units min-required vs
iter 800 deterministic winners at 5.9 — stochastic is finding a better buffer.
**Recommendation:** continue another 200-300 iters (target iter 1000-1100) with the
same config. Stop earlier if the 800-900 window shows the return_mean gains vanish.

### MAPPO — plateauing, favor consolidation over more raw iterations

Training-side return continues to climb (-0.24 → 0.00 → 0.13 → 0.29) but the
*rate* has flattened: each successive 100-iter window adds ~0.15 units of return,
same as the window before. Env-runner n_survivors moved from 3.05 → 3.49 across
the same four windows — real gain, but modest given MAPPO already sat at ~3 at
iter 500. Stochastic selfplay survival went 0.702 → 0.809 → 0.896 → 0.789 —
peaked at iter 700 and regressed. Deterministic selfplay survival showed the
same shape (0.032 → 0.255 → 0.341 → 0.209), also peaking at iter 700.

Two independent signals now say iter 700 was the sweet spot: the 1learned+3scripted
learned-seat survival also degrades 0.300 (iter 500) → 0.240 (iter 700) → 0.194
(iter 800) in stochastic, and its winner cash-share is essentially stable at
~180 across iters 500-800 (no meaningful trend). The stochastic gains from
500-700 did NOT consolidate into deterministic behavior — iter 800 deterministic
regressed by ~13 percentage points vs iter 700. The type-head entropy climbed
2.75 → 4.28 across the full 800-iter run — MAPPO's distribution is broadening,
not narrowing, so waiting for consolidation without an intervention is unlikely
to pay.

Terminal inventory tables tell the strategic story cleanly: at iter 700 stochastic,
survivors hold min-req 8.2 units (median 8), winners hold 8.1 (median 8) — a real
strategic buffer. At iter 800 both dropped: survivors 6.7 min-req, winners 6.1.
The policy learned a viable buffer strategy at iter 700 and is now drifting off it.

**Recommendation:** *Do not blindly train more*. Two productive next steps in
priority order:
1. **Roll back the "best" MAPPO checkpoint** — use `iter_0700` as the deployment
   MAPPO baseline; it is strictly better than iter_0800 on every closed-loop
   metric tracked here.
2. If more RL time is warranted, change ONE thing to force consolidation before
   burning more compute. The cheapest experiment is to **lower `entropy_coeff`
   from 0.01 to 0.003 for iters 800→1000**; this directly addresses the
   distribution-broadening pattern. Second-cheapest is to add an evaluation-mode
   rollout in RLlib and select the best checkpoint by deterministic survival
   rather than by training-side return. Do not add reward shaping — the terminal-
   only signal has already produced ~90% stochastic survival; the residual gap is
   optimization, not credit.

### Common threads

- Neither algo has yet closed the gap to Scripted-v1 Fast (99.9% selfplay
  survival, 100% reach-180, 105 trades/game) or to the MLP-head BC agent
  (98.4% selfplay survival) — the representation ceiling is not the bottleneck.
- Both algos' stochastic policies now materially outperform their deterministic
  argmax — this is the persistent v1→v2 issue. The `entropy_coeff` schedule and
  early stopping on deterministic-eval survival are the two most likely levers.
- The "hidden-imbalance" test (survivor min-required buckets) confirms that
  many of MAPPO's stochastic survivors do hold >5 units of their scarcest
  required food (frac_gt_5 = 0.66 at iter 700 stoch, 0.53 at iter 800 stoch) —
  they are actually managing food inventory, not just barely surviving on
  fumes. IPPO stoch survivors hold >5 units in only 45-50% of games —
  still relying more heavily on last-minute trades.
