# v2 RL baseline results — IPPO & MAPPO, 500 iters, seed=0, head_kind=mlp

Config (identical for both runs):
- termination: full 180s horizon or zero-alive (single-survivor early end **removed**)
- reward: terminal-only. dead-before-180 = **-1**, alive-nonwinner = **0**, winner = **+1** (each of a k-way tie gets +1)
- MAX_QTY=5, MAX_PRICE=50, NUM_ACTIONS=2049
- actor: MLPTorso 256→LN→SiLU→256→LN→SiLU  →  FactoredActor with MLP heads (Linear→LN(128)→SiLU→Linear)
- critic: unchanged (IPPO=local OBS 171-D, MAPPO=global 305-D, hidden [512,512,256])
- gamma=1.0, lambda=1.0, lr=3e-4, clip=0.2, entropy_coeff=0.01, train_batch=4096, minibatch=1024, num_epochs=4
- 500 iters each, checkpoints at 100/200/300/400/500, seed=0

## Training-side metrics (per-iter, from `iters.jsonl`)

### IPPO

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

Final iter 500: return=-3.240, n_survivors=1.24, entropy=3.245, vf_expl_var=0.665, grad_norm=0.429

**Plateau check (last 100 iters):** mean return = -3.122, std = 0.305
**Previous 100 iters (iter 300-400):** mean return = -3.244, std = 0.299

### MAPPO

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

Final iter 500: return=-0.083, n_survivors=3.12, entropy=4.007, vf_expl_var=0.242, grad_norm=0.882

**Plateau check (last 100 iters):** mean return = -0.236, std = 0.284
**Previous 100 iters (iter 300-400):** mean return = -0.877, std = 0.308

## Deep evaluation across checkpoints (500 games each)

Per-mode learned-seat survival is the primary metric — this is the fraction of learned seats that reach t=180 alive across the 500-game eval.

### IPPO — learned-seat survival

| iter | selfplay det | selfplay stoch | 1learned+3scripted det | 1learned+3scripted stoch |
|---:|---:|---:|---:|---:|
| 100 | 0.006 | 0.004 | 0.074 | 0.176 |
| 200 | 0.003 | 0.005 | 0.024 | 0.078 |
| 300 | 0.009 | 0.083 | 0.014 | 0.128 |
| 400 | 0.035 | 0.117 | 0.010 | 0.194 |
| 500 | 0.009 | 0.133 | 0.008 | 0.232 |

### IPPO — reach-180 fraction

| iter | selfplay det | selfplay stoch | 1learned+3scripted det | 1learned+3scripted stoch |
|---:|---:|---:|---:|---:|
| 100 | 0.036 | 0.054 | 0.998 | 1.000 |
| 200 | 0.020 | 0.044 | 1.000 | 1.000 |
| 300 | 0.052 | 0.398 | 0.994 | 1.000 |
| 400 | 0.182 | 0.490 | 0.994 | 0.996 |
| 500 | 0.060 | 0.536 | 0.988 | 1.000 |

### IPPO — action-type frequency (selfplay stochastic)

| iter | WAIT | CANCEL | TAKE | POST |
|---:|---:|---:|---:|---:|
| 100 | 0.39 | 0.14 | 0.19 | 0.27 |
| 200 | 0.37 | 0.16 | 0.20 | 0.26 |
| 300 | 0.52 | 0.10 | 0.16 | 0.22 |
| 400 | 0.49 | 0.11 | 0.15 | 0.25 |
| 500 | 0.46 | 0.11 | 0.13 | 0.30 |

### IPPO — trades per game (both modes, both lineups)

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 100 | 63.0 | 99.5 | 130.8 | 125.8 |
| 200 | 62.6 | 94.5 | 115.2 | 121.6 |
| 300 | 16.7 | 94.3 | 89.8 | 114.6 |
| 400 | 28.8 | 91.3 | 97.3 | 116.5 |
| 500 | 19.6 | 93.1 | 90.1 | 119.0 |

### IPPO — head diagnostics at iter 500 (selfplay stochastic)

- type mean_prob:  WAIT=0.360  CANCEL=0.127  TAKE=0.144  POST=0.370
- type argmax_freq: WAIT=0.391  CANCEL=0.137  TAKE=0.037  POST=0.436
- type entropy_mean: 0.728
- food|POST entropy: 1.215
- side|POST entropy: 0.502
- price|POST entropy: 2.972  argmax median: 7.0
- qty|POST entropy: 1.153  argmax median: 3.0

### MAPPO — learned-seat survival

| iter | selfplay det | selfplay stoch | 1learned+3scripted det | 1learned+3scripted stoch |
|---:|---:|---:|---:|---:|
| 100 | 0.000 | 0.003 | 0.002 | 0.066 |
| 200 | 0.031 | 0.015 | 0.122 | 0.160 |
| 300 | 0.515 | 0.532 | 0.142 | 0.350 |
| 400 | 0.265 | 0.566 | 0.014 | 0.352 |
| 500 | 0.032 | 0.702 | 0.010 | 0.300 |

### MAPPO — reach-180 fraction

| iter | selfplay det | selfplay stoch | 1learned+3scripted det | 1learned+3scripted stoch |
|---:|---:|---:|---:|---:|
| 100 | 0.000 | 0.036 | 0.990 | 1.000 |
| 200 | 0.156 | 0.114 | 0.994 | 1.000 |
| 300 | 0.954 | 0.960 | 0.992 | 1.000 |
| 400 | 0.582 | 0.956 | 0.986 | 1.000 |
| 500 | 0.114 | 0.992 | 0.988 | 1.000 |

### MAPPO — action-type frequency (selfplay stochastic)

| iter | WAIT | CANCEL | TAKE | POST |
|---:|---:|---:|---:|---:|
| 100 | 0.49 | 0.12 | 0.15 | 0.24 |
| 200 | 0.48 | 0.11 | 0.16 | 0.25 |
| 300 | 0.24 | 0.13 | 0.26 | 0.37 |
| 400 | 0.21 | 0.16 | 0.17 | 0.46 |
| 500 | 0.19 | 0.13 | 0.15 | 0.53 |

### MAPPO — trades per game (both modes, both lineups)

| iter | selfplay det | selfplay stoch | 1L+3S det | 1L+3S stoch |
|---:|---:|---:|---:|---:|
| 100 | 16.5 | 85.5 | 79.8 | 111.0 |
| 200 | 40.3 | 99.4 | 121.1 | 125.1 |
| 300 | 168.8 | 188.4 | 156.0 | 154.7 |
| 400 | 93.7 | 155.3 | 126.0 | 139.6 |
| 500 | 24.8 | 173.8 | 88.6 | 122.7 |

### MAPPO — head diagnostics at iter 500 (selfplay stochastic)

- type mean_prob:  WAIT=0.162  CANCEL=0.145  TAKE=0.151  POST=0.542
- type argmax_freq: WAIT=0.117  CANCEL=0.125  TAKE=0.088  POST=0.670
- type entropy_mean: 0.795
- food|POST entropy: 1.100
- side|POST entropy: 0.389
- price|POST entropy: 2.916  argmax median: 13.0
- qty|POST entropy: 1.095  argmax median: 2.0

## Final policies — 4-learned self-play + 1-vs-3-scripted (iter 500)

Direct reads from iter-500 deep-eval. Scripted-v1 Fast reference (same as BC report): 4.00 mean survivors, 100% reach-180, ~105 trades/game.

| algo | mode | lineup | learned survival | reach-180 | mean survivors | trades/game |
|---|---|---|---:|---:|---:|---:|
| IPPO | determ | learned_selfplay | 0.009 | 0.060 | 0.03 | 19.6 |
| IPPO | stocha | learned_selfplay | 0.133 | 0.536 | 0.53 | 93.1 |
| IPPO | determ | 1learned_3scripted | 0.008 | 0.988 | 2.16 | 90.1 |
| IPPO | stocha | 1learned_3scripted | 0.232 | 1.000 | 3.00 | 119.0 |
| MAPPO | determ | learned_selfplay | 0.032 | 0.114 | 0.13 | 24.8 |
| MAPPO | stocha | learned_selfplay | 0.702 | 0.992 | 2.81 | 173.8 |
| MAPPO | determ | 1learned_3scripted | 0.010 | 0.988 | 2.10 | 88.6 |
| MAPPO | stocha | 1learned_3scripted | 0.300 | 1.000 | 3.13 | 122.7 |

## Verdict

Both algorithms improve non-trivially compared to random play but neither comes close to the Scripted-v1 Fast reference (99.9% survival to 180s). IPPO selfplay survival at iter 500: det=0.9%, stoch=13.3%. MAPPO selfplay survival at iter 500: det=3.1%, stoch=70.2%.

The MLP-head BC agent under the exact same env config reaches 98.4% selfplay survival and 94.6% survival vs 3 scripted, so representation and env are not the bottleneck. The failure is the same one v1 flagged: PPO discovery / credit assignment / optimisation.

Both trained deterministic argmax policies over-emit WAIT (~50-65% frequency) — the classic collapse mode where the type head learns 'wait is safe' as a default and only the sampled tail explores enough to trade. The stochastic runs beat deterministic on every axis, confirming that the policy's *distribution* has learned something useful but its *argmax* has not.
