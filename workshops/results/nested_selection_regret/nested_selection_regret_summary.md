# E36: selection regret with a separate evaluation stage (deployment = v4 survival)

Selection on the original 100 games/ckpt (seeds 900000-900099); evaluation of the picked
checkpoint on 500 fresh games (seeds 950000-950499). Means over 8 healthy runs; SE over runs.

| selector | median ckpt | in-sample value (A) | in-sample regret (A, published) | out-of-sample value (B) | nested regret (B, vs oracle-A pick) | runs with regret>0 | regret vs B-oracle (ref.) |
|---|---|---|---|---|---|---|---|
| final checkpoint (default practice) | 1200 | 17.0 | 26.62 | 16.7 +- 4.48 | 24.23 +- 5.3 | 7/8 | 25.28 |
| self-play survival | 1200 | 21.88 | 21.75 | 22.48 +- 4.66 | 18.45 +- 4.97 | 7/8 | 19.5 |
| cross-play cash vs v4 (raw score) | 1200 | 20.0 | 23.62 | 18.98 +- 3.96 | 21.95 +- 4.98 | 7/8 | 23.0 |
| held-out portfolio survival (v2,v3,v5) | 1000 | 25.88 | 17.75 | 25.25 +- 4.31 | 15.68 +- 4.43 | 8/8 | 16.73 |
| oracle on selection games (v4 survival, n=100) | 600 | 43.62 | 0.0 | 40.92 +- 2.86 | 0.0 +- 0.0 | 0/8 | 1.05 |

- in-sample oracle optimism (A value of oracle-A pick minus its B value): **2.7 pts**
- B-oracle (argmax on B, in-sample at n=500) exceeds the oracle-A pick on B by 1.05 pts
- fresh-seed replication of the v4 trajectory: peak ckpt 600 (A: 700); 700 -> 1200: 35.0 -> 16.7 (runs down 6/8, up 2/8); 7/8 runs decline >5 pts from their B peak
- A vs B agreement over 96 (run, ckpt) cells: Pearson 0.898, mean |A-B| 3.95 pts, mean A-B 0.09 pts
- trajectories A: [19.5, 21.4, 24.4, 28.8, 32.1, 32.8, 36.8, 28.6, 26.8, 26.2, 24.4, 17.0]
- trajectories B: [19.4, 21.1, 26.6, 29.1, 32.5, 35.9, 35.0, 25.6, 26.2, 26.8, 22.7, 16.7]
