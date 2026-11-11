# E32: declared-target audit of the ckpt 700 -> 1200 transition (8 healthy runs)

E[r] = P(win) + P(survive) - 1 per cell. A run counts as up/down if its delta
exceeds +-2.0 pts (rates) or +-0.02 (returns). Verdict needs a majority.

| target | 700 | 1200 | delta | up/down (of 8) | verdict | peak ckpt (mean) |
|---|---|---|---|---|---|---|
| survival vs v4 (deployment target of Sec. 7) | 36.8 | 17.0 | -19.8 | 1/7 | DECLINE | 700 |
| win rate vs v4 | 9.4 | 6.8 | -2.6 | 2/5 | DECLINE | 800 |
| expected return vs v4 | -0.539 | -0.762 | -0.224 | 1/6 | DECLINE | 700 |
| survival vs v2 | 24.8 | 26.4 | 1.6 | 3/3 | mixed / flat | 1100 |
| survival vs v3 | 73.0 | 94.1 | 21.1 | 7/1 | improvement | 1100 |
| survival vs v5 | 13.2 | 16.2 | 3.0 | 3/1 | mixed / flat | 1000 |
| expected return vs v2 | -0.745 | -0.718 | 0.027 | 3/4 | mixed / flat | 1000 |
| expected return vs v3 | -0.191 | 0.264 | 0.455 | 8/0 | improvement | 1100 |
| expected return vs v5 | -0.823 | -0.711 | 0.111 | 6/1 | improvement | 1000 |
| mixture survival (uniform v2..v5) | 36.9 | 38.4 | 1.5 | 3/5 | DECLINE | 1100 |
| mixture expected return (uniform v2..v5) | -0.574 | -0.482 | 0.092 | 6/2 | improvement | 1100 |
| self-play survival | 73.6 | 90.3 | 16.8 | 7/0 | improvement | 1200 |
| self-play expected return (P(win)=1/4 by symmetry) | -0.014 | 0.153 | 0.168 | 7/0 | improvement | 1200 |

## Mean trajectories (ckpt 100..1200)

- survival vs v4 (deployment target of Sec. 7): [19.5, 21.4, 24.4, 28.8, 32.1, 32.8, 36.8, 28.6, 26.8, 26.2, 24.4, 17.0]
- win rate vs v4: [0.0, 0.0, 0.5, 0.6, 1.1, 4.4, 9.4, 10.1, 9.5, 9.6, 9.5, 6.8]
- expected return vs v4: [-0.805, -0.786, -0.751, -0.706, -0.667, -0.629, -0.539, -0.613, -0.637, -0.641, -0.661, -0.762]
- survival vs v2: [0.8, 1.4, 4.4, 4.5, 9.4, 19.5, 24.8, 19.9, 28.2, 29.1, 30.2, 26.4]
- survival vs v3: [10.2, 13.9, 24.8, 35.8, 45.8, 57.4, 73.0, 77.5, 86.9, 92.2, 96.1, 94.1]
- survival vs v5: [0.6, 0.5, 2.4, 3.9, 9.8, 9.4, 13.2, 19.0, 19.4, 22.8, 20.0, 16.2]
- expected return vs v2: [-0.992, -0.986, -0.956, -0.954, -0.906, -0.802, -0.745, -0.79, -0.703, -0.677, -0.679, -0.718]
- expected return vs v3: [-0.897, -0.861, -0.752, -0.641, -0.539, -0.4, -0.191, -0.096, 0.101, 0.201, 0.276, 0.264]
- expected return vs v5: [-0.991, -0.995, -0.974, -0.958, -0.881, -0.885, -0.823, -0.728, -0.704, -0.651, -0.688, -0.711]
- mixture survival (uniform v2..v5): [7.8, 9.3, 14.0, 18.2, 24.2, 29.8, 36.9, 36.2, 40.3, 42.6, 42.7, 38.4]
- mixture expected return (uniform v2..v5): [-0.922, -0.907, -0.858, -0.815, -0.748, -0.679, -0.574, -0.557, -0.486, -0.442, -0.438, -0.482]
- self-play survival: [0.2, 3.1, 16.2, 29.8, 54.9, 58.8, 73.6, 79.2, 82.7, 87.2, 89.0, 90.3]
- self-play expected return (P(win)=1/4 by symmetry): [-0.748, -0.719, -0.588, -0.452, -0.201, -0.163, -0.014, 0.042, 0.077, 0.122, 0.14, 0.153]
