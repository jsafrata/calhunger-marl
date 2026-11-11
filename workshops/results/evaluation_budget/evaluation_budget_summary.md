# E22: games-per-cell budget to decide each healthy pairwise verdict

SDs from the paired-seed bootstrap at n0=1000 (B=2000, seed 20260829);
1/sqrt(n) scaling verified empirically: mean sd(100)/sd(1000) = 3.17 (expected sqrt(10) = 3.16). Survival metric, macro over 9 cells.

Reference order: ippo_s2 > ippo_s1 > mappo_s3 > ippo_s4 > ippo_s3 > mappo_s0 > mappo_s2 > ippo_s0

| pair | adjacent | gap (pts) | sd@1000 (pts) | decided@1000 | n (50% power) | n (80% power) |
|---|---|---|---|---|---|---|
| ippo_s0 vs ippo_s2 |  | 13.98 | 0.65 | yes | 8 | 17 |
| ippo_s0 vs ippo_s1 |  | 13.02 | 0.89 | yes | 18 | 37 |
| ippo_s0 vs mappo_s3 |  | 12.54 | 0.87 | yes | 18 | 38 |
| ippo_s0 vs ippo_s4 |  | 12.08 | 0.98 | yes | 25 | 52 |
| ippo_s0 vs ippo_s3 |  | 11.01 | 0.83 | yes | 22 | 45 |
| ippo_s2 vs mappo_s2 |  | 10.43 | 0.61 | yes | 13 | 27 |
| ippo_s1 vs mappo_s2 |  | 9.48 | 0.75 | yes | 24 | 49 |
| mappo_s2 vs mappo_s3 |  | 9.00 | 0.77 | yes | 28 | 58 |
| ippo_s4 vs mappo_s2 |  | 8.53 | 0.91 | yes | 43 | 88 |
| ippo_s3 vs mappo_s2 |  | 7.47 | 0.64 | yes | 28 | 58 |
| ippo_s0 vs mappo_s0 |  | 7.37 | 0.84 | yes | 50 | 101 |
| ippo_s2 vs mappo_s0 |  | 6.61 | 0.85 | yes | 63 | 129 |
| ippo_s1 vs mappo_s0 |  | 5.66 | 1.03 | yes | 127 | 259 |
| mappo_s0 vs mappo_s3 |  | 5.18 | 0.82 | yes | 97 | 197 |
| ippo_s4 vs mappo_s0 |  | 4.71 | 1.15 | yes | 228 | 465 |
| mappo_s0 vs mappo_s2 | Y | 3.82 | 0.82 | yes | 179 | 365 |
| ippo_s3 vs mappo_s0 | Y | 3.64 | 0.81 | yes | 188 | 383 |
| ippo_s0 vs mappo_s2 | Y | 3.54 | 0.74 | yes | 166 | 339 |
| ippo_s2 vs ippo_s3 |  | 2.97 | 0.70 | yes | 213 | 434 |
| ippo_s1 vs ippo_s3 |  | 2.01 | 0.71 | yes | 485 | 990 |
| ippo_s2 vs ippo_s4 |  | 1.90 | 0.80 | yes | 683 | 1,394 |
| ippo_s3 vs mappo_s3 |  | 1.53 | 0.71 | yes | 821 | 1,676 |
| ippo_s2 vs mappo_s3 |  | 1.43 | 0.73 | NO | 1,004 | 2,049 |
| ippo_s3 vs ippo_s4 | Y | 1.07 | 0.84 | NO | 2,372 | 4,840 |
| ippo_s1 vs ippo_s2 | Y | 0.96 | 0.71 | NO | 2,110 | 4,306 |
| ippo_s1 vs ippo_s4 |  | 0.94 | 0.66 | NO | 1,878 | 3,832 |
| ippo_s1 vs mappo_s3 | Y | 0.48 | 0.67 | NO | 7,446 | 15,197 |
| ippo_s4 vs mappo_s3 | Y | 0.47 | 0.72 | NO | 9,154 | 18,681 |

## Summary

- 6/28 healthy pairs remain UNDECIDED at n=1000
- certifying the full healthy-8 ranking (all adjacent pairs, 80% power) needs **18,681 games per cell** (x90 cells = 1,681,326 games total)
- the original 100-game protocol could only ever decide pairs with gaps of roughly >= 4.9 points (1.96 x sd at n=100)
