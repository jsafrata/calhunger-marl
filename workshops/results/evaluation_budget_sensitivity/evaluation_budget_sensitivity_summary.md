# E33: the certification budget, made explicit

Criterion: two-sided test of a zero pairwise macro-survival difference at alpha=.05 with 80% power, per adjacent pair of the reference survival ordering; SD from the paired-seed bootstrap at n0 (shared game seeds across cells, so cross-cell dependence is inside sd); 1/sqrt(n) scaling (verified empirically, sd(100)/sd(1000)=3.17 vs 3.16); ranking certified when every adjacent pair is decided; budget = max over adjacent pairs.

Reference order: ippo_s2 > ippo_s1 > mappo_s3 > ippo_s4 > ippo_s3 > mappo_s0 > mappo_s2 > ippo_s0

| adjacent pair | gap (pts) | sd@1000 (pts) | n, 80% power, alpha .05 | n, 80% power, Bonferroni(7) |
|---|---|---|---|---|
| ippo_s2 > ippo_s1 | 0.96 | 0.71 | 4,313 | 6,854 |
| ippo_s1 > mappo_s3 | 0.48 | 0.67 | 15,222 | 24,190 |
| mappo_s3 > ippo_s4 | 0.47 | 0.72 | 18,712 | 29,736 |
| ippo_s4 > ippo_s3 | 1.07 | 0.84 | 4,848 | 7,705 |
| ippo_s3 > mappo_s0 | 3.64 | 0.81 | 384 | 610 |
| mappo_s0 > mappo_s2 | 3.82 | 0.82 | 365 | 581 |
| mappo_s2 > ippo_s0 | 3.54 | 0.74 | 339 | 539 |

- published plug-in budget (max over adjacent pairs): **18,712** games/cell, binding pair mappo_s3 > ippo_s4
- binding gap 0.47 +- 0.72 pts (bootstrap SE): budget 2,891 at gap+1SE, inf at gap-1SE
- budget recomputed on each bootstrap replicate's gaps: 5/25/50/75/95th pct = 8,324, 26,581, 82,520, 407,504, 8,251,955; 5.5% of replicates exceed 10M (a replicate gap at ~0)
- Bonferroni over the 7 adjacent comparisons (per-pair 80% power, FWER 5%): **29,736** games/cell
- joint power (all 7 adjacent verdicts correct and significant) at the published budget: **68%**
- budget for 80% JOINT power: **22,789** (per-pair alpha .05) / 34,288 (Bonferroni 7)
- joint-power curve (per-pair threshold): n=100: 0%, n=1000: 0%, n=5000: 1%, n=10000: 23%, n=18681: 68%, n=30000: 92%, n=50000: 99%, n=100000: 100%

Adjacent-difference correlation matrix (paired bootstrap):

    +1.00  -0.43  +0.11  -0.19  -0.28  +0.12  -0.17
    -0.43  +1.00  -0.55  +0.08  +0.35  -0.38  +0.07
    +0.11  -0.55  +1.00  -0.59  -0.38  +0.35  +0.01
    -0.19  +0.08  -0.59  +1.00  -0.03  -0.19  -0.11
    -0.28  +0.35  -0.38  -0.03  +1.00  -0.69  +0.22
    +0.12  -0.38  +0.35  -0.19  -0.69  +1.00  -0.43
    -0.17  +0.07  +0.01  -0.11  +0.22  -0.43  +1.00
