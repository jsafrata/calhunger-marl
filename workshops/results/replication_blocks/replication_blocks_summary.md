# E21: ten independent 100-game replications of the peer protocol

Blocks are disjoint 100-seed slices of the merged 1000-game matrix; block 0
is the original campaign. Reference = merged 1000-game leaderboard (top-1 ippo_s2). Healthy-8 survival metric throughout.

| block | seeds | top-1 | =ref? | rho vs ref | pair agree | E17 rho | E17 sig? |
|---|---|---|---|---|---|---|---|
| 0 (orig) | 900000-900099 | ippo_s2 | Y | 0.786 | 23/28 | +0.26 | no |
| 1 | 900100-900199 | mappo_s3 | N | 0.833 | 24/28 | +0.24 | no |
| 2 | 900200-900299 | ippo_s2 | Y | 0.952 | 26/28 | -0.10 | no |
| 3 | 900300-900399 | ippo_s2 | Y | 0.833 | 24/28 | +0.19 | no |
| 4 | 900400-900499 | ippo_s2 | Y | 0.929 | 26/28 | +0.10 | no |
| 5 | 900500-900599 | ippo_s2 | Y | 0.905 | 25/28 | +0.05 | no |
| 6 | 900600-900699 | ippo_s1 | N | 0.857 | 24/28 | +0.21 | no |
| 7 | 900700-900799 | ippo_s2 | Y | 0.905 | 25/28 | +0.10 | no |
| 8 | 900800-900899 | ippo_s2 | Y | 0.976 | 27/28 | +0.02 | no |
| 9 | 900900-900999 | ippo_s2 | Y | 0.976 | 27/28 | +0.02 | no |

## Summary

- top-1 reproduced in 8/10 replications
- Spearman vs reference: mean 0.895, min 0.786
- pairwise verdict agreement with reference: mean 25.1/28, min 23/28
- E17 ladder-validity rho across replications: [-0.10, +0.26]; significant in 0/10 (threshold |rho|>=0.738 at n=8)
