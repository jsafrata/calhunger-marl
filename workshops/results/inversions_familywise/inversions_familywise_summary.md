# E34: strict metric inversions (healthy 8, n=1000 paired games/cell) under familywise control

Family = 28 pairs x 6 metrics = 168 tests (normal-approx p from the paired bootstrap SD, B=4000).
Strict inversion = survival verdict significant AND an alternative-metric verdict significant with the opposite sign.

| rule | strict inversions / 28 |
|---|---|
| per_test_95_percentile | 15 |
| per_test_99_percentile | 14 |
| per_test_95_normal | 15 |
| bonferroni_168_normal | 14 |
| holm_168_normal | 14 |
| families_per_test_95_normal | 18 |
| families_bonferroni_84_normal | 18 |
| families_holm_84_normal | 18 |

Pairs under Holm(168):

- ippo_s0 vs ippo_s1: cash, cash_given_survival, win_given_survival
- ippo_s0 vs ippo_s2: cash, cash_given_survival
- ippo_s0 vs ippo_s3: win, cash, cash_given_survival, win_given_survival
- ippo_s0 vs ippo_s4: win, cash, cash_given_survival, win_given_survival
- ippo_s0 vs mappo_s0: win, cash, cash_given_survival, win_given_survival
- ippo_s0 vs mappo_s3: cash, win_given_survival
- ippo_s1 vs mappo_s2: win, cash, cash_given_survival, win_given_survival
- ippo_s2 vs mappo_s2: cash, cash_given_survival, win_given_survival
- ippo_s3 vs mappo_s0: cash
- ippo_s3 vs mappo_s2: win, cash, cash_given_survival, win_given_survival
- ippo_s4 vs mappo_s0: cash, cash_given_survival
- ippo_s4 vs mappo_s2: win, cash, cash_given_survival, win_given_survival
- mappo_s0 vs mappo_s2: win, cash, cash_given_survival, win_given_survival
- mappo_s2 vs mappo_s3: win, cash, cash_given_survival, win_given_survival
