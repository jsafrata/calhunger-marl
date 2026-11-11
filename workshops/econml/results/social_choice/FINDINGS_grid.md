# E27-grid. The full measurement portfolio as an electorate — combined (population x metric) voter grid [EconML theme 1]

Merged 1000-game peer matrix (n=1000/cell). Voter = one (population judge, metric) grid cell: judge j x metric m ranks the healthy-8 candidates i != j by metric m of cell (i vs 3x j) — the E12 population convention crossed with E20's six peer metrics (survival, win, cash, cash|survival, survival_time, win|survival). Judge universe = the 10 populations (pinned by the E27 cross-check; each candidate is judged by its 9 single-population 1v3 contexts), i.e. 60 grid voter cells; healthy-8-judges grid (48 cells) as a secondary voter set. Mean-of-normalized-scores is ill-defined across metric units, so rank-based rules only: Borda, Copeland, plurality, exact Kemeny (Held-Karp, n=8). Bootstrap: paired scheme, B=2000, n=1000, rng 20260903.

## Cross-check vs E27 (asserted in-script)

The grid restricted to (all 10 populations, survival) reproduces E27's `pop_all10judges_healthy8` EXACTLY: identical pairwise preference counts, identical majority structure (vs the stored `pairwise_majority.json`: no CW, weak CW ippo_s2, 0 cycles, 4 majority ties), identical Borda/Copeland/plurality/Kemeny rankings (vs the stored `rules_ranking.csv`).

## Voter accounting

| voter set | potential | dropped (all-NaN ballot) | seated | partial-ballot NaN cells |
|---|---|---|---|---|
| grid_pop10xmetrics6_healthy8 | 60 | 0 | 60 | 0 |
| grid_pop8xmetrics6_healthy8 | 48 | 0 | 48 | 0 |

## Condorcet structure on the grid

| voter set | Condorcet winner | weak CW (loses no majority) | cycle triples | majority ties |
|---|---|---|---|---|
| grid_pop10xmetrics6_healthy8 | mappo_s2 | mappo_s2 | 0/56 | 0 |
| grid_pop8xmetrics6_healthy8 | mappo_s2 | mappo_s2 | 0/56 | 2 |
| grid_restricted_pop10_survival | NONE | ippo_s2 | 0/56 | 4 |

## Rule dependence (full rankings, best to worst)

### grid_pop10xmetrics6_healthy8

| rule | ranking |
|---|---|
| borda | mappo_s2 > ippo_s2 > mappo_s3 > ippo_s1 > ippo_s0 > mappo_s0 > ippo_s3 > ippo_s4 |
| copeland | mappo_s2 > ippo_s2 > mappo_s3 > ippo_s0 > ippo_s1 > mappo_s0 > ippo_s3 > ippo_s4 |
| plurality | mappo_s2 > ippo_s2 > ippo_s1 > mappo_s3 > ippo_s0 > ippo_s4 > mappo_s0 > ippo_s3 |
| kemeny | mappo_s2 > ippo_s2 > mappo_s3 > ippo_s0 > ippo_s1 > mappo_s0 > ippo_s3 > ippo_s4 |

Distinct winners across rules: ['mappo_s2']

### grid_pop8xmetrics6_healthy8

| rule | ranking |
|---|---|
| borda | mappo_s2 > ippo_s2 > mappo_s3 > ippo_s1 > ippo_s0 > ippo_s3 > ippo_s4 > mappo_s0 |
| copeland | mappo_s2 > ippo_s2 > ippo_s0 > mappo_s3 > ippo_s1 > ippo_s3 > mappo_s0 > ippo_s4 |
| plurality | mappo_s2 > ippo_s2 > ippo_s1 > mappo_s3 > ippo_s0 > ippo_s4 > ippo_s3 > mappo_s0 |
| kemeny | mappo_s2 > ippo_s2 > mappo_s3 > ippo_s0 > ippo_s1 > mappo_s0 > ippo_s3 > ippo_s4 |

Distinct winners across rules: ['mappo_s2']

## 2-D manipulability: which policies can some (population, metric) voter crown #1?

| policy | crownable | #crowning voters | crowning metrics | rank range |
|---|---|---|---|---|
| ippo_s0 | True | 3 | cash, cash_given_survival, win_given_survival | 1-7 |
| ippo_s1 | True | 7 | cash, cash_given_survival, survival, survival_time | 1-6 |
| ippo_s2 | True | 13 | cash, cash_given_survival, survival, survival_time, win, win_given_survival | 1-7 |
| ippo_s3 | False | 0 | - | 2-8 |
| ippo_s4 | True | 3 | survival, survival_time | 1-8 |
| mappo_s0 | True | 2 | survival_time, win | 1-7 |
| mappo_s2 | True | 26 | cash, cash_given_survival, survival, win, win_given_survival | 1-8 |
| mappo_s3 | True | 5 | survival, survival_time, win | 1-7 |

Crownable set on the grid: **7/8** (ippo_s0, ippo_s1, ippo_s2, ippo_s4, mappo_s0, mappo_s2, mappo_s3) vs E27's 5/8 under population-voters alone (survival metric, healthy judges) — adding the metric dimension adds ippo_s0, mappo_s0 (5/8 -> 7/8). Survival-column-only crownable under the 10-judge universe: 5/8 (ippo_s1, ippo_s2, ippo_s4, mappo_s2, mappo_s3).

## Rule stability under paired bootstrap (B=2000, n=1000, rng 20260903)

| voter set | rule | top-1 stable | mean self-Spearman | p5 | winner | alternate winners |
|---|---|---|---|---|---|---|
| grid_pop10xmetrics6_healthy8 | borda | 60% | 0.965 | 0.905 | mappo_s2 | ippo_s2:38.9%, mappo_s3:0.9% |
| grid_pop10xmetrics6_healthy8 | copeland | 100% | 0.976 | 0.952 | mappo_s2 | - |
| grid_pop10xmetrics6_healthy8 | plurality | 100% | 0.969 | 0.922 | mappo_s2 | ippo_s2:0.1% |
| grid_pop10xmetrics6_healthy8 | kemeny | 100% | 0.976 | 0.952 | mappo_s2 | - |
| grid_pop8xmetrics6_healthy8 | borda | 90% | 0.991 | 0.970 | mappo_s2 | ippo_s2:9.6% |
| grid_pop8xmetrics6_healthy8 | copeland | 100% | 0.977 | 0.916 | mappo_s2 | - |
| grid_pop8xmetrics6_healthy8 | plurality | 100% | 0.987 | 0.951 | mappo_s2 | - |
| grid_pop8xmetrics6_healthy8 | kemeny | 100% | 0.970 | 0.905 | mappo_s2 | - |

## Interpretation (EconML: the measurement portfolio as one electorate)

E27 asked what happens when evaluation populations vote, and separately when metrics vote. Here the full measurement portfolio votes at once: every (population, metric) pair one has defensible grounds to publish is a voter, so the grid is the electorate an evaluator implicitly samples from when they pick 'the' leaderboard. The headline inverts E27: on the 60-voter grid the majority relation yields a STRICT Condorcet winner: mappo_s2 (it majority-beats runner-up ippo_s2 27-20 among the 48 voters ranking both), with 0/56 cycle triples and 0 majority ties — E27's Condorcet cycle among survival-voters and the metric-voters' split crown both dissolve once the portfolio votes jointly, and every rule agrees: borda -> mappo_s2; copeland -> mappo_s2; plurality -> mappo_s2; kemeny -> mappo_s2 (1 distinct champion from identical ballots, vs 2-3 in E27). The consensus champion is mappo_s2 — the policy the status-quo survival mean ranks 7th of 8 — because breadth across the portfolio (top-1 for 26/60 voter cells, strong on the cash and win channels in nearly every population context) beats depth on the single survival channel. Yet aggregation only relocates the evaluator's discretion, it does not remove it: manipulability now has two dimensions, and with both a population AND a metric free to choose, 7/8 policies can be crowned #1 by some single defensible voter cell (vs 5/8 with the population dimension alone; only ippo_s3 is never crownable) — the two degrees of evaluator freedom compose. Bootstrap: the Condorcet verdict is solid — borda top-1 60%; copeland top-1 100%; plurality top-1 100%; kemeny top-1 100% on the main grid — with the one soft spot that Borda's positional scores put the runner-up within sampling noise (ippo_s2:38.9%, mappo_s3:0.9%) while the majority-relation rules (Copeland, Kemeny) flip in 0/2000 replicates and plurality in 1/2000. Widening the electorate to the full measurement portfolio thus buys BOTH a Condorcet-consistent, rule-robust, sampling-stable champion AND strictly more room for single-cell cherry-picking: the choice being made is no longer which rule, but which electorate.

## Sanity checks

- restricted-grid (all populations, survival) == E27 pop_all10judges_healthy8: pref counts, majority structure, and all four rule rankings match exactly (asserted).
- dropped voter cells: 0 of 60 on the main grid; partial-ballot NaN candidate cells: 0.

Data: `workshops/econml/results/social_choice/` (grid_pairwise_majority.json, grid_rules_ranking.csv, grid_crownable.csv, grid_rule_stability.csv); script `workshops/econml/analyze_social_choice_grid.py` (rng seed 20260903).
