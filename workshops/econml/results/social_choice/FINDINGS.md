# E27. Leaderboards as social choice — aggregation & manipulability [EconML theme 1]

Merged 1000-game peer matrix (n=1000/cell); population-voters = single-opponent populations judging focals by survival (self-play diagonal excluded, E12 convention); metric-voters = the 6 E20 peer metrics. Rules: mean-score (status quo), Borda (pairwise-support), Copeland, plurality, exact Kemeny (Held-Karp). Bootstrap: paired scheme, B=2000, n=1000, rng 20260830.

## Condorcet structure

| voter set | Condorcet winner | weak CW (loses no majority) | cycle triples | majority ties |
|---|---|---|---|---|
| pop_all10judges_healthy8 | NONE | ippo_s2 | 0/56 | 4 |
| pop_healthy8judges_healthy8 | NONE | NONE | 1/56 | 0 |
|  | cycle: ippo_s1 > ippo_s2 > mappo_s3 > ippo_s1 | | | |
| pop_all10judges_all10 | NONE | ippo_s2 | 0/120 | 4 |
| metrics6_healthy8 | NONE | ippo_s2, mappo_s2 | 0/56 | 2 |
| metrics6_all10 | NONE | ippo_s2, mappo_s2 | 0/120 | 2 |

## Rule dependence (full rankings, best to worst)

### pop_all10judges_healthy8

| rule | ranking |
|---|---|
| mean_score | ippo_s2 > ippo_s1 > mappo_s3 > ippo_s4 > ippo_s3 > mappo_s0 > mappo_s2 > ippo_s0 |
| borda | ippo_s2 > mappo_s3 > ippo_s1 > ippo_s4 > ippo_s3 > mappo_s0 > mappo_s2 > ippo_s0 |
| copeland | ippo_s2 > ippo_s1 > mappo_s3 > ippo_s4 > ippo_s3 > mappo_s0 > mappo_s2 > ippo_s0 |
| plurality | ippo_s1 > ippo_s2 > ippo_s4 > mappo_s2 > mappo_s3 > ippo_s0 > ippo_s3 > mappo_s0 |
| kemeny | ippo_s2 > ippo_s1 > mappo_s3 > ippo_s4 > ippo_s3 > mappo_s2 > mappo_s0 > ippo_s0 |

Distinct winners across rules: ['ippo_s1', 'ippo_s2']

### pop_healthy8judges_healthy8

| rule | ranking |
|---|---|
| mean_score | ippo_s1 > mappo_s3 > ippo_s2 > ippo_s3 > ippo_s4 > mappo_s0 > mappo_s2 > ippo_s0 |
| borda | ippo_s2 > mappo_s3 > ippo_s1 > ippo_s3 > ippo_s4 > mappo_s2 > mappo_s0 > ippo_s0 |
| copeland | ippo_s1 > ippo_s2 > mappo_s3 > ippo_s3 > ippo_s4 > mappo_s2 > mappo_s0 > ippo_s0 |
| plurality | ippo_s1 > ippo_s2 > ippo_s4 > mappo_s2 > mappo_s3 > ippo_s0 > ippo_s3 > mappo_s0 |
| kemeny | mappo_s3 > ippo_s2 > ippo_s1 > ippo_s3 > ippo_s4 > mappo_s2 > mappo_s0 > ippo_s0 |

Distinct winners across rules: ['ippo_s1', 'ippo_s2', 'mappo_s3']

### pop_all10judges_all10

| rule | ranking |
|---|---|
| mean_score | ippo_s2 > ippo_s1 > mappo_s3 > ippo_s4 > ippo_s3 > mappo_s0 > mappo_s2 > ippo_s0 > mappo_s1 > mappo_s4 |
| borda | ippo_s2 > mappo_s3 > ippo_s1 > ippo_s4 > ippo_s3 > mappo_s0 > mappo_s2 > ippo_s0 > mappo_s1 > mappo_s4 |
| copeland | ippo_s2 > ippo_s1 > mappo_s3 > ippo_s4 > ippo_s3 > mappo_s0 > mappo_s2 > ippo_s0 > mappo_s1 > mappo_s4 |
| plurality | ippo_s1 > ippo_s2 > ippo_s4 > mappo_s2 > mappo_s3 > ippo_s0 > ippo_s3 > mappo_s0 > mappo_s1 > mappo_s4 |
| kemeny | ippo_s2 > ippo_s1 > mappo_s3 > ippo_s4 > ippo_s3 > mappo_s2 > mappo_s0 > ippo_s0 > mappo_s1 > mappo_s4 |

Distinct winners across rules: ['ippo_s1', 'ippo_s2']

### metrics6_healthy8

| rule | ranking |
|---|---|
| borda | ippo_s2 > mappo_s3 > mappo_s2 > ippo_s1 > ippo_s0 > ippo_s3 > ippo_s4 > mappo_s0 |
| copeland | ippo_s2 > mappo_s2 > ippo_s0 > mappo_s3 > ippo_s1 > ippo_s3 > mappo_s0 > ippo_s4 |
| plurality | mappo_s2 > ippo_s2 > mappo_s3 > ippo_s0 > ippo_s1 > ippo_s3 > ippo_s4 > mappo_s0 |
| kemeny | mappo_s2 > ippo_s2 > mappo_s3 > ippo_s0 > ippo_s1 > ippo_s3 > mappo_s0 > ippo_s4 |

Distinct winners across rules: ['ippo_s2', 'mappo_s2']

### metrics6_all10

| rule | ranking |
|---|---|
| borda | ippo_s2 > mappo_s3 > mappo_s2 > ippo_s1 > ippo_s0 > ippo_s3 > ippo_s4 > mappo_s0 > mappo_s1 > mappo_s4 |
| copeland | ippo_s2 > mappo_s2 > ippo_s0 > mappo_s3 > ippo_s1 > ippo_s3 > mappo_s0 > ippo_s4 > mappo_s1 > mappo_s4 |
| plurality | mappo_s2 > ippo_s2 > mappo_s3 > ippo_s0 > ippo_s1 > ippo_s3 > ippo_s4 > mappo_s0 > mappo_s1 > mappo_s4 |
| kemeny | mappo_s2 > ippo_s2 > mappo_s3 > ippo_s0 > ippo_s1 > ippo_s3 > mappo_s0 > ippo_s4 > mappo_s1 > mappo_s4 |

Distinct winners across rules: ['ippo_s2', 'mappo_s2']

## Manipulability: which policies can some single-population judge crown #1?

| group | crownable | who crowns whom |
|---|---|---|
| all10 (9 candidates/judge) | 5/10 | ippo_s1 by ippo_s2+ippo_s4+mappo_s2; ippo_s2 by ippo_s0+mappo_s3+mappo_s4; ippo_s4 by ippo_s1+mappo_s1; mappo_s2 by ippo_s3; mappo_s3 by mappo_s0 |
| healthy8 (7 candidates/judge) | 5/8 | ippo_s1 by ippo_s2+ippo_s4+mappo_s2; ippo_s2 by ippo_s0+mappo_s3; ippo_s4 by ippo_s1; mappo_s2 by ippo_s3; mappo_s3 by mappo_s0 |

Cross-check vs `results/population_sensitivity_1000/`: rank ranges match exactly = True (e.g. mappo_s2 spans 1-9, matching E12's 1-9).

## Rule stability under paired bootstrap (B=2000, n=1000)

| voter set | rule | top-1 stable | mean self-Spearman | p5 | winner |
|---|---|---|---|---|---|
| pop_all10judges_healthy8 | borda | 80% | 0.971 | 0.922 | ippo_s2 |
| pop_all10judges_healthy8 | copeland | 81% | 0.966 | 0.903 | ippo_s2 |
| pop_all10judges_healthy8 | plurality | 38% | 0.843 | 0.660 | ippo_s1 |
| pop_all10judges_healthy8 | mean_score | 90% | 0.977 | 0.929 | ippo_s2 |
| pop_all10judges_healthy8 | kemeny | 41% | 0.904 | 0.786 | ippo_s2 |
| pop_healthy8judges_healthy8 | borda | 49% | 0.974 | 0.922 | ippo_s2 |
| pop_healthy8judges_healthy8 | copeland | 45% | 0.972 | 0.933 | ippo_s1 |
| pop_healthy8judges_healthy8 | plurality | 63% | 0.867 | 0.728 | ippo_s1 |
| pop_healthy8judges_healthy8 | mean_score | 96% | 0.942 | 0.833 | ippo_s1 |
| pop_healthy8judges_healthy8 | kemeny | 91% | 0.964 | 0.857 | mappo_s3 |
| metrics6_healthy8 | borda | 99% | 0.993 | 0.970 | ippo_s2 |
| metrics6_healthy8 | copeland | 83% | 0.990 | 0.970 | ippo_s2 |
| metrics6_healthy8 | plurality | 99% | 0.984 | 0.837 | mappo_s2 |
| metrics6_healthy8 | kemeny | 99% | 0.988 | 0.952 | mappo_s2 |

## Interpretation (EconML: preference aggregation in model evaluation)

A leaderboard is a social-choice object: evaluation populations (and metrics) are voters, policies are candidates, and the published ranking is an aggregation rule applied to their preferences. Here no voter set yields a strict Condorcet winner. Under the defensible healthy-8 judge universe the failure is not a technicality: the top three policies form a genuine Condorcet cycle (ippo_s2 maj-beats ippo_s1 4-2, mappo_s3 maj-beats ippo_s2 4-2, ippo_s1 maj-beats mappo_s3 4-2, and each beats all five others), so WHICH of the three is champion is decided purely by the aggregation rule: mean-score and Copeland crown ippo_s1, Borda crowns ippo_s2, Kemeny crowns mappo_s3, plurality crowns ippo_s1 — three distinct champions from the same ballots. Under the all-10 judge universe ippo_s2 is a weak Condorcet winner (loses no majority, blocked only by a 4-4 tie with mappo_s3), yet plurality still elects ippo_s1. Metric-voters split the crown between the two weak co-winners ippo_s2 (Borda, Copeland) and mappo_s2 (plurality, Kemeny) — the E20 metric disagreement expressed as an electoral outcome. Manipulability is first-order: 5/8 healthy policies can be crowned #1 by choosing a single sympathetic judge population (mappo_s2 spans ranks 1-9 all-10, matching E12), so an evaluator free to pick the opponent population can strategically elect most of the field. Bootstrap: the status-quo mean-score is the MOST stable rule on its own terms (top-1 90-96%), while the rules with better axiomatic credentials are fragile (Kemeny 41%, plurality 38% top-1 under all-10 judges) — the cycle sits inside sampling noise, so rule choice and sampling interact: the leaderboard is an artifact of both.

## Sanity checks

- mean-score leaderboard reproduces `results/leaderboard_stability_1000/` exactly: ordering match = True, max |score diff| = 0.00e+00.
- crownability rank ranges match `population_sensitivity_1000` for all 10 policies: True.

Data: `workshops/econml/results/social_choice/` (pairwise_majority.json, rules_ranking.csv, crownable.csv, rule_stability.csv); script `workshops/econml/analyze_social_choice.py` (rng seed 20260830).
