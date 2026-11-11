# E28. Nash averaging vs the unstable leaderboard [EconML theme 1]

Setup: empirical 10x10 meta-game from the merged 1000-game peer matrix (E19 seed
extension; loader `eval/crossplay_raw_common.load_peer_1v3(extended=True)` +
`load_peer_selfplay`): M[i,j] = mean focal-i outcome vs 3xj, self-play diagonal from the
mixed cells; survival and win versions; all-10 and healthy-8. Nash averaging (Balduzzi et
al. 2018): maxent Nash p* of the antisymmetrized game A = (M - M^T)/2 — one Nash via LP
(HiGHS; antisymmetric => value 0), then unique maxent point of the optimal polytope
{p in simplex: (Ap)_j <= 0} via SLSQP with analytic gradients; rating_i = (Ap*)_i.
Script `econml/analyze_nash_averaging.py`; data `econml/results/nash_averaging/`
(`meta_game_matrix.csv`, `nash_ratings.csv`, `nash_vs_mean_stability.csv`,
`replicator_cycles.json`, full record `nash_summary.json`).

**Validation**: equilibrium residual max_j (Ap*)_j <= 2.4e-13 on all four point estimates
(requirement 1e-6); game value p*'Ap* ~ 1e-29; solver checks out on RPS (uniform) and
dominated-strategy test games. Macro-mean sanity: recomputed 9-cell macro-mean scores
match `results/leaderboard_stability_1000/` references to 1.1e-16, ranking identical; the
macro-mean rows of the bootstrap (rng 20260831) reproduce the E19 reference rows (rng
20260827) to ~1 pt (e.g. healthy-8 survival top-1 89.2% vs 90.6%).

**Nash ratings and support (point estimates, n=1000/cell)**:
- **Survival, all-10: the unique Nash is pure mappo_s1** — the *marginal* policy, 9th of
  10 on the mean leaderboard (23.1% survival). Every other policy rates negative (-0.044
  to -0.118); mean-leaderboard #1 ippo_s2 ranks **8th of 10**. Mechanism: A scores
  *mirrored relative* performance, and mappo_s1 is the catastrophic column (healthy
  policies survive 0.5-16.3% vs 3x mappo_s1 while mappo_s1 survives 14-33% against them)
  — being toxic to others counts as beating them. **Equilibrium rating crowns the
  spoiler.**
- Survival, healthy-8: mixed Nash {ippo_s3 0.775, mappo_s0 0.123, ippo_s1 0.102} — an
  empirical rock-paper-scissors triple in mirrored matchups (A entries: ippo_s3 > ippo_s1
  +0.014, ippo_s1 > mappo_s0 +0.092, mappo_s0 > ippo_s3 +0.012). Mean-leaderboard #1
  ippo_s2 is NOT in the support (rating -0.049, rank 6 of 8).
- **Win (all-10 and healthy-8): unique pure Nash ippo_s2 = the mean-leaderboard #1**; the
  win channel is nearly exclusive (approximately constant-sum), so the antisymmetrization
  is benign and the two concepts agree. E10a's mirrored win-dominance (ippo_s2<->ippo_s4
  = 30.3<->4.9) is exactly what A encodes: A_win[s2,s4] = +0.127.

**Stability (the E28 question): Nash averaging is LESS stable than the macro-mean on the
survival leaderboard, and only marginally better where it agrees with it (win,
healthy-8).** Same bootstrap as E19: game-level resampling, paired scheme (one index
vector for all 100 cells), n=1000, **B=2000 (no reduction needed; 2000 maxent solves/combo
in 3-11 s, 0 LP fallbacks)**, rng 20260831. Ranking-level, vs macro-mean on identical
replicates (`nash_vs_mean_stability.csv`):

| metric/group | estimator | mean rho | top-1 | mean flip | pairs >=5% | adj. flip |
|---|---|---|---|---|---|---|
| surv/all10 | nash | 0.962 | 100%* | 5.3% | 12/45 | 18.4% |
| surv/all10 | macro-mean | 0.988 | 89% | 1.8% | 5/45 | 7.7% |
| surv/healthy8 | nash | 0.960 | 91%* | 5.7% | 11/28 | 14.9% |
| surv/healthy8 | macro-mean | 0.977 | 89% | 3.0% | 5/28 | 9.9% |
| win/all10 | nash | 0.967 | 100% | 6.7% | 10/45 | 23.0% |
| win/all10 | macro-mean | 0.990 | 100% | 1.8% | 4/45 | 8.1% |
| win/healthy8 | nash | 0.985 | 100% | 2.1% | 3/28 | 6.9% |
| win/healthy8 | macro-mean | 0.980 | 100% | 2.8% | 4/28 | 10.4% |

*Nash top-1 is tie-aware (support members all rate exactly 0; point top policy in the
replicate's maximal-rating set within 1e-7; healthy-8 survival has a 3-way point tie),
so it is not directly comparable to the macro-mean argmax; ratings are snapped to 1e-7
before rank/sign tests, flips vs the strict point verdict (E19 "ties count" convention).
Support-level stability: survival all-10 support = {mappo_s1} in **2000/2000** replicates
(the spoiler verdict is rock-solid — stably wrong); healthy-8 survival support
exact-match 77.7% (Jaccard 0.86; member frequencies 0.92/0.90/0.79, mappo_s3 enters 3.9%);
win healthy-8 {ippo_s2} 98.3%; win all-10 exact-match only 49.4% (mappo_s1 rates -0.0005,
a near-tie that pulls it into the support in 48.8% of replicates).

**Replicator dynamics** (healthy-8 survival, dt=0.1 Euler, 5e5 steps, uniform start;
`replicator_cycles.json`): on the raw meta-game M (incl. the ~0.9 self-play diagonal) the
population **converges to a pure ippo_s1 monoculture** (final |dp| = 0) — the ecology is
coordination-like, monocultures absorb. On the antisymmetrized A it **never converges:
persistent cycling** with tail amplitude 0.87 and estimated period ~1450 time units,
orbiting the Nash-support triple (mass mostly sloshing ippo_s1 <-> ippo_s3) — E2/E10a's
non-transitivity showing up as a literal economic cycle.

**Interpretation (EconML theme 1)**: an economic solution concept does not rescue the
unstable leaderboard — honestly, it is *less* stable at the ranking level (structural
rating ties + support churn add variance), and on the survival metric it is invalid
before it is unstable: Nash averaging presumes a zero-sum symmetric game, and the
antisymmetrization A = (M - M^T)/2 of a non-constant-sum 1v3 ecology (both focals can
survive; M[i,j] and M[j,i] come from different lineups) converts mutual destruction into
skill, crowning the policy every mean-based analysis calls broken (mappo_s1) with perfect
bootstrap confidence. Where the payoff channel actually is near-zero-sum (wins), Nash
averaging agrees with the mean leaderboard's winner and is marginally more stable
(healthy-8). Lesson: the value of importing an equilibrium concept into evaluation hinges
on the game-theoretic structure of the *measured* payoff, not on the concept — a
misapplied solution concept is confidently, stably wrong.
