# E28 follow-up. Alpha-rank as the third solution concept — it sides with Nash, crowns the spoiler, and is the least stable of the three [EconML theme 1]

Setup: same empirical meta-game as E28 (merged 1000-game peer matrix, loader
`eval/crossplay_raw_common.load_peer_1v3(extended=True)` + self-play diagonal): M[i,j] =
mean focal-i outcome vs 3xj; survival and win; all-10 and healthy-8. Alpha-rank
(Omidshafiei et al. 2019): single-population response-graph Markov chain over monomorphic
states, mutant fixation prob rho = (1-e^{-aD})/(1-e^{-maD}) with D = M[tau,sigma] -
M[sigma,tau], population size m=50, transition C[s,t] = rho/(n-1); score = stationary
mass. Alpha sweep {0.01, 0.1, 1, 10, 100, inf-limit} (inf = exact limit + 1e-12 uniform
mixing for irreducibility). Script `econml/analyze_alpha_rank.py`; data
`econml/results/alpha_rank/` (`alpha_rank_ratings.csv`, `alpha_sweep.csv`,
`three_concepts_stability.csv`, full record `alpha_rank_summary.json`).

**Validation**: RPS gives the uniform stationary distribution at every swept alpha and at
inf (max dev < 1e-9); a strictly dominant strategy absorbs (pi > 0.999 at alpha >= 10 and
inf); every stationary solve verified to residual <= 1e-10 (0 eigen fallbacks across all
16,016 solves); recomputed macro-mean scores match `nash_averaging/nash_ratings.csv` to
1e-6. Structural note stated up front: with the paper's fixation formula the chain depends
on M *only through the mirrored differences* M[i,j] - M[j,i] — exactly the antisymmetric
part A = (M - M^T)/2 that Nash averaging uses — so E28's caveat (1v3 payoffs are not a
symmetric game; mutual destruction reads as skill) transfers to alpha-rank wholesale.

**Point rankings (inf limit; sweep in `alpha_sweep.csv`)**:
- **Survival, all-10: alpha-rank crowns the spoiler mappo_s1 — mass 1.0, at EVERY alpha
  from 0.01 to inf** (already 0.988 at alpha=1). mappo_s1 is the unique sink of the
  response graph (no mutant invades it: its toxic column means every healthy policy loses
  the mirrored comparison), the exact Markov-chain analogue of E28's pure Nash. Ranking
  correlation with the mean leaderboard is *negative* (rho = -0.41); with Nash +0.52;
  mean-#1 ippo_s2 gets mass ~7e-13.
- Survival, healthy-8: no sink — the recurrent class is a 6-state cycle and the mass
  ordering reproduces the Nash support: **ippo_s3 0.444 > ippo_s1 0.278 > mappo_s0
  0.111** > {ippo_s2, ippo_s4, mappo_s3} 0.056 each; rho(nash, alpharank) = **0.94** vs
  rho(mean, alpharank) = 0.43. Mean-#1 ippo_s2 ranks 5th (mass 0.056). Top-1 switches
  ippo_s1 -> ippo_s3 at alpha >= 10 (the only sweep top-1 change in any combo).
- Win (all-10 and healthy-8): **ippo_s2 is the unique sink, mass ~1.0 at all alphas — all
  three concepts agree on the mean-leaderboard #1**, consistent with E28's near-zero-sum
  reading of the win channel.
- Large-alpha regime: rankings match the inf limit exactly from alpha=10 (surv/all10) and
  alpha=100 (win/healthy8); in the other two combos the head of the ranking is stable
  from alpha>=10 (rank-vs-inf rho 0.98 surv/healthy8) and the residual disagreement is
  tie-granularity among ~zero-mass transient states (win groups, rho 0.52 vs inf because
  inf ties them while finite alpha orders their noise floor).

**Spoiler verdict (the E28 question): alpha-rank does NOT rescue the evaluation — it
agrees with Nash, not with the mean, on both survival games, and its spoiler verdict is
bootstrap-certain** (mappo_s1 top-1 in 2000/2000 replicates at inf AND at alpha=10;
`bootstrap_top1_frequency`). This is structural, not empirical accident: both concepts
consume only A, so any A-based method must reward being catastrophic to others.

**Stability** (same paired game-level bootstrap as E19/E28: n=1000, **B=2000**, rng
20260902; E28 nash/macro-mean rows copied verbatim for one-look comparison, full table
`three_concepts_stability.csv`; alpharank rows tie-aware, Nash convention):

| metric/group | estimator | mean rho | top-1 | mean flip | pairs >=5% | adj. flip |
|---|---|---|---|---|---|---|
| surv/all10 | alpharank_inf | 1.000 | 100% | 0.0% | 0/45 | 0.0% |
| surv/all10 | nash | 0.962 | 100% | 5.3% | 12/45 | 18.4% |
| surv/all10 | macro-mean | 0.988 | 89% | 1.8% | 5/45 | 7.7% |
| surv/healthy8 | alpharank_inf | **0.885** | **81%** | **16.6%** | **27/28** | 20.5% |
| surv/healthy8 | nash | 0.960 | 91% | 5.7% | 11/28 | 14.9% |
| surv/healthy8 | macro-mean | 0.977 | 89% | 3.0% | 5/28 | 9.9% |
| win/all10 | alpharank_inf | **0.757** | 98% | **40.1%** | 36/45 | 44.4% |
| win/all10 | nash | 0.967 | 100% | 6.7% | 10/45 | 23.0% |
| win/all10 | macro-mean | 0.990 | 100% | 1.8% | 4/45 | 8.1% |
| win/healthy8 | alpharank_inf | 0.989 | 100% | 1.0% | 0/28 | 1.3% |
| win/healthy8 | nash | 0.985 | 100% | 2.1% | 3/28 | 6.9% |
| win/healthy8 | macro-mean | 0.980 | 100% | 2.8% | 4/28 | 10.4% |

Reading: alpha-rank's stability is bimodal. Where a robust sink exists it is
*degenerately* perfect (surv/all10: every replicate reproduces the mappo_s1 point mass —
"stably wrong", even more so than Nash) or genuinely best-in-table (win/healthy8, 0/28
flipping pairs). Where the recurrent structure is contested it is the **worst of the
three**: surv/healthy8 (cycling MCC; mass sloshes ippo_s3 <-> ippo_s1, top-1 81% vs
Nash 91%/mean 89%, 27/28 pairs flip >= 5%) and win/all10 (the ippo_s2 -> mappo_s1
response edge sits on a 0.0005 mirrored-difference near-tie — E28's near-tied Nash
support — and when it flips, the sink dissolves into a cycle and the whole tie block
reshuffles: mean flip 40%). The graded alpha=10 rows (`alpharank_a10` in the CSV) confirm
the diagnosis: win/all10 becomes stable (rho 0.979, top-1 99.95%) once masses are graded
rather than winner-take-all, while surv/healthy8 top-1 drops to 62% (argmax churn between
the two heaviest cycle members); the low a10 rho on win/healthy8 (0.814) is cosmetic
noise-floor churn among ~1e-8..1e-40 masses and not a ranking claim.

**Interpretation (three economic concepts, one dataset)**: the three concepts split 2-vs-1
exactly along the payoff channel. On the near-zero-sum win channel all three crown
ippo_s2 and are all stable. On the non-constant-sum survival channel the two
*relative-performance* concepts (Nash averaging, alpha-rank) agree with each other
(rho 0.94 on healthy-8; identical spoiler verdict on all-10 with 2000/2000 bootstrap
confidence) and against the mean — because both are functions of the same antisymmetrized
matrix, alpha-rank inherits Nash's pathology rather than adjudicating it: evolutionary
dynamics offer no escape from a mis-specified payoff comparison. And as an estimator,
inf-limit alpha-rank is a step function of pairwise mirrored-difference signs, so its
ranking variance concentrates precisely where the game is interesting (cycles, near-tied
edges): it is either trivially stable or the least stable concept in the table. The mean
leaderboard remains the most stable ranking; the equilibrium/evolutionary concepts add
agreement with each other, not reliability.
