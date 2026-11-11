# EXPERIMENTS.md — single source of truth for all paper numbers

Every experiment, its setup, headline numbers, figure/data pointers, one-paragraph
interpretation, and venue tags. **Papers cite THIS file's numbers; when a number changes,
change it here first.** Venue tags: [IAEval] [WV]=Who Verifies [TAE] [IAB] [ICLR].

Evaluation conventions (apply everywhere unless stated): seat-balanced lineups (each policy
occupies each seat equally), fixed game-seed bases so every checkpoint/pair sees identical
games, stochastic sampling unless "deterministic", 100 games per scripted-eval cell, 50 per
self-play cell, checkpoint 1200 unless stated. "Survival" = alive at horizon T; "win" =
max cash among survivors (chance level 25%).

---

## E1. Environment verification [all venues, methods]
TS↔Python parity tests (byte-equivalence vs the live game), determinism tests,
property-based stress testing; 4 subtle order-replacement/reservation bugs found and
preserved as executable MREs (`research/bugs/*.md`). Reference floor: four v1 bots survive
to T in 99.75% of seats over 1000 games (~10^5 trades/game).
**Use**: "verifying the verifier" [WV]; benchmark-quality evidence [IAEval, ICLR].

## E2. Heuristic ladder calibration tournament [all venues]
Setup: v2–v5, T=30 variant, 100 seat-balanced games + all pairwise 2v2; fast cadence.
Numbers (four-way): survival 99/97/99/100%; **wins v4 54% > v5 22% > v2 16% > v3 8%**;
cash $130.2/$101.8/$87.0/$81.0. Pairwise non-transitivity: v4>v2 (39–11), v4>v3 (38–12),
v5>v3 (45–5), but v5<v4 (17–34) and v5<v2 (22–28). v1 strictly dominated (0% wins
four-way, $16 mean cash).
Data: `results/scripted_v*_tournament_small_30s/SCRIPTED_VERSION_TOURNAMENT.md`. → Table T1.
**Interpretation**: the opponent suite has a genuine skill ordering with rock–paper–scissors
structure; the economy rewards intertemporal position-taking (v4) over spread-earning.
Note the difficulty ordering is STABLE across opponent types (E6): the bots that dominate
the tournament (v4, v5) are also the hardest for RL; the weakest (v3) is easiest. The
ladder's calibration transfers — never phrase this as an "inverse" relation (user
correction, Aug 13): bot difficulty is the same whether bots face bots or policies.

## E3. BC control — capacity control AND paradigm control [IAEval, WV, TAE, ICLR]
Behavior cloning from scripted-v1-population demonstrations, identical network/encoding.
**In-population**: 97.0% survival among clones / 34% wins (100 games, seed base 700000;
98.4% in the original 1,000-game diagnostic), 94.6% vs the v1 population its demos came
from. Single-linear-head ablation: 40.9% (fixes factored-head design).
**Cross-play vs unseen archetypes (NEW, Aug 13, HPC job 778608, 100 seat-balanced paired
games each)**: **v2 22% / v3 8% / v4 0% / v5 0% survival, 0% wins everywhere**. Data:
`results/bc_vs_ladder_100/bc_vs_ladder_report.json`; harness `research/hpc/bc_vs_ladder/`.
**Interpretation (two claims, keep both)**: (1) capacity is not the bottleneck — the
function class represents a population-mastering policy; (2) imitation is population-bound
exactly like self-play RL — the generalization failure is paradigm-independent. NEVER cite
the BC control on self-play numbers alone (user requirement: the papers argue self-play
metrics are unreliable, so the control must rest on cross-play). Note: scripted v1 itself
*survives* vs v2–v4; the clone is worse than its source out-of-population (covariate shift).
**(c) v4 clone from mixed-lineup demos (Aug 14, HPC job 783598)**: per-step fidelity 93%
(type 95%, side 99.8%) but the **price head plateaus at ~71–73% exact** and behavior
diverges: real v4 four-way 99% surv / 53% win vs clone 57% / 31%; vs the v5 pair 86% vs
6.5%. **Cause: v4 prices off an internal fair-value EMA over ALL executed trades — hidden
state absent from the 171-d observation — so a memoryless MLP cannot fully represent it.**
v1 (reactive) clones near-perfectly; v4 (intertemporal) can't: policies sit between
representability (MLP can't express v4) and trainability (LSTM could but fails, E9).
Data: cluster `~/hunger_runs/bc_v4_clone/bc_v4_report.json` (not yet synced locally).
Capacity claim pending upgrade via E15.

## E15. Fine-tune vs the ladder — capacity by construction [COMPLETE Aug 14; IAEval/WV/TAE headline control]
Warm-started the 8 healthy best-config default_180_mlp checkpoints (ippo s0–4, mappo
s0/2/3, @1200); 300 iterations vs the eval lineup format (1 learned + 3× one archetype,
uniform {v2..v5}/episode, fast cadence, terminal-only reward unchanged; entropy/lr held at
annealed terminal values; warm start verified logits-identical). HPC jobs 808775 (train),
830143–830150 (eval). Data: `results/finetune_vs_ladder/` (training telemetry on cluster).
**Result — ALL 8 runs succeed** (100 paired games/cell, iter_1500, stochastic):
- vs v2: surv 71 [61–82], win 5 [0–17]
- vs v3: surv 98 [94–100], win 60 [23–79]
- vs v4: surv **72 [51–85]**, win **39 [26–57]** — was 16–17%/7% pre-fine-tune; best run
  (ippo_s3) 85/57, out-winning v4's own tournament rate (54%). Every run's win rate vs v4
  exceeds the 25% chance level.
- vs v5: surv 56 [31–78], win 35 [20–49] — was 14/10.
**Claim (exact framing)**: existence proof by construction — the identical architecture,
observation, and (initially) weights contain a ladder-mastering policy reachable by 300
iterations of standard PPO once the training population matches the test population.
Self-play's failure is the population, not representation or the optimizer. DELIBERATELY
trains on the test population: capacity control, NOT a generalization claim (papers must
say so). Nuance to report honestly: the maker's win channel stays hard (5% mean at 71%
survival) — surviving v2 ≠ out-cashing v2.
[Pending, secondary]: self-play/peer transfer evals of the tuned policies.

## E4. v2 5-seed baseline (fixed 4096-step batch, 180 s) [methods; WV appendix]
IPPO/MAPPO × seeds 0–4 × 1500 iters. Mean self-play survivors ~flat: 1.17→1.53 (IPPO),
1.17→1.42 (MAPPO) over 1500 iters. Vs scripted at 1500: vs v1 surv 0–28%, win 0–4%; vs v2
surv 0–14%, **win 0% in all 10 runs**. Cross-check (E10c): 9 of 10 final checkpoints fail
their own self-play on fresh seeds.
**Interpretation**: the naive configuration fails at full horizon; motivates episode-matched
batching (E6) — with γ=λ=1 and terminal-only reward, the effective sample count per PPO
update is *games*, not steps.

## E5. Entropy ablation [methods]
entropy_coeff 0.05 vs 0.01 (both algos, seed 0, 180 s): 0.05 strictly worse — 0% wins at
every checkpoint. Under-exploration is not the simple fix.

## E6. Episode-matched horizon sweep — NOW n=5 SEEDS [IAEval, WV, ICLR — headline]
Setup: 45 complete games per PPO update across T∈{60,90,120,150,180}; IPPO+MAPPO × seeds
0–4 × 1200 iters; scripted eval 100 games × v2–v5; self-play 50 games. 10 runs per cell.
**F2 matrix (ckpt 1200, mean survival %, [min–max]):**

| T | self-play | v3 (JIT) | v2 (maker) | v5 (sniper) | v4 (stockpiler) |
|---|---|---|---|---|---|
| 60 s | 83 [76–91] | 95 [91–99] | 79 [72–84] | 54 [41–61] | 28 [21–42] |
| 90 s | 82 [76–88] | 94 [84–99] | 59 [44–68] | 37 [12–47] | 26 [18–38] |
| 120 s | 83 [66–89] | 92 [83–99] | 49 [36–63] | 25 [14–42] | 33 [10–49] |
| 150 s | 5 [0–52] | 15 [5–56] | 6 [1–21] | 1 [0–4] | 23 [15–44] |
| 180 s | 71 [0–94] | 75 [7–100] | 16 [1–33] | 6 [1–19] | 16 [5–34] |

Figure: `figures/f2_crossplay_matrix.{png,pdf}` (`eval/plot_crossplay_matrix.py`).
Extended metrics (from resummarize, 180 s ckpt 1200): win|survival vs v2 ≈ 0–9%; vs v3
28–38% despite ~100% survival; **vs v5 survivors win 86–88% of the time (survival is the
bottleneck); vs v3 conversion is the bottleneck**. Near-death survivor fraction
(min required food 0–1): 0.09–0.25 vs v2.
**Interpretation**: (i) self-play ≫ cross-play vs every competent archetype, gap widens
with horizon; (ii) per-opponent difficulty follows the bots' own tournament ordering (E2 note);
(iii) which *channel* fails (survival vs conversion-to-win) depends on the opponent.

## E7. Discovery cliff — T=150 quantified; T=180 also seed-bimodal [IAEval, WV, ICLR]
Cliff campaign: +16 runs (IPPO/MAPPO × seeds 2–9) at T=150, episode-matched. Self-play
eval @1200 (50 games): **3/16 discover (92.5, 82.5, 51.5%); 13/16 at exactly 0.0%**.
Combined with the 4 original runs (one at 86.5%): **4/20 ≈ 20% discovery, outcomes binary**
(nothing between 0% and 51%). At T=180, n=5: **8/10 runs succeed** — seed 3 fails for BOTH
algorithms (~1.2 survivors/4 in training; F2's 180 s range 0–94 reflects this).
Data: `results/seeds234/cliff/_selfplay_survival_50/small_150/`.
**Interpretation**: not gradual degradation — a seed-dependent *discovery* event whose
probability falls with horizon (variance held constant by the episode-matched design).
Deaths pin near t≈90 in failed runs. [Optional add: scripted eval of the 3 discoverers.]

## E8. Best-config bundle [ICLR; brief elsewhere]
Everything-on config (survival features, trade-history critic, entropy 0.01→0.001 anneal,
lr anneal, episode-matched). n=5 for MLP arms @1200:
- small_90_mlp: self-play 90% [76–95]; vs v2 61 [43–85] win 8; vs v3 99 win 27; vs v4 32
  [23–47] win 5; vs v5 52 [37–60] win 24 [14–36]. Improves on E6's 90 s row (esp. vs v5).
- default_180_mlp: self-play 77% [0–94] (seed-bimodality persists); vs v2 22 win 2; vs v3
  80 win 30; vs v4 13 win 7; vs v5 14 win 10 (provisional: n=6 rows aggregated; reconcile
  old-dir schema to n=10 before camera-ready).
**Interpretation**: the best configuration buys real but modest cross-play gains; no
configuration change closes the generalization gap or removes seed-bimodality.

## E9. LSTM failure is horizon-dependent [ICLR; IAB one line]
small_90_lstm: 0–1% self-play at EVERY checkpoint (price-head entropy never sharpens:
stays 3.0–3.7 vs MLP decaying to ~2.0). default_180_lstm: partial training — self-play
29–57%, vs v3 33–62%, still ≪ MLP. Caveat: Ray 2.53 sequence-axis minibatching not yet
ruled out as a contributor at 90 s ([pending] post-mortem before strong claims).

## E10. Policy-vs-policy cross-play (learned peers) [IAEval, WV, TAE — F3]
Harness: `eval/checkpoint_policy_crossplay.py` (seat-balanced, seed-identical, 1v3 +
mixed modes, self-play diagonal mandatory). 100 games/cell.
**(a) BEST-ARCHITECTURE matrix (paper figure)** — default_180_mlp best-config,
IPPO/MAPPO × seeds 0–4, all @1200; identical architecture AND training config, differing
only in seed (+critic input across algos). F3 = `figures/f3_peer_crossplay_bestconfig_180s`.
Data: `results/seeds234/_peer_crossplay_100/`. HPC campaign `best_config_peer_crossplay`,
job 639242.
- Self-play diagonal: 8/10 healthy (88–95%, mean 92.2); mappo_s1 marginal (56.8%);
  mappo_s4 failed (0.0%) — the 180 s seed-bimodality (E7) reappearing under best config.
- **Healthy-vs-healthy (56 ordered pairs): survival mean 74.0 [45–93]** (≈18 pts below
  self-play), **win mean 16.7 [2–42]; 21/56 pairs win ≤10%** (baseline 25%). Strong
  non-transitive dominance among identical-config seeds. Canonical mirrored-pair example
  (user-preferred, abstract + body): **ippo_s2 ↔ ippo_s4, wins 33↔4 with near-identical
  survival both directions (74 vs 77) — dominance entirely in the win channel**. Others:
  42↔6 (cross-algo), 41↔11, 37↔7.
- **Catastrophic peer columns**: vs 3× the *marginal* mappo_s1, every healthy policy
  survives 0–17% with cash $235–285 — incl. **mappo_s2: 93.5% self-play → 0.0% surv / 0%
  win / $285** (rich-and-dead induced purely by a peer seed). Vs 3× the *failed* mappo_s4:
  healthy policies survive 17–40% holding **$330–380** — the bidirectional weak-partner
  effect (Prop. 1 with learned peers), replicated without any deprecated condition.
**(a-1000) 1000-game refresh (Aug 29, from the E19 seed extension; figure
`f3_peer_crossplay_bestconfig_1000g`, matrix `seeds234/_peer_crossplay_1000/`, builder
`eval/build_peer_matrix_1000.py`)**: all headline structure holds with tighter estimates —
diagonal 8/10 healthy (87.6–94.5), mappo_s1 54.2, mappo_s4 0.0; healthy-vs-healthy surv
73.2 [45–90], win 16.7 [3–37], 18/56 pairs win ≤10%; mirror ippo_s2↔s4 = 30.3↔4.9 wins at
75.6/78.6 surv; healthy vs 3×mappo_s1: surv 0.5–16.3 at $237–285 (mappo_s2: 92.1 self →
0.5 surv / 0.5 win / $285); vs 3×mappo_s4: surv 14.0–45.1 at $334–378. IAEval cites these;
the n=100 values remain the published-budget baseline that E19/E21 audit.
**(a-legacy) v3 campaign matrix @1100** (superseded as the paper figure — includes the
deprecated shared+ID condition; kept as supporting data): shared_s1 75%/24% self →
26%/0% vs 3× shared_s2; everyone vs 3× weak sharedid_s0: 9–30% surv at $253–342.
**@1400 dominates its own lineage** (checkpoint-lineage experiment, from the v3 runs; NOT
in the peers figure): shared_s1@1400 vs 3× shared_s1@1100 = 66% surv / 62% win / $282 —
while the same checkpoint collapses vs Scripted-V1 (E11).
**(b) small 30 s @1000**: all four checkpoints healthy in self-play (84–92%); the peer gap
is in the competitive channel: mirrored pairings ippo_s0 vs 3×ippo_s1 = 7% win / $57 vs
ippo_s1 vs 3×ippo_s0 = 41% win / $136; survival stays 67–95%.
**(c) hpc_v2 180 s @1500**: self-play diagonal first — 9/10 checkpoints fail their own
self-play (ippo 54/0/0.2/8.8/0; mappo 0/19/31.5/0.2/0.2) → off-diagonal uninterpretable as
cross-play; documents baseline failure + why the diagonal is mandatory.
Full writeup: `results/policy_crossplay/CROSSPLAY_FINDINGS.md`.

## E11. Rich and dead — longitudinal, MULTI-SEED on best architecture [WV, TAE — F1; all venues]
**PRIMARY (paper figure)**: default_180_mlp best-config, the 8 (of 10) runs healthy in
self-play, checkpoints 100→1200, cross-play = 1 learned + 3× Scripted-v4, 100 games/ckpt
paired seeds. Figure `figures/f1_rich_and_dead_bestconfig`
(`eval/plot_rich_and_dead_bestconfig.py`; healthy-seed filter documented in script).
- Self-play survival rises monotonically 0→**90.3%** (band 76–95 at 1200).
- Cross-play survival peaks **36.8% @ckpt 700**, declines to **17.0% @1200**; **7/8 runs
  decline >5 pts from their peak** (examples: 54→7, 50→5, 42→4).
- Mean terminal cash in the same games rises monotonically **$15→$174**, never decreasing
  while survival halves over the second half of training. With survival at 17%, most of
  that cash is held by agents that die.
- **Multi-seed redistribution (same runs, same checkpoints)**: vs v2 survival still
  *rising* through ckpt 1100 (1→25%) and vs v5 rising-then-plateau (0→20→17), while vs v4
  peaks-then-falls — per-opponent trends diverge within one training run set.
**LEGACY (earlier campaign, single run, sharper)**: ippo_shared_seed1 ckpts 1000→1400 vs
3× Scripted-V1: survival 52.5→9.5%, cash $88→$239 (dead-seat cash $135→$239), self-play
flat 83–90%; @1400 dominates its own lineage (62% wins). Keep attributed to "an earlier
training campaign" when cited; the best-config multi-seed version is the paper claim.

## E12. Population sensitivity — evaluation verdicts flip [TAE lead; IAEval]
v3 conditions vs one-of-each scripted populations (100 games): **shared_s2 in {v2,v3,v4}:
3% win / $28; in {v2,v4,v5}: 25% win / $147** — same policy, same seeds, 8× win-rate swing.
(Also: shared_s1 0%→7%, sharedid_s1 2%→10%.) 1v3 vs each archetype: no v3-condition policy
exceeds 9% wins vs any bot; parameter sharing (shared vs shared+ID) does not rescue 180 s
generalization. Data: `results/policy_crossplay/v3_vs_scripted_180s/`.
**REBASED Aug 28 (cite this version; the v3 numbers above are deprecated)**: population
sensitivity recomputed on the best-config merged 1000-game peer matrix
(`results/population_sensitivity_1000/`, script
`workshops/eval/analyze_population_sensitivity.py`). Same policy, different population:
mappo_s2 survives 92.1% in self-play but 0.5%-77.7% across single-peer populations;
ippo_s2 spans 2.7%-88.7%. Rank is a property of the evaluators: mappo_s2 places
anywhere from 1st to 9th depending on which single population judges it (ippo_s3:
2nd-8th; 7 of 8 healthy policies span >=5 rank positions). Two defensible
single-population leaderboards (healthy strangers) correlate at mean rho=0.35, min
**-0.77** — two reasonable evaluation populations can produce ANTI-correlated
leaderboards. n=1000/cell makes every quoted spread decided (binomial SE ~1.5 pts).

## E13. Argmax collapse / behavioral forensics [IAB lead; IAEval protocol point]
Deterministic argmax over-emits WAIT (50–65%) and dies; stochastic sampling of the SAME
policy trades competently and beats deterministic on every axis (v2-era deep evals; det
rows also in E10b small30 matrix). Head-entropy fingerprints: price|POST entropy decays
3.7→~2.0 when learning succeeds (MLP), never when it fails (LSTM, E9). Final-checkpoint
action mix (ckpt 1100, 500 games): wait 11.4 / cancel 9.8 / take 48.3 / post 30.6%.
**39.9% of surviving learned seats end within one consumption tick of death** (knife-edge
inventory management, not buffer-building).
**Interpretation**: *how you query the policy changes the verdict* (grader design);
behavior-level signals predict generalization failure before cross-play evaluation shows it.

## E14. Reproducibility note [methods]
Cliff seeds 2–4 reproduce episode-matched small_150 runs bit-identically across separate
campaigns (same seed/config → identical iters.jsonl), EXCEPT mappo_seed2 (small telemetry
divergence, likely PYTHONHASHSEED×task-index mapping). [pending] resolve before claiming
bit-exact reproducibility; claim "seeded and reproducible" until then.

## Result-to-venue emphasis map

| | IAEval | WV | TAE | IAB |
|---|---|---|---|---|
| Lead | E6+E10 (protocol + matrices) | E11+E10a (broken verifier) | E12+E11 (invalid instruments) | E13 (behavior) |
| Second | E2, E3, E13 | E1, E2 (verifier portfolio) | E10c (diagonal rule) | E11, E10a |
| Brief | E7, E8 | E6, E7 | E6 | E6 |
| Omit/1-line | E4, E5, E9 | E4, E5, E8, E9 | E3–E9 | E2–E9 |

## Known caveats to keep honest
- **Protocol-component claims (user, Aug 13)**: only THREE components have documented
  wrong-conclusion anecdotes (outcome conditioning ← cash inversion; dual sampling modes ←
  argmax "broke" a competent policy; self-play diagonal ← 9/10-fail-own-selfplay would
  read as incompatibility). Seat balancing and seed pairing are DESIGN GUARANTEES
  (fairness by construction / paired comparisons) — never claim their absence produced a
  wrong number. Framing "two design guarantees + three failure-motivated components" is
  now in the IAEval abstract, contributions list, and shared protocol.tex.
- **Two-config disclosure (user requirement, Aug 13)**: every paper's methods must name
  BOTH configurations (episode-matched sweep → horizon-matrix figure; best config =
  +survival features +trade-history critic +entropy/lr anneals → peers + rich-and-dead
  figures), state nothing was tuned per figure, and cite the cross-config replication
  (vs-v4 180 s endpoint: 16% episode-matched vs 17% best config, "independent training
  runs"). Sequencing wording (user, Aug 13): the ablation pass is "intermediate —
  conducted after the horizon sweep and frozen before the best-configuration runs"; never
  say "prior" or a bare "fixed in advance" (in advance of *what* must be explicit: each
  config frozen before the runs that produced its figures). Present in all four drafts.
- "Constant interaction" phrasing: agents can WAIT; the defensible claim is **no agent
  remains viable without repeatedly completing trades** (Prop. 1). Never claim "must
  interact every step."
- E8 default_180 aggregation provisional (n=6 vs expected 10).
- E9 LSTM: minibatching confound unresolved.
- E14 single-run telemetry divergence.
- Uniqueness claim: exhaustive vs Melting Pot (49 substrates, D/M/N), representative vs
  the broader landscape (see `shared/comparison.tex`).

## E16. Horizon transfer — the difficulty is social, not temporal [IAEval; Aug 27, job 1721772]
All 10 best-config small_90_mlp policies (@1200), unmodified, evaluated in the DEFAULT
180s env: self-play survival **79–99% (mean 89.3), n=10, 100 paired games each** — matches
at-home performance (mean 90) and EXCEEDS native 180s policies (mean 77 incl. 2 failed
seeds), because transfer sidesteps the long-horizon discovery cliff. Transfer is clean by
design: obs time is horizon-fractional; survival features are cycle-denominated. Claim:
doubling the horizon costs ~nothing; changing the company collapses policies — the
benchmark's generalization failures are specifically social. NOT a cross-horizon-failure
claim (we removed "horizons" from the failure axes; keep it that way).
Data: `results/horizon_transfer_90to180/`.

## E17. Ladder validity — scripted bots do not proxy peer competence [IAEval; Aug 27, job 1724783]
All 10 best-config default_180 policies (@1200) evaluated 1v3 vs each scripted rung v1–v5
(100 paired games, seed base 900000, same crossplay harness as the peer matrix). Question
(IAEval "do simulated partners proxy real ones?" transposed): does vs-ladder survival
predict mean off-diagonal peer survival (E12 matrix)? Answer: **only for detecting
degenerate policies, not for ranking competent ones.** On the healthy 8, no rung's
Spearman rho reaches significance (mean v2–v5: rho=+0.30, p=0.47; vs v4: rho=−0.07). The
all-10 correlations (e.g. vs v2 rho=+0.70) are carried entirely by the two degenerate
seeds (mappo_s4 scores ~0 on both axes). Headline inversions: mappo_s1 is the BEST of all
10 against the two hardest bots (45% vs v4, 33% vs v5) yet second-worst against peers
(24.9% survival); ippo_s3 has the best ladder mean (52.8) but the worst healthy peer win
rate (9.4%); ippo_s2 is near-bottom on the ladder (31.8) with the top peer win rate
(26.3%). Framing: a hand-built partner population is a valid smoke test and an invalid
leaderboard — even simulated partners engineered for this exact game mismeasure social
competence. n=8 rank correlations are low-powered; lead with the inversions, report rho
as supporting. Data: `results/seeds234/_ladder_validity_100/`, summary in
`results/seeds234/ladder_validity_dose_response_summary.md`.

## E18. Population dose-response — each stranger costs ~6 points [IAEval; Aug 27, job 1724783]
For focal F and stranger peer S, k = stranger copies among F's 3 co-players. k=0 (self-
play), k=1 (secondary rows of E12's 1v3 runs), k=3 (E12 1v3 rows) already existed; this
campaign added k=2 (mixed 2+2, all 45 unordered pairs, 100 paired games; each run yields
k=2 for both members). All 90 ordered pairs have complete curves. Healthy-focal+healthy-
stranger (n=56 pairs): **92.2 → 87.3 → 80.7 → 74.0%** at k=0,1,2,3 — near-linear, ≈6
points per stranger. All pairs (n=90): 79.4 → 68.2 → 59.0 → 51.5. Claim: the social
generalization failure is GRADED, not a cliff — one stranger at the table already costs
~5–6 points even with two familiar copies alongside (k=1 min among healthy pairs: 71.3%),
and each additional stranger costs about as much again. Complements E12 (which is the
k=3 endpoint) and E16 (social vs temporal). Note k-points differ in seat-count precision
(n_focal_seat_games = 400/300/200/100 for k=0/1/2/3). Data:
`results/seeds234/_peer_mix22_100/` + analysis script
`workshops/eval/analyze_ladder_validity_dose_response.py`.

## E19. Bootstrap stability of the peer leaderboard — the evaluation does not decide its own ranking [TAE lead candidate; Aug 27, offline reanalysis of E12 raw data]
Bootstrap (B=2000) of the 10-policy peer cross-play leaderboard (score = macro-mean over
the 9 focal 1v3 cells) from per-game raw outcomes, at budgets n=5/10/20/50/100 games per
cell, under two schemes: "paired" (one seed-index vector shared by all cells — what the
campaign's paired-seed protocol actually delivers) and "independent" (fresh indices per
cell). Survival leaderboard, healthy 8: even at the FULL n=100 budget the leaderboard is
not settled — mean self-Spearman 0.85 (paired), top-1 identity stable in only 59% of
replicates, 13.1% of pairwise verdicts flip on average, 15/28 pairs flip in ≥5% of
replicates. At n=5 the healthy-8 ranking is mostly noise (mean rho 0.42, 5th pct −0.09).
Protocol finding: seed pairing across cells makes cell scores positively correlated, so
the SAME budget with fresh seeds per cell yields a strictly more stable leaderboard
(n=100 healthy-8: mean rho 0.90 vs 0.85; top-1 67% vs 59%) — pairing helps paired
per-cell contrasts but hurts macro-average leaderboards, and a naive bootstrap that
ignores the pairing overstates precision. Conclusion-level: E17's headline ladder-validity
Spearman (healthy 8) has bootstrap 95% CI [−0.02, +0.65], P(rho≤0)=3.9%, and reaches the
n=8 significance threshold (0.74) in only 0.5% of replicates — the "ladder does not
predict peers" null is itself stable under resampling. NB: raw-data point estimate is
rho=0.262 (vs E17's published +0.30) because full precision breaks the ippo_s1/mappo_s0
tie at 62.4% that CSV rounding created; cite 0.26 going forward. Data:
`results/leaderboard_stability/` + script
`workshops/eval/analyze_bootstrap_leaderboard_stability.py` (loader
`workshops/eval/crossplay_raw_common.py`, rng seed 20260827).
**1000-game extension (Aug 28, campaign peer_crossplay_900ext, job 1791932, 100/100
tasks COMPLETED)**: every peer cell extended by 900 games (seeds 900100-900999,
disjoint+contiguous with the base set; merged loader `load_peer_1v3(extended=True)`).
Findings on the merged set (`results/leaderboard_stability_1000/`):
(a) **out-of-sample validation of the bootstrap** — the actual n=100 leaderboard
correlates with the n=1000 leaderboard at rho=0.786 (healthy 8; 4 of 8 policies change
rank, e.g. ippo_s3 2nd->5th, ippo_s1 4th->2nd), inside the band the n=100 bootstrap
predicted (mean 0.85, 5th pct 0.64) — the instability estimate is not a bootstrap
artifact; (b) at n=1000 the leaderboard is still not settled: top-1 stability 91%
(paired), 5/28 healthy pairs flip in >=5% of replicates, adjacent-pair flip 9.6% —
deciding the full healthy-8 ranking needs multiple thousands of games per cell;
(c) paired-vs-independent gap persists at every n (n=1000 top-1: 95% independent vs 91%
paired); (d) **E17's ladder-validity rho collapses from +0.26 to 0.000** (CI [-0.29,
+0.26], P(rho<=0)=61.6%) when peer competence is measured on 1000 games — the better the
peer measurement, the weaker the ladder's claim to proxy it; cite rho=0.00 going
forward.

## E20. Metric-portfolio disagreement — most pairwise verdicts are metric-dependent [TAE lead candidate; Aug 27, offline reanalysis of E12 raw data]
On the SAME peer cross-play games, each policy scored under six defensible peer metrics
(survival, win, mean terminal cash, cash|survival, survival time, win|survival) plus two
self-play instruments (selfplay survival, selfplay cash|survival). Healthy 8: survival
and cash leaderboards correlate NEGATIVELY (rho=−0.33); survival~win only +0.19; the two
self-play instruments correlate with peer survival at +0.44 and −0.45 respectively.
**23/28 healthy pairwise verdicts flip under at least one alternative peer metric; 9 are
STRICT inversions** (both directions individually significant under a paired-seed
bootstrap 95% CI, B=2000), e.g. ippo_s0-vs-mappo_s0 and mappo_s2-vs-mappo_s3 reverse
between survival and cash-family metrics. Structural observation: mean self-play terminal
cash is 100.0 for every policy BY CONSERVATION (trading only moves cash between the four
seats of one policy), so any self-play cash instrument is informative only when
survivor-conditioned — the raw quantity carries zero signal. (E11's $15→$174 is
cross-play cash vs 3×v4 and is NOT constrained by this.)
Framing: "which of two policies is better" is not a property of the games played but of
the metric chosen; a trustworthy protocol must report the portfolio and the
disagreements, not a scalar. Data: `results/metric_disagreement/` + script
`workshops/eval/analyze_metric_disagreement.py` (rng seed 20260828).
**1000-game extension (Aug 28, same campaign as E19)**: with 10x the games
(`results/metric_disagreement_1000/`), strict inversions RISE from 9 to **15 of 28
healthy pairs** (20/28 flip under >=1 alternative metric) — more data does not resolve
metric disagreement, it sharpens it: the contradictions are structural, not sampling
noise. Lead with the 1000-game numbers. **Multiplicity-hardened (Aug 28)**: 14 of the 15
strict inversions survive at 99% CIs, and collapsing the six metrics into three families
(immune to "you counted near-duplicates": cash~cash|surv rho=0.98) still yields
survival-vs-cash 17/28, survival-vs-win 14/28, win-vs-cash 4/28 cross-family strict
inversions at 99% — cite the family-collapsed numbers when space is tight.

## E21. Ten independent replications — the protocol disagrees with itself [TAE; Aug 28, offline, data from E19 extension]
The merged 1000-game peer matrix splits into TEN DISJOINT 100-game seed blocks — ten
genuinely independent runs of the original protocol (block 0 IS the original campaign),
not bootstrap simulations. Healthy-8 survival leaderboard per block vs the 1000-game
reference: **top-1 reproduced in 8/10** (block 1 crowns mappo_s3, block 6 crowns
ippo_s1); Spearman vs reference mean 0.895, min 0.786 (the ORIGINAL campaign is the
worst replicate); pairwise verdict agreement mean 25.1/28, min 23/28 — i.e. a typical
independent rerun of the published protocol contradicts ~3 of the 28 published pairwise
verdicts. E17 replication: ladder-validity rho spans [-0.10, +0.26] across the ten
reruns, significant in 0/10 — the null replicates perfectly. Framing: this is an actual
replication study, free, from seed-extension data; lead with it over the bootstrap
(reviewers trust disjoint-data replication more). Data: `results/replication_blocks/` +
script `workshops/eval/analyze_replication_blocks.py`.

## E22. Evaluation-budget prescription — certifying the ranking costs ~19k games/cell [TAE; Aug 28, offline]
Power analysis from the paired-seed bootstrap at n0=1000 (1/sqrt(n) scaling verified
empirically: sd(100)/sd(1000) = 3.17 vs sqrt(10) = 3.16). Per healthy pair: games/cell
needed to decide the verdict at alpha=0.05 (50% and 80% power). Big gaps are cheap
(>=7-point gaps decide at n<=100); **6/28 healthy pairs remain undecided even at
n=1000**; the tightest adjacent pairs (ippo_s1 vs mappo_s3: 0.48 pts; ippo_s4 vs
mappo_s3: 0.47 pts) need ~15,000-19,000 games/cell, so **certifying the full healthy-8
ranking at 80% power costs ~18,700 games per cell (~1.7M games total)** — vs the 100
games/cell the protocol ran, which could only ever decide gaps >= ~4.9 points.
Constructive claim: report which verdicts the budget can decide and mark the rest
undecided, instead of publishing a totally ordered leaderboard. Data:
`results/evaluation_budget/` + script `workshops/eval/analyze_evaluation_budget.py`
(rng seed 20260829).

## E23. Generated verifier — trained exploiter vs frozen champion [WV lead; Aug 28, jobs 1786802/1804018 (train), 1804019/1822433 (audit)]
For each of the 8 healthy best-config default_180_mlp champions @1200, a PPO exploiter was
warm-started FROM that champion and trained 300 episode-matched iterations vs 3x its frozen
self (stochastic inference, eval convention; entropy 0.001 / lr 1e-4 terminal-anneal values;
harness `research/hpc/exploiter_vs_champion/`). The adversary is thus GENERATED from the
artifact under audit by vanilla PPO — nothing hand-crafted.
**Audit (100 seat-balanced seed-paired games/cell, seed base 900000, both 1v3 directions):**
- Exploiter 1v3 champions: **win 31.0% mean [21–39]** (mirror baseline 25%), exploiter
  survival 88.9 [82–93]. Best: ippo_s4 39% win / $138 cash.
- Champion 1v3 exploiters: champion survival **76.6 [54–93]** (self-play diagonal 92.2),
  win **17.4 [11–26]**. Channel varies by seed: mappo_s0 loses survival (54%); mappo_s3
  keeps 89% survival but wins 11% at $61 cash vs the exploiters' $113 — exploited purely
  in the cash channel.
**Interpretation (exact framing)**: the generated worst-case verifier refutes the self-play
certificate only MODESTLY — the champions are fairly mirror-robust, plausibly because forced
interdependence bounds pure adversarial play (starving your counterparties starves you,
Prop. 1). Three verifiers of increasing construction cost now form a gradient on the same
checkpoints: generated exploiter (win 25→31, −16 champion survival pts) ≈ natural seed
diversity (E10a: 74.0 surv / 16.7 win) ≪ population shift (E11/E17 collapse). The
certificate fails from distribution shift, not adversarial fragility — and only the
composed portfolio distinguishes the two. Bonus claim: 300 iters of PPO cannot generate a
stronger adversary than seed diversity provides for free.
Caveats: exploiter budget 300 iters (a floor on exploitability, not a ceiling); mappo_s3's
first training task stalled on a degraded node (~0 samples; clean rerun used — task-level
telemetry in `mappo_seed3_headmlp.failed_1786802_7/`).
**(b) AGGRESSIVE arm — re-exploration does NOT strengthen the exploiter [Aug 28, jobs
1828981 (train), 1851196 (audit)]**: same 8 warm starts, same 300 iters, but optimizer
reset for re-exploration (entropy re-annealed 0.01→0.001, lr 3e-4→1e-4). Identical audit:
- Exploiter 1v3 champions: win **13.6 mean [6–27] — BELOW the 25% baseline**; exploiter
  cash $39–111 vs the champions' ~$96–120. The exploiter still survives (81–97) but loses
  the cash game it was warm-started knowing how to play.
- Champion 1v3 aggressive exploiters: champion survival 76.8 [44–96] — the SAME ~-15pt
  drop as vs the conservative exploiters (76.6) — but champion win now **33.3 [16–54]**,
  ABOVE its 25% mirror rate (mappo_s2: 54% wins / $213 vs the exploiters' $62).
**Interpretation (exact framing, user hypothesis test)**: parameter distance per se is not
what breaks the champion. The aggressive exploiter IS behaviorally farther from the warm
start, yet it flatters the champion's win rate (a distant-but-weaker stranger) while the
conservative exploiter (close-but-optimized) beats it. Meanwhile champion SURVIVAL degrades
identically (~77%) against both — deviation of any kind from the co-adapted partner
distribution costs survival (the bidirectional weak-partner effect, Prop. 1), while the win
channel tracks opponent competence, not opponent distance. Also replicates E5 at the
fine-tune stage: extra entropy is strictly worse — 300 iters cannot rebuild what the
high-entropy/high-lr phase destroys (partial catastrophic forgetting of the warm start).
**(c) Continuation arm — exploitability grows with budget, no plateau [Aug 29, jobs
1878216/1886618 (train, two tasks requeued off sick node hpc-24-21), 1916848 (audit)]**:
conservative exploiters resumed iter_1500 -> 1800 (same frozen champion @1200, same
lr/entropy). Identical audit at iter_1800: exploiter win **36.0 mean [29–47]** (was 31.0
@300 iters; mirror baseline 25; best mappo_s2 47%/$129, ippo_s4 45%/$154); champion 1v3
exploiters: survival 75.5 [52–100] (unchanged vs 76.6 @300), win **13.1 [9–22]** (was
17.4). Dose-response 0/300/600 iters: exploiter win 25→31.0→36.0; champion win at the
exploiter table 25→17.4→13.1; champion survival 92→77→76. **Cite these as the E23
headline**: exploitation deepens steadily in the cash/win channel (~5 pts per 300 iters,
curves still rising at 1800) while the survival channel saturates at the same ~15-pt
deviation cost as ANY stranger — even at 2x budget the generated adversary's damage
(75.5% survival) remains comparable to seed diversity (73.2, E10a-1000g) and far short of
population shift (17.0, E11). Caveat: mappo_s0's continuation stalled at iter 1653 (same
node-stall class), so its iter_1800 checkpoint carries ~450/600 iterations (win 0.42 at
freeze).
Data: `results/exploiter_vs_champion/` and `results/exploiter_vs_champion_aggr/`
(`_audit_100/*/{exploit,reverse}/crossplay_matrix.*`, per-run `iters.jsonl` exploit curves).

## E24. LLM arena — four production LLMs under the identical protocol [IAEval headline addition; WV, TAE]
Setup: claude-haiku-4-5, gpt-5.6-luna, deepseek-v4-flash, gemini-3.5-flash-lite as seats in
the unchanged small_90 env via a strict JSON wire protocol (frozen prompt `llm-arena-v3`:
rules + derived survival state + market view + mask-derived legality block, NO strategy
advice; template hash in every job identity). Same protocol as E6/E10: seat rotation,
paired seeds (base 910000), stochastic, 100 games/cell. Cells: ladder v2–v5, four selected
policies (small_90 best-config @1200: ippo s0/s2 + mappo s1/s3, chosen from E8 ladder
evals), and a 100-game 4-LLM mixed tournament. Harness `workshops/llm_arena/`; data +
per-decision trajectory logs (raw responses, latency, tokens, env verdicts):
`workshops/results/llm_arena/campaign_90s/` (canonical CSV `llm_matrix.csv`, findings doc
`LLM_ARENA_FINDINGS.md`). Figure F5 `figures/f5_llm_matrix.{png,pdf}`.
**Status (FROZEN Aug 29 ~12:00)**: claude/gpt/deepseek 100% complete (8 opponent cells +
selfplay diagonal + mixed, all n=100). Gemini: complete at n=100 for v2, v3, v4, ippo_s0,
ippo_s2, mixed; v5 partial at n=47; mappo_s1/mappo_s3 + its selfplay diagonal NOT run —
its prepaid API credits depleted (429 RESOURCE_EXHAUSTED) after earlier 503 throttling;
workers stopped. Finishing needs ~$4–6 credit at ai.studio; on top-up, relaunch
`campaign/gemini_worker.sh` (cache-resumes), then rerun `summarize_campaign.py` +
`analyze_campaign.py` + `plot_llm_matrix.py` to upgrade all numbers/CIs/F5.
**Headlines** (gemini partial cells marked with n in CSV/F5):
- **Survive-but-never-win**: vs every fixed opponent, wins 0–8% (chance 25%; 24 wins in
  ~2,600 seat-games) while survival vs v2/v3 is 84–100% and terminal cash $0.5–5 — the
  economic channel fails, not the survival channel. PPO wins 8–27% on the same cells.
- **Ladder ordering transfers — for 3 of 4 models**: v3 easiest … v4/v5 hardest, as for
  RL (E2/E6), for claude/gpt/deepseek; LLMs out-survive PPO vs v2 (84–99 vs 61 mean) yet
  never out-cash anyone. **Gemini inverts it (n=100, Aug 29): vs v3 only 59.0 surv / 3.0
  win / $23.9 — v3 is its second-hardest bot while being everyone else's easiest**
  (claude/gpt/deepseek 92–97 vs v3; gemini 98/96 vs v2/v4). Mechanism = style×population:
  gemini waits 74–85% and feeds off liquid books; the JIT procurer posts nothing early, so
  two passive parties leave nothing to take and gemini starves. Difficulty is a property
  of the (style, population) pair, not of the opponent alone — the strongest
  fixed-styles/population-relativity datum in the LLM grid.
- **Policy populations harsher + seed-dependent**: vs ippo_s0 survival 28–56% across
  models (≈ v4/v5 level, 30–60 pts below v2/v3); other seeds 20–30 pts gentler — peer-seed
  sensitivity (F3's column effect) reproduced with LLMs.
- **Population-relative ranking inversion**: mixed-economy winner Gemini (49% wins, $167)
  is passive vs bots (0–4% wins); best survivor Claude (93%) wins 3% at $26. Same models,
  opposite verdicts by population (E12's point, cross-paradigm).
- **Self-play diagonals healthy** (added Aug 28; 100 games × 4 same-model seats each):
  survival gpt 99.0 / deepseek 95.0 / claude 89.5 (gemini deferred, API-throttled); win
  rate ≈25% and mean cash exactly $100 in every diagonal (symmetry + conservation — two
  built-in integrity checks). Cross-population failures are therefore incompatibility,
  not incapacity. Claude's mask-illegal rate peaks in its own diagonal (15.5%): four
  hyperactive traders race for the same book. Diagonal ordering mirrors the wait-share
  fingerprints (calmer population → healthier economy).
- **Late-death signature**: 64–90% of LLM deaths land at t=60–90 (modal t=80–90), only
  3–11% in the first three feedings — starvation is the *endpoint* of economic stripping,
  not a separate failure (data `death_times.csv`).
- **Fixed styles, not adaptation**: action fingerprints are near-disjoint across models
  (wait share: claude 4–30% / deepseek 24–43% / gpt 46–73% / gemini 74–85%) and stable
  across all nine populations (`fingerprints.csv`); 95% bootstrap CIs for every cell in
  `llm_matrix_ci.csv` (10k resamples over games).
- **Reliability/cost axes** (CFP-native): malformed ≤0.1%; mask-illegal 0.3–7.4%
  (prompt-version sensitive: legality block cut it from 12–62%, and changed behavior —
  the prompt is part of the grader, cf. E13); latency ~1 s/decision except Gemini's 20.8 s
  p95 retry tail + 10 aborted games (Google 503 load-shedding) — an infrastructure
  reliability finding; campaign $228 actual at list prices incl. self-play diagonals
  (claude $169 / gpt $27 / deepseek $25 / gemini $7; ~371k decisions), ~74% Claude
  (tokenizer 1.5× on same prompt).
**Interpretation**: the benchmark's generalization claims are paradigm-independent in a
third way (E3 showed imitation; E24 shows in-context reasoning agents): any agent defined
without exposure to a population inherits that population's boundary, and the protocol's
outcome-conditioned, population-portfolio measurements are what make the failure legible.

---

## E25. The portfolio, longitudinally — signal-by-signal verdicts, selection regret, held-out validity, detection budget [WV lead exhibits; Aug 28, offline reanalysis of E11 data]
All on the E11 longitudinal data (8 healthy best-config runs, ckpts 100..1200, 100
games/cell vs each archetype, 50 self-play games/ckpt). Script:
`eval/analyze_wv_portfolio_longitudinal.py`; data `results/wv_portfolio_longitudinal/`.
**(a) Signal-by-signal ablation on the ckpt-700→1200 transition** (per-run deltas, ±2pt
flat band; "flags" needs a majority of runs declining):
| signal | 700→1200 | verdict |
|---|---|---|
| self-play survival | 73.6→90.3 (7/8 up) | certifies |
| price-head entropy fingerprint | 2.62→2.29 nats (8/8 sharpen) | certifies |
| cross-play cash vs v4 | $97→$174 (8/8 up) | certifies |
| cross-play survival vs v3 | 73→94 (7/8 up) | **certifies — an external verifier can also be fooled** |
| cross-play survival vs v2 / v5 | 25→26 / 13→16 | silent (redistribution) |
| cross-play survival vs v4 | 36.8→17.0 (7/8 down) | FLAGS |
| cross-play win vs v4 | 9.4→6.8 (5/8 down) | FLAGS |
| portfolio mean (v2..v5) | 36.9→38.4 mean UP, yet 5/8 runs down | flags only per-run — **averaging archetypes masks the collapse** (v3's gain cancels v4's loss) |
| held-out portfolio (v2,v3,v5) | 37.0→45.6 (6/8 up) | certifies — MISSES the collapse |
Claim: 5 of 8 defensible signals certify the transition, 2 are silent; only the
v4-family signals flag it — heterogeneity is load-bearing AND aggregation must stay
per-archetype (no pooling) and per-run (no seed-mean).
**(b) Checkpoint-selection regret (deployment = v4 survival; selector argmax over ckpts):**
ship final ckpt: 17.0% deployed, **regret 26.6 pts** vs oracle (43.6% @ ~ckpt 600; worst
run 47); select by self-play: 21.9%/21.8; by cash: 20.0%/23.6; by held-out {v2,v3,v5}
portfolio: 25.9%/17.8 — the held-out verifier recovers only ~1/3 of the regret.
Selection does not transfer across archetypes (consistent with E17's invalid-leaderboard
result and E2's non-transitivity).
**(c) Held-out verifier validity:** dropping v4 from the portfolio, the reduced verifier's
700→1200 verdict CERTIFIES (a-table); peak-based, it flags a (much smaller) decline in
6/8 runs — held-out detection exists but is weak and ~20× more expensive (d).
**(d) Detection budget** (one-sided two-prop, alpha .05, power .80, independent-binomial —
upper bounds vs the paired design): detecting each run's peak→1200 v4 drop needs
**median 21 games/cell (max 205; 1 run has no drop)** — the actual 100/cell over-detects;
without v4 (pooled v2,v3,v5) the median is ~473/cell. Soundbite: "with the right
archetype in the verifier population, refuting the certificate costs ~two dozen CPU
games; without it, ~20× more and the standard endpoint-verdict still certifies."
Caveats: verdicts use endpoint 700→1200 (the F1 claim) except where "peak-based" is
stated; entropy band 0.05 nats; regret is in-sample for the oracle row by construction.

---

# EconML block (E26–E31) — added Aug 29 for the EconML@NeurIPS submission
Venue tag [EconML]. Designs in `workshops/econml/DESIGN.md`; all scripts and data
under `workshops/econml/`. E26–E30 are offline/local ($0 API cost — the "replay" in
E26 re-executes the env on recorded actions, never the LLM APIs); E31 is HPC job
1896332 (campaign `research/hpc/econml_microstructure/`, commit 3b36057).

## E26. LLM economy reconstructed — terms of trade & wealth transfer [EconML headline; IAEval/WV color; Aug 29, offline replay]
All 3,239 campaign_90s games replayed deterministically (recorded LLM action
indices; scripted seats re-seeded per game; policy seats re-run from the original
local checkpoints under the arena's torch-stream discipline). **Validation
3,070/3,239 (94.8%)**: five per-game checks (every recorded env_applied verdict,
n_trades, per-seat terminal cash/survival time/trade counts); ALL scripted/
selfplay/mixed cells 100%; failures isolated to gemini policy cells (12–16% —
API-throttled resumes restarted the torch stream; treat gemini-vs-policy prices as
directional, n=26) and deepseek_vs_ippo_s2 (93%). Non-validated games excluded.
**Headlines** (markup = price vs the game's per-food VWAP, within-game):
- **Vs policy populations every LLM buys dear AND sells cheap**: buy markups
  +24..+77% (deepseek worst), sell −2..−16%; incumbents mirror (buy −6..−17%,
  sell +8..+15%). **Vs scripted bots the same models trade near-fair (−11..+4%)**
  — expropriation is price-mediated and specific to TRAINED incumbents.
- **Crisis pricing**: last-10-ticks-before-death LLM buy markups +27..+72% vs
  policies.
- **Wealth transfer**: LLM seat runs a net trade deficit ~$40–100/game in every
  population; vs bots it is food-at-fair-prices, vs policies the same deficit is
  price-amplified — the channel behind E24's $0.5–5 terminal cash. Mixed 4-LLM
  economy: claude net-pays gemini $58.40/game (reproduces E24's verdict as flow).
- **Stripping→starvation**: death-aligned cash decays monotonically into death
  (vs ippo_s0: claude $28.6→$12.6, gpt $24.6→$7.8, deepseek $70.3→$23.7 over the
  final 60 ticks) — starvation is the endpoint of expropriation (E24's late-death
  signature, now with mechanism).
Data `workshops/econml/results/llm_economy/` (validation.csv, terms_of_trade.csv,
wealth_transfer_matrix.csv, stripping_trajectories.csv, FINDINGS.md); scripts
`econml/replay_llm_economy.py`, `econml/analyze_llm_economy.py`.

## E27. Leaderboards as social choice — Condorcet cycle at the top, rule-dependent champion [EconML theme 1; TAE crossover; Aug 29, offline]
Each evaluation population (E12's judges) and each metric (E20's six) treated as a
voter over the healthy-8 policies (1000-game matrix; B=2000 paired bootstrap, rng
20260830). **No strict Condorcet winner in any voter set.** Healthy-8 judges: a
genuine **Condorcet paradox among the top three** — ippo_s1, ippo_s2, mappo_s3 each
majority-beat all five others but cycle (s2>s1 4–2, ms3>s2 4–2, s1>ms3 4–2), and
the champion is a pure function of the aggregation rule: **mean-score/Copeland/
plurality → ippo_s1; Borda → ippo_s2; Kemeny → mappo_s3** — three champions from
identical ballots. All-10 judges: ippo_s2 is a weak Condorcet winner. Metric-voters:
0 cycles, co-winners ippo_s2/mappo_s2 (E20 as an electoral outcome).
**Manipulability**: 5/8 healthy policies are crownable #1 by choosing a single
defensible judge population (crowning judge named per policy in crownable.csv).
**Rule stability (paired bootstrap)**: the status-quo mean-score is the MOST stable
on its own terms (top-1 96% healthy-8 / 90% all-10); Kemeny 91% healthy-8 but 41%
all-10; plurality 38% all-10. Validation: mean-score leaderboard reproduces
`leaderboard_stability_1000` exactly (diff 0.0); crownability ranks match
`population_sensitivity_1000` for all 10 (mappo_s2 spans 1–9). Data
`workshops/econml/results/social_choice/`; script `econml/analyze_social_choice.py`.

## E28. Nash averaging crowns the spoiler and is LESS stable than the mean [EconML theme 1; TAE; Aug 29, offline]
Nash averaging (Balduzzi 2018) on the 1000-game meta-game M[i,j] (focal i vs 3×j;
A=(M−Mᵀ)/2 antisymmetrized — caveat documented: 1v3 payoffs are not a symmetric
2-player game; maxent Nash via LP+SLSQP, equilibrium residuals ≤2.4e-13).
- **Survival, all-10: unique pure Nash = mappo_s1 — the marginal SPOILER** (9th of
  10 by mean, 23.1% survival) is crowned because healthy policies survive only
  0.5–16.3% vs 3×mappo_s1; mean-#1 ippo_s2 ranks 8th under Nash. The verdict holds
  in 2000/2000 bootstrap replicates: "confidently, stably wrong."
- Survival, healthy-8: mixed Nash {ippo_s3 .775, mappo_s0 .123, ippo_s1 .102} — an
  empirical RPS triple. Win channel (near-constant-sum): pure Nash = ippo_s2,
  agreeing with the mean.
- **Stability, identical paired bootstrap (B=2000, rng 20260831): Nash averaging is
  LESS stable than the macro-mean** (survival pairwise flips 5.3/5.7% vs 1.8/3.0%;
  adjacent flips 18.4/14.9% vs 7.7/9.9%). An economic solution concept is not a
  free fix — it answers a different question (beatability) than the mean (typical
  performance), and inherits the spoiler.
- Replicator dynamics (healthy-8): raw M converges to pure ippo_s1 monoculture;
  antisymmetrized A cycles forever (tail amplitude 0.87, period ≈1450) around the
  Nash triple — E10a's non-transitivity as a literal economic cycle.
Validation: macro-mean scores match E19 refs to 1e-16. Data
`workshops/econml/results/nash_averaging/`; script `econml/analyze_nash_averaging.py`.

## E29. Welfare accounting — dead wealth triples with strangers; monocultures efficient but NOT equal [EconML theme 2; Aug 29, offline]
All-four-seats accounting over 104,500 games (145 cells; per-seat k = number of
co-players of a different spec). **Cash conservation verified EXACTLY: total
terminal cash = 400.0 in every game** — cash is purely redistributive; survival is
the aggregate output. E18 reconciliation exact at the 100-game budget
(92.2→87.3→80.7→74.0 healthy pairs).
- **Dead-wealth share** (cash held by dead seats), healthy pairs, k=0→3:
  **10.4 → 15.9 → 23.2 → 32.2%** (all pairs 22.3→58.1%). Dead/alive mean-cash
  ratio >1 at every k (1.23→1.30): rich-and-dead is structural, present even in
  healthy monocultures. Worst cells ~91.7% of the economy's cash held by the dead.
- **Inequality**: healthy self-play per-game Gini 0.411 vs 1v3 0.394 vs 2+2 0.386
  — **monocultures are high-welfare but NOT low-Gini (design expectation
  refuted)**. Gini explodes only with a degenerate seed (healthy-vs-sick mean
  0.588, max 0.708); corr(survival, Gini) −0.73 overall but +0.10 healthy-only.
- **Volume falls slower than survival** (k=0→3 normalized: 1.00→0.87 vs
  1.00→0.80) — the market does not thin before it kills.
Data `workshops/econml/results/welfare/` (+ welfare_frontier.pdf, 145-cell
survival-vs-Gini frontier); script `econml/analyze_welfare.py`.

## E30. Ladder microstructure — roles invert vs design labels [EconML supporting; Aug 29, local]
Tournament re-run reproduces the published E2 numbers **byte-identically** (80/80
metrics), now with full ledgers (8,410 trades; maker/taker attribution complete).
Four-way small_30: the market is take-dominated (88.5% TAKE) and thin (two-sided
book 37% of food-ticks). **The nominal spread-earner v2 earns no realized spread**
(sell−buy −$0.06/unit, 60% taker, net −$13.0/seat-game); **the position-taker v4
is the de-facto maker-extractor** (78.4% maker share, +$0.55/unit carry, +$30.2/
seat-game — the only big extractor). v5: pure taker (7.5% maker), largest per-unit
edge (+$4.22) on tiny volume; v3's 60.8% maker share is adverse selection
(−$2.05/unit). Pairwise flows: v5 farms v3 $75.5/game (matches 45–5); v4 farms v2
$20.3 and v3 $32.6; but v5 extracts from v2 yet LOSES the pairing 22–28, and v4
beats v5 34–17 at ~zero net flow — **non-transitivity is not a single cash-flow
cycle**. Price formation is lineup-dependent: v2v4 economy 5.88±1.40 vs v3v5
11.77±8.25 (2× level, 6× dispersion). Data
`workshops/econml/results/ladder_microstructure/`; script
`econml/scripted_tournament_ledger.py`.

## E31. Monoculture vs mixed-economy microstructure — conventions, not collusion [EconML theme 2; Aug 29, HPC job 1896332]
32 peer cells × 100 games (seed base 900000 = the published matrix's paired games)
re-run with full ledger (`HG_LEDGER=1`). **Validation: all 26 cells with a base
twin reproduce game-for-game** (26/26); the 6 mixed 2+2 cells are new.
- **Seed-specific price conventions**: monoculture VWAP spans 2.06 (mappo_s2) to
  5.02 (mappo_s1) — 2.4× across co-adapted economies differing only in seed;
  failed mappo_s4 is a scarcity economy (VWAP 32.8, volume 38 vs 164–248).
- **Mixing breaks conventions**: volume falls in 20/22 mixed cells (median ratio
  0.90 vs constituent monocultures); catastrophic pairings collapse trade
  (mappo_s2 vs 3×mappo_s1: 0.449; ippo_s0 vs 3×mappo_s4: 0.467).
- **Strangers pay worse prices**: cross-type buy markups exceed within-type in
  16/22 cells, often ~10×: +24.4% vs +2.2% (ippo_s1 economy), +17.9% vs +1.0%
  (ippo_s4), +75.8% vs +0.4% (mappo_s4-majority) — the price face of E18's
  k-dose response.
- **Dominance/collapse as measured flows**: ippo_s4 net-pays ippo_s2 $39–64/game
  across k (the E10a 33↔4 win-channel pair as wealth transfer); mappo_s1 seats
  net-pay the dying mappo_s2 $178–185/game (rich-and-dead as FLOW: the cash
  arrives, the food does not); mappo_s4 net-pays a visiting ippo_s0 $230/game.
**Framing (exact)**: co-adaptation is a tacit trading convention, not collusion in
the antitrust sense — monocultures clear MORE volume at seed-specific prices, and
entry thins the market and worsens the entrant's terms of trade. Data
`workshops/econml/results/microstructure_hpc/`; harness
`research/hpc/econml_microstructure/`; analysis `econml/analyze_microstructure.py`.

## EconML follow-ups (Aug 29, second pass — all offline/local, $0)

**E26b. Maker attribution — LLMs cross the spread into incumbent quotes** [mechanism
behind E26]. Order-registry attribution over the validated replays (coverage
91–99%/cell): vs policies, LLMs are TAKERS on 74–84% of their buys (claude ~50%) at
+29..+95% markup — they lift expensive incumbent asks — and takers on 67–88% of sells
at −5..−18% — they hit low incumbent bids. Vs bots the identical taking behavior meets
fair books (~0%). Gemini's rare maker buys are pathological (+132..+156% posted bids).
Producer confound dismissed: forced-import buys (buyer cannot produce the food) still
carry +24..+69% vs policies (`producer_strata.csv`). Data
`results/llm_economy/maker_attribution.csv`; script `econml/analyze_maker_attribution.py`.

**E26c. Discrimination control — quote traps, not bespoke discrimination** [local
ledger runs: 4 arena-policy monocultures + 2 policy-vs-v2, small_90, seed base
910000]. The overpriced asks exist at HOME too (monoculture best asks 13.5–15.8 vs
executed VWAP 3.69–4.28): co-adapted peers don't lift them, LLM entrants do, paying
1.5–2.9× the home VWAP while aggregate prices move only +10–30%. Vs bots the same
policies quote tight (best ask ~5.0). Mechanism: a standing screen incumbents' own
kind reads and strangers cannot — adverse selection against naive entrants. Caveat:
within-game state-dependent quoting not fully ruled out. Data + FINDINGS:
`results/policy_small90_ledger/`.

**E26d. Deficit decomposition — the deficit is food vs bots, PRICE vs policies**
[unifies E26/E24]. LLM net outflow split into volume (at game VWAP) + price
components: vs scripted $79.0/game = $74.1 volume + $4.9 price (**price share
6.2%**); vs policies $87.0/game = −$1.2 volume + $88.1 price (**price share 101%** —
at fair prices the LLM trade pattern would net-RECEIVE cash; 7/14 cells negative
volume component). Exception: claude_vs_v2 price share 47.8%. Validation: cash
conservation exact in 3,070/3,070 replayed games; replay terminal cash = campaign
records in 12,280/12,280 seat comparisons. LLM welfare on E29's frontier: all 34
cells in/beyond the healthy cluster, never on the sick-seed arm (gemini/gpt-vs-bots
reach Gini 0.24–0.32 at 98–100% survival). Data `results/llm_welfare/`
(deficit_decomposition.csv, unified_frontier.pdf); script `econml/analyze_llm_welfare.py`.

**E27b. Full-portfolio electorate — the paradox dissolves, and crowns the mean's
7th** [extends E27]. All 60 (population×metric) voters over the healthy 8: a STRICT
Condorcet winner exists — **mappo_s2** (27–20 vs runner-up, 0 cycles) — and Borda,
Copeland, plurality, and Kemeny ALL crown it, though the survival mean ranks it 7th
of 8 (breadth across cash/win channels beats depth on survival). 2-D crownability
7/8 (only ippo_s3 never crownable). Bootstrap: Copeland/Kemeny 100% top-1, plurality
99.95%, Borda 60% (flips to ippo_s2). Cross-check: grid restricted to survival
reproduces E27 exactly. Data `results/social_choice/grid_*`; script
`econml/analyze_social_choice_grid.py`.

**E28b. Alpha-rank sides with Nash — the spoiler verdict is structural** [extends
E28]. Single-population alpha-rank (m=50, alpha sweep 0.01→∞; RPS/dominance
self-validated): survival all-10 crowns **mappo_s1 (mass 1.0, unique sink) — the
same spoiler as Nash** (mean rho vs alpha-rank −0.41); healthy-8 reproduces the
Nash support (rho 0.94, mean-#1 ippo_s2 ranks 5th); win channel: all three concepts
agree (ippo_s2). Structural reason: Fermi fixation depends on M only through
M[i,j]−M[j,i] — exactly Nash averaging's antisymmetrization — so mutual-destruction-
reads-as-skill transfers wholesale. Stability is bimodal: degenerately perfect where
a robust sink exists ("stably wrong"), worst-of-three where contested (surv/healthy-8
mean flip 16.6% vs mean's 3.0%). The mean leaderboard stays the most stable; the two
economic concepts add agreement with each other, not reliability. Data
`results/alpha_rank/`; script `econml/analyze_alpha_rank.py`.

**E31b. Price dose-response — the singleton premium** [extends E31/E18]. Per-seat k
over all E31 cells: buy markup +7.2 → +5.5 → +7.0 → **+13.4%** at k=0..3
(healthy-only; CIs disjoint for the k=3 jump) — unlike E18's graded survival curve,
the PRICE penalty is convex: majority seats (k=1) pay nothing extra, the k=3
singleton pays +6–8pp, buy-side only. Survives the producer control (forced-import
stratum: −3.7 → +2.8%, monotone from k=1). k=0 sanity: volume-weighted markup 0 to
machine precision; the +7% trade-weighted offset is small-lot own-produce buys
(+92% markup, 10.4% of trades — flat in k, cannot manufacture the curve). Bonus:
a structural reseller premium (+63–83% selling food you don't produce, all k). Data
`results/microstructure_hpc/price_dose_response.{csv,pdf}`, `producer_confound.csv`;
script `econml/analyze_price_dose_response.py`.

---

# TAE camera-ready block (E32–E36) — added Nov 9, 2026 in response to the TAE reviews

Accepted at TAE (NeurIPS 2026 workshop), poster. The camera-ready added, in response to the
reviews: (1) a declared-target audit that names the deployment archetype before any verdict is
called a degradation; (2) a convergence assessment of the generated exploiters, with the stress-test
diagnosis narrowed accordingly; (3) explicit joint-power and familywise statistics behind the
"certifying the ranking" budget and the metric-inversion count; (4) a separate evaluation stage for
the selection-regret estimate; plus a reproducibility appendix and fixed cross-references.

## E32. Declared-target audit — the 700→1200 verdict is a property of the target [TAE §3, Table tab:targets; Nov 9, offline]
Same E11 data, every target scored: E[r] = P(win)+P(surv)−1 per cell. 700→1200, 8 healthy
runs (±2 pts / ±0.02 return band): **v4 survival 36.8→17.0 (1 up/7 down), v4 win 9.4→6.8
(2/5), v4 return −0.54→−0.76 (1/6) = decline**; v3 survival 73.0→94.1 (7/1), v3 return
−0.19→+0.26 (8/0) = improvement; v2/v5 survival mixed (3/3, 3/1); **uniform-mixture survival
36.9→38.4 yet 3 up/5 down; mixture return −0.57→−0.48 (6/2) = improvement**; self-play
survival 73.6→90.3 (7/0), self-play return (P(win)=1/4 by symmetry) −0.01→+0.15 (7/0).
Framing adopted everywhere: *target-specific degradation, not general instrument invalidity*;
"ground truth" → "reference measurements" + a "Declared targets" paragraph in §2.
Data `results/objective_targets/`, script `eval/analyze_objective_targets.py`.

## E33. Certification budget made explicit — joint power, familywise, gap uncertainty [TAE §5 + App. stats; Nov 9, offline]
Per adjacent pair k: n_k = n0((z.975+z.8)·sd_k/|d_k|)², sd from the paired-seed bootstrap at
n0=1000 (B=2000, seed 20260829), budget = max_k n_k = **18,712** (binding mappo_s3>ippo_s4, gap
0.47±0.72). Joint power (all 7 adjacent verdicts correct & significant; MVN with bootstrap
covariance) at that budget = **68%**; 80% joint power needs **22,789**; Bonferroni(7) per-pair
budget **29,736**, 80% joint under Bonferroni **34,288**. Gap uncertainty: budget 2,891 at
gap+1SE, infinite at gap−1SE; recomputed on replicate gaps: median 82,520, 5th/95th pct
8,324 / 8.25M, 5.5% of replicates >10M. Data `results/evaluation_budget_sensitivity/`, script
`eval/analyze_evaluation_budget_sensitivity.py`.

## E34. Strict inversions under familywise control [TAE §5 + App. stats; Nov 9, offline]
Family = 28 healthy pairs × 6 peer metrics = 168 tests; normal-approx p from paired-bootstrap
SD (B=4000, seed 20260828). Strict inversions: 15 per-test 95% (reproduces E20), 14 at 99%,
**14 under Bonferroni(168) and Holm(168)** — the same 14 as at 99%; the casualty is
ippo_s2 vs ippo_s4. Data `results/inversions_familywise/`, script
`eval/analyze_inversions_familywise.py`.

## E35. Exploiter convergence — the win channel has not plateaued [TAE §6 + App. conv, Figure f8; Nov 9, offline]
iters.jsonl of the 8 conservative exploiters, 1201–1800. Pooled win rate 26.2 (1201–1300) →
30.6 (1401–1500) → 32.5 (1501–1600) → 35.7 (1701–1800); OLS slope over 1501–1800 **+1.60±0.18
pts/100 it (pooled)**, significantly positive in **6/8** runs (ippo_s2 −1.7, mappo_s3 −1.0 n.s.);
exploiter's own survival flat ~91%. mappo_s0 continuation stalled @1653. Diagnosis in the paper
narrowed to "ranking of the tested stressors at the tested budgets; adversarial fragility not
excluded; no survival-targeted adversary trained". Figure `figures/f8_exploiter_convergence`,
data `results/exploiter_convergence/`, script `eval/analyze_exploiter_convergence.py`.

## E36. Selection regret with a separate evaluation stage [TAE §7; Nov 9, HPC job 6552675, commit 5701f27]
Fresh-seed re-evaluation of ALL 12 checkpoints of the 10 best-config default_180_mlp runs vs
3×v4: 500 games/ckpt, seeds 950000–950499 (disjoint from the 900000–900099 selection games);
harness `research/hpc/best_config_eval/run_array_scripted_fresh_v4.sbatch` (+ `merge_fresh_v4.sh`),
campaign root `~/hunger_runs/best_config_fresh_eval_v4_500/`. Selectors pick on the original
100 games; picks are valued on the fresh 500. Script `eval/analyze_nested_selection_regret.py`,
data `results/best_config_fresh_eval_v4_500/` + `results/nested_selection_regret/`.
RESULTS (Nov 9, 120/120 tasks COMPLETED, ~9.5 min each): oracle-A pick valued on fresh games
**40.9%** (in-sample 43.6 → optimism only **2.7 pts**); nested regret (B value of oracle-A pick
minus B value of rule pick, mean±SE over 8 runs): **final 24.2±5.3 (7/8 runs >0)** [in-sample
26.6], self-play 18.5±5.0 (7/8) [21.8], cash 22.0±5.0 (7/8) [23.6], held-out portfolio
15.7±4.4 (8/8) [17.8]. Ordering + directions unchanged; magnitudes shrink 2–3 pts → paper now
quotes **16–24 pts** (was 18–27). B-oracle (argmax on B, in-sample at n=500) beats the oracle-A
pick by only 1.05 pts. Fresh-seed replication of F1: peak ckpt 600 (35.9) vs 700 on A; 700→1200
= 35.0→16.7 (6/8 down, 2/8 up); 7/8 decline >5 pts from their B peak; A vs B over 96 cells
Pearson 0.898, mean |A−B| 3.95 pts (≈ binomial noise at n=100), mean A−B 0.09 pts (no offset).
Trajectory B: [19.4, 21.1, 26.6, 29.1, 32.5, 35.9, 35.0, 25.6, 26.2, 26.8, 22.7, 16.7].

---

# Who-Verifies camera-ready block (E37–E39) — added Nov 9, 2026 in response to the WV reviews
Accepted (poster). The camera-ready added the three analyses below in response to the reviews.

## E37. Headline robustness — CIs, 36.8 vs 43.6 reconciliation, all ten runs [WV §3.1 + App. robust; Nov 9, offline]
Script `eval/analyze_camera_ready_headline_ci.py`, data `results/camera_ready_wv/headline_ci/`
(same E11 CSVs as F1/Table 2). 95% t-intervals over runs.
- Healthy 8: v4 survival 700→1200 = 36.75→17.0, paired Δ **−19.75 [−33.3, −6.2]**, 7/8 down
  (ippo_s4 +7); self-play Δ +16.75 [6.2, 27.3], 7/8 up. Per-checkpoint v4 means with CIs:
  100: 19.5 [16.6,22.4] … 700: 36.75 [28.8,44.7] … 1200: 17.0 [6.2,27.8].
- **Reconciliation** (camera-ready request): 36.8 = fixed-checkpoint mean at ckpt 700; 43.6 = mean of
  each run's OWN peak (per-run argmax; peaks at 400,500,400,1100,1200,700,600,500), which
  necessarily exceeds every fixed-checkpoint mean; 21.9 = mean v4 survival at each run's
  self-play-selected checkpoint (picks 900–1200); 17.0 = mean at 1200.
- **All ten runs** (camera-ready request; adds marginal mappo_s1 = 58% self-play and failed mappo_s4 =
  0%): v4 32.1→17.8, Δ −14.3 [−27.8, −0.8], 7/10 down (mappo_s1 +17 from a 2% base, mappo_s4
  −2). Table 2 verdicts unchanged except "cross-play win vs v4" (5/10 down → silent) and the
  pooled portfolio (4 up/5 down → silent instead of flags-per-run); regret (in-sample oracle):
  last ckpt 23.9 [12.3,35.5], self-play 20.0, cash 19.5, v4-free portfolio 15.6.

## E38. Held-out seed block — Table 2 replication + two-fold cross-validated regret [WV §3.1, §4, App. F; Nov 9, HPC]
Campaign `research/hpc/camera_ready_heldout` Part A (job 6553393, 16 tasks, all COMPLETED; seed base
960000; 100 seat-balanced paired-seed games per cell; 10 runs × 12 ckpts × v2–v5). Data
`results/camera_ready_wv/_scripted_eval_100_heldout/`; analysis `eval/analyze_camera_ready_heldout.py`
→ `results/camera_ready_wv/analysis/` (ablation_blocks, trajectories_ci, regret_crossvalidated,
cell_agreement_A_vs_B). Original block = A (seed base 900000), held-out block = B (960000).
- Per-cell agreement A vs B: v4 survival r=0.86 over 96 healthy cells, mean B−A 0.0, SD 6.2 (binomial
  expectation 6.3 at n=100); v2 0.95, v3 0.99, v5 0.92. No systematic shift between blocks.
- Healthy 8 on B: v4 36.25→17.88 (run-mean peak 38.5 @600), paired Δ **−18.4 [−30.4, −6.4]**,
  6 down / 0 up by >2 pts. All 10 on B: 31.5→18.4, Δ −13.1 [−25.8, −0.4], 7/10 down.
- Table 2 on B: every certify verdict and both v4 flags replicate; v2 silent→flags (3/8→5/8 down,
  25.4→22.9); pooled portfolio flags-per-run→silent (5/8→4/8 down; mean 38.2→37.5, flat). All 10 on
  either block: win vs v4 and pooled silent (5/10 split), everything else as Table 2.
- Two-fold CV regret (pick on one block, score on the other, oracle included; folds averaged per run,
  t-CI over runs): ship last **24.5 [11.8, 37.3]** (in-sample 26.6); self-play **20.0 [7.6, 32.4]**
  (21.8); cash 22.6 (23.6); v4-free portfolio 15.4 (17.8); oracle deploys 41.9 vs 43.6 in-sample
  (≈2 pts of selection-on-noise). All 10: last 21.5, self-play 17.9, oracle 39.6. Consistent with the
  TAE E36 nested-selection estimate on a 500-game fresh v4 block (final 24.2, self-play 18.5, optimism 2.7).
Paper: §3.1 "replicates on a disjoint block"; §4 replication sentence; regret paragraph; abstract 24.5;
intro 20–25; Appendix F Tables 4–6.

## E39. Held-out learned populations — checkpoint trajectory vs never-audited peers [WV §3.2, Table 2 row, Limitations, App. F; Nov 9, HPC]
Part B (job 6553394, 30 tasks, all COMPLETED, longest 1h24m). Opponent populations: 3× ep_ippo_s2,
3× ep_mappo_s2, mixed ep_ippo_s0+ep_ippo_s4+ep_mappo_s1 (episode-matched default_180 @ iter_1200;
never used in any audit or selection). 10 runs × 12 ckpts × 100 games, seed base 960000; harness
`checkpoint_policy_crossplay.py` (1v3 and mixed modes). Data `results/camera_ready_wv/_heldout_peers_100/`;
analysis output `heldout_peer_populations.csv`; figure `figures/f6_wv_heldout_trajectories.{png,pdf}`.
- RESULT (opposite of the v4 trajectory): survival vs all three populations rises monotonically,
  ~0% @100 → **97.4 / 93.4 / 91.5 @1200** (healthy 8; 700→1200: 52.8→97.4, 42.1→93.4, 41.0→91.5;
  8/8 up each; Δ CIs [25.3,64.0], [34.6,67.9], [35.7,65.3]); win share 2–4% → 27 / 17 / 29% (chance
  25); mean cash ≈$9 → $103 / $71 / $98. Post-peak declines 0–5 pts in 23/24 run×population pairs
  (mappo_s3 vs mixed: 10). Spearman(run-mean traj, v4 traj A) = −0.01 / −0.01 / +0.02.
  All 10 runs: 42.2→80.6, 33.7→76.4, 32.8→75.6 (mappo_s4 stays at 0 throughout).
- Verdict under the Table 2 rule: **certifies** → added as a Table 2 row ("certifies (learned)");
  defensible-signal count in abstract and caption is now "six of nine" (was five of eight).
- Interpretation written into the paper: the regression is specific to the out-of-distribution
  storage-arbitrage style, not a uniform collapse (confirms §3.2 redistribution; v4 explicitly not
  ground truth); independently trained peers of the same recipe inherit the blind spot (they
  themselves sit at the configuration's 16% vs v4 at iter 1200), so "held-out" ≠ "independent
  verifier". Limitations and Discussion habit (1) rewritten accordingly.

### Correction to E37/E39 (Nov 10): excluded-run bases and "monotone" wording
- E37 said the marginal run mappo_s1 improves "+17 from a 2% base": 2% is its *self-play* survival at
  ckpt 700 (0% through 600, 58% at 1200). Its v4 survival is 25% @700 → 42% @1200 (block A; 20→40 on
  block B). The failed run mappo_s4 is 15–21% vs v4 through ckpt 600 and ends at 0% (block A) / 1% (B);
  it does not "stay near zero" throughout. Paper Appendix F states the corrected values.
- E39 "rises monotonically": true of the run means for 3× ep_ippo_s2 and 3× ep_mappo_s2; the mixed
  population's run mean dips 92.4→91.5 at the last checkpoint. Per run, post-peak declines are 0–5 pts
  in 23/24 run×population pairs (10 in one). Paper wording changed to "rises throughout training".

### Correction (Nov 11): full-precision intervals and tie handling
- E38: the two-fold CV regret interval for "ship last" is [11.7, 37.3] at full precision (11.7467, 37.2533);
  the earlier "[11.8, 37.3]" double-rounded the 2-decimal CSV value 11.75. Rule: quote prose numbers from
  full-precision values, never from already-rounded CSV columns.
- E37: the last checkpoint attains each run's self-play maximum in 4 of 8 healthy runs (mappo_s3 ties
  1100/1200 at 94%; the selector breaks ties toward the later checkpoint, consistent with the 21.9%
  figure); the paper said "3 of 8" (unique maxima) — changed to 4 of 8. The oracle's per-run v4 peaks
  (400, 500, 400, 1100, 1200, 700, 600, 500) have median 550, not the "median checkpoint 600" inherited
  from the submitted text — corrected. Failed run mappo_s4 vs v4 on block A over ckpts 100–600 is
  11–21% (not 15–21%).
