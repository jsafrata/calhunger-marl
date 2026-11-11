# Policy-vs-policy cross-play — findings (first full run, Aug 9 2026)

163 jobs, 100 games each, seat-balanced, seed base 900000, harness
`workshops/eval/checkpoint_policy_crossplay.py`. All numbers stochastic sampling unless noted.
One transient job failure (Ray init during a laptop sleep/wake) was rerun; matrices complete.

## 1. small 30 s (`small_2seed_partial` @ iter 1000, 4 policies)

All four checkpoints are healthy in self-play (84–92% survival, 25% wins by symmetry).
**The peer-generalization gap at short horizon lives in the competitive channel, not
survival**: mirrored pairings are strongly asymmetric — ippo_s0 vs 3×ippo_s1: 7% wins /
cash 57; ippo_s1 vs 3×ippo_s0: 41% wins / cash 136. Survival stays 67–95% everywhere.
Pecking order: ippo_s1 > mappo_s0 > ippo_s0 ≈ mappo_s1.

## 2. v3 180 s (`v3_shared_vs_independent` @ iter 1100 + shared_s1 @ 1400) — the headline

Figure: `workshops/figures/f3_policy_crossplay_v3_180s.{png,pdf}`.

- **Peer cross-play mirrors scripted cross-play**: shared_s1 falls from 75% survival / 24%
  wins (self-play) to **26% / 0%** vs 3× shared_s2 — same algorithm, same architecture,
  different seed. sharedid_s1: 66% self → 15% vs 3× sharedid_s0.
- **Bidirectional failure — weak partners kill you**: every policy seated with 3× the weak
  sharedid_s0 checkpoint (its own self-play: 27.5%) survives only 9–30% while holding
  $253–342 — the economy collapses under the strong policy regardless of its skill.
  Empirical Prop 1 with learned peers: there is no unilateral survival, so a policy's
  performance is a property of the *population*, not the policy.
- **The rich-and-dead checkpoint is adapted, not degraded**: shared_s1@1400 (which collapsed
  52.5%→9.5% vs Scripted-V1, fig F1) *dominates its own lineage* — 66% survival / 62% wins
  vs 3× shared_s1@1100 — while still failing against scripted opponents. Continued self-play
  training produced within-lineage improvement and out-of-lineage collapse simultaneously:
  the strongest possible form of the "adaptation, not general improvement" claim.

## 3. hpc_v2 180 s (`hpc_v2_5seed` @ iter 1500, 10 policies)

**Self-play diagonal first** (the protocol rule the smoke test forced): 9 of 10 final
checkpoints fail their *own* self-play on fresh seeds — survival ippo: 54.0 / 0.0 / 0.2 /
8.8 / 0.0; mappo: 0.0 / 19.0 / 31.5 / 0.2 / 0.2. Only ippo_s0 clears 50%, so no healthy
pair exists and the off-diagonal is uninterpretable as cross-play. This matrix therefore
documents the fixed-batch 180 s baseline's failure (consistent with the campaign's
1.17–1.53 mean survivors during training) and demonstrates *why* the diagonal is mandatory
in the protocol: without it, checkpoint weakness and cross-play incompatibility are
indistinguishable.

## 4. v3 conditions vs the scripted ladder, 180 s (`v3_vs_scripted_180s/`, 28 jobs)

- **No v3-condition policy ever beats the calibrated bots**: win rate ≤ 9% against every
  archetype (chance 25%), survival ≤ 76% best-case (vs v3-JIT), mostly ≤ 44%. The
  parameter-sharing axis (shared vs shared+ID) does not rescue 180 s generalization.
- **The rich-and-dead checkpoint's "collapse" is population-specific**: @1400 vs @1100
  survival is *better* against v3-JIT (76 vs 55) and v2-maker (27 vs 11) but *worse* against
  v4-stockpiler (23 vs 40) and v5-sniper (8 vs 14). The monotone 52.5→9.5 collapse curve
  (F1) was measured against Scripted-V1 specifically. Refinement for the paper: continued
  self-play training doesn't degrade uniformly — it *redistributes* competence across
  opponent types unpredictably, which is precisely why a single fixed opponent (or any
  single number) is an invalid verifier and a portfolio is required.
- **Population composition flips verdicts** (mixed one-of-each lineups): shared_s2 wins 3%
  with cash 28 in {v2,v3,v4} but **25% with cash 147** in {v2,v4,v5} — same policy, same
  seeds, 8× the win rate. Direct evidence for the TAE framing: evaluation conclusions are
  unstable in the choice of opponent population unless reported as a portfolio.

## Paper placement

- F3 (v3 matrix) joins F1/F2 as a lead result: the generalization claim now covers scripted
  opponents AND independently trained peers, in both directions of relative strength.
- The @1400-dominates-@1100 result becomes a panel or paragraph tied to F1.
- small30 asymmetry: one paragraph — the win/cash channel degrades before survival as
  horizon shrinks.
- hpc_v2 diagonal: methods/appendix — evidence for the diagonal rule + baseline failure.

## Raw locations

`workshops/results/policy_crossplay/{small30_iter1000, v3_180s_iter1100,
hpc_v2_180s_iter1500}/crossplay_matrix.csv` (+ raw/ caches; deterministic-mode rows included
for small30 fully and hpc_v2 self rows).
