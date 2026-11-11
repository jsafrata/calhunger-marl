# ladder_validity_dose_response — two IAEval gap evals on the best architecture

Two evaluation-only experiments on the same 10 policies as best_config_peer_crossplay
(default_180_mlp best-config arms, IPPO/MAPPO x seeds 0-4, iter_1200), sharing one
Slurm array. Both are analysis-ready joins against the existing
_peer_crossplay_100 matrix (same harness, same seed base 900000, same columns).

## Experiment A — ladder validity (does the scripted ladder proxy real peers?)

The IAEval question "do simulated interaction partners validly proxy real ones?"
transposed to MARL: does a policy's performance against the cheap scripted ladder
predict its performance against expensive independently trained peers?

Runs: each of the 10 policies as 1v3 focal vs scripted:v1..v5, 100 seat-balanced
games on paired seeds -> 50 jobs -> `_ladder_validity_100/`.

Analysis: per policy, x = survival (and win rate) vs each bot / vs mean(v2-v5);
y = mean off-diagonal peer survival from the existing `_peer_crossplay_100`
matrix (the `focal vs 3xPEER` rows). Report Spearman rank correlation across the
10 policies, per-bot and aggregate. Either sign is a finding: the ladder is a
valid cheap proxy, or even hand-built partners mismeasure social competence.

## Experiment B — population dose-response (how many strangers break a policy?)

How gradually does the peer-crossplay failure kick in as co-players change from
familiar to unfamiliar? For a focal policy F and stranger peer S, count k =
strangers among F's 3 co-players:

- k=0: 4xF self-play            — EXISTS (`F vs self` rows, mixed diagonal)
- k=1: lineup S,F,F,F           — EXISTS (the `F vs S` secondary rows of the
                                  1v3 runs where F was the 3x opponent)
- k=2: lineup F,F,S,S           — NEW: this campaign
- k=3: lineup F,S,S,S           — EXISTS (the `F vs 3xS` 1v3 rows)

Runs: all 45 unordered pairs {F,S}, mixed mode 2+2, 100 seat-balanced games each
-> 45 jobs -> `_peer_mix22_100/`. Each run yields the k=2 point for BOTH members
of the pair (rows `F vs 2xS` and `S vs 2xF`).

Analysis: per ordered (F,S) pair, survival at k=0,1,2,3 (n_focal_seat_games =
400/300/200/100); aggregate curve = mean over pairs (report healthy-8 subset
alongside all-10). Answers whether one stranger already collapses survival or
degradation is graded in k.

## Array layout

10 tasks (one per focal label, same order as best_config_peer_crossplay). Task t
runs its 5 ladder jobs, plus the pairs whose index in the fixed enumeration
(i<j, lexicographic) satisfies index % 10 == t (4-5 pairs/task) -> 9-10 jobs per
task, the same shape as the prior campaign (walltime 8h for slack; jobs are
cache-resumable, shared output dirs are safe because raw files are keyed by
lineup).

## Deploy

Standard DEPLOY.md flow (eval-only: no preflight/validate_plan; the harness
fail-fasts on missing checkpoints). Copy campaign.env.example -> campaign.env,
set account hippo. After completion, rsync `_ladder_validity_100/` and
`_peer_mix22_100/` to `workshops/results/seeds234/`.
