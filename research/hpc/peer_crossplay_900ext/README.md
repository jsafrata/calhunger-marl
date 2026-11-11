# peer_crossplay_900ext — seed extension of the best-config peer matrix (E19 follow-up)

Purpose: E19 (bootstrap stability) showed the 100-game peer leaderboard does not decide
its own ranking (healthy-8 top-1 identity stable in only 59% of resampled reruns; 15/28
pairwise verdicts flip in >=5% of replicates). This campaign buys the instability down
empirically: 900 additional games for every cell of the best-config peer cross-play
matrix, at seed base 900100 (seeds 900100–900999, disjoint from the existing 900000–
900099), for a combined 1000 paired games per cell.

Design: identical to `best_config_peer_crossplay` in every respect except n_games and
seed base — same 10 policies (IPPO/MAPPO x seeds 0–4, iter_1200, default_180_mlp
best-config arms), same full ordered 1v3 matrix (90 cells) + self-play diagonal (10
cells), same harness, stochastic sampling, seat rotation balanced (900 % 4 == 0). New
games write `*_stochastic_900.json` into a separate output dir
(`$HUNGER_CAMPAIGN_ROOT/_peer_crossplay_900ext/`), so nothing existing is touched;
the local analysis merges `_peer_crossplay_100` + `_peer_crossplay_900ext` on the
disjoint seed ranges.

Sharding: one Slurm array task per cell (array 0–99; focal = id/10, cell = id%10,
cell 0 = self-play diagonal, cells 1–9 = ordered 1v3 vs the other nine labels in label
order). ~900 games x ~2 s ≈ 30–45 min per task; walltime 2:30 h. Tasks write disjoint
raw files; concurrent rebuilds of crossplay_matrix.csv are benign (it is regenerated
from raw files; rerun any one job at the end for a clean rebuild). Cache-resumable:
a task whose raw JSON already exists (matching identity) exits immediately.

Deploy: standard DEPLOY.md flow; copy campaign.env.example -> campaign.env, set account
hippo. `bash plan.sh` prints the 100-task mapping for eyeball validation before
submitting. After completion, rsync `_peer_crossplay_900ext/` to
`workshops/results/seeds234/` and rerun the E19/E20 analyses on the merged 1000-game
set.
