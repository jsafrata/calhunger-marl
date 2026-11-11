# camera_ready_heldout — two eval-only campaigns for the Who-Verifies camera-ready

Reviewer-driven robustness checks for paper 302 (NeurIPS 2026 Workshop "Who
Verifies the Agents?"). Both parts are evaluation-only (no training), use the
existing harnesses unchanged, and write to a fresh campaign root.

## Part A — held-out game-seed block for the Figure 1 / Table 2 evaluation

Reviewers noted that the "oracle" checkpoint in the selection-regret analysis
is chosen and scored on the same 100 games (selection on noise biases the
oracle upward), and asked for confidence intervals on the n=8 headline.

Replays the longitudinal scripted evaluation of `hpc/best_config_eval` on a
**fresh seed base (960000)**: all 10 best-config `default_180_mlp` runs
(IPPO/MAPPO x seeds 0-4), checkpoints 100..1200, 100 seat-balanced games per
cell vs 3x each archetype v2..v5. Harness `eval.scaled_checkpoint_scripted_diagnostics`
with exactly the flags of `hpc/best_config_eval/run_array_scripted.sbatch`
except `--game-seed-base` and the output directory. 16 tasks (2 roots x 4
opponents x 2 algorithms), 24-36 cells per task, ~1-2 min per cell.

Output: `$HUNGER_CAMPAIGN_ROOT/_scripted_eval_100_heldout/<root>/<opponent>/<algorithm>/`
(`checkpoint_diagnostics.csv` + `raw/`). Analysis (laptop):
`workshops/eval/analyze_camera_ready_heldout.py` — selectors pick checkpoints on
the ORIGINAL block (base 900000), deployment survival is scored on THIS block;
the Table 2 verdicts are recomputed on this block as a replication.

## Part B — the checkpoint trajectory against a held-out learned population

Reviewers noted that the headline's magnitude is measured against the v4
archetype and asked for a reference population never trained or audited
against. The episode-matched `default_180` policies (same horizon, different
training configuration, iter_1200, healthy in self-play) were never used in
any audit or checkpoint selection. Each of the 10 best-config runs is scored
at all 12 checkpoints as the focal seat against three held-out populations:

| p | population | mode |
|---|---|---|
| 0 | 3x `ep_ippo_s2` (self-play 94.0%) | 1v3 |
| 1 | 3x `ep_mappo_s2` (self-play 94.5%) | 1v3 |
| 2 | mixed table `ep_ippo_s0` + `ep_mappo_s1` + `ep_ippo_s4` (88.5 / 91.0 / 88.0%) | mixed |

100 seat-balanced games per cell, seed base 960000. Harness:
`checkpoint_policy_crossplay.py` (copy of `hpc/best_config_peer_crossplay/`,
reads each checkpoint's declared observation contract, so best-config and
episode-matched policies encode correctly at the same table). 30 tasks (3
populations x 10 runs), 12 cells each, ~3 min per cell.

Output: `$HUNGER_CAMPAIGN_ROOT/_heldout_peers_100/{3x_ep_ippo_s2,3x_ep_mappo_s2,mixed_ep3}/`
(`crossplay_matrix.csv` rebuilt from `raw/` by every task; the laptop analysis
re-reads `raw/` directly, so a last-writer race on the CSV is harmless).

## Deploy

Standard `DEPLOY.md` flow (eval-only: no preflight/validate_plan; both
harnesses fail fast on missing checkpoints). `python hpc/camera_ready_heldout/plan.py`
prints the task mapping. Copy `campaign.env.example` -> `campaign.env`, set the
account, source it, smoke-test one tiny mixed-table job on the login node
(`--n-games 4`) to confirm the two observation contracts coexist, then
`bash hpc/camera_ready_heldout/submit_all.sh`.

After completion, rsync `_scripted_eval_100_heldout/` and `_heldout_peers_100/`
to `workshops/results/camera_ready_wv/` and run the analysis script.
