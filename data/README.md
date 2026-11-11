# The external data archive

The repository ships the per-cell summaries every analysis script reads (`workshops/results/`, 18 MB).
Everything larger lives in three tarballs, archived on Zenodo:

**DOI: 10.5281/zenodo.XXXXXXX** (placeholder until the record is published)

| archive | size | contents |
|---|---|---|
| `calhunger-results-v1.tar.gz` | 89 MB | the complete `workshops/results/` and `workshops/econml/results/` trees, i.e. the shipped summaries **plus** every raw per-game JSON (`raw/` directories, 8,400 files) and the full LLM-arena transcripts (`llm_arena/campaign_90s/*/games/`, one JSON per game with every prompt hash, raw response, parsed action, latency, and token count) |
| `calhunger-checkpoints-v1.tar.gz` | 1.93 GB | 203 RLlib checkpoints (see below) plus each run's training telemetry (`iters.jsonl`) |
| `calhunger-checkpoints-v1-small90.tar.gz` | 97 MB | the ten 90-second best-configuration policies (`small_90_mlp`, iteration 1200) that the LLM arena and the EconML controls seated |

Checksums (`*.sha256`) and file manifests (`*.MANIFEST.txt`) for all three are in this directory.

## Extracting

From the repository root:

```bash
# Raw per-game outputs: extracts in place under workshops/ (adds raw/ and games/ next to the summaries)
tar -xzf calhunger-results-v1.tar.gz -C workshops/

# Checkpoints: paths inside the tarballs are campaign-relative
mkdir -p checkpoints
tar -xzf calhunger-checkpoints-v1.tar.gz -C checkpoints/
tar -xzf calhunger-checkpoints-v1-small90.tar.gz -C checkpoints/
```

`checkpoints/` and the raw directories are git-ignored, so the working tree stays clean.

## What the checkpoint archive contains

Paths follow the cluster layout `<campaign>/<arm>/<algo>_seed<k>_headmlp/checkpoints/iter_NNNN`;
every checkpoint is a standard RLlib 2.53.0 checkpoint directory (load with `ray[rllib]==2.53.0`).

| campaign | arm | runs | checkpoints | role in the papers |
|---|---|---|---|---|
| `best_config_2seed_ippo_mappo_1200` | `default_180_mlp` | ippo/mappo × seeds 0,1 | all twelve (100 … 1200) | best configuration, seeds 0–1: Figure 1, the declared-target audit, selection regret, the exploiters' champions |
| `best_config_mlp_seeds234_ippo_mappo_1200` | `default_180_mlp` | ippo/mappo × seeds 2,3,4 | all twelve | best configuration, seeds 2–4 (the ten runs of the papers = these two campaigns together) |
| `episode_matched_5env_2seed_ippo_mappo_1200`, `..._seeds234_...` | `default_180`, `small_60`, `small_90`, `small_120`, `small_150` | ippo/mappo × seeds 0–4 | iteration 1200 | the horizon × population matrix; the 180 s runs are the peer population of the cross-play matrix and the held-out learned peers |
| `exploiter_vs_champion_180` | — | 8 champions | iterations 1500 and 1800 (= 300 and 600 exploiter iterations after the 1200-iteration warm start) | generated adversaries, conservative arm |
| `exploiter_vs_champion_180_aggr` | — | 8 champions | iteration 1500 | generated adversaries, aggressive (re-exploration) arm |
| `finetune_vs_ladder_180` | — | 8 runs | iteration 1500 | the fine-tuning capacity control |
| `bc_v4_clone` | — | — | `bc_actor.pt` + reports | the behaviour-cloning control (its 19 GB of cloning datasets are not included) |
| `best_config_*/small_90_mlp` (the `-small90` tarball) | `small_90_mlp` | ippo/mappo × seeds 0–4 | iteration 1200 | policy seats of the LLM arena; EconML policy-economy controls |

The LLM-arena identity blocks and `workshops/econml/run_policy_small90_controls.sh` refer to the 90 s
policies under a single local name, `best_config_small90_mlp/<run>/checkpoints/iter_1200`. Recreate
that layout with symlinks after extracting the `-small90` tarball:

```bash
mkdir -p checkpoints/best_config_small90_mlp
for r in checkpoints/best_config_2seed_ippo_mappo_1200/small_90_mlp/*_headmlp \
         checkpoints/best_config_mlp_seeds234_ippo_mappo_1200/small_90_mlp/*_headmlp; do
  ln -s "$(cd "$r" && pwd)" "checkpoints/best_config_small90_mlp/$(basename "$r")"
done
```

## Pointing the harnesses at the checkpoints

- The checkpoint harnesses take `--campaign-root <dir containing the run directories>`, e.g.
  `checkpoints/best_config_2seed_ippo_mappo_1200/default_180_mlp`, with `--algorithms` and `--seeds`
  selecting runs by name (`ippo_seed0_headmlp`, ...). The Slurm scripts under `research/hpc/` show
  every invocation.
- The cross-play harness and the LLM arena take explicit seat specs:
  `policy:LABEL=checkpoints/.../checkpoints/iter_1200`.
- The plain runners take `--agents policy:NAME --checkpoint NAME=<checkpoint dir>`.

Every number in the papers was produced from the summaries that are already in the repository or
from the raw games in `calhunger-results-v1.tar.gz`; the checkpoints are provided so that the
evaluations themselves can be re-run or extended (new opponents, new budgets, new seeds).
