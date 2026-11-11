# CalHunger

Code, result summaries, and analysis scripts behind two NeurIPS 2026 workshop papers that audit
multi-agent evaluation practice in **CalHunger**, a four-player market-survival game in which no
agent can survive without repeatedly trading with its competitors:

- **When the Metric Lies: A Full-Stack Audit of a Multi-Agent Evaluation Protocol.**
  Jan Safrata. NeurIPS 2026 Workshop on Trust-AI-Eval (TAE): Can We Trust AI Evaluation?
  [`papers/2026_NeurIPS_TAE_When_the_Metric_Lies.pdf`](papers/2026_NeurIPS_TAE_When_the_Metric_Lies.pdf)
- **Self-Play Is a Broken Verifier: Measuring, Auditing, and Red-Teaming Verification Signals in a
  Forced-Interdependence Market Game.** Jan Safrata. NeurIPS 2026 Workshop: Who Verifies the Agents?
  [`papers/2026_NeurIPS_WhoVerifies_Self_Play_Is_a_Broken_Verifier.pdf`](papers/2026_NeurIPS_WhoVerifies_Self_Play_Is_a_Broken_Verifier.pdf)

Both are non-archival workshop papers (posters). Everything needed to re-run every number in them is
either in this repository (code, per-cell summaries, figures) or in the accompanying data archive
(raw per-game outputs and trained checkpoints; see [`data/README.md`](data/README.md)).

## The game in one paragraph

Four players, four foods (A, B, C, D). Each player produces two foods and must eat the other two, and
no player produces what it eats, so required food can only be bought from a competitor. Every 10
simulated seconds is a feeding: a living player first gains 2 units of each food it produces, then must
eat 2 units of each food it requires; a player whose *available* stock (inventory minus units locked
in its own open asks) of either required food is below 2 at that moment dies. Trading happens in one
limit order book per food with price-time priority and maker-price execution (actions: `wait`,
`post`, `take`, `cancel`; prices 1..50, quantities 1..5, 2,049 discrete actions, legality mask).
Players start with $100 and 100 food units (at least 20 per food). The game ends after T=180 seconds
(variants: 30, 60, 90, 120, 150 s with rescaled endowments); dead players lose, the richest survivor
wins (ties broken by total food), a sole survivor wins regardless of cash. Reward is terminal only:
−1 die, 0 survive, +1 win. Autarky is provably fatal (the papers' Proposition 1), so the co-player
population can never be factored out of what an evaluation measures.

The code base calls the game by its internal name, *Nash Hunger* (`NashHungerMultiAgentEnv`,
"Run one Nash Hunger game"); CalHunger is the same game.

## What is in this repository

```
research/            The simulator, agents, RL stack, evaluation harnesses, tests, and Slurm campaigns
  env/               Game logic (config, order books, feeding/death, actions + legality mask, observations)
  env/rl/            RLlib wrapper, observation encoders, factored masked actor, PPO config (IPPO / MAPPO)
  env/small_env.py   Duration-scaled variants (30/60/90/120/150 s)
  agents/            Scripted archetypes v1–v5 (the calibrated ladder), cadence wrapper, legacy bots
  eval/              Training entry point (long_baseline.py), checkpoint evaluation harnesses, plots
  hpc/               One directory per Slurm campaign: locked campaign.json, preflight, sbatch, runbook
  tests/             pytest suite (simulator invariants, encoders, RL module, harnesses, campaign plans)
  bugs/              Minimal reproductions of simulator bugs found and fixed during development
workshops/           The analysis layer that produced every paper number and figure
  eval/              Analysis + plotting scripts (peer cross-play harness, bootstrap audits, regret, ...)
  llm_arena/         LLM-seat extension of the harness (strict JSON wire protocol, frozen prompt v3)
  econml/            Market-microstructure and social-choice analyses (EconML study of the same data)
  results/           Per-cell summaries (CSV/JSON/MD) of every evaluation campaign; raw games in the archive
  figures/           Every figure used in the papers, as PNG + PDF
  paper/EXPERIMENTS.md  The experiment ledger: one entry per experiment with its data, script, and numbers
papers/              Camera-ready PDFs of the two workshop papers
data/README.md       The external data archive: raw per-game outputs and trained checkpoints
```

The two top-level directories are deliberately separate: `research/` is the frozen game and training
stack; `workshops/` imports it read-only (via `PYTHONPATH=research`) and never modifies it.

## Installation

Python 3.11 or 3.12. The simulator and scripted agents need only NumPy; the RL stack is pinned to the
versions that produced every released checkpoint (RLlib checkpoints load only with
`ray[rllib]==2.53.0`).

```bash
git clone https://github.com/jsafrata/calhunger.git
cd calhunger
python3 -m venv .venv && source .venv/bin/activate

pip install -e './research[dev]'     # simulator + scripted agents + pytest (NumPy only)
pip install -r requirements.txt      # + RLlib 2.53.0, PyTorch 2.12 (CPU wheel), SciPy, matplotlib, pandas
```

Linux pulls the CPU-only PyTorch wheel from the extra index declared in `requirements.txt`; macOS
uses the default wheel. Nothing here needs a GPU: every learned policy in the papers was trained on
6 CPU cores.

## Quickstart

```bash
cd research

# One game: three storage traders (v4, the hardest archetype) and one market maker (v2)
python run_game.py --seed 42 --agents scripted_v4,scripted_v4,scripted_v4,scripted_v2

# Watch a game tick by tick in the terminal
python watch_game.py --seed 7 --agents scripted_v4,scripted_v3,scripted_v2,scripted_v5 --delay 0.05

# Many games, per-seat statistics (seat balance is the caller's job; see the harnesses below)
python run_batch.py --agents scripted_v2,scripted_v3,scripted_v4,scripted_v5 --num_games 200

# Seat a trained checkpoint (from the checkpoint archive) against three scripted copies
HG_MAX_QTY=5 HG_MAX_PRICE=50 python run_game.py --seed 42 --stochastic-policy \
  --agents policy:a,scripted_v4,scripted_v4,scripted_v4 \
  --checkpoint a=../checkpoints/best_config_2seed_ippo_mappo_1200/default_180_mlp/ippo_seed0_headmlp/checkpoints/iter_0700

# Tests (simulator invariants, encoders, RL module, harness plans; the RLlib ones take a few minutes)
pytest -q
```

`HG_MAX_QTY=5 HG_MAX_PRICE=50` selects the action space every checkpoint was trained with (they are
also the defaults); set them before any RL import. `HUNGER_ENVIRONMENT_VARIANT` selects the horizon
variant for the checkpoint harnesses (`default` = 180 s, `small` = 30 s, `small_60` ... `small_150`).

## The scripted ladder

Five fixed archetypes with a calibrated, non-transitive skill ordering (v4 > v5 ≈ v2 > v3 ≫ v1),
all played through the same decision cadence (`agents/cadenced_scripted.py`):

| name | archetype | behaviour |
|---|---|---|
| `scripted_v1` | frozen reference | port of the original live-game bot: reactive, post-only, tiered on inventory |
| `scripted_v2` | market maker | two-sided quotes around an inventory-skewed reservation price (Avellaneda–Stoikov shape) |
| `scripted_v3` | just-in-time procurer | buys at prices that escalate as its feeding deadline approaches; passive otherwise |
| `scripted_v4` | storage trader | front-loads deep safety stock and liquidates into late-game scarcity (the deployment target of the papers) |
| `scripted_v5` | edge-only taker | waits by default, trades almost only by crossing the spread at displaced prices |

`research/eval/scripted_version_tournament.py` reproduces the calibration tournament (Appendix A of both
papers).

## Training

Every learned policy is PPO in RLlib (one policy shared by the four seats; IPPO uses the local
observation for the critic, MAPPO a centralized one), a 2×256 LayerNorm/SiLU MLP torso, factored
masked action heads, and a terminal-only reward with γ = λ = 1. `research/eval/long_baseline.py` is the
training entry point; the Slurm campaigns under `research/hpc/<campaign>/` wrap it with a locked
`campaign.json` (every hyperparameter and library version), a preflight check, an array script, and a
runbook. The two configurations used in the papers:

| configuration | campaign directories | what differs |
|---|---|---|
| episode-matched (horizon sweep) | `hpc/episode_matched_5env_2seed_cpu`, `hpc/episode_matched_5env_seeds234_cpu` | 45 complete games per update, lr 3e-4, entropy 0.01, constant |
| best configuration (longitudinal) | `hpc/best_config_2seed_cpu`, `hpc/best_config_mlp_seeds234_cpu` | + survival features, trade-history critic input, entropy 0.01→0.001 and lr 3e-4→1e-4 annealing |

A local smoke run of the same code path (two iterations on the 30 s variant, one CPU runner):

```bash
cd research
HG_MAX_QTY=5 HG_MAX_PRICE=50 python -m eval.long_baseline --algos ippo --env_variant small \
  --iters 2 --games_per_update 4 --num_env_runners 1 --checkpoint_every 1 --out_root /tmp/calhunger_smoke
```

`research/hpc/DEPLOY.md` is the deployment checklist used for every campaign (bundle, verify, preflight,
submit); the cluster-specific values (account, partition, paths) live in each campaign's
`campaign.env.example`. The generated-adversary campaign is `hpc/exploiter_vs_champion`, the
fine-tuning control `hpc/finetune_vs_ladder`, the behaviour-cloning control `hpc/bc_v4_clone`, and the
held-out evaluation blocks `hpc/best_config_eval` and `hpc/camera_ready_heldout`.

## Evaluation harnesses and protocol

All evaluations follow the same conventions: **seat balancing** (every policy occupies every physical
seat equally often), **paired game seeds** (every lineup sees identical initial states), stochastic
sampling with argmax as a reported second query mode, **outcome-conditioned metrics** (cash and win
rate given survival alongside raw cash), and a mandatory **self-play diagonal**.

| harness | what it measures | entry point |
|---|---|---|
| checkpoint vs scripted ladder | survival / win / cash of one learned seat against three copies of an archetype, per checkpoint | `python -m eval.scaled_checkpoint_scripted_diagnostics --help` (set `HUNGER_ENVIRONMENT_VARIANT`) |
| self-play survival + head entropies | four-seat self-play per checkpoint, with per-head entropy fingerprints | `python -m eval.checkpoint_selfplay_survival_diagnostics --help` |
| learned vs learned (peer cross-play) | 1v3 and mixed lineups of independently trained checkpoints and/or scripted seats | `workshops/eval/checkpoint_policy_crossplay.py` (see `workshops/eval/CROSSPLAY_README.md`) |
| LLM arena | production LLMs seated in the same games through a strict JSON wire protocol | `workshops/llm_arena/run_llm_arena.py` (see `workshops/llm_arena/README.md`; keys via `.env`, never committed) |
| generated adversary | PPO exploiter warm-started from a champion and trained against three frozen copies of it | `research/hpc/exploiter_vs_champion/` |

Worked examples of every harness invocation are the `*.sbatch` files under `research/hpc/`.

## Reproducing the papers' figures and tables

Run from the repository root with `PYTHONPATH=research`. "Summaries" means the inputs are already in
`workshops/results/`; "raw" means the per-game JSON from the data archive is required (extract it in
place first, see `data/README.md`). The ledger `workshops/paper/EXPERIMENTS.md` lists every experiment
(E1–E39) with its script, data directory, and the numbers quoted in the papers.

| paper item | script | inputs |
|---|---|---|
| TAE Fig. 1, WV Fig. 1 (rich and dead, best configuration) | `workshops/eval/plot_rich_and_dead_bestconfig.py --results-root workshops/results --out-dir <out>` | summaries |
| TAE Fig. 2, WV Fig. (LLM matrix) | `workshops/llm_arena/plot_llm_matrix.py --matrix workshops/results/llm_arena/campaign_90s/llm_matrix.csv --out-dir <out>` | summaries |
| TAE Fig. 3, WV Fig. (horizon × population matrix) | `workshops/eval/plot_crossplay_matrix.py --results-root workshops/results --out-dir <out>` | summaries |
| TAE Fig. 4 + Table 7 (exploiter convergence) | `workshops/eval/analyze_exploiter_convergence.py --results-root workshops/results --out-dir <out> --fig-dir <out>` | summaries |
| TAE Table 1 (declared-target audit) | `workshops/eval/analyze_objective_targets.py --results-root workshops/results --out-dir <out>` | summaries |
| TAE Table 2 (selection regret, held-out stage) | `workshops/eval/analyze_nested_selection_regret.py --results-root workshops/results --out-dir <out>` | summaries |
| TAE Table 3 (ladder calibration tournament) | `cd research && python -m eval.scripted_version_tournament --out-dir <out>` | re-plays the games |
| TAE Table 4 (ten disjoint-seed replications) | `workshops/eval/analyze_replication_blocks.py` | raw |
| TAE Table 5, WV Table 2 (signal-by-signal verdicts, regret, detection budget) | `workshops/eval/analyze_wv_portfolio_longitudinal.py --results-root workshops/results --out-dir <out>` | summaries |
| TAE Table 6 + §5 budget (power analysis) | `workshops/eval/analyze_evaluation_budget.py`, then `analyze_evaluation_budget_sensitivity.py` | raw |
| TAE §5 population / metric / replication audits | `analyze_population_sensitivity.py`, `analyze_metric_disagreement.py`, `analyze_inversions_familywise.py`, `analyze_bootstrap_leaderboard_stability.py` (all in `workshops/eval/`) | raw |
| WV peer-matrix figure | `workshops/eval/plot_policy_crossplay.py --matrix workshops/results/seeds234/_peer_crossplay_1000/crossplay_matrix.csv --out-dir <out> --name f3_peer_crossplay_bestconfig_1000g --policies ippo_s0,ippo_s1,ippo_s2,ippo_s3,ippo_s4,mappo_s0,mappo_s1,mappo_s2,mappo_s3,mappo_s4` (the matrix itself is rebuilt from raw games by `build_peer_matrix_1000.py`) | summaries (raw to rebuild the matrix) |
| WV held-out replication (CIs, held-out seed block, held-out learned peers) | `workshops/eval/analyze_camera_ready_headline_ci.py` and `analyze_camera_ready_heldout.py` (both `--results-root workshops/results --out-dir <out>`) | summaries |
| Appendix head-entropy fingerprints | `workshops/eval/plot_head_entropy.py --results-root workshops/results --out-dir <out>` | summaries |

Example:

```bash
PYTHONPATH=research python workshops/eval/analyze_objective_targets.py \
  --results-root workshops/results --out-dir /tmp/calhunger_out/objective_targets
```

Regenerating a result directory that already exists in `workshops/results/` reproduces the shipped
files; the scripts are deterministic given the data and their fixed bootstrap seeds.

## Data and checkpoints

The repository ships per-cell summaries (18 MB). Two archives hold the rest; see
[`data/README.md`](data/README.md) for their layout, checksums, and where to extract them:

- `calhunger-results-v1.tar.gz` — every raw per-game JSON behind `workshops/results/` and
  `workshops/econml/results/` (bootstrap, replication, and inversion analyses read these), including the
  complete LLM-arena transcripts;
- `calhunger-checkpoints-v1.tar.gz` (+ `-small90`) — the trained policies the papers evaluate: all
  twelve checkpoints of the ten best-configuration runs, the final checkpoint of every episode-matched
  run and horizon, the generated exploiters at 300 and 600 iterations, the fine-tuned policies, the
  behaviour-cloning actor, and the 90 s policies the LLM arena seated.

## Citing

```bibtex
@inproceedings{safrata2026metric,
  title     = {When the Metric Lies: A Full-Stack Audit of a Multi-Agent Evaluation Protocol},
  author    = {Safrata, Jan},
  booktitle = {NeurIPS 2026 Workshop on Trust-AI-Eval (TAE): Can We Trust AI Evaluation?},
  year      = {2026},
  note      = {Non-archival workshop paper}
}

@inproceedings{safrata2026selfplay,
  title     = {Self-Play Is a Broken Verifier: Measuring, Auditing, and Red-Teaming Verification Signals
               in a Forced-Interdependence Market Game},
  author    = {Safrata, Jan},
  booktitle = {NeurIPS 2026 Workshop: Who Verifies the Agents?},
  year      = {2026},
  note      = {Non-archival workshop paper}
}
```

## License

MIT (see `LICENSE`). The NeurIPS style file is not included; the paper sources are not part of this
release.
