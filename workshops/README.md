# workshops/ — the analysis layer

Everything in this directory consumes `../research/` read-only (`PYTHONPATH=research` from the
repository root) and produced the numbers, tables, and figures in the two papers.

| directory | contents |
|---|---|
| `eval/` | Analysis and plotting scripts (`analyze_*.py`, `plot_*.py`), the learned-vs-learned cross-play harness `checkpoint_policy_crossplay.py` (`CROSSPLAY_README.md`), and local copies of the checkpoint harnesses used for one-off evaluations. |
| `llm_arena/` | LLM seats for the same harness: strict JSON wire protocol, the frozen prompt (`llm-arena-v3`), provider adapters (OpenAI, Anthropic, Gemini, DeepSeek, CLI, mock), campaign scripts, and the summariser/plotter. API keys are read from `llm_arena/.env` (copy `.env.example`; never committed). |
| `econml/` | Market-microstructure, welfare, and social-choice analyses of the same games (price formation on the ladder, terms of trade in the LLM economy, Nash averaging and α-rank versus the mean leaderboard). `DESIGN.md` documents each analysis. |
| `results/` | One directory per evaluation campaign with its per-cell summaries (`*.csv`, `*.json`, `*.md`, `iters.jsonl`). Raw per-game JSON (`raw/`, `games/`) is distributed in the data archive and extracts in place here. |
| `figures/` | Every figure in the papers (`f1` … `f8`), PNG and PDF. |
| `paper/EXPERIMENTS.md` | The experiment ledger: for each experiment E1–E39 the question, method, data directory, script, and the numbers quoted in the papers. Start here to trace any number. |

Most analysis scripts take `--results-root workshops/results --out-dir <out>`; the few that read raw
games (bootstrap, replication, and inversion audits; see the table in the top-level README) need the
data archive extracted first.
