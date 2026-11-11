# CalHunger LLM arena

Extends the paper's evaluation protocol to **LLM seats**: one
seat-balanced, seed-paired harness that can seat language models,
trained PPO checkpoints, and the scripted heuristic ladder in the *same
games*. Built for the IAEval submission ("evaluation protocols for
collaborative agents", "benchmarks for long-horizon interaction …
reliability", trajectory-level assessment incl. latency and cost).

Provenance: the wire protocol, prompt, and provider adapters are
adapted from the preliminary arena (`jsafrata/calhunger-llm`); the seat
runners, rotation, seed pairing, checkpoint loading, and summary schema
are reused from `workshops/eval/checkpoint_policy_crossplay.py`. The
game itself is the unchanged research env (imported read-only), so LLM
results are directly comparable to every existing matrix — same seeds,
same rotation, same metrics.

## What one simulated second looks like

1. Scripted seats (cadence-gated) and policy seats (factored actor over
   the encoded observation) choose actions synchronously.
2. Each living LLM seat gets a prompt with the shared rules + a minimal
   JSON state (own private state, public book — the game's real partial
   observability) and must return one bare JSON action object
   (`wait`/`cancel`/`take`/`post`). Malformed responses are retried with
   the byte-identical prompt (no repair hints), up to `--llm-attempts`.
3. Parsed actions are checked against the env's legal-action mask — the
   same constraint policy and scripted seats live under by construction.
   A mask-illegal action (e.g. taking from an empty book, selling
   unavailable food) is recorded as rejected and a WAIT is submitted in
   its place; `rejected_action_percentage` reports the rate. (Mask
   enforcement is also required for env integrity: the
   simultaneous-clearing pipeline assumes mask-legal actions, and
   unmasked replacement posts can crash it — see the comment on
   `DecisionRecord.mask_illegal`.)
4. All four actions are stepped through the unchanged env.

**The prompt is FROZEN at `llm-arena-v3`** (debugged on the 60 s
variant, Aug 27): rules + derived survival state + market view + a
mask-derived LEGAL ACTIONS block (parity with the legal-action mask RL
policies receive; consistency asserted by test) + output contract, no
strategy advice, `bid`/`ask` side semantics. The legality block dropped
rejection rates from 12–62% to 0–5%, so rejections now measure
coherence, and failures are strategic by construction. Do not edit
`prompt.py` before scoring runs — the template hash in each job's
identity marks which prompt produced which data.

A model that exhausts its parse attempts scores a WAIT and the failure
is recorded (`--on-malformed wait`, default) — protocol reliability is a
*measurement*, not a crash. `--on-malformed abort` restores the
preliminary arena's fail-closed exhibition behavior.

## Seat spec grammar

- `llm:LABEL=PROVIDER/MODEL` — e.g. `llm:haiku=anthropic_api/claude-haiku-4-5-20251001`.
  Providers: `openai_api`, `anthropic_api`, `gemini_api`, `deepseek_api`,
  `codex_cli`, `claude_cli`, `mock`.
- `policy:LABEL=CHECKPOINT[@MODULE]` — RLlib checkpoint (cross-play grammar).
- `scripted:v1` .. `scripted:v5` — the calibrated ladder, fast cadence.

## Running

```bash
cd Hunger_games_2
PY=research/.venv-mappo/bin/python

# Tests + offline smoke (no tokens)
$PY -m pytest workshops/llm_arena/tests -q
$PY workshops/llm_arena/run_llm_arena.py --mode 1v3 \
    --focal "llm:mm=mock/market-maker" --opponent scripted:v1 \
    --environment-variant small --n-games 4 --name smoke

# Key checks without spending tokens, then one live decision
$PY workshops/llm_arena/run_llm_arena.py --mode 1v3 \
    --focal "llm:haiku=anthropic_api/claude-haiku-4-5-20251001" \
    --opponent scripted:v3 --preflight
$PY workshops/llm_arena/run_llm_arena.py --mode 1v3 \
    --focal "llm:haiku=anthropic_api/claude-haiku-4-5-20251001" \
    --opponent scripted:v3 --probe 0

# LLM vs the ladder (30 s variant: 30 decisions/seat/game)
$PY workshops/llm_arena/run_llm_arena.py --mode 1v3 \
    --focal "llm:haiku=anthropic_api/claude-haiku-4-5-20251001" \
    --opponent scripted:v3 --environment-variant small \
    --n-games 20 --name haiku_vs_v3

# LLM seated inside a trained-policy population
$PY workshops/llm_arena/run_llm_arena.py --mode 1v3 \
    --focal "llm:haiku=anthropic_api/claude-haiku-4-5-20251001" \
    --opponent "policy:ippo_s0=research/eval/results/.../checkpoints/iter_1200" \
    --environment-variant small --n-games 20 --name haiku_vs_ippo

# Four models against each other (mixed mode, rotated)
$PY workshops/llm_arena/run_llm_arena.py --mode mixed \
    --seat "llm:gpt=openai_api/gpt-5.2" \
    --seat "llm:claude=anthropic_api/claude-haiku-4-5-20251001" \
    --seat "llm:gemini=gemini_api/gemini-3.5-flash-lite" \
    --seat "llm:deepseek=deepseek_api/deepseek-v4-flash" \
    --environment-variant small --n-games 8 --name four_llms
```

API keys: export them or put `KEY=VALUE` lines in
`workshops/llm_arena/.env` (gitignored; see `.env.example`). Defaults:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`DEEPSEEK_API_KEY`; override per label with `--llm-config` TOML
(`[providers.LABEL] api_key_env/base_url/executable`).

## Protocol conventions (inherited from the paper)

- **Seat balance**: cyclic rotation; every slot occupies every physical
  seat exactly `n_games/4` times (`--n-games` must be a multiple of 4).
- **Seed pairing**: game `g` always plays seed `--game-seed-base + g`,
  so every lineup sees identical initial states; policy sampling is
  seeded per job (`seed_everything`).
- **Self-play diagonal**: a uniform lineup (`--focal X --opponent X`)
  is the required baseline row for any cross-play claim.
- **Query mode**: `--deterministic` switches policy seats to argmax;
  LLM temperature is whatever the provider default is (a benchmark
  axis, recorded in the identity block).

## Output

`workshops/results/llm_arena/<name>/`:

- `identity.json` — full job identity incl. `protocol_version` and a
  hash of the prompt template (changing the prompt invalidates caches).
- `games/game_NNNN.json` — one file per game: outcomes per seat plus
  the complete LLM trajectory (per decision: prompt hash, raw response,
  parsed action, attempts, latency, token usage, and the env's
  applied/rejected verdict). Interrupted or rate-limited runs resume at
  game granularity; `--overwrite` replays.
- `arena_summary.csv` / `.json` — one row per distinct spec. The first
  columns match `checkpoint_policy_crossplay.py` exactly (survival, win,
  win|survival, outcome-conditioned cash, trades), so LLM cells join the
  existing matrices; LLM-only columns append reliability and cost:
  `malformed_decision_percentage`, `rejected_action_percentage`,
  `wait_action_percentage`, `mean_decision_latency_seconds`,
  `mean_attempts_per_decision`, `total_input/output_tokens`.

## Cost planning

One decision per living seat per simulated second: an N-second variant
costs ≤ N API calls per LLM seat per game (fewer once seats die), plus
parse retries. Prefer `small` (30) / `small_60` for sweeps; `default`
(180 s) is 180 calls/seat/game. Wall-clock per tick is the slowest
provider; games in a job run sequentially (paired seeds), so parallelize
across jobs/lineups, not within a game.

## Caveats

- Resuming a partially cached run with policy seats restarts the torch
  sampling stream at the job seed for the remaining games (game seeds
  and lineups are unaffected). Fully cached and fully fresh runs are
  exactly reproducible.
- Do not commit `.env`; raw responses in `games/` can be large for long
  horizons.
