# LLM arena campaign — 90 s, frozen prompt v3 (Aug 27–28, 2026)

Four production LLMs evaluated with the paper's exact protocol (seat
rotation, paired seeds base 910000, stochastic mode, 100 games/cell) on
the small_90 variant: vs the scripted ladder v2–v5, vs four selected
trained policies (small_90 best-config @1200: ippo s0/s2, mappo s1/s3),
and a 100-game four-way mixed LLM tournament. Prompt frozen at
`llm-arena-v3` (rules + derived survival state + market view +
mask-derived legality block; no strategy advice); template hash in every
job identity. Models: claude-haiku-4-5-20251001, gpt-5.6-luna,
deepseek-v4-flash, gemini-3.5-flash-lite.

Canonical numbers: `llm_matrix.csv` (regenerate with
`workshops/llm_arena/summarize_campaign.py`); 95% bootstrap CIs for
every cell in `llm_matrix_ci.csv`, death timing in `death_times.csv`,
action fingerprints in `fingerprints.csv` (all from
`workshops/llm_arena/analyze_campaign.py`, 10k resamples over games). Figure:
`workshops/figures/f5_llm_matrix.{png,pdf}`. Gemini cells marked (n<100)
were throttled by sustained Google-side 503 load-shedding — itself a
reliability result, see §6 — and are still filling in; rerun the
summarizer to upgrade.

## 1. Headline matrix (survival% / win% / mean terminal cash $)

| model | 3-LLM mix | v3 JIT | v2 maker | v5 sniper | v4 stockp. | ippo_s0 | ippo_s2 | mappo_s1 | mappo_s3 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5.6-luna | 94/20/65 | 97/0/0.8 | 99/0/3.6 | 69/8/59 | 100/0/3.3 | 56/1/7.4 | 90/1/13.7 | 86/0/11.6 | 82/0/2.8ⁿ |
| DeepSeek v4-flash | 75/28/142 | 95/0/2.2 | 99/0/4.6 | 52/2/55 | 95/0/2.4 | 36/0/22.8 | 66/0/13.2 | 62/0/22.1 | 61/0/6.9 |
| Gemini 3.5 FL | 87/49/167 | 100/0/42ⁿ | 98/4/53ⁿ | 0/0/51ⁿ | 96/0/39ⁿ | 29/2/17.8 | 57/0/13ⁿ | — | — |
| Claude Haiku 4.5 | 93/3/26 | 92/0/0.5 | 84/0/4.7 | 18/0/19.1 | 39/0/8.9 | 28/0/12.6 | 59/1/8.4 | 53/0/10.9 | 59/0/6.7 |

ⁿ = n<100 at time of writing (see llm_matrix.csv for exact n).

## 2. Findings

**F-LLM-1 — Survive-but-never-win vs every fixed opponent.** Against
bots and trained policies alike, wins are 0–8% (chance 25%): 24 wins in
~2,600 seat-games. Survival vs the weak bots is high (84–100% vs v2/v3)
while terminal cash collapses to $0.5–5 — the opponents strip every
dollar. PPO reference (10-seed mean, same seeds): the policies *win*
8–27% vs the ladder. The LLM failure is concentrated in the economic
channel; outcome-conditioned metrics are what reveal it.

**F-LLM-2 — The ladder's difficulty ordering transfers for 3 of 4
models; Gemini inverts it.** For claude/gpt/deepseek: v3 easiest, v2
easy, v4/v5 hard — the same ordering RL faces. PPO comparison per
opponent (LLM range vs PPO mean): v3 92–97 vs 99; v2 84–99 vs 61; v4
39–100 vs 32; v5 0–69 vs 52. The LLMs *out-survive* PPO against v2
while losing the win channel everywhere. **Gemini (n=100): 59% survival
vs v3 — its second-hardest bot, everyone else's easiest** — because its
high-wait style needs resting orders to take and the JIT procurer posts
nothing early; difficulty is a property of the (style, population)
pair, not the opponent alone (see F-LLM-5b).

**F-LLM-3 — Trained-policy populations are harsher than any single
bot's survival column suggests, and seed-dependent.** ippo_s0 is the
killer column: survival 28–56% across models — below or near each
model's v4/v5 numbers, and 30–60 points below their v2/v3 numbers.
The other three seeds are 20–30 points gentler (e.g. GPT: 56 vs 90;
Claude: 28 vs 59). A single peer seed flips the verdict — the LLM
version of the peer-column sensitivity in the F3 policy matrix.

**F-LLM-4 — Population-relative rankings invert (rich-and-alive vs
poor-and-alive).** In the all-LLM economy, Gemini wins 49% of games
(2× chance) holding $167 and DeepSeek 28%/$142, while Claude — the
most *active* trader, and the best survivor (93%) — wins 3% holding
$26. Against the scripted ladder the same Gemini merely sits on its
cash (0–4% wins) and the same Claude looks comparable to the others.
Which model "is better" is a property of the table it sits at.

**F-LLM-5 — Activity without pricing skill is a wealth transfer.**
Action mixes are model-signatures, stable across opponents: Claude
posts 39–48% of turns (a compulsive maker), GPT and Gemini wait
55–85%, DeepSeek is balanced. Cash outcomes track waits, not trades:
the more an LLM trades against competent opponents, the poorer it ends
(Claude: most trades, least cash everywhere). An LLM rich-and-dead
also appears: vs v5, dying seats hold $19–59 (DeepSeek dead at $160+
in individual games).

**F-LLM-5b — Fingerprints are near-disjoint across models and stable
across populations** (`fingerprints.csv`). Wait share across all
populations: Claude 4–30%, DeepSeek 24–43%, GPT 46–73%, Gemini 74–85% —
the between-model ranges barely overlap, while each model's range across
nine different populations is far narrower than the between-model gaps.
Claude's post share never drops below 39% against anyone. The models
carry fixed styles into every economy rather than adapting to it; the
scripted bots, by contrast, are state-contingent by construction.

**F-LLM-7 — LLMs die late, of economic exhaustion, not early
incompetence** (`death_times.csv`). 64–90% of all LLM deaths occur in
the last third of the game (t=60–90), with modal death at the t=80–90
feedings; only 3–11% die in the first three feedings. The failure
trajectory is: trade → get stripped → run out of cash → starve within
sight of the horizon. This inverts the failed-RL-run signature (deaths
pinned early-mid game) and matches the cash curves: survival fails as a
*consequence* of the economic-channel failure, not separately from it.

**F-LLM-6 — Reliability under the strict protocol.** Malformed-output
rate ≤0.1% for all four models (~0 with structured output). Rejected
(mask-illegal) actions: 0.3–7.4% per cell; highest for Claude
post-bankruptcy (futile buy attempts). Mean decision latency 0.9–1.2 s
(Claude/DeepSeek/GPT); Gemini 0.8 s median but a 20.8 s p95 tail and
10 aborted games from sustained 503 load-shedding — the only
infrastructure-unreliable provider, which capped its n in several
cells. Cost for the full 3,300-game campaign ≈ $150–170 total, ~70%
of it Claude (tokenizer counts the same prompt ~1.5×).

## 3. Protocol notes

- Same env, seeds, rotation, and metrics as the policy/scripted
  matrices; LLM cells join those tables directly.
- LLM actions are mask-checked like every other seat kind; a
  mask-illegal action scores a WAIT and increments the rejection
  metric (also required for env integrity — unmasked replacement
  posts can crash simultaneous clearing; see
  `llm_arena/arena.py:DecisionRecord.mask_illegal`).
- Prompt-version sensitivity is itself measurable: on identical seeds,
  adding the legality block (v2.1→v3) cut rejections 12–62% → 0–5%
  and *changed behavior* (Claude's wait rate halved; Gemini went from
  near-inert to winning games). The prompt is part of the grader; we
  report the template hash with every cell.
