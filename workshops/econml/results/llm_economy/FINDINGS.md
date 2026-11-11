# E26 — the LLM economy, reconstructed (deterministic replay, $0 API cost)

Replayed all 3,239 cached campaign_90s games with recorded LLM actions fed back
verbatim; scripted seats re-run from per-game seeds; policy seats re-run from the
original local checkpoints with the arena's torch-stream discipline. Scripts:
`econml/replay_llm_economy.py` (replay + per-game validation) and
`econml/analyze_llm_economy.py`. Data: `econml/results/llm_economy/`.

## Validation (per-game, five checks: env_applied verdicts, n_trades, per-seat
## terminal cash / survival time / trade counts)
- 3,070/3,239 games validated (94.8%). ALL scripted, self-play, and mixed-LLM
  cells: 100%. Failures concentrate exactly where predicted: gemini policy cells
  (12/100 and 14/88 — Gemini's API-throttled resumes restarted the torch stream)
  and deepseek_vs_ippo_s2 (93/100). Non-validated games are excluded everywhere.
- Consequence: gemini-vs-policy price numbers rest on n=26 validated games —
  treat as directional only.

## Terms of trade (markup vs the game's per-food VWAP; thousands of trades/cell)
- **Vs trained policy populations, every LLM buys dear and sells cheap**: buy
  markups +24% to +77% (claude +24..+49, gpt +35..+60, deepseek +46..+77,
  gemini +41..+62), sell markups −2% to −16%. The incumbents mirror it exactly:
  buy −6% to −17%, sell +8% to +15%.
- **Vs scripted bots the SAME models trade near-fair**: buy markups −11% to +4%.
  The expropriation is price-mediated and specific to trained incumbents — the
  bots don't exploit; the learned policies do.
- **Crisis pricing**: in the last 10 ticks before death, LLM buy markups reach
  +27% to +72% vs policy populations (desperation purchases at scarcity prices).

## Wealth transfer (net cash flow between seat classes per game, $400 economy)
- The LLM seat runs a persistent net trade deficit of ~$40–100/game in every
  population (it must buy survival), but the decomposition differs: vs bots the
  deficit is food bought at fair prices; vs policies the same deficit is
  amplified by adverse prices — the channel behind E24's $0.5–5 terminal cash.
- Mixed 4-LLM economy internally reproduces E24's verdict: claude net-pays
  gemini $58.40/game (gemini: 49% wins, $167; claude: 93% survival, $26).

## Stripping → starvation
- Death-aligned cash paths decay monotonically into death (e.g. vs ippo_s0:
  claude $28.6 → $12.6 over the final 60 ticks; gpt $24.6 → $7.8; deepseek
  $70.3 → $23.7). Starvation is the endpoint of expropriation, not a separate
  failure — the mechanism behind E24's late-death signature.

## Caveats
- Replay determinism is verified per game by the five checks; excluded games are
  counted and reported per cell (`validation.csv`).
- Markups are relative to within-game VWAP, so they measure *relative* prices in
  the same games — immune to cross-game price-level differences.
- Gemini policy-cell numbers: small validated n (above).
