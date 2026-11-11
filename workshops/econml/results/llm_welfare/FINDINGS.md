# E26+E29 unified — LLM economies on the welfare frontier; the LLM deficit decomposed into volume vs price [EconML; Aug 29, offline]

Data: the 34 E26 replay cells (3070 validated games; validation.ok only) welfare-accounted with E29's exact definitions, overlaid on E29's 145-cell frontier. Script: `workshops/econml/analyze_llm_welfare.py`; outputs `workshops/econml/results/llm_welfare/`.

**Sanity**: (a) cash conservation — total final-tick cash across the four seats is exactly 400 in 3070/3070 validated games (max |dev| 0.00e+00); the LLM economies conserve the 4x100 endowment exactly like the policy economies. (b) replay final-tick cash equals the campaign's own per-seat terminal_cash in 12280 seat comparisons with 0 mismatches (0 alive/survived mismatches, 0 originals missing) — reserved cash is returned to the player before the terminal snapshot; final-tick `cash` IS the campaign's terminal cash.

## 1. LLM economies on the E29 frontier (`llm_welfare_by_cell.csv`, `unified_frontier.pdf/.png`)

- LLM self-play/mixed (4 cells): survival 87.2-99.0%, Gini 0.380-0.479, dead-wealth share 0.7-15.5%.
- LLM vs scripted (16 cells): survival 71.2-100.0%, Gini 0.240-0.450, dead share 0.0-13.2%.
- LLM vs policies (14 cells): survival 58.3-92.0%, Gini 0.361-0.501, dead share 7.5-41.4%.
- E29 healthy policy economies for reference: Gini 0.289-0.515, survival 63.2-94.5%.

## 2. Deficit decomposition (`deficit_decomposition.csv`)

Per LLM-vs-X game: actual net outflow = paid-as-buyer - received-as-seller; volume component = the same trades at the game's per-food VWAP; price component = actual - volume (the cost of adverse prices).

- **Vs scripted bots (16 cells)**: mean actual outflow $79.0/game, of which volume $74.1 and price $4.9 — price share 6.2% (per-cell price share -29.4-47.8%).
- **Vs trained policies (14 cells)**: mean actual outflow $87.0/game, volume $-1.2, price $88.1 — price share 101.3% (per-cell 35.8-153.7%).
- **The whole deficit is price vs policies**: the volume component is NEGATIVE in 7/14 vs-policy cells (all four vs-ippo_s0 cells; e.g. deepseek_vs_ippo_s0 $-41) — at fair prices the LLM's own trade pattern would RECEIVE net cash; the observed ~$80-95 outflow is adverse pricing alone (price share >100%).
- One vs-scripted exception: claude_vs_v2 price share 47.8% (price $45.5 of $95.3) — Claude's hyperactive selling into the v2 book (47 sells/game) leaks cash through prices even against bots; the other 15 vs-scripted cells range -29 to +20%.
