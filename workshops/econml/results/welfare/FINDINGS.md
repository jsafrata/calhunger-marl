# E29. Welfare accounting — monoculture efficiency, dead wealth, inequality [EconML theme 2; offline reanalysis of E10/E18 raw data]

Data: 145 cells / 104500 games / 418000 seat-games — 90 ordered 1v3 pairs @1000g (`results/seeds234/_peer_crossplay_100/raw` + `_peer_crossplay_900ext/raw`), 10 self-play diagonals @1000g, 45 2+2 cells @100g (`_peer_mix22_100/raw`). Per-seat stranger dose k = #co-players with a different spec (E18 construction over ALL seats). Script: `workshops/econml/analyze_welfare.py`; outputs `workshops/econml/results/welfare/`.

**Sanity (cash conservation)**: total terminal cash across the four seats is exactly 400.0 in 104500/104500 games (max |dev| 0.00e+00); 0 violations. Also per game sum(trades_as_buyer) = sum(trades_as_seller) = n_trades in 104500/104500 games; min seat cash 0.0 (0 negative-cash seat-games). Trading only redistributes the 4x100 endowment; survival is the only welfare output, cash is purely distributional.

**E18 reconciliation (per-seat k construction reproduces the published dose-response)**:

- 100g (E18 budget) | all 90 pairs: 79.4 -> 68.2 -> 59.0 -> 51.5
- 100g (E18 budget) | healthy 56 pairs: 92.2 -> 87.3 -> 80.7 -> 74.0
- merged (1000g 1v3/self, 100g 2v2) | all 90 pairs: 78.5 -> 68.0 -> 59.0 -> 50.6
- merged (1000g 1v3/self, 100g 2v2) | healthy 56 pairs: 91.4 -> 86.9 -> 80.7 -> 73.2
- E18 published (healthy 56): 92.2 -> 87.3 -> 80.7 -> 74.0; (all 90): 79.4 -> 68.2 -> 59.0 -> 51.5.

The 100-game-budget reconstruction reproduces E18 EXACTLY (to a decimal) — no definitional difference; the merged curve differs only through the 900-seed extension of the 1v3/self-play cells (the k=2 2+2 cells have no extension). Seat-level pooled per-k means (`welfare_by_k.csv`) equal the pair-level means because every pair contributes equal game counts within each k.

## 1. Dead wealth — the k=0->3 trend (`welfare_by_k.csv`, `dead_wealth.csv`)

- Share of terminal cash held by dead seats, healthy pairs, k=0->3: **10.4 -> 15.9 -> 23.2 -> 32.2%** — each stranger roughly adds 5-9 pts of the economy's endowment locked in corpses; all pairs: 22.3 -> 26.8 -> 39.9 -> 58.1%.
- **Rich and dead at every dose**: dead/alive mean-cash ratio, healthy pairs, k=0->3: **1.23 -> 1.26 -> 1.26 -> 1.30** (dead-seat mean cash $121 -> $124 -> $120 -> $114; alive $98 -> $98 -> $95 -> $88) — even in healthy monocultures the dead die richer than the survivors live. All-pairs ratio is non-monotonic (1.05 -> 0.78 -> 0.96 -> 1.42): at k=1 the dying majority seats of sick-singleton cells die POOR (stripped), at k=3 healthy singletons vs sick majorities die rich.
- Worst cells: mixed_mappo_s4_mappo_s4_mappo_s4_mappo_s4 100.0% dead-held ($400/400); mappo_s1_vs_mappo_s4 91.7% dead-held ($367/400); mappo_s2_vs_mappo_s1 91.7% dead-held ($367/400).

## 2. Inequality (`gini_by_population.csv`)

- Mean per-game terminal-cash Gini: healthy self-play 0.411 [0.297 mappo_s3 - 0.515 ippo_s4]; healthy 1v3 cells 0.394 [0.289-0.485]; healthy 2+2 0.386. **Monocultures are high-welfare but NOT low-inequality** — the k=0 Gini is slightly ABOVE mixed healthy economies; composition moves survival, not (healthy) inequality.
- Inequality explodes only with a sick seed at the table: healthy-vs-sick 1v3 columns mean Gini 0.588, max 0.708 (ippo_s1 vs 3xmappo_s4, survival 5.2%); min over all cells 0.289 (ippo_s1_vs_mappo_s3_1v3).
- Across all 145 cells corr(survival, Gini) = -0.73; healthy-only cells +0.10 — the survival-inequality tradeoff is entirely driven by sick-seed economies.

## 3. Trade intensity (`welfare_by_k.csv`, `role_asymmetry.csv`)

- Trades per seat (buyer+seller), healthy pairs, k=0->3: 53.1 -> 51.0 -> 48.7 -> 46.0; normalized to k=0: volume 1.000 -> 0.961 -> 0.918 -> 0.867 vs survival 1.000 -> 0.951 -> 0.883 -> 0.801 — **volume falls SLOWER than survival**: the market does not thin before it kills; per-alive-tick trade rate 29.9 -> 28.9 -> 28.0 -> 27.0 trades/100 ticks (a ~10% slowdown at k=3, vs a 20% survival drop).
- Buyer/seller role asymmetry (healthy lineups, mixed-lineup seats): small but systematic — most buyer-skewed mappo_s0 (+1.93 trades/seat-game, ratio 1.09), most seller-skewed mappo_s2 (-1.84, ratio 0.93); |diff| <= 2 trades on ~50 per seat-game, so learned styles differ in role at the margin, not in kind.

## 4. Welfare frontier (`welfare_frontier.pdf/.png`)

One point per cell (145): x = aggregate survival of all four seats, y = mean per-game cash Gini; color/marker = composition (self-play / 1v3 / 2+2), open markers = lineup includes a sick seed (mappo_s1/s4). Healthy economies cluster at high survival with Gini 0.29-0.51 regardless of composition; sick-seed economies form the low-survival/high-Gini arm.

**EconML theme-2 reading**: in a fixed-endowment agent economy, population composition is a pure welfare lever — cash conservation makes survival the only aggregate output, and every stranger added to a monoculture destroys ~6 pts of it (E18) while shifting 5-9 pts of the endowment into dead hands. The 'rich and dead' externality is not a pathology of bad seeds: dead seats out-hold survivors by 23-30% even in all-healthy economies. Monoculture buys efficiency without buying equality; catastrophic inequality appears only when a degenerate model enters the ecosystem, and then cash concentrates precisely on the seats that die.
