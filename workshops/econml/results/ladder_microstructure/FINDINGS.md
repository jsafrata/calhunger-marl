# E30. Price formation on the heuristic ladder — local microstructure [EconML theme 2]

Setup: exact re-run of the E2 tournament (small 30 s, fast cadence, 100 seat-balanced
four-way + 6×50 pairwise 2v2, seed base 1,300,000) with a full ledger: every trade
(asdict + maker/taker attribution), per-second L1 book per food, seat→version map,
per-seat terminal outcomes. Script `econml/scripted_tournament_ledger.py` (copy of
`eval/scripted_version_tournament.py`; env stepped identically, additions read-only).
**Validation: EXACT reproduction of the published run — all 80 per-version metrics
(7 matches × survival/win/cash/time) byte-identical** to
`results/scripted_v2_v5_tournament_small_30s/scripted_version_tournament.json`
(wins v4 54% > v5 22% > v2 16% > v3 8%; survival 99/97/99/100; cash
$130.24/$101.79/$86.99/$80.98). `validation.json`.

Attribution rule (documented from `research/env/market.py`): every order id comes from
a POST (`actions_applied` carries it in both clearing pathways) → order_id→seat registry
is complete (0 unattributed makers / 8,410 trades). `taker_order_id == -1` = TAKE
execution; maker = owner of `maker_order_id` (post-crossed: the earlier-sequence order);
taker = the counterparty. Trades always print at the maker's price. 88.5% of all
trades (7,439/8,410) are TAKE executions; the book is two-sided on only 37.0% of
food-ticks (mean quoted spread 1.96 when it is) — a take-dominated, thin market.

## 1. Archetype microstructure (four-way, 100 games) — `archetype_micro.csv`
| | maker share | vol/seat-game | VW buy | VW sell | sell−buy /unit | net trade cash/seat-game |
|---|---:|---:|---:|---:|---:|---:|
| v2 dealer | 39.5% | 18.7 | 7.25 | 7.19 | **−0.06** | **−$13.0** |
| v3 deadline | 60.8% | 17.2 | 8.47 | 6.42 | −2.05 | −$19.0 |
| v4 stockpiler | **78.4%** | 13.6 | 6.92 | 7.47 | +0.55 | **+$30.2** |
| v5 sniper | 7.5% | 8.7 | 7.31 | **11.53** | **+4.22** | +$1.8 |

**The nominal spread-earner (v2) earns no spread as measured**: highest volume, realized
sell ≈ buy (−$0.06/unit), 60% of its fills as taker (94.6% of those TAKE actions) — its
quoting is real but it crosses the spread away. **The position-taker (v4) is also the
market's de-facto maker**: 78% maker share, net seller (8.65 sold vs 4.97 bought
/seat-game), +$0.55/unit carry margin, and the only large net extractor (+$30.2 ≈ the
entire $130 vs $100 cash gap over v5). **v5 is the pure taker**: 92.5% of fills as
aggressor, lowest volume, but +$4.22/unit — sells into late scarcity at 11.5. v3's 61%
maker share is adverse selection, not dealing: its deadline-concession resting orders
get picked off (buys at 8.47, sells at 6.42). Maker share ≠ spread income; the
spread-earner vs position-taker distinction survives, but the ROLES INVERT vs design
intent: carry (v4) and snipe-edge (v5) are profitable; dealing (v2) and scheduled
procurement (v3) pay the market.

## 2. Pairwise net cash flows (who farms whom) — `pairflow.csv`
Net trade cash per game (positive = named version extracts): v4 farms v2 ($20.3/game,
wins 39–11) and v3 ($32.6, 38–12); v5 farms v3 (**$75.5/game**, 45–5) and v2 ($13.2);
v2 farms v3 ($20.4, 35–15). Two decouplings from the win table: (i) **v5 extracts cash
from v2 yet loses the pairing** (28–22 to v2; terminal cash still favors v5
$106.6 vs $93.4) — v2 beats v5 on per-game wealth ranking, not on flows; (ii) **v4
beats v5 34–17 with ~zero net flow** ($0.36/game, lowest cross-volume 12.6 units/game)
— that pairing is decided by survival economics, not by trading one another. The
non-transitivity is thus not a single cash-flow cycle: v4>v2/v3 and v5>v3 are
farming relations; v4>v5 and v2>v5 are not.

## 3. Price formation by lineup — `price_formation.csv`
VW mean price (std): four-way 7.57 (5.11). Pairwise: v2v3 7.54 (5.65), v2v5 7.28
(4.64), **v3v5 11.77 (8.25)** vs v4-containing v2v4 **5.88 (1.40)**, v3v4 7.34 (5.49),
v4v5 7.34 (4.29). **v4's presence compresses the price level and dispersion**
(mean of v4 lineups 6.85 vs 8.86 without; tightest market is dealer+stockpiler v2v4);
the v3v5 economy — conceder meets sniper with no storage — produces the highest and
noisiest prices (v5's late rich-bid bait + v3's deadline concession, no v4 supply
into scarcity). Price level is lineup-dependent: which archetypes are present moves
the market-wide price by up to 2× and dispersion by 6×.

## 4. Cross-experiment comparison row (for E26.4 LLM economy / E31 monocultures)
`E30_ladder_fourway_small30_fast`: VW mean price **7.57**, VW std **5.11**,
**21.7 trades/game**, **29.1 units/game** (100 games, 30 s, fast cadence). Last row
of `price_formation.csv`.

**Interpretation (EconML theme 2)**: the heuristic archetypes are distinguishable
economic agents in measured microstructure, not just in outcomes — maker share spans
7.5→78.4%, per-unit margin −2.05→+4.22 across the ladder. The economy rewards
intertemporal position-taking exactly as E2 claims, and the ledger shows the
mechanism: v4's carry trade (buy at 6.9 early, sell at 7.5 into scarcity while being
the standing book) is the profit center, v5's snipe edge is high-margin/low-volume,
and continuous dealing (v2) is competed to zero spread by a take-dominated market.
Data: `ledger/*.jsonl` (400 games, 8,410 trades, per-second L1).
