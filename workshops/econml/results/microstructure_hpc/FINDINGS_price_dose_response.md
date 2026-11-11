# E31 follow-up — per-seat-k price dose-response + producer-role confound

Same data as FINDINGS.md (32 cells x 100 games, seed base 900000, HG_LEDGER=1).
Per-seat k = co-players of a different spec (diagonals k=0; 1v3 majority k=1,
2+2 k=2, singleton k=3). Trade markup = (price - game-food-VWAP)/VWAP, pooled
trade-level per k; 95% CIs = game-cluster bootstrap (B=2000, rng 20260901).
3,200 games (2,400 in the healthy-only variant, which drops every game
containing mappo_s1/mappo_s4). Script `econml/analyze_price_dose_response.py`;
outputs `price_dose_response.csv`, `producer_confound.csv`,
`price_dose_response.pdf/.png`.

## The price face of the k-dose response is a BUY-side singleton effect
- **Buy markup k=0..3: +6.7% -> +6.6% -> +8.6% -> +12.8%** (all cells);
  healthy-only **+7.2% -> +5.5% -> +7.0% -> +13.4%**. The k=3 singleton pays
  **+6.2pp over the k=1 majority** sharing its games (+7.9pp healthy-only);
  k=3 vs k=1 CIs disjoint in both variants.
- Sell markup is far flatter: +6.7 -> +7.4 -> +8.6 -> +10.1 (all); healthy-only
  **flat at +7.1–7.7 for every k** (all CIs overlap). The out-group penalty is
  paid when strangers BUY, not when they sell — consistent with E24/E26's
  stripping running through purchase prices.
- Shape vs E18's survival dose-response (92.2 -> 87.3 -> 80.7 -> 74.0): survival
  degrades **smoothly and monotonically per unit k**; the price curve is NOT
  similarly graded — it is flat k=0->1, rises from k>=2, and is dominated by a
  convex jump at k=3 (all-cells buy curve monotone only from k=1). Prices
  punish being the lone stranger more than proportionally; survival punishes
  every stranger added.

## Sanity check: k=0 is +6.7%, not 0 — explained, and it is not a seat asymmetry
At k=0 the buy and sell trade sets are identical (both parties of every trade
have k=0), and the buy/sell rows match exactly, so no seat asymmetry exists.
The volume-weighted mean markup at k=0 is **0 to machine precision** (6e-19);
the +6.7% offset is purely small-lots-trade-rich (corr(qty, markup) = -0.11)
driven by a producer-role asymmetry: **buys of food the buyer itself produces
(10.4% of trades, 8.7% of volume, median lot 1) average +92.4% markup (median
0.0% — heavy right tail)** while import buys average -3.2% (median -16.6%);
0.896(-3.2%) + 0.104(+92.4%) = +6.7%. Same-lineup seats buying back their own
product at silly prices is a monoculture feature, present at every k — it sets
the level of the curve, not its slope (next section).

## Producer-role confound: dismissed
Stratify every buy by whether the buyer produces the food (`food_type in
producers[buyer_idx]`) vs must import it (`producer_confound.csv`):
- **Forced purchases (buyer does NOT produce), the economically meaningful
  stratum, keep the dose-response**: buy markup k=0..3 = -3.2% -> -4.3% ->
  -2.4% -> **+0.9%** (all cells); healthy-only -3.7% -> -4.1% -> -1.1% ->
  **+2.8%** — monotone from k=1, k=3 vs k=0 shift **+4.1pp (+6.4pp healthy)**
  with disjoint CIs (healthy k=0 [-4.2, -3.1] vs k=3 [+1.2, +4.3]).
- Own-produce buys sit at ~+90–110% at EVERY k (92/101/99/111), and their share
  of buy trades is flat in k (10.4/10.3/10.8/10.8%) — so neither the stratum's
  level nor a composition shift can manufacture the k-curve. The singleton
  premium is paid on food the buyer has no choice but to import.
- Side observation, same table: selling food you do not produce (re-selling)
  carries +63–83% markup at all k, selling your own product ~0% — the
  lot-size/reseller premium is a structural feature of the economy, orthogonal
  to k.

## Interpretation (paper framing)
The behavioral dose-response (E18) has a price mechanism, and it is asymmetric:
each added stranger costs survival roughly linearly, but the market only
reprices meaningfully when you are the lone out-group buyer — the k=3 singleton
pays +6–8pp over majority seats in the same games, and the effect survives,
indeed sharpens, when restricted to forced imports and to healthy seeds. Being
economically surrounded is what raises your cost of eating.
