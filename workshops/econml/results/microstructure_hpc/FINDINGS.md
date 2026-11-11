# E31 — monoculture vs mixed-economy microstructure (HPC job 1896332)

32 cells × 100 games (default 180s, best-config @1200, seed base 900000 = the
published peer matrix's paired games) re-run with `HG_LEDGER=1`: full trade
ledger + per-tick L1 tops/depth + per-tick cash. Campaign
`research/hpc/econml_microstructure/` (commit 3b36057); analysis
`econml/analyze_microstructure.py`; data `econml/results/microstructure_hpc/`.

## Validation
All 26 cells that exist in the base campaign reproduce it **game-for-game**
(seeds, survival, terminal cash, trade counts) — 26/26; the 6 mixed 2+2 cells
are new. The ledger fields are therefore attached to exactly the games the
published matrices scored.

## Monocultures sustain seed-specific price conventions (k=0 diagonals)
- VWAP across the 9 non-failed monocultures spans **2.06 (mappo_s2) to 5.02
  (mappo_s1)** — a 2.4× price-level difference between co-adapted economies of
  identical architecture/config differing only in seed. Volume 164–248
  units/game; the failed mappo_s4 monoculture is a scarcity economy: VWAP 32.8,
  volume 38.
- So "the" market price is a convention of the co-adapted population, not a
  property of the environment.

## Mixing breaks the conventions (k=2, k=3 vs constituent monocultures)
- Volume falls in 20/22 mixed cells (volume ratio vs constituent-monoculture
  mean: median 0.90); catastrophic pairings collapse trade: mappo_s2 vs
  3×mappo_s1 at **0.449**, the 2+2 version at 0.568, ippo_s0 vs 3×mappo_s4 at
  0.467.
- Price levels move toward the majority's convention (e.g. mappo_s4-majority
  cells trade near mappo_s4-adjusted levels; the visiting singleton cannot
  impose its home prices).

## Strangers pay worse prices (within- vs cross-type buy markups, same games)
- Cross-type buy markups exceed within-type in 16/22 cells, often by an order
  of magnitude: ippo_s1 economy +24.4% cross vs +2.2% within; ippo_s4 economy
  +17.9% vs +1.0%; mappo_s1-majority +30.5% vs +7.1%; mappo_s4-majority +75.8%
  vs +0.4%. Trading with a stranger costs more than trading with your own
  co-adapted kind — the price face of the k-dose response (E18).

## Dominance and collapse as measured cash flows
- The canonical win-channel dominance pair (E10a: ippo_s2 beats ippo_s4 33↔4
  at equal survival): **ippo_s4 net-pays ippo_s2 $39–57/game (1v3, both
  directions) and $64/game in the 2+2 economy** — non-transitive dominance is a
  literal wealth transfer.
- The catastrophic column: mappo_s1 seats net-pay mappo_s2 **$178–185/game**
  while mappo_s2 still dies (93.5% self-play survival → 0.7 survivors/4) —
  "rich and dead" observed as flow, not just terminal stock: the cash arrives,
  the food does not.
- The failed seed hemorrhages: mappo_s4 net-pays a visiting ippo_s0 $230/game.

## Interpretation (exact framing for the paper)
Co-adaptation behaves like a tacit trading convention, not price collusion in
the antitrust sense: monocultures clear MORE volume at seed-specific price
levels, and entry by a stranger thins the market and worsens the stranger's
terms of trade. The generalization failure the benchmark measures behaviorally
(E10a/E18) is visible in the market data as convention breakdown — same games,
now with the mechanism.
