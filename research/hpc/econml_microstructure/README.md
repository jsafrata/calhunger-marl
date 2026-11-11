# E31 — monoculture vs mixed-economy microstructure (EconML camera-ready)

Re-runs 32 peer cross-play cells (100 games each, seed base 900000 — the SAME
paired games as the published best-config peer matrix) with `HG_LEDGER=1`, so
every game record additionally carries the full executed-trade ledger
(price/qty/buyer/seller/maker-taker ids/tick), per-tick L1 book tops+depth, and
per-tick player cash. Cells: 10 self-play monoculture diagonals (k=0), 8
ordered 1v3 pairs both directions (k=3), 6 mixed 2+2 cells (k=2).

Question (EconML theme 2): do co-adapted monocultures sustain trading
conventions — price levels, spreads, volumes — that strangers break
(algorithmic-collusion-like structure and its collapse under entry)?

Validation: with `HG_LEDGER=0` the harness is byte-identical to the base
campaign's; with `=1` the count-level fields (survival/win/cash/trade counts)
must reproduce the published `_peer_crossplay_100` records game-for-game
(same seeds, same torch-stream discipline). Check before analysis.

Deploy per `research/hpc/DEPLOY.md`. Eval-only harness: no separate preflight
(checkpoint paths are validated at startup). Output:
`$HUNGER_CAMPAIGN_ROOT/_econml_microstructure_100/`; sync back to
`workshops/econml/results/microstructure_hpc/` (raw JSONs ~3-6 MB/cell).
Analysis script lives laptop-side: `workshops/econml/analyze_microstructure.py`
(to be written against the synced raw).
