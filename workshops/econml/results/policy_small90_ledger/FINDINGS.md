# E26c — price-discrimination control: quote traps, not bespoke discrimination

Local ledger runs (harness `hpc_staging/econml_microstructure/checkpoint_policy_crossplay.py`,
HG_LEDGER=1, small_90, seed base 910000 = the arena's, 100 games/cell): the four arena
policies in self-play monoculture + ippo_s0/ippo_s2 vs 3×v2. Script
`econml/run_policy_small90_controls.sh`; raw in `results/policy_small90_ledger/raw/`.

Question: are the expensive books LLM entrants face (E26/E26b) bespoke discrimination,
or the incumbents' home convention?

| | home (monoculture) | LLM-present cells |
|---|---|---|
| VWAP | 3.69–4.28 | 4.19–5.37 (+10–30%) |
| mean best ask | 13.5–15.8 | (spread_mean 11.5–16.0, similar books) |
| LLM mean buy price | — | 6.25–11.74 (**1.5–2.9× home VWAP**) |
| LLM mean sell price | — | 2.95–4.79 (≈ home VWAP or below) |

**Verdict: the standing overpriced asks exist at home too** (monoculture best asks
13.5–15.8 vs executed VWAP ~4) — co-adapted peers simply do not lift them, so home
trades clear near 4. LLM entrants DO lift them (E26b: 74–84% buy-taker share at
+29..+95% markups), paying 1.5–2.9× the home price while the aggregate market moves
only +10–30%. So the mechanism is **quote traps / adverse selection against naive
entrants**, not entrant-specific quoting: the incumbents' book is a standing screen
that their own kind reads and strangers cannot. Caveat: state-dependent quoting
(quotes reacting to LLM behavior within a game) is not fully ruled out by aggregate
book stats; the claim to publish is the conservative one above. Control lineage: the
policy-vs-v2 cells clear at VWAP 4.4–4.8 with best asks ~5.0 — against bots the same
policies quote tight; wide traps are a peer-economy phenomenon.
