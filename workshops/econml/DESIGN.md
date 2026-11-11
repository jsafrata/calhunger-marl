# EconML @ NeurIPS 2026 — experiment designs (E26–E31)

Venue: EconML: Economics for Machine Learning (Atlanta, deadline Aug 29 2026 AoE).
Theme 1: economic tools for ML training/alignment/evaluation (social choice, mechanism
design, strategic behavior in model evaluation). Theme 2: ecosystems of many interacting
models (algorithmic collusion, model monoculture, pricing of interacting agents,
emerging economic phenomena). CFP explicitly welcomes empirical evidence of economic
phenomena.

All numbers produced here feed `workshops/paper/EXPERIMENTS.md` (E26+). Everything is
offline/local except E31 (HPC). Nothing in `../research/` is modified (HPC campaign is
staged in `workshops/hpc_staging/` and ADDED under `research/hpc/` at launch only).

Conventions inherited: seat-balanced cyclic rotation, paired game seeds, stochastic
sampling, existing seed bases. RNG seeds for any bootstrap here: 20260830+.

---

## E26. The LLM economy, reconstructed — terms of trade & wealth transfer [HEADLINE, theme 2]

**Question**: through which *price channel* are production LLM agents economically
stripped in a machine economy of incumbent agents (E24's survive-but-never-win)?

**Method**: deterministic replay of all campaign_90s games (3,232 games, 39 cells,
seed base 910000, FoodTradingEnv90). For each game: rebuild the lineup (cyclic
rotation `(seat - game) % 4`), instantiate `environment_class(seed=game_seed)`, step
the env feeding each recorded LLM `action_index` (from `llm_decisions`) at its tick;
scripted seats re-run their deterministic heuristics; policy seats re-run the actual
checkpoints (local paths in identity) with `seed_everything(910000)` and games replayed
in ascending order to reproduce the torch sampling stream. Dump per game:
full `env.state.trades` ledger (price, qty, buyer, seller, maker/taker, tick),
per-tick per-seat cash/inventory snapshots, and per-tick best bid/ask per food.

**Validation (mandatory, per game)**: replay must reproduce (a) every recorded
`env_applied` verdict at each LLM decision tick, (b) `n_trades`,
(c) `trades_as_buyer/seller` per seat, (d) `terminal_cash` per seat,
(e) `survival_time_ticks`. A game failing any check is excluded and counted;
report the validation rate per cell. Policy-seat cells are the at-risk group
(sampling-stream resume caveat, arena.py docstring): if a policy cell validates
below threshold, it drops to count-level analyses only (the vs-scripted, selfplay,
and mixed_4llm cells have no policy seats and must validate ~100%).

**Analyses** (on validated games):
1. **Terms of trade**: for each executed trade, markup vs the contemporaneous
   opposite-side best quote and vs the game's rolling trade-price EMA; distribution
   per (seat-type buyer, seat-type seller) pair. Claim shape: LLMs systematically buy
   dear / sell cheap by X% vs incumbents' trades in the same games.
2. **Wealth-transfer matrix**: net cash flow between seat types (LLM/policy/scripted
   archetype), i.e. who extracts surplus from whom; per cell and pooled.
3. **Stripping → starvation**: per-seat cash+inventory trajectories aligned on death
   time; show the cash path collapsing before starvation (E24's late-death signature,
   now with the mechanism). Death as the *endpoint of expropriation*.
4. Side product for E31 comparison: per-cell price level/dispersion, volumes, spreads.

**Outputs**: `results/llm_economy/` — per-cell `replay/game_XXXX.json` (ledger +
series), `validation.csv`, `terms_of_trade.csv`, `wealth_transfer_matrix.csv`,
`stripping_trajectories.csv` + `.md` findings. Scripts:
`econml/replay_llm_economy.py`, `econml/analyze_llm_economy.py`.

**Cost**: $0 (no API calls — replay feeds recorded actions).

---

## E27. Leaderboards as social choice — aggregation & manipulability [theme 1]

**Question**: if each evaluation population (and each metric) is a *voter* over
policies, what do social-choice tools say about the leaderboard?

**Data**: existing raw — merged 1000-game peer matrix
(`workshops/eval/crossplay_raw_common.py`, `load_peer_1v3(extended=True)`), the
E12 population-sensitivity results, E20 metric portfolio.

**Analyses** (healthy-8 policies unless stated):
1. **Condorcet structure**: pairwise majority relation where voter v (an evaluation
   population = the 9 focal 1v3 columns; separately, the 6 metrics) prefers A to B iff
   A beats B under v. Count Condorcet cycles; is there a Condorcet winner?
2. **Rule dependence**: Borda, Copeland, plurality, Kemeny (n=8 ⇒ exact), and
   mean-score (the status quo) over population-voters and metric-voters; report how
   the winner and full ranking change with the aggregation rule.
3. **Manipulability of evaluation**: for each policy, does there exist a defensible
   single-population judge that ranks it #1 (E12 rebased says mappo_s2 spans ranks
   1–9)? Report |{policies that can be crowned}| / 8 and the judge that crowns each —
   "strategic behavior in model evaluation" made quantitative.
4. Stability interaction: rerun (2) under the E19 bootstrap (paired scheme, B=2000,
   rng 20260830) — is any rule's winner stable?

**Outputs**: `results/social_choice/` — `pairwise_majority.json`, `rules_ranking.csv`,
`crownable.csv`, `rule_stability.csv` + `.md`. Script: `econml/analyze_social_choice.py`.

---

## E28. Nash averaging vs the unstable leaderboard [theme 1]

**Question**: does an economic solution concept (equilibrium rating) beat the
macro-mean leaderboard's instability (E19)?

**Data**: merged 1000-game peer matrix (survival and win payoffs; 10 policies and
healthy 8), plus the scripted 4-way/pairwise tournament for a ladder demo.

**Method**: build the empirical 2-population meta-game M[i,j] = focal i's score vs
3×j (1v3 matrix, self-play diagonal included). Nash averaging à la Balduzzi et al.
2018: maxent Nash of the antisymmetrized zero-sum game A = (M − Mᵀ)/2 (solved as an
LP / regret minimization — no new deps: scipy.optimize.linprog); policy rating =
expected payoff vs the Nash mixture. Report ratings + Nash support. Then push the
SAME estimator through the E19 bootstrap machinery (paired + independent schemes,
n=100 and n=1000, B=2000, rng 20260831): top-1 stability, mean self-Spearman,
pairwise flip rates — side by side with the macro-mean leaderboard's numbers.
Also: replicator dynamics on the empirical game (cycle detection = the
non-transitivity, E2/E10a, as an economic phenomenon).

**Honest caveats**: 1v3 payoffs are not a symmetric 2-player game (document the
antisymmetrization choice); Nash support sensitivity to sampling noise is itself a
finding either way.

**Outputs**: `results/nash_averaging/` — `nash_ratings.csv`, `nash_vs_mean_stability.csv`,
`replicator_cycles.json` + `.md`. Script: `econml/analyze_nash_averaging.py`.

---

## E29. Welfare accounting — monoculture efficiency, dead wealth, inequality [theme 2]

**Question**: what does training-population composition do to *aggregate* economic
outcomes? Cash is conserved by trade (zero-sum in cash), so survival is the real
output; cash measures redistribution.

**Data**: peer cross-play raw per-seat records (terminal_cash, survived,
survival_time, buyer/seller trade counts) across k = 0,1,2,3 strangers (E18's
dose-response cells + self-play diagonal + 1v3 + 2v2), 100–1000 games/cell.

**Analyses**:
1. **Dead-wealth share**: fraction of terminal cash held by dead seats, per k and per
   population pair — the "rich and dead" externality as a deadweight number.
2. **Inequality**: terminal-cash Gini per game, aggregated per k; monoculture (k=0)
   vs mixed economies. Expected: monocultures are high-welfare/low-Gini; each
   stranger degrades welfare (~6 pts survival, E18) — quantify the inequality side.
3. **Trade-intensity economics**: total volume (trade counts) per k; buyer/seller
   role asymmetry per policy (maker-ish vs taker-ish learned styles); volume collapse
   vs welfare collapse — does the market thin before it kills?
4. **Welfare frontier figure**: aggregate survival (x) vs cash Gini (y), one point
   per population composition, k-coded.

**Outputs**: `results/welfare/` — `welfare_by_k.csv`, `dead_wealth.csv`,
`gini_by_population.csv`, frontier figure `.png/.pdf` + `.md`.
Script: `econml/analyze_welfare.py`.

---

## E30. Price formation on the heuristic ladder — local microstructure [supporting, theme 2]

**Question**: do the ladder's economic archetypes produce measurably different market
microstructure (the spread-earner v2 vs the position-taker v4, E2's economy rewarding
intertemporal position-taking)?

**Method**: copy (never edit in place) `workshops/eval/scripted_version_tournament.py`
→ `econml/scripted_tournament_ledger.py`, adding a full-ledger dump: per game, all
`env.state.trades` (asdict) + per-second best bid/ask. Re-run the v2–v5 four-way +
all pairwise 2v2 tournament locally (small_30 variant, 100 games four-way + 6×50
pairwise, seed base as original — exactly reproducible, scripted-only, minutes).

**Analyses**: per-archetype realized spread earned vs paid, maker share, price
dispersion by lineup, per-pair net cash flow (who farms whom — the non-transitivity
v4>v2, v5<v2 as measured cash flows), price-level differences across lineups.

**Outputs**: `results/ladder_microstructure/` — `ledger/` per game, `archetype_micro.csv`,
`pairflow.csv` + `.md`. Cost: local CPU minutes.

---

## E31. Monoculture vs mixed-economy microstructure — collusion test [HPC; camera-ready]

**Question**: do co-adapted self-play monocultures sustain trading conventions
(prices/spreads/volumes) that strangers break — tacit-collusion-like structure from
co-adaptation, and its collapse under entry?

**Method**: patched cross-play harness (same 4-line ledger dump as E30, applied to a
COPY in the campaign dir) re-running, with full trade + L1-book logging:
(a) the 10 self-play diagonals (k=0), (b) the 8 healthy-pair 1v3 extremes from E10a
(the 33↔4 mirrored pair, the catastrophic mappo_s1/mappo_s4 columns), (c) a spread of
k=1,2 mixed cells matching E18 — ~30 cells × 100 games, default_180_mlp best-config
@1200 checkpoints (cluster paths). Slurm array, one cell per task, structured like
`peer_crossplay_900ext` (same env/venv/account conventions, additive campaign dir
`research/hpc/econml_microstructure/` staged from `workshops/hpc_staging/`).

**Analyses** (offline after rsync): price level/dispersion, effective spread, volume,
trade-network density per cell; monoculture vs k>0 contrasts; does the co-adapted
price structure predict which stranger pairings collapse (link to E10a's win-channel
dominance)? Comparison row vs the LLM economy (E26.4) and the ladder (E30).

**Timeline**: submission cites design + E26/E30 as the empirical base; E31 lands for
camera-ready (non-archival venue; notification ~Sep 29).

---

## Venue-tag updates

E26 [EconML headline; also IAEval/WV color], E27+E28 [EconML theme 1; TAE crossover],
E29 [EconML theme 2], E30 [EconML supporting], E31 [EconML camera-ready].
