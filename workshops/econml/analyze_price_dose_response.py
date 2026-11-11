"""E31 follow-up: per-seat-k price dose-response + producer-role confound.

Task 1 — for each SEAT in each game, k = number of co-players (other 3 seats)
whose spec differs from the seat's spec (diagonals: k=0 for all seats; 1v3:
singleton k=3, majority k=1; 2+2: k=2 for all).  Pool the seat's BUY and SELL
markups over all cells/games per k, where a trade's markup is
(price - game-food-VWAP)/VWAP with VWAP the volume-weighted mean executed
price of that food in that game.  Two variants: all 32 cells, and a
HEALTHY-ONLY variant excluding any game whose lineup contains mappo_s1 or
mappo_s4.  Means are trade-level pooled; 95% CIs are percentile bootstrap
over GAMES (cluster bootstrap, B=2000, rng 20260901).

Task 2 — the same k-curve stratified by whether the buyer PRODUCES the food
bought (food_type in producers[buyer_idx]) vs must import it; the sell-side
analogue (seller produces vs not) is included for completeness.  If the
dose-response survives in the buyer-does-NOT-produce stratum, the
producer-role confound is dismissed.

Inputs : results/microstructure_hpc/raw/*.json  (32 cells x 100 games)
Outputs: results/microstructure_hpc/price_dose_response.csv
         results/microstructure_hpc/producer_confound.csv
         results/microstructure_hpc/price_dose_response.pdf / .png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE / "results" / "microstructure_hpc"

FAILED_SEEDS = {"mappo_s1", "mappo_s4"}
B = 2000
RNG_SEED = 20260901

# ---------------------------------------------------------------------------
# per-game extraction
# ---------------------------------------------------------------------------


def game_markups(game: dict) -> dict:
    """Return per-(k, side, stratum) -> (sum_of_markups, n_trades) for one game.

    side    : 'buy' | 'sell'
    stratum : 'all' | 'produces' | 'imports'
    """
    lineup = game["lineup"]
    producers = [set(p) for p in game["producers"]]
    k_of_seat = [
        sum(1 for j in range(4) if j != i and lineup[j] != lineup[i])
        for i in range(4)
    ]

    # per-food VWAP over this game's executed trades
    vol = defaultdict(float)
    notional = defaultdict(float)
    for t in game["trades"]:
        vol[t["food_type"]] += t["quantity"]
        notional[t["food_type"]] += t["price_per_unit"] * t["quantity"]
    vwap = {f: notional[f] / vol[f] for f in vol if vol[f] > 0}

    acc = defaultdict(lambda: [0.0, 0])  # (k, side, stratum) -> [sum, n]
    for t in game["trades"]:
        f = t["food_type"]
        if f not in vwap:
            continue
        m = (t["price_per_unit"] - vwap[f]) / vwap[f]
        bi, si = t["buyer_idx"], t["seller_idx"]
        b_strat = "produces" if f in producers[bi] else "imports"
        s_strat = "produces" if f in producers[si] else "imports"
        for key in ((k_of_seat[bi], "buy", "all"),
                    (k_of_seat[bi], "buy", b_strat),
                    (k_of_seat[si], "sell", "all"),
                    (k_of_seat[si], "sell", s_strat)):
            a = acc[key]
            a[0] += m
            a[1] += 1
    return acc


# ---------------------------------------------------------------------------
# pooled mean + cluster bootstrap over games
# ---------------------------------------------------------------------------


def pooled_ci(per_game: list[list[float]], rng: np.random.Generator):
    """per_game: list of [sum, n] pairs (one per contributing game).

    Returns (mean, lo, hi, n_trades) of the trade-level pooled mean with a
    percentile bootstrap resampling games.
    """
    sums = np.array([p[0] for p in per_game])
    ns = np.array([p[1] for p in per_game], dtype=float)
    n_trades = int(ns.sum())
    mean = float(sums.sum() / ns.sum())
    ng = len(sums)
    idx = rng.integers(0, ng, size=(B, ng))
    boot = sums[idx].sum(axis=1) / np.maximum(ns[idx].sum(axis=1), 1e-12)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi), n_trades


def write_csv(name: str, rows: list[dict]) -> None:
    path = ROOT / name
    cols = list(rows[0].keys())
    with path.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    # bins[(healthy_only, k, side, stratum)] = list of [sum, n] per game
    bins = defaultdict(list)
    n_games = {True: 0, False: 0}

    for path in sorted((ROOT / "raw").glob("*.json")):
        payload = json.loads(path.read_text())
        for g in payload["games"]:
            healthy = not (set(g["lineup"]) & FAILED_SEEDS)
            acc = game_markups(g)
            n_games[False] += 1
            n_games[True] += healthy
            for key, sn in acc.items():
                bins[(False,) + key].append(sn)
                if healthy:
                    bins[(True,) + key].append(sn)
        print(f"  {path.stem}: {len(payload['games'])} games")

    print(f"pooled games: all={n_games[False]} healthy_only={n_games[True]}")

    rng = np.random.default_rng(RNG_SEED)

    # ---- Task 1: price_dose_response.csv ---------------------------------
    dose_rows = []
    curve = {}  # (healthy, side, k) -> (mean, lo, hi) in pct, for figure
    for healthy in (False, True):
        for k in range(4):
            row = {"healthy_only": healthy, "k": k}
            for side in ("buy", "sell"):
                per_game = bins.get((healthy, k, side, "all"))
                mean, lo, hi, n = pooled_ci(per_game, rng)
                row[f"n_trades_{side}"] = n
                row[f"n_games_{side}"] = len(per_game)
                row[f"{side}_markup_pct"] = round(100 * mean, 3)
                row[f"{side}_ci_lo_pct"] = round(100 * lo, 3)
                row[f"{side}_ci_hi_pct"] = round(100 * hi, 3)
                curve[(healthy, side, k)] = (100 * mean, 100 * lo, 100 * hi)
            dose_rows.append(row)
    write_csv("price_dose_response.csv", dose_rows)

    # ---- Task 2: producer_confound.csv -----------------------------------
    conf_rows = []
    for healthy in (False, True):
        for k in range(4):
            for side in ("buy", "sell"):
                for strat in ("produces", "imports"):
                    per_game = bins.get((healthy, k, side, strat))
                    if not per_game:
                        continue
                    mean, lo, hi, n = pooled_ci(per_game, rng)
                    conf_rows.append({
                        "healthy_only": healthy, "k": k, "side": side,
                        "stratum": strat, "n_trades": n,
                        "n_games": len(per_game),
                        "markup_pct": round(100 * mean, 3),
                        "ci_lo_pct": round(100 * lo, 3),
                        "ci_hi_pct": round(100 * hi, 3),
                    })
    write_csv("producer_confound.csv", conf_rows)

    # ---- figure ----------------------------------------------------------
    make_figure(curve)

    # console recap
    for healthy in (False, True):
        tag = "healthy-only" if healthy else "all games"
        for side in ("buy", "sell"):
            xs = [f"{curve[(healthy, side, k)][0]:+.2f}%" for k in range(4)]
            print(f"{tag:12s} {side:4s} markup k=0..3: " + " -> ".join(xs))


def make_figure(curve: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # colorblind-safe pair (validated categorical slots 1-2)
    C_ALL = "#2a78d6"      # blue  — all games
    C_HEALTHY = "#eb6834"  # orange — healthy-only

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharex=True,
                             sharey=True)
    ks = np.arange(4)
    for ax, side, title in zip(
        axes, ("buy", "sell"),
        ("Buy markup vs game VWAP", "Sell markup vs game VWAP"),
    ):
        for healthy, color, label, dx in (
            (False, C_ALL, "all 32 cells", -0.06),
            (True, C_HEALTHY, "healthy-only", +0.06),
        ):
            mean = np.array([curve[(healthy, side, k)][0] for k in ks])
            lo = np.array([curve[(healthy, side, k)][1] for k in ks])
            hi = np.array([curve[(healthy, side, k)][2] for k in ks])
            ax.errorbar(
                ks + dx, mean, yerr=[mean - lo, hi - mean],
                color=color, marker="o", markersize=5, linewidth=2,
                capsize=3, label=label,
            )
        ax.axhline(0.0, color="#b0afa9", linewidth=1, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k = co-players of a different spec")
        ax.set_xticks(ks)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("markup vs game-food VWAP (%)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "Price dose-response: per-seat markup vs number of out-group "
        "co-players (95% game-bootstrap CIs)",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = ROOT / f"price_dose_response.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
