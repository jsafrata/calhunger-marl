"""E26 analysis: terms of trade, wealth transfer, and economic stripping.

Consumes the validated replay output of ``replay_llm_economy.py``
(``results/llm_economy/<cell>/replay/game_*.json``) and produces the
paper numbers.  Only games with ``validation.ok`` are used; every table
carries the n it was computed from.

Definitions (documented for the paper's methods):
- VWAP_{g,f}: volume-weighted mean executed price of food f in game g.
- Buy markup of a trade: (price − VWAP_{g,f}) / VWAP_{g,f}; positive =
  bought dearer than the game's average market. Sell markup is the same
  quantity from the seller side; positive = sold dearer (good for the
  seller). "Buys dear, sells cheap" ⇒ buy markup > 0 > sell markup.
- Pre-trade quote reference: best bid/ask of the PREVIOUS tick's
  post-clearing snapshot (the market state the trader could see).
- Wealth transfer: every trade moves total_price cash buyer→seller;
  net flow between seat classes is summed per game and averaged.
- Stripping trajectory: per-seat cash path aligned on death tick
  (x = tick − death_tick); survivors aligned on horizon end.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE / "results" / "llm_economy"


def seat_class(key: str) -> str:
    if key.startswith("llm_"):
        return key  # keep the model identity
    if key.startswith("scripted_"):
        return key
    return f"policy_{key}"


def coarse_class(key: str) -> str:
    if key.startswith("llm_"):
        return "llm"
    if key.startswith("scripted_"):
        return "scripted"
    return "policy"


def iter_validated_games(cell_dir: Path):
    for path in sorted((cell_dir / "replay").glob("game_*.json")):
        game = json.loads(path.read_text())
        if game["validation"]["ok"]:
            yield game


def main() -> None:
    cells = sorted(
        d for d in ROOT.iterdir()
        if d.is_dir() and (d / "replay").is_dir()
    )

    tot_rows = []          # terms of trade per (cell, seat_class)
    wt_rows = []           # wealth transfer per (cell, buyer_class, seller_class)
    strip_rows = []        # death-aligned mean cash paths per (cell, class)
    market_rows = []       # per-cell market summary

    for cell_dir in cells:
        cell = cell_dir.name
        # accumulators keyed by seat class
        buy_markups = defaultdict(list)
        sell_markups = defaultdict(list)
        buy_vs_quote = defaultdict(list)   # paid − prev best ask (ticks of slippage)
        sell_vs_quote = defaultdict(list)  # received − prev best bid
        buy_px = defaultdict(list)
        sell_px = defaultdict(list)
        buy_qty = defaultdict(float)
        sell_qty = defaultdict(float)
        flows = defaultdict(float)         # (buyer_class, seller_class) -> cash
        n_games = 0
        vwaps, vols, spreads = [], [], []
        # death-aligned cash paths: class -> offset -> [cash...]
        aligned = defaultdict(lambda: defaultdict(list))
        crisis = defaultdict(list)         # buy markup in last 10 ticks before death

        for game in iter_validated_games(cell_dir):
            n_games += 1
            lineup = game["lineup"]
            classes = [seat_class(k) for k in lineup]
            trades = game["trades"]
            ticks = game["ticks"]

            # per-food VWAP
            px = defaultdict(list)
            for t in trades:
                px[t["food_type"]].append(
                    (t["price_per_unit"], t["quantity"])
                )
            vwap = {
                f: sum(p * q for p, q in v) / max(1, sum(q for _, q in v))
                for f, v in px.items()
            }
            all_q = sum(t["quantity"] for t in trades)
            if all_q:
                vwaps.append(
                    sum(t["price_per_unit"] * t["quantity"] for t in trades)
                    / all_q
                )
            vols.append(all_q)
            sp = [
                b["best_ask"] - b["best_bid"]
                for snap in ticks for b in snap["books"].values()
                if b["best_ask"] is not None and b["best_bid"] is not None
            ]
            if sp:
                spreads.append(float(np.mean(sp)))

            # snapshots by tick for prev-quote reference
            snap_by_tick = {s["tick"]: s for s in ticks}

            # death ticks per seat
            death_tick = [None] * 4
            last_alive = [None] * 4
            for s in ticks:
                for i, seat in enumerate(s["seats"]):
                    if seat["alive"]:
                        last_alive[i] = s["tick"]
            for i in range(4):
                # died if not alive in final snapshot
                final_alive = ticks[-1]["seats"][i]["alive"] if ticks else True
                death_tick[i] = None if final_alive else last_alive[i]

            for t in trades:
                f = t["food_type"]
                m = (t["price_per_unit"] - vwap[f]) / vwap[f] if vwap[f] else 0.0
                bc, sc = classes[t["buyer_idx"]], classes[t["seller_idx"]]
                buy_markups[bc].append(m)
                sell_markups[sc].append(m)
                buy_px[bc].append(t["price_per_unit"])
                sell_px[sc].append(t["price_per_unit"])
                buy_qty[bc] += t["quantity"]
                sell_qty[sc] += t["quantity"]
                flows[(bc, sc)] += t["total_price"]
                prev = snap_by_tick.get(t["elapsed_second"] - 1)
                if prev is not None:
                    book = prev["books"].get(f)
                    if book:
                        if book["best_ask"] is not None:
                            buy_vs_quote[bc].append(
                                t["price_per_unit"] - book["best_ask"]
                            )
                        if book["best_bid"] is not None:
                            sell_vs_quote[sc].append(
                                t["price_per_unit"] - book["best_bid"]
                            )
                dt = death_tick[t["buyer_idx"]]
                if dt is not None and dt - t["elapsed_second"] <= 10:
                    crisis[classes[t["buyer_idx"]]].append(m)

            # death-aligned cash paths
            for s in ticks:
                for i in range(4):
                    anchor = (
                        death_tick[i]
                        if death_tick[i] is not None
                        else ticks[-1]["tick"]
                    )
                    off = s["tick"] - anchor
                    aligned[classes[i]][off].append(s["seats"][i]["cash"])

        if not n_games:
            continue

        market_rows.append({
            "cell": cell, "n_games": n_games,
            "vwap": round(float(np.mean(vwaps)), 3) if vwaps else "",
            "volume_mean": round(float(np.mean(vols)), 2),
            "price_dispersion": round(float(np.std(vwaps)), 3) if vwaps else "",
            "spread_mean": round(float(np.mean(spreads)), 3) if spreads else "",
        })

        for cls in sorted(set(list(buy_markups) + list(sell_markups))):
            bm, sm = buy_markups[cls], sell_markups[cls]
            tot_rows.append({
                "cell": cell, "seat_class": cls, "n_games": n_games,
                "n_buys": len(bm), "n_sells": len(sm),
                "buy_qty": int(buy_qty[cls]), "sell_qty": int(sell_qty[cls]),
                "buy_markup_pct": round(100 * float(np.mean(bm)), 2) if bm else "",
                "sell_markup_pct": round(100 * float(np.mean(sm)), 2) if sm else "",
                "buy_slippage_vs_prev_ask": round(float(np.mean(buy_vs_quote[cls])), 3) if buy_vs_quote[cls] else "",
                "sell_slippage_vs_prev_bid": round(float(np.mean(sell_vs_quote[cls])), 3) if sell_vs_quote[cls] else "",
                "mean_buy_price": round(float(np.mean(buy_px[cls])), 3) if bm else "",
                "mean_sell_price": round(float(np.mean(sell_px[cls])), 3) if sm else "",
                "crisis_buy_markup_pct": round(100 * float(np.mean(crisis[cls])), 2) if crisis[cls] else "",
                "n_crisis_buys": len(crisis[cls]),
            })

        pair_net = defaultdict(float)
        for (bc, sc), cash in flows.items():
            pair_net[(bc, sc)] += cash
        seen = set()
        for (a, b) in list(pair_net):
            if (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            net_ab = pair_net.get((a, b), 0.0) - pair_net.get((b, a), 0.0)
            wt_rows.append({
                "cell": cell,
                "payer_class": a, "receiver_class": b,
                "gross_a_pays_b": round(pair_net.get((a, b), 0.0) / n_games, 2),
                "gross_b_pays_a": round(pair_net.get((b, a), 0.0) / n_games, 2),
                "net_a_to_b_per_game": round(net_ab / n_games, 2),
            })

        for cls, offsets in aligned.items():
            for off in sorted(offsets):
                vals = offsets[off]
                strip_rows.append({
                    "cell": cell, "seat_class": cls,
                    "offset_from_death": off,
                    "mean_cash": round(float(np.mean(vals)), 2),
                    "n": len(vals),
                })

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        path = ROOT / name
        cols = list(rows[0].keys())
        with path.open("w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
        print(f"wrote {path} ({len(rows)} rows)")

    write_csv("terms_of_trade.csv", tot_rows)
    write_csv("wealth_transfer_matrix.csv", wt_rows)
    write_csv("stripping_trajectories.csv", strip_rows)
    write_csv("cell_market_summary.csv", market_rows)


if __name__ == "__main__":
    main()
