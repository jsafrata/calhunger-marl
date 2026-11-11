"""E26 follow-up: WHO sets the bad prices, and the producer-role control.

Question 1 (mechanism): decompose each LLM trade by maker/taker role.
"LLM accepts an expensive incumbent quote" (LLM is taker on an incumbent
maker order) is a different failure from "LLM posts a bad order that gets
picked off" (LLM is maker).  Attribution: every trade carries
``maker_order_id``; the replay's per-tick book scan gives an order
registry (order id -> owner seat, side).  The maker order's side settles
which counterparty is maker: an ask-side maker order means the SELLER is
the maker and the buyer is the taker, and vice versa.  Orders that fill
in the same tick they were posted may be absent from the registry
(scanned post-step) — those trades are counted as unattributed and the
coverage rate is reported.

Question 2 (confound): stratify buy markups by whether the buyer
PRODUCES the food bought.  Forced imports (buyer does not produce) are
the economically meaningful stratum.

Only validated games are used.  Outputs to results/llm_economy/:
maker_attribution.csv, producer_strata.csv.
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
        return key
    if key.startswith("scripted_"):
        return key
    return f"policy_{key}"


def main() -> None:
    maker_rows = []
    strata_rows = []
    cells = sorted(
        d for d in ROOT.iterdir() if d.is_dir() and (d / "replay").is_dir()
    )
    for cell_dir in cells:
        cell = cell_dir.name
        # accumulators keyed by (seat_class, side, role)
        markup = defaultdict(list)
        counts = defaultdict(int)
        unattributed = 0
        total = 0
        strat = defaultdict(list)
        n_games = 0
        for path in sorted((cell_dir / "replay").glob("game_*.json")):
            game = json.loads(path.read_text())
            if not game["validation"]["ok"]:
                continue
            n_games += 1
            lineup = game["lineup"]
            classes = [seat_class(k) for k in lineup]
            producers = game.get("producers") or [[] for _ in range(4)]
            orders = game.get("orders", {})
            trades = game["trades"]
            px = defaultdict(list)
            for t in trades:
                px[t["food_type"]].append((t["price_per_unit"], t["quantity"]))
            vwap = {
                f: sum(p * q for p, q in v) / max(1, sum(q for _, q in v))
                for f, v in px.items()
            }
            for t in trades:
                f = t["food_type"]
                m = (t["price_per_unit"] - vwap[f]) / vwap[f] if vwap[f] else 0.0
                bc, sc = classes[t["buyer_idx"]], classes[t["seller_idx"]]
                total += 1
                order = orders.get(str(t["maker_order_id"]))
                if order is None:
                    unattributed += 1
                    role_b, role_s = "unknown", "unknown"
                else:
                    if order["side"] == "ask":
                        role_b, role_s = "taker", "maker"
                    else:
                        role_b, role_s = "maker", "taker"
                markup[(bc, "buy", role_b)].append(m)
                markup[(sc, "sell", role_s)].append(m)
                counts[(bc, "buy", role_b)] += 1
                counts[(sc, "sell", role_s)] += 1
                # producer strata (buy side)
                key = (
                    bc,
                    "produces" if f in producers[t["buyer_idx"]] else "imports",
                )
                strat[key].append(m)

        if not n_games:
            continue
        cls_set = sorted({k[0] for k in markup})
        for cls in cls_set:
            row = {"cell": cell, "seat_class": cls, "n_games": n_games}
            for side in ("buy", "sell"):
                n_side = sum(
                    counts[(cls, side, r)] for r in ("maker", "taker", "unknown")
                )
                for role in ("maker", "taker"):
                    vals = markup[(cls, side, role)]
                    row[f"{side}_{role}_share_pct"] = (
                        round(100 * counts[(cls, side, role)] / n_side, 1)
                        if n_side else ""
                    )
                    row[f"{side}_{role}_markup_pct"] = (
                        round(100 * float(np.mean(vals)), 2) if vals else ""
                    )
            maker_rows.append(row)
        for (cls, stratum), vals in sorted(strat.items()):
            strata_rows.append({
                "cell": cell, "seat_class": cls, "stratum": stratum,
                "n_buys": len(vals),
                "buy_markup_pct": round(100 * float(np.mean(vals)), 2),
            })
        cov = 100 * (1 - unattributed / max(1, total))
        maker_rows[-1]["_cell_attribution_coverage_pct"] = round(cov, 1)

    def write_csv(name: str, rows: list[dict]) -> None:
        path = ROOT / name
        cols = sorted({c for r in rows for c in r}, key=lambda c: (c[0] == "_", c))
        with path.open("w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
        print(f"wrote {path} ({len(rows)} rows)")

    write_csv("maker_attribution.csv", maker_rows)
    write_csv("producer_strata.csv", strata_rows)


if __name__ == "__main__":
    main()
