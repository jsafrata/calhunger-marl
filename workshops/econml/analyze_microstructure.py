"""E31 analysis: monoculture vs mixed-economy market microstructure.

Consumes the synced raw of the ``econml_microstructure`` HPC campaign
(32 cells x 100 games at seed base 900000 with HG_LEDGER=1) and asks the
collusion question: do co-adapted self-play monocultures sustain trading
conventions (price level, spread, volume) that strangers break?

Validation: the campaign replays the SAME games (seeds, torch-stream
discipline) as the published best-config peer matrix, so for every cell
that also exists in ``workshops/results/seeds234/_peer_crossplay_100/raw``
the count-level per-game fields (survival, terminal cash, trade counts)
must reproduce game-for-game.  Reported per cell; analysis proceeds only
if validation passes.

Outputs (results/microstructure_hpc/):
- validation_vs_base.csv
- cell_micro.csv       — per cell: VWAP, dispersion, volume, spread, ...
- monoculture_prices.csv — per-seed price conventions (k=0)
- mixing_contrasts.csv — mixed cells vs their constituent monocultures
- crosstype_terms.csv  — within- vs cross-type markups + net flows in
                         mixed/1v3 cells (the E26 machinery on peers)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE / "results" / "microstructure_hpc"
BASE_RAW = (
    _HERE.parent / "results" / "seeds234" / "_peer_crossplay_100" / "raw"
)


def load_cells() -> dict[str, dict]:
    cells = {}
    for path in sorted((ROOT / "raw").glob("*.json")):
        payload = json.loads(path.read_text())
        cells[path.stem] = payload
    return cells


def validate(cells: dict[str, dict]) -> list[dict]:
    rows = []
    for name, payload in cells.items():
        base_path = BASE_RAW / f"{name}.json"
        if not base_path.is_file():
            rows.append({"cell": name, "base_found": False, "n_match": "",
                         "n_games": len(payload["games"]), "ok": ""})
            continue
        base = json.loads(base_path.read_text())["games"]
        by_index = {g["game_index"]: g for g in base}
        n_match = 0
        for g in payload["games"]:
            b = by_index.get(g["game_index"])
            if b is None:
                continue
            same = (
                g["game_seed"] == b["game_seed"]
                and g["n_trades"] == b["n_trades"]
                and all(
                    gs["survived"] == bs["survived"]
                    and gs["terminal_cash"] == bs["terminal_cash"]
                    and gs["trades_as_buyer"] == bs["trades_as_buyer"]
                    and gs["trades_as_seller"] == bs["trades_as_seller"]
                    for gs, bs in zip(g["seats"], b["seats"])
                )
            )
            n_match += same
        rows.append({
            "cell": name, "base_found": True,
            "n_match": n_match, "n_games": len(payload["games"]),
            "ok": n_match == len(payload["games"]),
        })
    return rows


def game_micro(game: dict) -> dict:
    trades = game["trades"]
    qty = sum(t["quantity"] for t in trades)
    vwap = (
        sum(t["price_per_unit"] * t["quantity"] for t in trades) / qty
        if qty else None
    )
    prices = [t["price_per_unit"] for t in trades]
    spreads = [
        b["ba"] - b["bb"]
        for tick in game.get("ticks", []) for b in tick["books"].values()
        if b["ba"] is not None and b["bb"] is not None
    ]
    two_sided = (
        len(spreads)
        / max(1, sum(len(t["books"]) for t in game.get("ticks", [])))
    )
    return {
        "vwap": vwap,
        "volume": qty,
        "n_trades": len(trades),
        "price_std": float(np.std(prices)) if prices else None,
        "spread": float(np.mean(spreads)) if spreads else None,
        "two_sided_share": two_sided,
        "survivors": sum(s["survived"] for s in game["seats"]),
    }


def kind_of(name: str, lineup: list[str]) -> tuple[str, int]:
    uniq = sorted(set(lineup))
    if len(uniq) == 1:
        return "monoculture", 0
    counts = {u: lineup.count(u) for u in uniq}
    if sorted(counts.values()) == [1, 3]:
        return "1v3", 3
    return "mixed22", 2


def main() -> None:
    cells = load_cells()
    print(f"loaded {len(cells)} cells")

    val = validate(cells)
    n_ok = sum(1 for r in val if r["ok"] is True)
    n_base = sum(1 for r in val if r["base_found"])
    print(f"validation vs base campaign: {n_ok}/{n_base} cells reproduce "
          f"game-for-game ({len(val) - n_base} cells have no base twin)")

    micro_rows = []
    mono_price = {}
    crosstype_rows = []
    for name, payload in cells.items():
        games = payload["games"]
        lineup0 = games[0]["lineup"]
        kind, k = kind_of(name, lineup0)
        stats = defaultdict(list)
        for g in games:
            m = game_micro(g)
            for key, v in m.items():
                if v is not None:
                    stats[key].append(v)
        row = {"cell": name, "kind": kind, "k": k,
               "n_games": len(games)}
        for key in ("vwap", "volume", "n_trades", "price_std", "spread",
                    "two_sided_share", "survivors"):
            row[key] = round(float(np.mean(stats[key])), 3) if stats[key] else ""
        micro_rows.append(row)
        if kind == "monoculture":
            mono_price[lineup0[0]] = {
                "vwap": float(np.mean(stats["vwap"])) if stats["vwap"] else None,
                "volume": float(np.mean(stats["volume"])),
                "spread": float(np.mean(stats["spread"])) if stats["spread"] else None,
                "survivors": float(np.mean(stats["survivors"])),
            }

        # cross-type vs within-type terms of trade (mixed cells only)
        if kind != "monoculture":
            buy_m = defaultdict(list)
            flows = defaultdict(float)
            vols = defaultdict(int)
            for g in games:
                lineup = g["lineup"]
                trades = g["trades"]
                qty = sum(t["quantity"] for t in trades)
                if not qty:
                    continue
                vwap = sum(
                    t["price_per_unit"] * t["quantity"] for t in trades
                ) / qty
                for t in trades:
                    b, s = lineup[t["buyer_idx"]], lineup[t["seller_idx"]]
                    scope = "within" if b == s else "cross"
                    buy_m[scope].append(
                        (t["price_per_unit"] - vwap) / vwap
                    )
                    vols[scope] += t["quantity"]
                    if b != s:
                        flows[(b, s)] += t["total_price"]
            pair_names = sorted({x for p in flows for x in p})
            net = 0.0
            if len(pair_names) == 2:
                a, b = pair_names
                # flows[(buyer, seller)] accumulates cash buyer -> seller;
                # net > 0 means a pays b on net (a is the net cash loser).
                net = (flows.get((a, b), 0) - flows.get((b, a), 0)) / len(games)
            crosstype_rows.append({
                "cell": name, "kind": kind, "k": k,
                "within_buy_markup_pct": round(
                    100 * float(np.mean(buy_m["within"])), 2
                ) if buy_m["within"] else "",
                "cross_buy_markup_pct": round(
                    100 * float(np.mean(buy_m["cross"])), 2
                ) if buy_m["cross"] else "",
                "within_volume_share": round(
                    vols["within"] / max(1, vols["within"] + vols["cross"]), 3
                ),
                "types": "|".join(pair_names),
                "net_first_pays_second_per_game": round(net, 2),
            })

    # mixing contrasts: mixed cell vs mean of constituent monocultures
    contrast_rows = []
    for row in micro_rows:
        if row["kind"] == "monoculture":
            continue
        types = sorted(set(cells[row["cell"]]["games"][0]["lineup"]))
        monos = [mono_price[t] for t in types if t in mono_price]
        if len(monos) != 2:
            continue
        exp_vwap = np.mean([m["vwap"] for m in monos if m["vwap"]])
        exp_vol = np.mean([m["volume"] for m in monos])
        exp_surv = np.mean([m["survivors"] for m in monos])
        contrast_rows.append({
            "cell": row["cell"], "kind": row["kind"], "k": row["k"],
            "vwap": row["vwap"], "expected_vwap_from_monos": round(exp_vwap, 3),
            "volume": row["volume"], "expected_volume": round(exp_vol, 1),
            "volume_ratio": round(row["volume"] / exp_vol, 3) if exp_vol else "",
            "survivors": row["survivors"],
            "expected_survivors": round(exp_surv, 2),
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

    write_csv("validation_vs_base.csv", val)
    write_csv("cell_micro.csv", micro_rows)
    write_csv(
        "monoculture_prices.csv",
        [
            {"policy": p, **{k: round(v, 3) if v else "" for k, v in d.items()}}
            for p, d in sorted(mono_price.items())
        ],
    )
    write_csv("mixing_contrasts.csv", contrast_rows)
    write_csv("crosstype_terms.csv", crosstype_rows)


if __name__ == "__main__":
    main()
