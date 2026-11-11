"""Audit — recompute the primary headline stats with proper 95% CIs,
using games as the independent unit for game-level metrics and
game-clustered bootstrap for seat-level metrics.

The default summarizer uses per-observation SE for CIs on seat-level
outcomes, which underestimates uncertainty because the 4 seats of a
single game are correlated (they share producers, initial inventory,
and market conditions). Clustered bootstrap fixes this by resampling
whole games (not seats).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


N_BOOT = 2000
ROOT = Path(__file__).resolve().parent / "results" / "main"


def _cluster_bootstrap(
    groups: list[list[float]],
    stat=np.mean,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Game-clustered bootstrap. `groups` is a list of lists — each inner
    list is the per-seat observations for one game. Returns (point, lo, hi)."""
    if not groups:
        return float("nan"), float("nan"), float("nan")
    flat_all = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    point = float(stat(flat_all))
    rng = np.random.default_rng(seed)
    n_games = len(groups)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_games, size=n_games)
        resampled = np.concatenate([np.asarray(groups[j], dtype=float) for j in idx])
        boots[i] = stat(resampled)
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _game_bootstrap(
    values: list[float], stat=np.mean, n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float, float]:
    """Plain bootstrap over one-value-per-game series."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    point = float(stat(arr))
    rng = np.random.default_rng(seed)
    boots = stat(arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))], axis=1)
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    # Per-config accumulators. Game-level = one-value-per-game.
    # Seat-level = per-seat observations grouped by game.
    per_cfg: dict[str, dict] = defaultdict(lambda: defaultdict(list))

    for f in sorted(ROOT.glob("*.jsonl")):
        cfg = f.stem
        with open(f) as fh:
            for line in fh:
                g = json.loads(line)
                if "error" in g:
                    continue
                final = g["final"]
                per_cfg[cfg]["reached_time_limit"].append(
                    1 if final["reached_time_limit"] else 0
                )
                per_cfg[cfg]["no_survivor"].append(1 if final["no_survivors"] else 0)
                per_cfg[cfg]["n_survivors"].append(final["n_survivors"])
                per_cfg[cfg]["n_trades_total"].append(final["n_trades_total"])
                # Seat-level: grouped by this game.
                per_cfg[cfg]["seat_survived"].append([
                    p["survived"] for p in final["players"]
                ])
                per_cfg[cfg]["seat_final_cash"].append([
                    p["final_cash"] for p in final["players"]
                ])
                per_cfg[cfg]["seat_units_bought"].append([
                    p["total_bought"] for p in final["players"]
                ])
                # Per-cadence: (cadence, game_grouped_values)
                # Store as (cadence, value) pairs; we'll re-group per cadence below.
                for pb, pf in zip(g["players_behavior"], final["players"]):
                    per_cfg[cfg][f"seat_survived_by_cad__{pb['cadence']}"].append([pf["survived"]])
                    per_cfg[cfg][f"decisions_by_cad__{pb['cadence']}"].append([pb["decisions_taken"]])

    out_path = ROOT / "ci_audit.csv"
    fields = [
        "config", "metric", "unit", "n", "point_estimate", "ci95_lo", "ci95_hi",
    ]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cfg, buckets in sorted(per_cfg.items()):
            # Game-level metrics — treat each game as the independent unit.
            for metric in ("reached_time_limit", "no_survivor", "n_survivors",
                           "n_trades_total"):
                point, lo, hi = _game_bootstrap(
                    buckets[metric], seed=hash((cfg, metric)) & 0xFFFF
                )
                w.writerow({
                    "config": cfg, "metric": metric, "unit": "game",
                    "n": len(buckets[metric]),
                    "point_estimate": f"{point:.4f}",
                    "ci95_lo": f"{lo:.4f}", "ci95_hi": f"{hi:.4f}",
                })
            # Seat-level metrics — game-clustered bootstrap so nested-seat
            # correlation isn't hidden.
            for metric in ("seat_survived", "seat_final_cash", "seat_units_bought"):
                grouped: list[list[float]] = buckets[metric]
                point, lo, hi = _cluster_bootstrap(
                    grouped, seed=hash((cfg, metric)) & 0xFFFF
                )
                w.writerow({
                    "config": cfg, "metric": metric, "unit": "seat (game-clustered)",
                    "n": sum(len(x) for x in grouped),
                    "point_estimate": f"{point:.4f}",
                    "ci95_lo": f"{lo:.4f}", "ci95_hi": f"{hi:.4f}",
                })
            # Per-cadence survival (seat-level, game-clustered).
            for cadence in ("fast", "medium", "slow"):
                key = f"seat_survived_by_cad__{cadence}"
                if key not in buckets or not buckets[key]:
                    continue
                point, lo, hi = _cluster_bootstrap(
                    buckets[key], seed=hash((cfg, "survival", cadence)) & 0xFFFF
                )
                w.writerow({
                    "config": cfg, "metric": f"seat_survived[{cadence}]",
                    "unit": "seat (game-clustered)",
                    "n": sum(len(x) for x in buckets[key]),
                    "point_estimate": f"{point:.4f}",
                    "ci95_lo": f"{lo:.4f}", "ci95_hi": f"{hi:.4f}",
                })
    print(f"[audit-ci] wrote {out_path}")


if __name__ == "__main__":
    main()
