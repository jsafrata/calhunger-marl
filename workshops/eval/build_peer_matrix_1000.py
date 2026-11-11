"""Merge the 100-game peer matrix with its 900-game seed extension and
emit a 1000-game crossplay_matrix.csv (plot_policy_crossplay schema),
plus the derived statistics the paper's peers section cites.

Output: workshops/results/seeds234/_peer_crossplay_1000/crossplay_matrix.csv
Figure: plot with plot_policy_crossplay.py --matrix <that csv>.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from crossplay_raw_common import HEALTHY, LABELS, load_peer_1v3, load_peer_selfplay

OUT = Path(__file__).resolve().parents[1] / "results" / "seeds234" / "_peer_crossplay_1000"


def main() -> None:
    pair = load_peer_1v3(extended=True)
    selfplay = load_peer_selfplay(extended=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for focal in LABELS:
        c = selfplay[focal]
        surv, won = c.values[:, 0], c.values[:, 1]
        rows.append({
            "focal": focal, "opponent": "self", "lineup_mode": "mixed",
            "mode": "stochastic", "n_games": c.n_games,
            "survival_percentage": round(100 * surv.mean(), 1),
            "win_percentage": round(100 * won.mean(), 1),
            "mean_terminal_cash": round(float(c.values[:, 2].mean()), 2),
        })
        for cell in pair[focal]:
            rows.append({
                "focal": focal, "opponent": f"3x{cell.opponent}",
                "lineup_mode": "1v3", "mode": "stochastic",
                "n_games": cell.n_games,
                "survival_percentage": round(100 * cell.values[:, 0].mean(), 1),
                "win_percentage": round(100 * cell.values[:, 1].mean(), 1),
                "mean_terminal_cash": round(float(cell.values[:, 2].mean()), 2),
            })
    with (OUT / "crossplay_matrix.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {OUT / 'crossplay_matrix.csv'}")

    # ---- paper statistics (healthy = 8 self-play-healthy policies) ------
    by = {(r["focal"], r["opponent"]): r for r in rows}
    diag = {f: by[(f, "self")]["survival_percentage"] for f in LABELS}
    print("\nself-play diagonal:",
          {f: diag[f] for f in LABELS})

    hh = [(f, o) for f in HEALTHY for o in HEALTHY if f != o]
    surv = [by[(f, f"3x{o}")]["survival_percentage"] for f, o in hh]
    win = [by[(f, f"3x{o}")]["win_percentage"] for f, o in hh]
    print(f"\nhealthy-vs-healthy ({len(hh)} ordered pairs): "
          f"survival mean {np.mean(surv):.1f} [{min(surv):.0f}-{max(surv):.0f}]  "
          f"win mean {np.mean(win):.1f} [{min(win):.0f}-{max(win):.0f}]  "
          f"pairs win<=10%: {sum(w <= 10 for w in win)}/{len(hh)}")

    for a, b in [("ippo_s2", "ippo_s4")]:
        r1, r2 = by[(a, f"3x{b}")], by[(b, f"3x{a}")]
        print(f"mirror {a}<->{b}: wins {r1['win_percentage']}<->{r2['win_percentage']}, "
              f"surv {r1['survival_percentage']} vs {r2['survival_percentage']}")

    for col in ["mappo_s1", "mappo_s4"]:
        cs = [by[(f, f"3x{col}")] for f in HEALTHY]
        print(f"healthy vs 3x{col}: surv "
              f"{min(c['survival_percentage'] for c in cs):.1f}-"
              f"{max(c['survival_percentage'] for c in cs):.1f}  cash "
              f"${min(c['mean_terminal_cash'] for c in cs):.0f}-"
              f"${max(c['mean_terminal_cash'] for c in cs):.0f}  win "
              f"{min(c['win_percentage'] for c in cs):.1f}-"
              f"{max(c['win_percentage'] for c in cs):.1f}")
    r = by[("mappo_s2", "3xmappo_s1")]
    print(f"mappo_s2 (self {diag['mappo_s2']}) vs 3xmappo_s1: "
          f"surv {r['survival_percentage']} win {r['win_percentage']} "
          f"cash ${r['mean_terminal_cash']:.0f}")


if __name__ == "__main__":
    main()
