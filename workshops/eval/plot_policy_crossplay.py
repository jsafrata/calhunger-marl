"""Figure F3 — policy-vs-policy cross-play (v3 campaign, default 180 s env).

Heatmap of focal-seat survival % (fill) with win % underneath, for every ordered
1v3 pairing among independently trained checkpoints, plus the uniform self-play
baseline as the first column. Rows are focal policies, columns the population the
other three seats play. Reads crossplay_matrix.csv produced by
checkpoint_policy_crossplay.py; stochastic rows only.

Usage:
  python workshops/eval/plot_policy_crossplay.py \
      --matrix workshops/results/policy_crossplay/v3_180s_iter1100/crossplay_matrix.csv \
      --out-dir workshops/figures --name f3_policy_crossplay_v3_180s
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2 = "#0b0b0b", "#52514e"
LABELS = {
    "shared_s1": "shared s1",
    "shared_s2": "shared s2",
    "sharedid_s0": "shared+ID s0",
    "sharedid_s1": "shared+ID s1",
    "shared_s1_1400": "shared s1 @1400",
}


def display(label: str) -> str:
    if label in LABELS:
        return LABELS[label]
    return label.replace("ippo_s", "IPPO s").replace("mappo_s", "MAPPO s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", default="f3_policy_crossplay")
    ap.add_argument("--policies", default=None,
                    help="comma-separated policy labels in display order; default: v3 set")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.matrix)) if r["mode"] == "stochastic"]
    self_rows = {r["focal"]: r for r in rows if r["opponent"] == "self"}
    pair_rows = {}
    for r in rows:
        opp = r["opponent"]
        if r["lineup_mode"] == "1v3" and opp.startswith("3x"):
            pair_rows[(r["focal"], opp[2:])] = r

    order = args.policies.split(",") if args.policies else list(LABELS)
    focals = [f for f in order if f in self_rows]
    opponents = ["self"] + [f for f in order if f != "shared_s1_1400"
                            and any((fo, f) in pair_rows for fo in focals)]

    def cell(focal: str, opp: str):
        r = self_rows.get(focal) if opp == "self" else pair_rows.get((focal, opp))
        if r is None or focal == opp:
            return None
        return float(r["survival_percentage"]), float(r["win_percentage"])

    grid = [[cell(f, o) for o in opponents] for f in focals]

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(1.05 + 0.62 * len(opponents), 0.55 + 0.62 * len(focals)))
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#2a78d6", "#123a6b"])
    data = [[c[0] if c else float("nan") for c in row] for row in grid]
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    short = {"shared_s1": "shared s1", "shared_s2": "shared s2",
             "sharedid_s0": "+ID s0", "sharedid_s1": "+ID s1"}
    ax.set_xticks(range(len(opponents)),
                  ["self-play" if o == "self" else f"3× {short.get(o, display(o))}" for o in opponents],
                  fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(focals)), [display(f) for f in focals], fontsize=7)
    ax.set_ylabel("focal policy (1 seat)")
    ax.set_xlabel("other three seats")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(1, len(focals)):
        ax.axhline(i - 0.5, color="white", lw=2)
    for j in range(1, len(opponents)):
        ax.axvline(j - 0.5, color="white", lw=2)

    for i, f in enumerate(focals):
        for j, o in enumerate(opponents):
            c = grid[i][j]
            if c is None:
                ax.text(j, i, "—", ha="center", va="center", color=INK2)
                continue
            surv, win = c
            color = "white" if surv > 55 else INK
            ax.text(j, i - 0.16, f"{surv:.0f}", ha="center", va="center",
                    color=color, fontsize=8.5, fontweight="bold")
            ax.text(j, i + 0.24, f"win {win:.0f}", ha="center", va="center",
                    color=color, fontsize=5.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("survival (%)", fontsize=8.5)
    cbar.outline.set_visible(False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{args.name}.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", args.out_dir / f"{args.name}.{{png,pdf}}")


if __name__ == "__main__":
    main()
