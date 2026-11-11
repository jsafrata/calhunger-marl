"""Figure F2 — generalization matrix: survival at checkpoint 1200 by horizon x opponent.

Rows: episode-matched training horizons (60/90/120/150/180 s).
Cols: self-play (4 copies of the checkpoint) then each scripted opponent, ordered
easy-to-hard for the learned policies (v3 JIT, v2 maker, v5 sniper, v4 stockpiler).
Cell: mean survival % of the learned seat over the 4 runs (IPPO/MAPPO x seeds 0-1),
annotated mean with min-max range underneath.

Sources (regenerated extended CSVs + selfplay diagnostics), nothing hand-entered:
  workshops/results/episode_matched_5env_scripted_eval_100/<env>/<bot>/checkpoint_diagnostics_extended.csv
  workshops/results/episode_matched_5env_selfplay_survival_50/<env>/selfplay_survival_diagnostics.csv

Usage:
  python workshops/eval/plot_crossplay_matrix.py --results-root workshops/results \
      --out-dir workshops/figures [--checkpoint 1200]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ENVS = [("small_60", "60 s"), ("small_90", "90 s"), ("small_120", "120 s"),
        ("small_150", "150 s"), ("default_180", "180 s")]
OPPONENTS = [("selfplay", "self-play"), ("v3", "v3 (JIT)"), ("v2", "v2 (maker)"),
             ("v5", "v5 (sniper)"), ("v4", "v4 (stockpiler)")]
INK, INK2 = "#0b0b0b", "#52514e"


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def cell_values(results_root: Path, env: str, opp: str, checkpoint: int) -> list[float]:
    if opp == "selfplay":
        paths = [
            results_root / "episode_matched_5env_selfplay_survival_50" / env / "selfplay_survival_diagnostics.csv",
            results_root / "seeds234/epmatch/_selfplay_survival_50" / env / "selfplay_survival_diagnostics.csv",
        ]
    else:
        paths = [
            results_root / "episode_matched_5env_scripted_eval_100" / env / opp / "checkpoint_diagnostics_extended.csv",
            results_root / "seeds234/epmatch/_scripted_eval_100" / env / opp / "checkpoint_diagnostics.csv",
        ]
    rows = [r for p in paths if p.exists() for r in read_rows(p)
            if int(r["checkpoint"]) == checkpoint and r["mode"] == "stochastic"]
    return [float(r["survival_percentage"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=int, default=1200)
    args = ap.parse_args()

    means, ranges, counts = {}, {}, {}
    for env, _ in ENVS:
        for opp, _ in OPPONENTS:
            vals = cell_values(args.results_root, env, opp, args.checkpoint)
            means[(env, opp)] = sum(vals) / len(vals)
            ranges[(env, opp)] = (min(vals), max(vals))
            counts[(env, opp)] = len(vals)

    n_runs = sorted(set(counts.values()))
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    # single-hue sequential ramp (light -> dark blue), white at 0
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#2a78d6", "#123a6b"])

    grid = [[means[(env, opp)] for opp, _ in OPPONENTS] for env, _ in ENVS]
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(OPPONENTS)), [lbl for _, lbl in OPPONENTS])
    ax.set_yticks(range(len(ENVS)), [lbl for _, lbl in ENVS])
    ax.set_ylabel("training/eval horizon")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # white gridlines between cells
    for i in range(1, len(ENVS)):
        ax.axhline(i - 0.5, color="white", lw=2)
    for j in range(1, len(OPPONENTS)):
        ax.axvline(j - 0.5, color="white", lw=2)

    for i, (env, _) in enumerate(ENVS):
        for j, (opp, _) in enumerate(OPPONENTS):
            m = means[(env, opp)]
            lo, hi = ranges[(env, opp)]
            color = "white" if m > 55 else INK
            ax.text(j, i - 0.12, f"{m:.0f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
            ax.text(j, i + 0.26, f"{lo:.0f}–{hi:.0f}", ha="center", va="center",
                    color=color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("survival (%)", fontsize=8.5)
    cbar.outline.set_visible(False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"f2_crossplay_matrix.{ext}", dpi=300, bbox_inches="tight")
    print(f"checkpoint {args.checkpoint}; runs per cell: {n_runs}")
    for env, lbl in ENVS:
        print(lbl, [f"{means[(env, opp)]:.0f}" for opp, _ in OPPONENTS])
    print("wrote", args.out_dir / "f2_crossplay_matrix.{png,pdf}")


if __name__ == "__main__":
    main()
