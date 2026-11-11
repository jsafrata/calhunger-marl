"""Figure F1 — "rich and dead": self-play metrics certify a collapsing policy.

Two stacked panels over training checkpoints 1000-1400 of the ippo_shared_seed1 run
(v3_shared_vs_independent campaign):
  top    — survival %: contemporaneous self-play (training-time game_n_survivors/4,
           windowed mean) vs cross-play (1 learned seat + 3 Scripted-V1, 200 games,
           fixed seeds, stochastic sampling).
  bottom — learned seat mean terminal cash in the same cross-play games.

Reads the real artifacts (deep_eval JSONs + iters.jsonl); nothing hand-entered.
No dual axes: cash and survival are different quantities, so they get separate panels.

Usage:
  python workshops/eval/plot_rich_and_dead.py \
      --run-dir research/eval/results/v3_shared_vs_independent/ippo_shared_seed1 \
      --out-dir workshops/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHECKPOINTS = [1000, 1100, 1200, 1300, 1400]
WINDOW = 25  # iterations either side for the training-metric mean

# Validated categorical palette (dataviz reference instance, light mode, slots 1-2)
BLUE = "#2a78d6"  # self-play (the deceptive within-training signal)
ORANGE = "#eb6834"  # cross-play vs fixed population (ground truth)
INK = "#0b0b0b"
INK2 = "#52514e"


def load_crossplay(run_dir: Path) -> tuple[list[float], list[float]]:
    surv, cash = [], []
    for it in CHECKPOINTS:
        p = run_dir / f"deep_eval_iter{it}_1learned_3scripted_stochastic_200.json"
        s = json.loads(p.read_text())["lineups"]["1learned_3scripted"]["stochastic"]
        surv.append(100.0 * s["learned_seat_survival_rate"])
        cash.append(s["learned_seat_mean_final_cash"])
    return surv, cash


def load_selfplay(run_dir: Path) -> list[float]:
    by_iter: dict[int, float] = {}
    with open(run_dir / "iters.jsonl") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "iter" in d and d.get("game_n_survivors") is not None:
                by_iter[d["iter"]] = d["game_n_survivors"]
    out = []
    for it in CHECKPOINTS:
        vals = [v for i, v in by_iter.items() if abs(i - it) <= WINDOW]
        out.append(100.0 * (sum(vals) / len(vals)) / 4.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cross_surv, cross_cash = load_crossplay(args.run_dir)
    self_surv = load_selfplay(args.run_dir)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.edgecolor": "#c9c8c4",
            "axes.linewidth": 0.8,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.labelcolor": INK,
        }
    )
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.0, 4.0), sharex=True, height_ratios=[3, 2],
        gridspec_kw={"hspace": 0.18},
    )

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e7e6e2", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax1.plot(CHECKPOINTS, self_surv, color=BLUE, lw=2, marker="o", ms=4.5, zorder=3)
    ax1.plot(CHECKPOINTS, cross_surv, color=ORANGE, lw=2, marker="o", ms=4.5, zorder=3)
    ax1.annotate(
        "self-play (training signal)",
        (CHECKPOINTS[2], self_surv[2]), xytext=(0, 8),
        textcoords="offset points", ha="center", color=INK, fontsize=8.5,
    )
    ax1.annotate(
        "vs fixed scripted population",
        (CHECKPOINTS[2], cross_surv[2]), xytext=(8, 10),
        textcoords="offset points", ha="left", color=INK, fontsize=8.5,
    )
    ax1.set_ylabel("survival (%)")
    ax1.set_ylim(0, 100)

    ax2.plot(CHECKPOINTS, cross_cash, color=INK2, lw=2, marker="o", ms=4.5, zorder=3)
    ax2.annotate(
        f"${cross_cash[-1]:.0f}",
        (CHECKPOINTS[-1], cross_cash[-1]), xytext=(0, -13),
        textcoords="offset points", ha="center", color=INK, fontsize=8.5,
    )
    ax2.annotate(
        f"${cross_cash[0]:.0f}",
        (CHECKPOINTS[0], cross_cash[0]), xytext=(0, 8),
        textcoords="offset points", ha="center", color=INK, fontsize=8.5,
    )
    ax2.set_ylabel("terminal cash ($)")
    ax2.set_xlabel("training checkpoint (PPO iteration)")
    ax2.set_xticks(CHECKPOINTS)
    ax2.set_ylim(0, max(cross_cash) * 1.25)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"f1_rich_and_dead.{ext}", dpi=300, bbox_inches="tight")
    print("self-play %:", [round(v, 1) for v in self_surv])
    print("cross-play %:", cross_surv)
    print("cash:", [round(v, 2) for v in cross_cash])
    print("wrote", args.out_dir / "f1_rich_and_dead.{png,pdf}")


if __name__ == "__main__":
    main()
