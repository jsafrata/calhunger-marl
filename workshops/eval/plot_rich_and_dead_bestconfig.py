"""Figure F1 (best-config, multi-seed) — "rich and dead" on the best architecture.

Two stacked panels over training checkpoints 100-1200 of the default_180_mlp best-config
campaign (IPPO+MAPPO x seeds 0-4; the 8 runs healthy in self-play, i.e. excluding the two
seeds that never learn self-play — including them only exaggerates the decline):
  top    — survival %: self-play (50 games/ckpt) vs cross-play (1 learned seat + 3x
           Scripted-v4 stockpiler, 100 games/ckpt, paired seeds). Mean line + min-max band.
  bottom — learned-seat mean terminal cash in the same cross-play games.

Reads the campaign CSVs directly; nothing hand-entered.

Usage:
  python workshops/eval/plot_rich_and_dead_bestconfig.py \
      --results-root workshops/results --out-dir workshops/figures [--opponent v4]
"""

from __future__ import annotations

import argparse
import statistics
from csv import DictReader
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CK = list(range(100, 1201, 100))
HEALTHY = {("ippo", "0"), ("ippo", "1"), ("ippo", "2"), ("ippo", "3"), ("ippo", "4"),
           ("mappo", "0"), ("mappo", "2"), ("mappo", "3")}
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        rows += [r for r in DictReader(open(p)) if r.get("mode", "stochastic") == "stochastic"
                 and (r["algorithm"], r["seed"]) in HEALTHY]
    return rows


def traj(rows: list[dict], field: str):
    mean, lo, hi = [], [], []
    for ck in CK:
        v = [float(r[field]) for r in rows if int(r["checkpoint"]) == ck]
        mean.append(statistics.mean(v)); lo.append(min(v)); hi.append(max(v))
    return mean, lo, hi


def band(ax, mean, lo, hi, color, lw=2.0):
    ax.fill_between(CK, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.plot(CK, mean, color=color, lw=lw, marker="o", ms=3.2, zorder=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--opponent", default="v4")
    args = ap.parse_args()
    rr = args.results_root

    sp = load([rr / "best_config_selfplay_survival_50/default_180_mlp/selfplay_survival_diagnostics.csv",
               rr / "seeds234/bestcfg/_selfplay_survival_50/default_180_mlp/selfplay_survival_diagnostics.csv"])
    sc = load([rr / f"best_config_scripted_eval_100/default_180_mlp/{args.opponent}/checkpoint_diagnostics.csv",
               rr / f"seeds234/bestcfg/_scripted_eval_100/default_180_mlp/{args.opponent}/checkpoint_diagnostics.csv"])

    sp_m, sp_lo, sp_hi = traj(sp, "survival_percentage")
    cx_m, cx_lo, cx_hi = traj(sc, "survival_percentage")
    ca_m, ca_lo, ca_hi = traj(sc, "mean_terminal_cash")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c4", "axes.linewidth": 0.8,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.labelcolor": INK})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 4.2), sharex=True,
                                   height_ratios=[3, 2], gridspec_kw={"hspace": 0.16})
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e7e6e2", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    band(ax1, sp_m, sp_lo, sp_hi, BLUE)
    band(ax1, cx_m, cx_lo, cx_hi, ORANGE)
    ax1.annotate("self-play (training signal)", (CK[6], sp_m[6]), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax1.annotate("vs storage trader v4\n(3 scripted copies)", (CK[8], cx_m[8]), xytext=(0, -30),
                 textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax1.set_ylabel("survival (%)")
    ax1.set_ylim(0, 100)

    band(ax2, ca_m, ca_lo, ca_hi, INK2)
    ax2.annotate(f"${ca_m[0]:.0f}", (CK[0], ca_m[0]), xytext=(2, 8),
                 textcoords="offset points", fontsize=8.5, color=INK)
    ax2.annotate(f"${ca_m[-1]:.0f}", (CK[-1], ca_m[-1]), xytext=(0, 8),
                 textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax2.set_ylabel("terminal cash ($)")
    ax2.set_xlabel("training checkpoint (PPO iteration)")
    ax2.set_xticks(CK[1::2])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"f1_rich_and_dead_bestconfig.{ext}", dpi=300,
                    bbox_inches="tight")
    print("selfplay:", [round(v) for v in sp_m])
    print("crossplay:", [round(v) for v in cx_m])
    print("cash:", [round(v) for v in ca_m])
    print("wrote", args.out_dir / "f1_rich_and_dead_bestconfig.{png,pdf}")


if __name__ == "__main__":
    main()
