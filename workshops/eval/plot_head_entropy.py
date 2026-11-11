"""Figure F4 — head-entropy fingerprints: the price head sharpens iff learning succeeds.

One panel: conditional entropy of the price head (price | POST, nats) across training
checkpoints, for the best-config small_90 arms — MLP (learning succeeds, entropy decays)
vs LSTM (learning fails, entropy never sharpens). One thin line per run (IPPO/MAPPO x
seeds), direct labels, uniform-distribution reference line at ln(50).

Usage:
  python workshops/eval/plot_head_entropy.py --results-root workshops/results \
      --out-dir workshops/figures
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def load(paths: list[Path]) -> dict[str, list[tuple[int, float]]]:
    runs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for p in paths:
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            v = r.get("price_given_post_entropy", "")
            if v not in ("", "nan"):
                runs[f"{r['algorithm']}_s{r['seed']}"].append((int(r["checkpoint"]), float(v)))
    return {k: sorted(v) for k, v in runs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    mlp = load([args.results_root / "best_config_selfplay_survival_50/small_90_mlp/selfplay_survival_diagnostics.csv",
                args.results_root / "seeds234/bestcfg/_selfplay_survival_50/small_90_mlp/selfplay_survival_diagnostics.csv"])
    lstm = load([args.results_root / "best_config_selfplay_survival_50/small_90_lstm/selfplay_survival_diagnostics.csv"])

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": "#c9c8c4", "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.grid(axis="y", color="#e7e6e2", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.axhline(math.log(50), color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.annotate("uniform over 50 prices (ln 50)", (100, math.log(50)), xytext=(0, 4),
                textcoords="offset points", fontsize=7.5, color=INK2)

    for series, color in ((lstm, ORANGE), (mlp, BLUE)):
        for pts in series.values():
            ax.plot([c for c, _ in pts], [e for _, e in pts], color=color, lw=1.4, alpha=0.85)
    mlp_last = [pts[-1][1] for pts in mlp.values()]
    lstm_last = [pts[-1][1] for pts in lstm.values()]
    ax.annotate("LSTM runs (0–1% survival):\nnever sharpens", (620, 3.62), fontsize=8.5,
                color=INK, ha="left")
    ax.annotate("MLP runs (84–95% survival):\nprice head sharpens", (620, 2.35), fontsize=8.5,
                color=INK, ha="left")

    ax.set_xlabel("training checkpoint (PPO iteration)")
    ax.set_ylabel("price-head entropy (nats)")
    ax.set_xlim(80, 1220)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"f4_head_entropy.{ext}", dpi=300, bbox_inches="tight")
    print("mlp final:", [round(v, 2) for v in sorted(mlp_last)])
    print("lstm final:", [round(v, 2) for v in sorted(lstm_last)])
    print("wrote", args.out_dir / "f4_head_entropy.{png,pdf}")


if __name__ == "__main__":
    main()
