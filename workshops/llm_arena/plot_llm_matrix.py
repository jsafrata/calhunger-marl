"""Figure F5 — LLM generalization matrix: survival by model x population.

Rows: the four LLMs. Cols: the mixed all-LLM economy, then the scripted
ladder ordered easy-to-hard for learned policies (v3, v2, v5, v4), then
the four selected trained-policy opponents (small_90 best-config @1200).
Cell: survival % of the LLM seat (bold) with win % underneath; cells
with n < 100 games additionally annotate n. Style mirrors F2
(plot_crossplay_matrix.py) so the two matrices read as one family.

Source (regenerated, nothing hand-entered):
  workshops/results/llm_arena/campaign_90s/llm_matrix.csv
  (produced by workshops/llm_arena/summarize_campaign.py)

Usage:
  python workshops/llm_arena/plot_llm_matrix.py \
      --matrix workshops/results/llm_arena/campaign_90s/llm_matrix.csv \
      --out-dir workshops/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MODELS = [("gpt", "GPT-5.6-luna"), ("deepseek", "DeepSeek v4-flash"),
          ("gemini", "Gemini 3.5 FL"), ("claude", "Claude Haiku 4.5")]
POPULATIONS = [("selfplay", "self-play"), ("3llm_mix", "3 LLMs"),
               ("v3", "v3"), ("v2", "v2"),
               ("v5", "v5"), ("v4", "v4"),
               ("ippo_s0", "IPPO s0"), ("ippo_s2", "IPPO s2"),
               ("mappo_s1", "MAPPO s1"), ("mappo_s3", "MAPPO s3")]
INK, INK2 = "#0b0b0b", "#52514e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = {}
    with open(args.matrix) as f:
        for r in csv.DictReader(f):
            rows[(r["model"], r["opponent"])] = r

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#2a78d6", "#123a6b"])

    grid = []
    for model, _ in MODELS:
        grid.append([
            float(rows[(model, pop)]["survival_percentage"])
            if (model, pop) in rows else float("nan")
            for pop, _ in POPULATIONS
        ])
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(POPULATIONS)), [lbl for _, lbl in POPULATIONS],
                  fontsize=8)
    ax.set_yticks(range(len(MODELS)), [lbl for _, lbl in MODELS], fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(1, len(MODELS)):
        ax.axhline(i - 0.5, color="white", lw=2)
    for j in range(1, len(POPULATIONS)):
        ax.axvline(j - 0.5, color="white", lw=2)

    for i, (model, _) in enumerate(MODELS):
        for j, (pop, _) in enumerate(POPULATIONS):
            row = rows.get((model, pop))
            if row is None:
                ax.text(j, i, "–", ha="center", va="center", color=INK2)
                continue
            surv = float(row["survival_percentage"])
            win = float(row["win_percentage"])
            n = int(row["n_games"])
            color = "white" if surv > 55 else INK
            ax.text(j, i - 0.16, f"{surv:.0f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
            sub = f"win {win:.0f}" + (f" · n{n}" if n < 100 else "")
            ax.text(j, i + 0.26, sub, ha="center", va="center",
                    color=color, fontsize=6.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("survival (%)", fontsize=8.5)
    cbar.outline.set_visible(False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"f5_llm_matrix.{ext}", dpi=300,
                    bbox_inches="tight")
    print("wrote", args.out_dir / "f5_llm_matrix.{png,pdf}")


if __name__ == "__main__":
    main()
