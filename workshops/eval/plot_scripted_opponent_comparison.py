"""Compare one environment's checkpoint diagnostics across scripted opponents."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps


OPPONENTS = ("v2", "v3", "v4", "v5")
SERIES = (("ippo", 0), ("ippo", 1), ("mappo", 0), ("mappo", 1))
METRICS = {
    "mean_survival_time_ticks": (
        "Average survival time against scripted bots",
        "Survival time (seconds)",
        "horizon",
    ),
    "mean_terminal_cash": (
        "Average terminal cash against scripted bots",
        "Terminal cash",
        "data",
    ),
    "mean_learned_trade_participations": (
        "Average learned-seat trades against scripted bots",
        "Trades per game",
        "data",
    ),
    "win_percentage": (
        "Win percentage against scripted bots",
        "Win percentage (%)",
        (0.0, 100.0),
    ),
    "survival_percentage": (
        "Survival percentage against scripted bots",
        "Survival percentage (%)",
        (0.0, 100.0),
    ),
}


def read_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for opponent in OPPONENTS:
        path = root / opponent / "checkpoint_diagnostics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "opponent": row["opponent"],
                        "algorithm": row["algorithm"],
                        "seed": int(row["seed"]),
                        "checkpoint": int(row["checkpoint"]),
                        "n_games": int(row["n_games"]),
                        "mode": row["mode"],
                        "game_duration_seconds": int(row["game_duration_seconds"]),
                        **{metric: float(row[metric]) for metric in METRICS},
                    }
                )
    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(row["opponent"], row["algorithm"], row["seed"])].append(
            row["checkpoint"]
        )
    expected_keys = {
        (opponent, algorithm, seed)
        for opponent in OPPONENTS
        for algorithm, seed in SERIES
    }
    if set(grouped) != expected_keys:
        raise ValueError(
            f"series mismatch: missing={sorted(expected_keys - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected_keys)}"
        )
    expected_checkpoints = list(range(100, 1201, 100))
    incomplete = {
        key: sorted(checkpoints)
        for key, checkpoints in grouped.items()
        if sorted(checkpoints) != expected_checkpoints
    }
    if incomplete:
        raise ValueError(f"incomplete series: {incomplete}")
    metadata = {
        (row["n_games"], row["mode"], row["game_duration_seconds"])
        for row in rows
    }
    if len(metadata) != 1:
        raise ValueError(f"unexpected evaluation metadata: {metadata}")


def metric_ylim(
    rows: list[dict[str, Any]], metric: str, scale: object
) -> tuple[float, float]:
    if scale == "horizon":
        return 0.0, float(rows[0]["game_duration_seconds"])
    if scale == "data":
        observed = max(float(row[metric]) for row in rows)
        step = 10.0
        return 0.0, max(step, step * math.ceil(observed * 1.08 / step))
    return scale  # type: ignore[return-value]


def series_style(algorithm: str, seed: int) -> dict[str, Any]:
    cmap = colormaps["Blues"] if algorithm == "ippo" else colormaps["Oranges"]
    return {
        "label": f"{algorithm.upper()} seed {seed}",
        "color": cmap(0.56 + 0.22 * seed),
        "linestyle": "-" if seed == 0 else "--",
        "marker": "o" if seed == 0 else "s",
        "linewidth": 2.8,
        "markersize": 6.8,
    }


def plot_metric(
    rows: list[dict[str, Any]],
    metric: str,
    output_dir: Path,
    configuration_label: str,
) -> None:
    title, ylabel, scale = METRICS[metric]
    ylim = metric_ylim(rows, metric, scale)
    duration = int(rows[0]["game_duration_seconds"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.15,
        top=0.87,
        hspace=0.34,
        wspace=0.24,
    )
    fig.suptitle(
        f"{title} — {duration}-second {configuration_label}",
        fontsize=23,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.91,
        "1 learned vs 3 scripted  •  stochastic  •  100 games per checkpoint",
        ha="center",
        fontsize=13,
        color="#555555",
    )
    for ax, opponent in zip(axes.ravel(), OPPONENTS):
        for algorithm, seed in SERIES:
            values = sorted(
                (
                    row
                    for row in rows
                    if row["opponent"] == opponent
                    and row["algorithm"] == algorithm
                    and row["seed"] == seed
                ),
                key=lambda row: row["checkpoint"],
            )
            ax.plot(
                [row["checkpoint"] for row in values],
                [row[metric] for row in values],
                **series_style(algorithm, seed),
            )
        ax.set_title(f"vs Scripted {opponent}", fontsize=18, fontweight="semibold")
        ax.set_xlabel("Training checkpoint", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xlim(80, 1220)
        ax.set_ylim(*ylim)
        ax.set_xticks((100, 300, 500, 700, 900, 1200))
        if ylim == (0.0, 100.0):
            ax.set_yticks(np.arange(0, 101, 20))
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.22, linewidth=0.8)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        fontsize=14,
        frameon=False,
        handlelength=3.0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / metric
    fig.savefig(stem.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration-label", default="best-config policy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.root.resolve())
    for metric in METRICS:
        plot_metric(
            rows,
            metric,
            args.output_dir.resolve(),
            args.configuration_label,
        )
    print(f"wrote {len(METRICS)} PNG/PDF comparison graphs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
