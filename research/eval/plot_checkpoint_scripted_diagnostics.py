"""Plot the five requested checkpoint-vs-scripted diagnostic metrics."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib import colormaps


METRICS = {
    "mean_survival_time_ticks": (
        "Average survival time",
        "Survival time (seconds)",
        "horizon",
    ),
    "mean_terminal_cash": ("Average terminal cash", "Cash", None),
    "mean_learned_trade_participations": (
        "Average learned-seat trades",
        "Trades per game",
        (0, None),
    ),
    "win_percentage": ("Win percentage", "Win percentage (%)", "percentage"),
    "survival_percentage": (
        "Survival percentage",
        "Survival percentage (%)",
        "percentage",
    ),
}


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "seed": int(row["seed"]),
                    "checkpoint": int(row["checkpoint"]),
                    "n_games": int(row["n_games"]),
                    "mode": row["mode"],
                    "environment_variant": row.get(
                        "environment_variant", "default"
                    ),
                    "game_duration_seconds": int(
                        row.get("game_duration_seconds") or 180
                    ),
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            )
    return rows


def validate_rows(
    rows: list[dict[str, object]], expected_series: int | None = None
) -> None:
    if not rows:
        raise ValueError("diagnostic CSV contains no rows")
    series = defaultdict(list)
    for row in rows:
        series[(row["algorithm"], row["seed"])].append(int(row["checkpoint"]))
    if expected_series is not None and len(series) != expected_series:
        raise ValueError(
            f"expected {expected_series} algorithm/seed series, found {len(series)}"
        )
    expected = sorted({int(row["checkpoint"]) for row in rows})
    incomplete = {
        key: sorted(values)
        for key, values in series.items()
        if sorted(values) != expected
    }
    if incomplete:
        raise ValueError(
            f"series do not share checkpoints {expected}: {incomplete}"
        )

    metadata = {
        (
            int(row["n_games"]),
            str(row["mode"]),
            str(row["environment_variant"]),
            int(row["game_duration_seconds"]),
        )
        for row in rows
    }
    if len(metadata) != 1:
        raise ValueError(f"diagnostic rows have inconsistent metadata: {metadata}")


def plot_metric(
    rows: list[dict[str, object]],
    metric: str,
    output_dir: Path,
    experiment_label: str | None = None,
    metric_ymax: Mapping[str, float] | None = None,
) -> None:
    title, ylabel, ylim = METRICS[metric]
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["algorithm"]), int(row["seed"]))].append(row)

    fig, ax = plt.subplots(figsize=(14.5, 8.5), constrained_layout=True)
    algorithm_colors = {"ippo": colormaps["Blues"], "mappo": colormaps["Oranges"]}
    markers = ("o", "s", "^", "D", "v")
    line_styles = ("-", "--", "-.", ":", (0, (5, 2)))
    for (algorithm, seed), values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["checkpoint"]))
        cmap = algorithm_colors.get(algorithm, colormaps["viridis"])
        color = cmap(0.42 + 0.12 * seed)
        ax.plot(
            [int(row["checkpoint"]) for row in values],
            [float(row[metric]) for row in values],
            label=f"{algorithm.upper()} seed {seed}",
            color=color,
            linestyle=line_styles[seed % len(line_styles)],
            marker=markers[seed % len(markers)],
            linewidth=2.8,
            markersize=7.5,
        )

    n_games = int(rows[0]["n_games"])
    mode = str(rows[0]["mode"])
    label = f" — {experiment_label}" if experiment_label else ""
    ax.set_title(f"{title}{label}", fontsize=22, pad=28, weight="semibold")
    ax.text(
        0.5,
        1.012,
        f"1 learned vs 3 scripted  •  {mode}  •  {n_games} games",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        color="#555555",
    )
    ax.set_xlabel("Training checkpoint (iteration)", fontsize=19, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=19, labelpad=10)
    ax.set_xticks(sorted({int(row["checkpoint"]) for row in rows}))
    ax.tick_params(axis="both", which="major", labelsize=15, width=1.2, length=6)
    if ylim == "percentage":
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 10))
    elif ylim == "horizon":
        ax.set_ylim(0, max(int(row["game_duration_seconds"]) for row in rows))
    elif ylim is not None:
        ax.set_ylim(*ylim)
    if metric_ymax and metric in metric_ymax:
        ax.set_ylim(0, metric_ymax[metric])
    ax.grid(axis="both", alpha=0.22, linewidth=0.7)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=14,
        frameon=False,
        handlelength=2.8,
        labelspacing=0.8,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / metric
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-series", type=int)
    parser.add_argument("--experiment-label")
    parser.add_argument(
        "--terminal-cash-ymax",
        type=float,
        help="fixed upper y-axis limit for terminal-cash plots",
    )
    parser.add_argument(
        "--trades-ymax",
        type=float,
        help="fixed upper y-axis limit for learned-trade plots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.csv.resolve())
    validate_rows(rows, expected_series=args.expected_series)
    metric_ymax = {
        metric: value
        for metric, value in (
            ("mean_terminal_cash", args.terminal_cash_ymax),
            ("mean_learned_trade_participations", args.trades_ymax),
        )
        if value is not None
    }
    for metric in METRICS:
        plot_metric(
            rows,
            metric,
            args.output_dir.resolve(),
            experiment_label=args.experiment_label,
            metric_ymax=metric_ymax,
        )
    print(f"wrote {len(METRICS)} PNG and PDF graphs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
