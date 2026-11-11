"""Plot survival percentage and average death tick across checkpoints."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps


METRICS = {
    "survival_percentage": (
        "Seat survival percentage in learned self-play",
        "Survival percentage (%)",
        "percentage",
    ),
    "average_death_tick": (
        "Average death tick in learned self-play",
        "Average death tick (seconds)",
        "horizon",
    ),
}


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        return [
            {
                "algorithm": row["algorithm"],
                "seed": int(row["seed"]),
                "checkpoint": int(row["checkpoint"]),
                "n_games": int(row["n_games"]),
                "mode": row["mode"],
                "environment_variant": row["environment_variant"],
                "game_duration_seconds": int(row["game_duration_seconds"]),
                **{metric: float(row[metric]) for metric in METRICS},
            }
            for row in csv.DictReader(handle)
        ]


def validate_rows(rows: list[dict[str, object]], expected_series: int = 4) -> None:
    if not rows:
        raise ValueError("self-play survival CSV contains no rows")
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["algorithm"]), int(row["seed"]))].append(
            int(row["checkpoint"])
        )
    if len(grouped) != expected_series:
        raise ValueError(f"expected {expected_series} series, found {len(grouped)}")
    checkpoints = sorted({int(row["checkpoint"]) for row in rows})
    incomplete = {
        key: sorted(values)
        for key, values in grouped.items()
        if sorted(values) != checkpoints
    }
    if incomplete:
        raise ValueError(f"series do not share checkpoints {checkpoints}: {incomplete}")
    variants = {str(row["environment_variant"]) for row in rows}
    durations = {int(row["game_duration_seconds"]) for row in rows}
    if len(variants) != 1 or len(durations) != 1:
        raise ValueError(
            f"plot input must contain one environment, found {variants}, {durations}"
        )


def plot_metric(
    rows: list[dict[str, object]], metric: str, output_dir: Path
) -> None:
    title, ylabel, ylim = METRICS[metric]
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["algorithm"]), int(row["seed"]))].append(row)

    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    colors = {"ippo": colormaps["Blues"], "mappo": colormaps["Oranges"]}
    markers = ("o", "s")
    styles = ("-", "--")
    for (algorithm, seed), values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["checkpoint"]))
        cmap = colors.get(algorithm, colormaps["viridis"])
        ax.plot(
            [int(row["checkpoint"]) for row in values],
            [float(row[metric]) for row in values],
            label=f"{algorithm.upper()} seed {seed}",
            color=cmap(0.45 + 0.22 * seed),
            linestyle=styles[seed % len(styles)],
            marker=markers[seed % len(markers)],
            linewidth=1.9,
            markersize=5,
        )

    checkpoints = sorted({int(row["checkpoint"]) for row in rows})
    duration = int(rows[0]["game_duration_seconds"])
    ax.set_title(
        f"{title} — {duration}-second environment "
        f"(stochastic, {int(rows[0]['n_games'])} games/checkpoint)"
    )
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(checkpoints)
    if ylim == "horizon":
        ax.set_ylim(0, duration)
    elif ylim == "percentage":
        ax.set_ylim(0, 100)
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8.5,
        frameon=False,
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
    parser.add_argument("--expected-series", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.csv.resolve())
    validate_rows(rows, expected_series=args.expected_series)
    for metric in METRICS:
        plot_metric(rows, metric, args.output_dir.resolve())
    print(f"wrote {len(METRICS)} PNG and PDF graphs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
