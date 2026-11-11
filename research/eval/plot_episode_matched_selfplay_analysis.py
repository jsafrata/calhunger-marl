"""Plot cross-horizon self-play survival and policy-head diagnostics.

The input is the ``_selfplay_survival_50`` output produced by
``eval.checkpoint_selfplay_survival_diagnostics``.  The plots deliberately
normalize death time by the environment horizon and entropy by each head's
theoretical maximum so the five episode lengths can be compared directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps


HORIZONS = (
    ("small_60", 60),
    ("small_90", 90),
    ("small_120", 120),
    ("small_150", 150),
    ("default_180", 180),
)
SERIES = (("ippo", 0), ("ippo", 1), ("mappo", 0), ("mappo", 1))
HEAD_MAX_ENTROPY = {
    "type": math.log(4),
    "food": math.log(4),
    "side": math.log(2),
    "price": math.log(50),
    "quantity": math.log(5),
}
RAW_NAME = re.compile(
    r"(?P<algorithm>ippo|mappo)_seed(?P<seed>[01])_iter(?P<checkpoint>\d+)_"
)


@dataclass(frozen=True)
class Row:
    horizon: int
    algorithm: str
    seed: int
    checkpoint: int
    n_games: int
    survival_percentage: float
    average_death_tick: float
    type_prob_post_mean: float
    type_entropy_mean: float
    price_given_post_entropy: float

    @property
    def death_time_percentage(self) -> float:
        return 100.0 * self.average_death_tick / self.horizon

    @property
    def post_probability_percentage(self) -> float:
        return 100.0 * self.type_prob_post_mean

    @property
    def normalized_type_entropy(self) -> float:
        return 100.0 * self.type_entropy_mean / HEAD_MAX_ENTROPY["type"]

    @property
    def effective_price_choices(self) -> float:
        return math.exp(self.price_given_post_entropy)


def _float(value: str, *, field: str, path: Path) -> float:
    if value == "":
        raise ValueError(f"missing {field} in {path}")
    return float(value)


def read_rows(root: Path) -> list[Row]:
    rows: list[Row] = []
    for directory, horizon in HORIZONS:
        path = root / directory / "selfplay_survival_diagnostics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                rows.append(
                    Row(
                        horizon=horizon,
                        algorithm=raw["algorithm"],
                        seed=int(raw["seed"]),
                        checkpoint=int(raw["checkpoint"]),
                        n_games=int(raw["n_games"]),
                        survival_percentage=_float(
                            raw["survival_percentage"],
                            field="survival_percentage",
                            path=path,
                        ),
                        average_death_tick=_float(
                            raw["average_death_tick"],
                            field="average_death_tick",
                            path=path,
                        ),
                        type_prob_post_mean=_float(
                            raw["type_prob_post_mean"],
                            field="type_prob_post_mean",
                            path=path,
                        ),
                        type_entropy_mean=_float(
                            raw["type_entropy_mean"],
                            field="type_entropy_mean",
                            path=path,
                        ),
                        price_given_post_entropy=_float(
                            raw["price_given_post_entropy"],
                            field="price_given_post_entropy",
                            path=path,
                        ),
                    )
                )
    validate_rows(rows)
    return rows


def validate_rows(rows: list[Row]) -> None:
    if not rows:
        raise ValueError("no self-play rows found")
    grouped: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(row.horizon, row.algorithm, row.seed)].append(row.checkpoint)
    expected_keys = {
        (horizon, algorithm, seed)
        for _, horizon in HORIZONS
        for algorithm, seed in SERIES
    }
    if set(grouped) != expected_keys:
        raise ValueError(
            f"series mismatch: missing={sorted(expected_keys - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected_keys)}"
        )
    checkpoint_sets = {tuple(sorted(values)) for values in grouped.values()}
    if len(checkpoint_sets) != 1:
        raise ValueError("not every horizon/algorithm/seed has the same checkpoints")
    checkpoints = next(iter(checkpoint_sets))
    if checkpoints != tuple(range(100, 1201, 100)):
        raise ValueError(f"expected checkpoints 100..1200, found {checkpoints}")


def read_head_diagnostics(root: Path) -> dict[tuple[int, str, int, int], dict[str, Any]]:
    diagnostics: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    for directory, horizon in HORIZONS:
        raw_dir = root / directory / "raw"
        for path in sorted(raw_dir.glob("*.json")):
            match = RAW_NAME.match(path.name)
            if match is None:
                continue
            report = json.loads(path.read_text())
            head = report.get("head_diagnostics")
            if not head:
                raise ValueError(f"missing head_diagnostics in {path}")
            key = (
                horizon,
                match.group("algorithm"),
                int(match.group("seed")),
                int(match.group("checkpoint")),
            )
            diagnostics[key] = head
    expected = len(HORIZONS) * len(SERIES) * 12
    if len(diagnostics) != expected:
        raise ValueError(f"expected {expected} raw head reports, found {len(diagnostics)}")
    return diagnostics


def _series_style(algorithm: str, seed: int) -> dict[str, Any]:
    cmap = colormaps["Blues"] if algorithm == "ippo" else colormaps["Oranges"]
    return {
        "color": cmap(0.56 + 0.22 * seed),
        "linestyle": "-" if seed == 0 else "--",
        "marker": "o" if seed == 0 else "s",
        "linewidth": 2.7,
        "markersize": 6.5,
        "label": f"{algorithm.upper()} seed {seed}",
    }


def _facet_axes(title: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), constrained_layout=True)
    fig.suptitle(title, fontsize=23, fontweight="semibold")
    return fig, axes.ravel()


def _format_axis(ax: plt.Axes, *, horizon: int, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_title(f"{horizon}-second environment", fontsize=17, fontweight="semibold")
    ax.set_xlabel("Training checkpoint", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlim(80, 1220)
    ax.set_ylim(*ylim)
    ax.set_xticks((100, 300, 500, 700, 900, 1200))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.22, linewidth=0.8)


def plot_faceted_metric(
    rows: list[Row],
    *,
    value_name: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    output_dir: Path,
    stem: str,
) -> None:
    fig, axes = _facet_axes(title)
    for ax, (_, horizon) in zip(axes, HORIZONS):
        for algorithm, seed in SERIES:
            values = sorted(
                (
                    row
                    for row in rows
                    if row.horizon == horizon
                    and row.algorithm == algorithm
                    and row.seed == seed
                ),
                key=lambda row: row.checkpoint,
            )
            ax.plot(
                [row.checkpoint for row in values],
                [float(getattr(row, value_name)) for row in values],
                **_series_style(algorithm, seed),
            )
        _format_axis(ax, horizon=horizon, ylabel=ylabel, ylim=ylim)
        if ylim == (0.0, 100.0):
            ax.set_yticks(np.arange(0, 101, 20))

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(
        handles,
        labels,
        loc="center",
        fontsize=15,
        frameon=False,
        handlelength=3.2,
    )
    _save(fig, output_dir / stem)


def plot_final_type_composition(
    diagnostics: dict[tuple[int, str, int, int], dict[str, Any]],
    output_dir: Path,
) -> None:
    fig, axes = _facet_axes("Final-checkpoint action-type probability profile")
    actions = ("wait", "cancel", "take", "post")
    action_labels = ("WAIT", "CANCEL", "TAKE", "POST")
    colors = (
        colormaps["Greys"](0.55),
        colormaps["Purples"](0.55),
        colormaps["Greens"](0.55),
        colormaps["Blues"](0.62),
    )
    labels = [f"{algorithm.upper()} s{seed}" for algorithm, seed in SERIES]
    for ax, (_, horizon) in zip(axes, HORIZONS):
        bottoms = np.zeros(len(SERIES))
        x = np.arange(len(SERIES))
        for action, action_label, color in zip(actions, action_labels, colors):
            values = np.array(
                [
                    100.0
                    * diagnostics[(horizon, algorithm, seed, 1200)][
                        f"type/mean_prob_{action}"
                    ]
                    for algorithm, seed in SERIES
                ]
            )
            ax.bar(
                x,
                values,
                bottom=bottoms,
                label=action_label,
                color=color,
                width=0.72,
            )
            bottoms += values
        ax.set_title(f"{horizon}-second environment", fontsize=17, fontweight="semibold")
        ax.set_ylabel("Mean probability (%)", fontsize=15)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)

    axes[-1].axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(
        handles,
        legend_labels,
        loc="center",
        fontsize=15,
        frameon=False,
        title="Type head",
        title_fontsize=16,
    )
    _save(fig, output_dir / "final_type_probability_composition")


def plot_final_head_heatmap(
    rows: list[Row],
    diagnostics: dict[tuple[int, str, int, int], dict[str, Any]],
    output_dir: Path,
) -> None:
    final_survival = {
        (row.horizon, row.algorithm, row.seed): row.survival_percentage
        for row in rows
        if row.checkpoint == 1200
    }
    row_keys = [
        (horizon, algorithm, seed)
        for _, horizon in HORIZONS
        for algorithm, seed in SERIES
    ]
    columns = (
        "Survival",
        "Type H",
        "Food|POST H",
        "Side|POST H",
        "Price|POST H",
        "Qty|POST H",
    )
    entropy_fields = (
        ("type/entropy_mean", HEAD_MAX_ENTROPY["type"]),
        ("food_given_post/entropy_mean", HEAD_MAX_ENTROPY["food"]),
        ("side_given_post/entropy_mean", HEAD_MAX_ENTROPY["side"]),
        ("price_given_post/entropy_mean", HEAD_MAX_ENTROPY["price"]),
        ("qty_given_post/entropy_mean", HEAD_MAX_ENTROPY["quantity"]),
    )
    matrix = []
    for horizon, algorithm, seed in row_keys:
        head = diagnostics[(horizon, algorithm, seed, 1200)]
        matrix.append(
            [final_survival[(horizon, algorithm, seed)]]
            + [100.0 * float(head[field]) / maximum for field, maximum in entropy_fields]
        )
    values = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(13.5, 13), constrained_layout=True)
    image = ax.imshow(values, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set_title(
        "Checkpoint 1200: survival and normalized policy-head entropy",
        fontsize=23,
        fontweight="semibold",
        pad=18,
    )
    ax.set_xticks(np.arange(len(columns)), columns, rotation=24, ha="right")
    ax.set_yticks(
        np.arange(len(row_keys)),
        [
            f"{horizon}s  {algorithm.upper()} seed {seed}"
            for horizon, algorithm, seed in row_keys
        ],
    )
    ax.tick_params(axis="both", labelsize=13)
    ax.axvline(0.5, color="white", linewidth=3.0)
    for boundary in (3.5, 7.5, 11.5, 15.5):
        ax.axhline(boundary, color="white", linewidth=2.5)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text_color = "white" if value < 40 or value > 78 else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="semibold",
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.03)
    colorbar.set_label("Percent (%)", fontsize=15)
    colorbar.ax.tick_params(labelsize=12)
    _save(fig, output_dir / "final_head_entropy_heatmap")


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: Iterable[Row], output_dir: Path) -> None:
    final_rows = sorted(
        (row for row in rows if row.checkpoint == 1200),
        key=lambda row: (row.horizon, row.algorithm, row.seed),
    )
    path = output_dir / "checkpoint_1200_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "horizon_seconds",
                "algorithm",
                "seed",
                "survival_percentage",
                "death_time_percentage_of_horizon",
                "post_probability_percentage",
                "normalized_type_entropy_percentage",
                "effective_price_choices",
            )
        )
        for row in final_rows:
            writer.writerow(
                (
                    row.horizon,
                    row.algorithm,
                    row.seed,
                    row.survival_percentage,
                    row.death_time_percentage,
                    row.post_probability_percentage,
                    row.normalized_type_entropy,
                    row.effective_price_choices,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    rows = read_rows(root)
    diagnostics = read_head_diagnostics(root)
    common = {"rows": rows, "output_dir": output_dir}
    plot_faceted_metric(
        **common,
        value_name="survival_percentage",
        title="Learned self-play survival across episode lengths",
        ylabel="Seat survival (%)",
        ylim=(0.0, 100.0),
        stem="survival_percentage_by_horizon",
    )
    plot_faceted_metric(
        **common,
        value_name="death_time_percentage",
        title="Mean self-play death time, normalized by episode length",
        ylabel="Death time (% of horizon)",
        ylim=(0.0, 100.0),
        stem="normalized_death_time_by_horizon",
    )
    plot_faceted_metric(
        **common,
        value_name="post_probability_percentage",
        title="Policy preference for posting orders",
        ylabel="Mean POST probability (%)",
        ylim=(0.0, 100.0),
        stem="post_probability_by_horizon",
    )
    plot_faceted_metric(
        **common,
        value_name="normalized_type_entropy",
        title="Action-type uncertainty during learned self-play",
        ylabel="Type entropy (% of ln 4)",
        ylim=(0.0, 100.0),
        stem="type_entropy_by_horizon",
    )
    plot_faceted_metric(
        **common,
        value_name="effective_price_choices",
        title="Effective number of prices considered after assuming POST",
        ylabel="Effective price choices, exp(H)",
        ylim=(0.0, 50.0),
        stem="effective_price_choices_by_horizon",
    )
    plot_final_type_composition(diagnostics, output_dir)
    plot_final_head_heatmap(rows, diagnostics, output_dir)
    write_summary(rows, output_dir)
    print(f"wrote 7 PNG/PDF figures and one summary CSV to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
