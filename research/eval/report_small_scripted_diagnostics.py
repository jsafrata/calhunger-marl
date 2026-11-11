"""Tabulate small-env checkpoint diagnostics across scripted opponent archetypes.

Reads the per-opponent `checkpoint_diagnostics.csv` files written by
`eval/small_checkpoint_scripted_diagnostics.py` and prints the five learned-seat
statistics per (algorithm, seed, checkpoint), plus per-opponent aggregates.

Usage:
    python -m eval.report_small_scripted_diagnostics \\
        --campaign-root eval/results/small_2seed_partial
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    ("mean_survival_time_ticks", "surv_ticks", "{:.2f}"),
    ("mean_terminal_cash", "cash", "{:.1f}"),
    ("mean_learned_trade_participations", "trades", "{:.2f}"),
    ("win_percentage", "win%", "{:.1f}"),
    ("survival_percentage", "surv%", "{:.1f}"),
)


def load_rows(campaign_root: Path, opponents: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for opponent in opponents:
        path = campaign_root / f"scripted_diag_{opponent}" / "checkpoint_diagnostics.csv"
        if not path.is_file():
            print(f"  (missing: {path})")
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                row["opponent"] = row.get("opponent") or opponent
                rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def print_detail(rows: list[dict[str, Any]], opponent: str) -> None:
    subset = [r for r in rows if r["opponent"] == opponent]
    if not subset:
        return
    header = f"{'algo':6s} {'seed':>4s} {'ckpt':>5s} " + " ".join(
        f"{label:>10s}" for _, label, _ in METRICS
    )
    print(f"\n--- learned seat vs 3x scripted {opponent} ---")
    print(header)
    for row in sorted(
        subset, key=lambda r: (r["algorithm"], int(r["seed"]), int(r["checkpoint"]))
    ):
        cells = " ".join(
            f"{fmt.format(float(row[key])):>10s}" for key, _, fmt in METRICS
        )
        print(f"{row['algorithm']:6s} {row['seed']:>4s} {row['checkpoint']:>5s} {cells}")


def print_aggregate(rows: list[dict[str, Any]], opponents: list[str]) -> None:
    print("\n=== mean over all seeds and checkpoints ===")
    print(f"{'opponent':10s} " + " ".join(f"{label:>10s}" for _, label, _ in METRICS))
    for opponent in opponents:
        subset = [r for r in rows if r["opponent"] == opponent]
        if not subset:
            continue
        cells = " ".join(
            f"{fmt.format(_mean([float(r[key]) for r in subset])):>10s}"
            for key, _, fmt in METRICS
        )
        print(f"{opponent:10s} {cells}")


def print_by_checkpoint(rows: list[dict[str, Any]], opponents: list[str]) -> None:
    """Progression over training, averaged across every seed present."""
    n_seeds = len({(r["algorithm"], r["seed"]) for r in rows})
    print(f"\n=== by checkpoint (mean over the {n_seeds} seeds) ===")
    checkpoints = sorted({int(r["checkpoint"]) for r in rows})
    for opponent in opponents:
        subset = [r for r in rows if r["opponent"] == opponent]
        if not subset:
            continue
        print(f"\n  vs {opponent}")
        print(f"  {'ckpt':>5s} " + " ".join(f"{label:>10s}" for _, label, _ in METRICS))
        for checkpoint in checkpoints:
            at = [r for r in subset if int(r["checkpoint"]) == checkpoint]
            if not at:
                continue
            cells = " ".join(
                f"{fmt.format(_mean([float(r[key]) for r in at])):>10s}"
                for key, _, fmt in METRICS
            )
            print(f"  {checkpoint:>5d} {cells}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--opponents", default="v1,v2,v3")
    parser.add_argument("--detail", action="store_true", help="per-seed breakdown")
    parser.add_argument(
        "--checkpoint-min",
        type=int,
        default=None,
        help=(
            "Drop checkpoints below this. Needed when a reference result set "
            "covers a wider range than the runs it is compared against — "
            "otherwise the aggregate means are taken over different x-axes."
        ),
    )
    parser.add_argument("--checkpoint-max", type=int, default=None)
    args = parser.parse_args()

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    rows = load_rows(args.campaign_root.resolve(), opponents)
    if args.checkpoint_min is not None:
        rows = [r for r in rows if int(r["checkpoint"]) >= args.checkpoint_min]
    if args.checkpoint_max is not None:
        rows = [r for r in rows if int(r["checkpoint"]) <= args.checkpoint_max]
    if not rows:
        print("no results found")
        return 1

    n_seeds = len({(r["algorithm"], r["seed"]) for r in rows})
    n_ckpt = len({r["checkpoint"] for r in rows})
    games = {int(r["n_games"]) for r in rows}
    print(
        f"{len(rows)} result rows | {n_seeds} seeds | {n_ckpt} checkpoints | "
        f"{sorted(games)} games each"
    )

    print_aggregate(rows, opponents)
    print_by_checkpoint(rows, opponents)
    if args.detail:
        for opponent in opponents:
            print_detail(rows, opponent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
