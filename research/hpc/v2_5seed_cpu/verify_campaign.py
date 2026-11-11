from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .common import checkpoint_iterations, load_campaign, run_dir, tasks


def verify_run(run_path: Path, campaign: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not run_path.is_dir():
        return [f"missing run directory: {run_path}"]

    metrics_path = run_path / "iters.jsonl"
    observed_iterations: list[int] = []
    if not metrics_path.is_file():
        errors.append(f"missing metrics file: {metrics_path}")
    else:
        with metrics_path.open() as fh:
            for line_number, line in enumerate(fh, start=1):
                try:
                    observed_iterations.append(int(json.loads(line)["iter"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(
                        f"invalid metrics line {metrics_path}:{line_number}: {exc}"
                    )
                    break

    expected_iterations = list(range(1, int(campaign["iterations"]) + 1))
    if observed_iterations != expected_iterations:
        missing = sorted(set(expected_iterations) - set(observed_iterations))
        duplicates = sorted(
            value
            for value in set(observed_iterations)
            if observed_iterations.count(value) > 1
        )
        errors.append(
            "iteration sequence mismatch: "
            f"observed={len(observed_iterations)}, expected={len(expected_iterations)}, "
            f"first_missing={missing[:5]}, duplicates={duplicates[:5]}"
        )

    observed_checkpoints = checkpoint_iterations(run_path)
    expected_checkpoints = [int(value) for value in campaign["checkpoint_iterations"]]
    if observed_checkpoints != expected_checkpoints:
        errors.append(
            f"checkpoint mismatch: observed={observed_checkpoints}, "
            f"expected={expected_checkpoints}"
        )
    for iteration in expected_checkpoints:
        marker = run_path / "checkpoints" / f"iter_{iteration:04d}" / "rllib_checkpoint.json"
        if not marker.is_file():
            errors.append(f"missing RLlib checkpoint marker: {marker}")

    latest_path = run_path / "latest_checkpoint.txt"
    if not latest_path.is_file() or not latest_path.read_text().strip():
        errors.append(f"missing or empty latest checkpoint pointer: {latest_path}")
    return errors


def verify_campaign(campaign_root: Path) -> dict[str, Any]:
    campaign = load_campaign()
    reports: list[dict[str, Any]] = []
    for algorithm, seed in tasks(campaign):
        path = run_dir(campaign_root, algorithm, seed)
        errors = verify_run(path, campaign)
        reports.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "path": str(path),
                "ok": not errors,
                "errors": errors,
            }
        )
    return {
        "ok": all(report["ok"] for report in reports),
        "campaign_root": str(campaign_root),
        "runs": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all campaign artifacts")
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=(
            Path(os.environ["HUNGER_CAMPAIGN_ROOT"])
            if os.environ.get("HUNGER_CAMPAIGN_ROOT")
            else None
        ),
        required=not bool(os.environ.get("HUNGER_CAMPAIGN_ROOT")),
    )
    args = parser.parse_args()
    report = verify_campaign(args.campaign_root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
