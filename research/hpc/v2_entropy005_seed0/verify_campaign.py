from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hpc.v2_5seed_cpu.common import run_dir
from hpc.v2_5seed_cpu.verify_campaign import verify_run

from .common import load_campaign


def verify_campaign(campaign_root: Path) -> dict[str, object]:
    campaign = load_campaign()
    reports: list[dict[str, object]] = []
    for algorithm in campaign["algorithms"]:
        seed = int(campaign["seeds"][0])
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
        "ok": all(bool(report["ok"]) for report in reports),
        "campaign_root": str(campaign_root),
        "entropy_coeff": campaign["training"]["entropy_coeff"],
        "runs": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the entropy-0.05 pilot")
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
