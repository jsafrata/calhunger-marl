from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hpc.v2_5seed_cpu.common import run_dir
from hpc.v2_5seed_cpu.verify_campaign import verify_run

from .common import arm_root, campaign_for_arm, load_campaign, tasks


def verify_campaign(campaign_root: Path) -> dict[str, object]:
    campaign = load_campaign()
    reports: list[dict[str, object]] = []
    for arm_name, algorithm, seed in tasks(campaign):
        output_root = arm_root(campaign_root, arm_name)
        path = run_dir(output_root, algorithm, seed)
        errors = verify_run(path, campaign_for_arm(campaign, arm_name))
        reports.append(
            {
                "arm": arm_name,
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
        "runs": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the best-config bundled campaign"
    )
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
