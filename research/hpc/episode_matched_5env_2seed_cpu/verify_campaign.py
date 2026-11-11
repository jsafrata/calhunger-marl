from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hpc.v2_5seed_cpu.common import run_dir
from hpc.v2_5seed_cpu.verify_campaign import verify_run

from .common import campaign_for_environment, environment_root, load_campaign, tasks


def verify_campaign(campaign_root: Path) -> dict[str, object]:
    campaign = load_campaign()
    reports: list[dict[str, object]] = []
    for environment_variant, algorithm, seed in tasks(campaign):
        output_root = environment_root(campaign, campaign_root, environment_variant)
        path = run_dir(output_root, algorithm, seed)
        environment_campaign = campaign_for_environment(
            campaign, environment_variant
        )
        errors = verify_run(path, environment_campaign)
        reports.append(
            {
                "environment_variant": environment_variant,
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
        description="Verify the episode-matched five-horizon campaign"
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
