from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from hpc.v2_5seed_cpu.run_task import _write_attempt_manifest, build_command
from hpc.v2_5seed_cpu.verify_campaign import verify_run

from .common import (
    campaign_for_environment,
    environment_root,
    load_campaign,
    resolve_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or execute one scaled-environment array task"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=(
            int(os.environ["SLURM_ARRAY_TASK_ID"])
            if os.environ.get("SLURM_ARRAY_TASK_ID")
            else None
        ),
        required=not bool(os.environ.get("SLURM_ARRAY_TASK_ID")),
    )
    parser.add_argument(
        "--research-root",
        type=Path,
        default=Path(os.environ.get("HUNGER_RESEARCH_ROOT", Path.cwd())),
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest unambiguous checkpoint for this task.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = load_campaign()
    research_root = args.research_root.resolve()
    campaign_root = args.campaign_root.resolve()
    environment_variant, algorithm, seed = resolve_task(campaign, args.task_id)
    environment_campaign = campaign_for_environment(campaign, environment_variant)
    output_root = environment_root(campaign_root, environment_variant)

    command, target, remaining = build_command(
        campaign=environment_campaign,
        research_root=research_root,
        campaign_root=output_root,
        algorithm=algorithm,
        seed=seed,
        resume=args.resume,
    )
    plan = {
        "execute": args.execute,
        "resume": args.resume,
        "task_id": args.task_id,
        "environment_variant": environment_variant,
        "algorithm": algorithm,
        "seed": seed,
        "remaining_iterations": remaining,
        "target": str(target),
        "command": command,
        "command_shell_escaped": shlex.join(command),
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    if remaining == 0:
        errors = verify_run(target, environment_campaign)
        if errors:
            raise RuntimeError("completed run failed verification: " + "; ".join(errors))
        print(f"task already complete and verified: {target}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _write_attempt_manifest(
        campaign=environment_campaign,
        campaign_root=output_root,
        research_root=research_root,
        task_id=args.task_id,
        algorithm=algorithm,
        seed=seed,
        command=command,
        resume=args.resume,
    )
    print(f"attempt manifest: {manifest}", flush=True)
    subprocess.run(command, cwd=research_root, check=True)

    errors = verify_run(target, environment_campaign)
    if errors:
        raise RuntimeError("run completed but verification failed: " + "; ".join(errors))
    print(f"run complete and verified: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
