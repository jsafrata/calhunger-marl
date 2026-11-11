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
    arm_by_name,
    arm_from_process,
    arm_root,
    campaign_for_arm,
    load_campaign,
    resolve_task,
)


def extend_command(command: list[str], campaign: dict, arm: dict) -> list[str]:
    """Append every best-config knob to the base long_baseline command.

    `build_command` covers the v2-era arguments; the bundle adds
    episode-matched batching, the torso choice, both annealing schedules,
    and both observation-contract flags.
    """
    training = campaign["training"]
    command = list(command)
    command.extend(["--games_per_update", str(training["games_per_update"])])
    command.extend(["--torso", str(arm["torso"])])
    command.extend(["--entropy_anneal_to", str(training["entropy_anneal_to"])])
    command.extend(["--lr_anneal_to", str(training["lr_anneal_to"])])
    command.append("--include_survival_features")
    command.append("--include_trade_history")
    return command


def parse_args() -> argparse.Namespace:
    campaign = load_campaign()
    parser = argparse.ArgumentParser(
        description="Plan or execute one best-config arm task"
    )
    parser.add_argument(
        "--arm",
        choices=[arm["name"] for arm in campaign["arms"]],
        default=os.environ.get("HUNGER_ARM"),
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
        help="Resume from this task's latest unambiguous checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = load_campaign()
    arm_name = args.arm if args.arm else arm_from_process(campaign)
    arm = arm_by_name(campaign, arm_name)
    algorithm, seed = resolve_task(campaign, arm_name, args.task_id)
    research_root = args.research_root.resolve()
    campaign_root = args.campaign_root.resolve()
    arm_campaign = campaign_for_arm(campaign, arm_name)
    output_root = arm_root(campaign_root, arm_name)

    command, target, remaining = build_command(
        campaign=arm_campaign,
        research_root=research_root,
        campaign_root=output_root,
        algorithm=algorithm,
        seed=seed,
        resume=args.resume,
    )
    command = extend_command(command, campaign, arm)
    plan = {
        "execute": args.execute,
        "resume": args.resume,
        "task_id": args.task_id,
        "arm": arm_name,
        "env_variant": arm["env_variant"],
        "torso": arm["torso"],
        "game_duration_seconds": arm["duration_seconds"],
        "games_per_update": campaign["training"]["games_per_update"],
        "train_batch_size": arm["train_batch_size"],
        "minibatch_size": arm["minibatch_size"],
        "entropy_anneal_to": campaign["training"]["entropy_anneal_to"],
        "lr_anneal_to": campaign["training"]["lr_anneal_to"],
        "walltime": arm["walltime"],
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
        errors = verify_run(target, arm_campaign)
        if errors:
            raise RuntimeError("completed run failed verification: " + "; ".join(errors))
        print(f"task already complete and verified: {target}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _write_attempt_manifest(
        campaign=arm_campaign,
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

    errors = verify_run(target, arm_campaign)
    if errors:
        raise RuntimeError("run completed but verification failed: " + "; ".join(errors))
    print(f"run complete and verified: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
