from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    checkpoint_iterations,
    load_campaign,
    resolve_task,
    run_dir,
    run_name,
)
from .verify_campaign import verify_run


def _last_metric_iteration(run_path: Path) -> int | None:
    metrics_path = run_path / "iters.jsonl"
    if not metrics_path.is_file():
        return None
    last: int | None = None
    with metrics_path.open() as fh:
        for line in fh:
            if line.strip():
                last = int(json.loads(line)["iter"])
    return last


def _git_snapshot(research_root: Path) -> dict[str, str | bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=research_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=research_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=research_root,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "revision": revision,
            "dirty": bool(status.strip()),
            "status": status.strip(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "revision": None,
            "dirty": True,
            "status": "git snapshot unavailable",
            "tracked_diff_sha256": None,
        }


def build_command(
    *,
    campaign: dict[str, Any],
    research_root: Path,
    campaign_root: Path,
    algorithm: str,
    seed: int,
    resume: bool,
) -> tuple[list[str], Path, int]:
    target = run_dir(campaign_root, algorithm, seed)
    completed = 0
    resume_checkpoint: Path | None = None

    if resume:
        checkpoints = checkpoint_iterations(target)
        if not checkpoints:
            raise ValueError(f"cannot resume; no checkpoints found in {target}")
        completed = checkpoints[-1]
        last_metric = _last_metric_iteration(target)
        if last_metric != completed:
            raise ValueError(
                "refusing an ambiguous resume: the metrics file ends at "
                f"iteration {last_metric}, but the latest checkpoint is {completed}. "
                "Archive/reconcile the post-checkpoint metrics before resuming."
            )
        resume_checkpoint = target / "checkpoints" / f"iter_{completed:04d}"
    elif target.exists():
        raise FileExistsError(
            f"fresh task would collide with existing run directory: {target}"
        )

    remaining = int(campaign["iterations"]) - completed
    if remaining < 0:
        raise ValueError(
            f"latest checkpoint {completed} exceeds campaign target "
            f"{campaign['iterations']}"
        )

    training = campaign["training"]
    command = [
        sys.executable,
        "-m",
        "eval.long_baseline",
        "--iters",
        str(remaining),
        "--checkpoint_every",
        str(campaign["checkpoint_every"]),
        "--checkpoint_start",
        str(campaign["checkpoint_start"]),
        "--seed",
        str(seed),
        "--algos",
        algorithm,
        "--train_batch_size",
        str(training["train_batch_size"]),
        "--minibatch_size",
        str(training["minibatch_size"]),
        "--num_epochs",
        str(training["num_epochs"]),
        "--num_env_runners",
        str(training["num_env_runners"]),
        "--out_root",
        str(campaign_root),
        "--variant_suffix",
        f"seed{seed}_headmlp",
        "--head_kind",
        training["head_kind"],
        "--gamma",
        str(training["gamma"]),
        "--lambda_",
        str(training["lambda"]),
        "--reward_kind",
        training["reward_kind"],
        "--reward_variant",
        training["reward_variant"],
        "--entropy_coeff",
        str(training["entropy_coeff"]),
    ]
    env_variant = training.get("env_variant")
    if env_variant:
        command.extend(["--env_variant", str(env_variant)])
    if resume_checkpoint is not None:
        command.extend(["--resume_from_checkpoint", str(resume_checkpoint)])
    else:
        command.append("--fail_if_output_exists")
    return command, target, remaining


def _write_attempt_manifest(
    *,
    campaign: dict[str, Any],
    campaign_root: Path,
    research_root: Path,
    task_id: int,
    algorithm: str,
    seed: int,
    command: list[str],
    resume: bool,
) -> Path:
    manifest_root = campaign_root / "_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    attempt = os.environ.get("SLURM_RESTART_COUNT", "0")
    path = manifest_root / (
        f"{run_name(algorithm, seed)}_job{job_id}_attempt{attempt}.json"
    )
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    captured_environment = {
        key: os.environ.get(key)
        for key in (
            "HG_MAX_QTY",
            "HG_MAX_PRICE",
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "RAY_TMPDIR",
        )
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign,
        "task_id": task_id,
        "algorithm": algorithm,
        "seed": seed,
        "resume": resume,
        "command": command,
        "command_shell_escaped": shlex.join(command),
        "research_root": str(research_root),
        "campaign_root": str(campaign_root),
        "git": _git_snapshot(research_root),
        "python": sys.version,
        "pip_freeze": pip_freeze,
        "environment": captured_environment,
        "slurm": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("SLURM_")
        },
    }
    with path.open("x") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or execute one array task")
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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually start training. Without this flag, only print the plan.",
    )
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
    algorithm, seed = resolve_task(campaign, args.task_id)
    command, target, remaining = build_command(
        campaign=campaign,
        research_root=research_root,
        campaign_root=campaign_root,
        algorithm=algorithm,
        seed=seed,
        resume=args.resume,
    )
    plan = {
        "execute": args.execute,
        "resume": args.resume,
        "task_id": args.task_id,
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
        errors = verify_run(target, campaign)
        if errors:
            raise RuntimeError("completed run failed verification: " + "; ".join(errors))
        print(f"task already complete and verified: {target}")
        return 0

    campaign_root.mkdir(parents=True, exist_ok=True)
    manifest = _write_attempt_manifest(
        campaign=campaign,
        campaign_root=campaign_root,
        research_root=research_root,
        task_id=args.task_id,
        algorithm=algorithm,
        seed=seed,
        command=command,
        resume=args.resume,
    )
    print(f"attempt manifest: {manifest}", flush=True)
    subprocess.run(command, cwd=research_root, check=True)

    errors = verify_run(target, campaign)
    if errors:
        raise RuntimeError("run completed but verification failed: " + "; ".join(errors))
    print(f"run complete and verified: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
