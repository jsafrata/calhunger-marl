from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import base_version, load_campaign


def _is_allowed_hpc_storage_path(path: Path, username: str | None) -> bool:
    """Return whether ``path`` is persistent Resnick storage for this user.

    Group storage is valid for any authorized group. Personal storage is
    restricted to the submitting user's own canonical VAST home. Resolving
    first also accepts the cluster's ``/home/<user>`` symlink.
    """
    resolved = path.resolve(strict=False)
    allowed_roots = [Path("/resnick/groups")]
    if username:
        allowed_roots.append(Path("/resnick/home") / username)
    return any(resolved == root or root in resolved.parents for root in allowed_roots)


def _git_status(research_root: Path) -> tuple[str | None, str | None]:
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
        ).stdout.strip()
        return revision, status
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def inspect(
    research_root: Path,
    campaign_root: Path | None,
    campaign: dict[str, Any] | None = None,
) -> dict[str, Any]:
    campaign = campaign or load_campaign()
    errors: list[str] = []
    warnings: list[str] = []
    observed_versions: dict[str, str] = {}

    locked = campaign["locked_versions"]
    python_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_major_minor != locked["python_major_minor"]:
        errors.append(
            f"Python {python_major_minor} does not match locked "
            f"{locked['python_major_minor']}"
        )

    for package in ("ray", "torch", "numpy", "gymnasium", "pyarrow", "scipy"):
        try:
            module = importlib.import_module(package)
            observed = str(module.__version__)
            observed_versions[package] = observed
            if base_version(observed) != locked[package]:
                errors.append(
                    f"{package} {observed} does not match locked {locked[package]}"
                )
        except Exception as exc:  # pragma: no cover - exact import errors vary
            errors.append(f"cannot import {package}: {exc}")

    try:
        from env.actions import MAX_PRICE, MAX_QTY, NUM_ACTIONS

        expected_qty = campaign["training"]["max_qty"]
        expected_price = campaign["training"]["max_price"]
        expected_actions = 1 + 8 + 8 * expected_qty + 8 * expected_qty * expected_price
        if (MAX_QTY, MAX_PRICE, NUM_ACTIONS) != (
            expected_qty,
            expected_price,
            expected_actions,
        ):
            errors.append(
                "action contract mismatch: observed "
                f"qty={MAX_QTY}, price={MAX_PRICE}, actions={NUM_ACTIONS}; expected "
                f"qty={expected_qty}, price={expected_price}, actions={expected_actions}"
            )
    except Exception as exc:  # pragma: no cover - reported to operator
        errors.append(f"cannot validate action contract: {exc}")

    revision, status = _git_status(research_root)
    if revision is None:
        errors.append(f"{research_root} is not a readable git worktree")
    elif status:
        warnings.append("research worktree is dirty; commit the campaign snapshot first")

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    required_cpus = int(campaign["resources"]["cpus_per_task"])
    if slurm_cpus is not None and int(slurm_cpus) < required_cpus:
        errors.append(
            f"SLURM allocated {slurm_cpus} CPUs, but campaign requires {required_cpus}"
        )

    if os.environ.get("SLURM_JOB_ID") and platform.machine() not in (
        "x86_64",
        "amd64",
    ):
        errors.append(f"unexpected HPC architecture: {platform.machine()}")

    if campaign_root is not None:
        if os.environ.get("SLURM_JOB_ID") and not _is_allowed_hpc_storage_path(
            campaign_root, os.environ.get("USER")
        ):
            errors.append(
                "HUNGER_CAMPAIGN_ROOT must be under /resnick/groups or the "
                "submitting user's /resnick/home/<username> on the HPC"
            )
        if campaign_root.exists() and not campaign_root.is_dir():
            errors.append(f"campaign root exists but is not a directory: {campaign_root}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "research_root": str(research_root),
        "campaign_root": str(campaign_root) if campaign_root else None,
        "git_revision": revision,
        "git_dirty": bool(status),
        "python": sys.version.split()[0],
        "architecture": platform.machine(),
        "versions": observed_versions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the v2 HPC environment")
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
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat a dirty git worktree as a fatal reproducibility error.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect(args.research_root.resolve(), args.campaign_root)
    if args.strict and report["git_dirty"]:
        report["errors"].append("strict mode rejects a dirty git worktree")
        report["ok"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
