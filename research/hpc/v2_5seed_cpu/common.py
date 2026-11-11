from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CAMPAIGN_DIR = Path(__file__).resolve().parent
CAMPAIGN_CONFIG = CAMPAIGN_DIR / "campaign.json"
CHECKPOINT_RE = re.compile(r"^iter_(\d{4})$")


def load_campaign() -> dict[str, Any]:
    with CAMPAIGN_CONFIG.open() as fh:
        campaign = json.load(fh)
    validate_campaign(campaign)
    return campaign


def validate_campaign(campaign: dict[str, Any]) -> None:
    algorithms = campaign["algorithms"]
    seeds = campaign["seeds"]
    if algorithms != ["ippo", "mappo"]:
        raise ValueError(f"unexpected algorithm order: {algorithms}")
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError(f"unexpected seed set: {seeds}")

    iterations = int(campaign["iterations"])
    every = int(campaign["checkpoint_every"])
    start = int(campaign["checkpoint_start"])
    derived = list(range(start, iterations + 1, every))
    if derived[-1] != iterations:
        derived.append(iterations)
    declared = [int(value) for value in campaign["checkpoint_iterations"]]
    if declared != derived:
        raise ValueError(
            f"declared checkpoints {declared} do not match schedule {derived}"
        )

    task_count = len(algorithms) * len(seeds)
    if campaign["resources"]["array"].split("%", 1)[0] != f"0-{task_count - 1}":
        raise ValueError("SLURM array range does not match campaign task count")


def tasks(campaign: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        (algorithm, int(seed))
        for algorithm in campaign["algorithms"]
        for seed in campaign["seeds"]
    ]


def resolve_task(campaign: dict[str, Any], task_id: int) -> tuple[str, int]:
    all_tasks = tasks(campaign)
    if not 0 <= task_id < len(all_tasks):
        raise ValueError(
            f"task id {task_id} is outside the valid range 0-{len(all_tasks) - 1}"
        )
    return all_tasks[task_id]


def run_name(algorithm: str, seed: int) -> str:
    return f"{algorithm}_seed{seed}_headmlp"


def run_dir(campaign_root: Path, algorithm: str, seed: int) -> Path:
    return campaign_root / run_name(algorithm, seed)


def checkpoint_iterations(run_path: Path) -> list[int]:
    checkpoint_root = run_path / "checkpoints"
    if not checkpoint_root.is_dir():
        return []
    found: list[int] = []
    for path in checkpoint_root.iterdir():
        match = CHECKPOINT_RE.match(path.name)
        if path.is_dir() and match:
            found.append(int(match.group(1)))
    return sorted(found)


def base_version(version: str) -> str:
    """Normalize build suffixes such as PyTorch's ``+cpu``."""
    return version.split("+", 1)[0]
