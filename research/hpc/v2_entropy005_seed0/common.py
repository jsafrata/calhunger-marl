from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CAMPAIGN_DIR = Path(__file__).resolve().parent
CAMPAIGN_CONFIG = CAMPAIGN_DIR / "campaign.json"


def load_campaign() -> dict[str, Any]:
    with CAMPAIGN_CONFIG.open() as fh:
        campaign = json.load(fh)
    validate_campaign(campaign)
    return campaign


def validate_campaign(campaign: dict[str, Any]) -> None:
    if campaign["algorithms"] != ["ippo", "mappo"]:
        raise ValueError("pilot must contain IPPO followed by MAPPO")
    if campaign["seeds"] != [0]:
        raise ValueError("pilot is frozen to seed 0")
    if float(campaign["training"]["entropy_coeff"]) != 0.05:
        raise ValueError("pilot entropy coefficient must be 0.05")

    iterations = int(campaign["iterations"])
    every = int(campaign["checkpoint_every"])
    start = int(campaign["checkpoint_start"])
    expected = list(range(start, iterations + 1, every))
    if expected[-1] != iterations:
        expected.append(iterations)
    declared = [int(value) for value in campaign["checkpoint_iterations"]]
    if declared != expected:
        raise ValueError(
            f"declared checkpoints {declared} do not match schedule {expected}"
        )

    if campaign["resources"]["array"] != "0-1%2":
        raise ValueError("pilot SLURM array must be 0-1%2")


def resolve_task(campaign: dict[str, Any], task_id: int) -> tuple[str, int]:
    tasks = [
        (algorithm, int(seed))
        for algorithm in campaign["algorithms"]
        for seed in campaign["seeds"]
    ]
    if not 0 <= task_id < len(tasks):
        raise ValueError(f"task id {task_id} is outside the valid range 0-1")
    return tasks[task_id]
