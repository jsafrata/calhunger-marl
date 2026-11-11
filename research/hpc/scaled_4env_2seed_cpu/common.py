from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


CAMPAIGN_DIR = Path(__file__).resolve().parent
CAMPAIGN_CONFIG = CAMPAIGN_DIR / "campaign.json"
EXPECTED_ENVIRONMENT_VARIANTS = [
    "small_60",
    "small_90",
    "small_120",
    "small_150",
]


def load_campaign() -> dict[str, Any]:
    with CAMPAIGN_CONFIG.open() as fh:
        campaign = json.load(fh)
    validate_campaign(campaign)
    return campaign


def validate_campaign(campaign: dict[str, Any]) -> None:
    if campaign["environment_variants"] != EXPECTED_ENVIRONMENT_VARIANTS:
        raise ValueError(
            "scaled campaign environments must be "
            f"{EXPECTED_ENVIRONMENT_VARIANTS} in duration order"
        )
    if campaign["algorithms"] != ["ippo", "mappo"]:
        raise ValueError("scaled campaign must contain IPPO followed by MAPPO")
    if campaign["seeds"] != [0, 1]:
        raise ValueError("scaled campaign must use seeds 0 and 1")
    if int(campaign["iterations"]) != 1200:
        raise ValueError("scaled campaign must train for 1200 iterations")

    iterations = int(campaign["iterations"])
    every = int(campaign["checkpoint_every"])
    start = int(campaign["checkpoint_start"])
    expected_checkpoints = list(range(start, iterations + 1, every))
    if expected_checkpoints[-1] != iterations:
        expected_checkpoints.append(iterations)
    declared = [int(value) for value in campaign["checkpoint_iterations"]]
    if declared != expected_checkpoints:
        raise ValueError(
            f"declared checkpoints {declared} do not match schedule "
            f"{expected_checkpoints}"
        )

    task_count = (
        len(campaign["environment_variants"])
        * len(campaign["algorithms"])
        * len(campaign["seeds"])
    )
    array_range = campaign["resources"]["array"].split("%", 1)[0]
    if array_range != f"0-{task_count - 1}":
        raise ValueError("SLURM array range does not match campaign task count")


def tasks(campaign: dict[str, Any]) -> list[tuple[str, str, int]]:
    return [
        (environment, algorithm, int(seed))
        for environment in campaign["environment_variants"]
        for algorithm in campaign["algorithms"]
        for seed in campaign["seeds"]
    ]


def resolve_task(campaign: dict[str, Any], task_id: int) -> tuple[str, str, int]:
    all_tasks = tasks(campaign)
    if not 0 <= task_id < len(all_tasks):
        raise ValueError(
            f"task id {task_id} is outside the valid range 0-{len(all_tasks) - 1}"
        )
    return all_tasks[task_id]


def campaign_for_environment(
    campaign: dict[str, Any], environment_variant: str
) -> dict[str, Any]:
    """Return the shared runner's single-environment campaign view."""
    if environment_variant not in campaign["environment_variants"]:
        raise ValueError(f"unknown campaign environment {environment_variant!r}")
    environment_campaign = copy.deepcopy(campaign)
    environment_campaign["training"]["env_variant"] = environment_variant
    return environment_campaign


def environment_root(campaign_root: Path, environment_variant: str) -> Path:
    return campaign_root / environment_variant
