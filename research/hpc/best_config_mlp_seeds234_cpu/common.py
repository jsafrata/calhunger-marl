from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


CAMPAIGN_DIR = Path(__file__).resolve().parent
CAMPAIGN_CONFIG = CAMPAIGN_DIR / "campaign.json"

EXPECTED_ARM_NAMES = [
    "small_90_mlp",
    "default_180_mlp",
]
EXPECTED_DURATIONS = {"small_90": 90, "default": 180}
EXPECTED_SEEDS = [2, 3, 4]


def load_campaign() -> dict[str, Any]:
    with CAMPAIGN_CONFIG.open() as fh:
        campaign = json.load(fh)
    validate_campaign(campaign)
    return campaign


def _expected_batch(duration_seconds: int, games_per_update: int) -> tuple[int, int]:
    train_batch_size = duration_seconds * games_per_update
    return train_batch_size, train_batch_size // 4


def validate_campaign(campaign: dict[str, Any]) -> None:
    arm_names = [arm["name"] for arm in campaign["arms"]]
    if arm_names != EXPECTED_ARM_NAMES:
        raise ValueError(
            f"best-config MLP campaign arms must be {EXPECTED_ARM_NAMES}, "
            f"got {arm_names}"
        )
    if campaign["algorithms"] != ["ippo", "mappo"]:
        raise ValueError("campaign must contain IPPO followed by MAPPO")
    if campaign["seeds"] != EXPECTED_SEEDS:
        raise ValueError("campaign must use seeds 2, 3, and 4")
    if int(campaign["iterations"]) != 1200:
        raise ValueError("campaign must train for 1200 iterations")

    iterations = int(campaign["iterations"])
    every = int(campaign["checkpoint_every"])
    start = int(campaign["checkpoint_start"])
    expected_checkpoints = list(range(start, iterations + 1, every))
    declared = [int(value) for value in campaign["checkpoint_iterations"]]
    if declared != expected_checkpoints:
        raise ValueError(
            f"declared checkpoints {declared} do not match "
            f"schedule {expected_checkpoints}"
        )

    training = campaign["training"]
    games_per_update = int(training["games_per_update"])
    if games_per_update != 45:
        raise ValueError("campaign must hold 45 complete games per PPO update")
    if int(training["num_minibatches"]) != 4:
        raise ValueError("campaign must divide each train batch into four minibatches")
    # The whole point of this campaign: every update we currently believe in
    # is ON. Refuse to run a partial bundle by accident.
    if not training["include_survival_features"]:
        raise ValueError("best-config campaign requires survival features")
    if not training["include_trade_history"]:
        raise ValueError("best-config campaign requires critic trade history")
    if float(training["entropy_anneal_to"]) >= float(training["entropy_coeff"]):
        raise ValueError("entropy must anneal downward")
    if float(training["lr_anneal_to"]) >= 3e-4:
        raise ValueError("learning rate must anneal downward from 3e-4")

    for arm in campaign["arms"]:
        variant = arm["env_variant"]
        if variant not in EXPECTED_DURATIONS:
            raise ValueError(f"unknown arm environment {variant!r}")
        duration = EXPECTED_DURATIONS[variant]
        if int(arm["duration_seconds"]) != duration:
            raise ValueError(
                f"arm {arm['name']} declares duration {arm['duration_seconds']}, "
                f"expected {duration}"
            )
        expected_train, expected_minibatch = _expected_batch(
            duration, games_per_update
        )
        if int(arm["train_batch_size"]) != expected_train:
            raise ValueError(f"arm {arm['name']} train batch is not episode-matched")
        if int(arm["minibatch_size"]) != expected_minibatch:
            raise ValueError(f"arm {arm['name']} minibatch is not batch/4")
        if arm["torso"] != "mlp":
            raise ValueError(
                f"arm {arm['name']} has torso {arm['torso']!r}; this campaign "
                "is MLP-only (LSTM excluded pending post-mortem)"
            )
        if not arm["name"].endswith(f"_{arm['torso']}"):
            raise ValueError(
                f"arm name {arm['name']!r} must end with its torso for "
                "unambiguous output directories"
            )

    if campaign["resources"]["array"] != "0-5%6":
        raise ValueError("each arm must submit a six-task 0-5%6 array")


def arm_by_name(campaign: dict[str, Any], arm_name: str) -> dict[str, Any]:
    for arm in campaign["arms"]:
        if arm["name"] == arm_name:
            return arm
    raise ValueError(f"unknown campaign arm {arm_name!r}")


def tasks_for_arm(campaign: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        (algorithm, int(seed))
        for algorithm in campaign["algorithms"]
        for seed in campaign["seeds"]
    ]


def tasks(campaign: dict[str, Any]) -> list[tuple[str, str, int]]:
    return [
        (arm["name"], algorithm, seed)
        for arm in campaign["arms"]
        for algorithm, seed in tasks_for_arm(campaign)
    ]


def resolve_task(
    campaign: dict[str, Any], arm_name: str, task_id: int
) -> tuple[str, int]:
    arm_by_name(campaign, arm_name)  # validate the arm exists
    arm_tasks = tasks_for_arm(campaign)
    if not 0 <= task_id < len(arm_tasks):
        raise ValueError(
            f"task id {task_id} is outside the valid range 0-{len(arm_tasks) - 1}"
        )
    return arm_tasks[task_id]


def campaign_for_arm(campaign: dict[str, Any], arm_name: str) -> dict[str, Any]:
    """Per-arm campaign dict shaped for `hpc.v2_5seed_cpu.run_task.build_command`."""
    arm = arm_by_name(campaign, arm_name)
    arm_campaign = copy.deepcopy(campaign)
    training = arm_campaign["training"]
    training["env_variant"] = arm["env_variant"]
    training["train_batch_size"] = int(arm["train_batch_size"])
    training["minibatch_size"] = int(arm["minibatch_size"])
    training["torso"] = arm["torso"]
    return arm_campaign


def arm_root(campaign_root: Path, arm_name: str) -> Path:
    return campaign_root / arm_name


def arm_from_process(campaign: dict[str, Any]) -> str:
    arm_name = os.environ.get("HUNGER_ARM")
    if not arm_name:
        raise ValueError(
            "HUNGER_ARM is unset; submit with submit_all.sh or pass --arm"
        )
    arm_by_name(campaign, arm_name)
    return arm_name
