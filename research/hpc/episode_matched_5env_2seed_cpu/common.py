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
    "default",
]
EXPECTED_DURATIONS = {
    "small_60": 60,
    "small_90": 90,
    "small_120": 120,
    "small_150": 150,
    "default": 180,
}
EXPECTED_WALLTIMES = {
    "small_60": "08:00:00",
    "small_90": "12:00:00",
    "small_120": "16:00:00",
    "small_150": "20:00:00",
    "default": "24:00:00",
}


def load_campaign() -> dict[str, Any]:
    with CAMPAIGN_CONFIG.open() as fh:
        campaign = json.load(fh)
    validate_campaign(campaign)
    return campaign


def _expected_batch(duration_seconds: int, games_per_update: int) -> tuple[int, int]:
    train_batch_size = duration_seconds * games_per_update
    return train_batch_size, train_batch_size // 4


def validate_campaign(campaign: dict[str, Any]) -> None:
    if campaign["environment_variants"] != EXPECTED_ENVIRONMENT_VARIANTS:
        raise ValueError(
            "episode-matched campaign environments must be "
            f"{EXPECTED_ENVIRONMENT_VARIANTS} in duration order"
        )
    if campaign["algorithms"] != ["ippo", "mappo"]:
        raise ValueError("campaign must contain IPPO followed by MAPPO")
    if campaign["seeds"] != [0, 1]:
        raise ValueError("campaign must use seeds 0 and 1")
    if int(campaign["iterations"]) != 1200:
        raise ValueError("campaign must train for 1200 iterations")

    iterations = int(campaign["iterations"])
    every = int(campaign["checkpoint_every"])
    start = int(campaign["checkpoint_start"])
    expected_checkpoints = list(range(start, iterations + 1, every))
    if expected_checkpoints[-1] != iterations:
        expected_checkpoints.append(iterations)
    declared_checkpoints = [int(value) for value in campaign["checkpoint_iterations"]]
    if declared_checkpoints != expected_checkpoints:
        raise ValueError(
            f"declared checkpoints {declared_checkpoints} do not match "
            f"schedule {expected_checkpoints}"
        )

    training = campaign["training"]
    games_per_update = int(training["games_per_update"])
    if games_per_update != 45:
        raise ValueError("campaign must hold 45 complete games per PPO update")
    if int(training["num_minibatches"]) != 4:
        raise ValueError("campaign must divide each train batch into four minibatches")
    declared_batches = training["episode_matched_batches"]
    for environment, duration in EXPECTED_DURATIONS.items():
        expected_train, expected_minibatch = _expected_batch(
            duration, games_per_update
        )
        declared = declared_batches[environment]
        observed = (
            int(declared["duration_seconds"]),
            int(declared["train_batch_size"]),
            int(declared["minibatch_size"]),
        )
        expected = (duration, expected_train, expected_minibatch)
        if observed != expected:
            raise ValueError(
                f"episode-matched batch for {environment} is {observed}, "
                f"expected {expected}"
            )

    resources = campaign["resources"]
    if resources["array"] != "0-3%4":
        raise ValueError("each horizon must submit a four-task 0-3%4 array")
    if resources["walltime_by_environment"] != EXPECTED_WALLTIMES:
        raise ValueError(
            "wall times must scale linearly from 12 hours at the 90-second "
            "reference horizon"
        )


def tasks_for_environment(
    campaign: dict[str, Any], environment_variant: str
) -> list[tuple[str, int]]:
    if environment_variant not in campaign["environment_variants"]:
        raise ValueError(f"unknown campaign environment {environment_variant!r}")
    return [
        (algorithm, int(seed))
        for algorithm in campaign["algorithms"]
        for seed in campaign["seeds"]
    ]


def tasks(campaign: dict[str, Any]) -> list[tuple[str, str, int]]:
    return [
        (environment, algorithm, seed)
        for environment in campaign["environment_variants"]
        for algorithm, seed in tasks_for_environment(campaign, environment)
    ]


def resolve_task(
    campaign: dict[str, Any], environment_variant: str, task_id: int
) -> tuple[str, int]:
    environment_tasks = tasks_for_environment(campaign, environment_variant)
    if not 0 <= task_id < len(environment_tasks):
        raise ValueError(
            f"task id {task_id} is outside the valid range "
            f"0-{len(environment_tasks) - 1}"
        )
    return environment_tasks[task_id]


def campaign_for_environment(
    campaign: dict[str, Any], environment_variant: str
) -> dict[str, Any]:
    if environment_variant not in campaign["environment_variants"]:
        raise ValueError(f"unknown campaign environment {environment_variant!r}")
    environment_campaign = copy.deepcopy(campaign)
    training = environment_campaign["training"]
    training["env_variant"] = environment_variant
    batch = training["episode_matched_batches"][environment_variant]
    # Pass the derived sizes explicitly as well as --games_per_update. The
    # latter recomputes them in long_baseline; these explicit values keep the
    # printed command and attempt manifest self-describing.
    training["train_batch_size"] = int(batch["train_batch_size"])
    training["minibatch_size"] = int(batch["minibatch_size"])
    return environment_campaign


def environment_output_name(
    campaign: dict[str, Any], environment_variant: str
) -> str:
    return str(campaign["environment_output_names"][environment_variant])


def environment_root(
    campaign: dict[str, Any], campaign_root: Path, environment_variant: str
) -> Path:
    return campaign_root / environment_output_name(campaign, environment_variant)


def environment_variant_from_process(campaign: dict[str, Any]) -> str:
    import os

    environment_variant = os.environ.get("HUNGER_ENVIRONMENT_VARIANT")
    if not environment_variant:
        raise ValueError(
            "HUNGER_ENVIRONMENT_VARIANT is unset; submit with submit_all.sh "
            "or pass --environment-variant"
        )
    if environment_variant not in campaign["environment_variants"]:
        raise ValueError(f"unknown campaign environment {environment_variant!r}")
    return environment_variant
