from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from env.config import DEFAULT_CONFIG
from env.env import FoodTradingEnv
from env.rl.rllib_config import episode_matched_batch
from env.rl.rllib_env import NashHungerMultiAgentEnv
from env.small_env import SCALED_ENVIRONMENTS
from hpc.v2_5seed_cpu.preflight import inspect as inspect_base

from .common import EXPECTED_DURATIONS, load_campaign


def inspect(
    research_root: Path,
    campaign_root: Path | None,
    environment_variant: str | None = None,
) -> dict[str, object]:
    campaign = load_campaign()
    report = inspect_base(research_root, campaign_root, campaign=campaign)
    errors = report["errors"]
    assert isinstance(errors, list)

    environments = (
        [environment_variant]
        if environment_variant is not None
        else campaign["environment_variants"]
    )
    observed_environments: list[dict[str, object]] = []
    for variant in environments:
        if variant not in campaign["environment_variants"]:
            errors.append(f"unknown campaign environment {variant!r}")
            continue
        if variant == "default":
            expected_config = DEFAULT_CONFIG
            expected_class = FoodTradingEnv
        else:
            expected_config, expected_class = SCALED_ENVIRONMENTS[variant]

        observed: dict[str, object] = {"variant": variant}
        try:
            adapter = NashHungerMultiAgentEnv(
                {
                    "seed": 123,
                    "reward_kind": "terminal_only",
                    "env_variant": variant,
                }
            )
            adapter.reset(seed=123)
            inner = adapter._env
            if not isinstance(inner, expected_class):
                errors.append(
                    f"RLlib adapter constructed {type(inner).__name__} for "
                    f"env_variant={variant!r}, expected {expected_class.__name__}"
                )
            if inner.config != expected_config:
                errors.append(f"{variant} did not use its frozen default config")

            duration = inner.config.game_duration_seconds
            if duration != EXPECTED_DURATIONS[variant]:
                errors.append(
                    f"{variant} duration is {duration}, expected "
                    f"{EXPECTED_DURATIONS[variant]}"
                )
            train_batch_size, minibatch_size = episode_matched_batch(
                duration,
                games_per_update=int(campaign["training"]["games_per_update"]),
                num_minibatches=int(campaign["training"]["num_minibatches"]),
            )
            declared = campaign["training"]["episode_matched_batches"][variant]
            if train_batch_size != int(declared["train_batch_size"]):
                errors.append(f"{variant} train batch does not match campaign")
            if minibatch_size != int(declared["minibatch_size"]):
                errors.append(f"{variant} minibatch does not match campaign")
            observed.update(
                {
                    "inner_environment": type(inner).__name__,
                    "duration_seconds": duration,
                    "games_per_update": campaign["training"]["games_per_update"],
                    "train_batch_size": train_batch_size,
                    "minibatch_size": minibatch_size,
                    "walltime": campaign["resources"]["walltime_by_environment"][
                        variant
                    ],
                }
            )
        except Exception as exc:
            errors.append(f"cannot validate {variant}: {exc}")
        observed_environments.append(observed)

    report["episode_matched_environments"] = observed_environments
    report["ok"] = not errors
    return report


def main() -> int:
    campaign = load_campaign()
    parser = argparse.ArgumentParser(
        description="Validate the episode-matched five-horizon HPC campaign"
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
    )
    parser.add_argument(
        "--environment-variant",
        choices=campaign["environment_variants"],
        default=os.environ.get("HUNGER_ENVIRONMENT_VARIANT"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = inspect(
        args.research_root.resolve(), args.campaign_root, args.environment_variant
    )
    if args.strict and report["git_dirty"]:
        errors = report["errors"]
        assert isinstance(errors, list)
        errors.append("strict mode rejects a dirty git worktree")
        report["ok"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
