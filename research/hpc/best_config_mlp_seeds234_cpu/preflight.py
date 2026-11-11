from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from env.config import DEFAULT_CONFIG
from env.env import FoodTradingEnv
from env.rl.global_observation_encoder import global_observation_dim
from env.rl.observation_encoder import observation_dim
from env.rl.rllib_config import episode_matched_batch
from env.rl.rllib_env import NashHungerMultiAgentEnv
from env.small_env import SCALED_ENVIRONMENTS
from hpc.v2_5seed_cpu.preflight import inspect as inspect_base

from .common import load_campaign


def inspect(
    research_root: Path,
    campaign_root: Path | None,
    arm_name: str | None = None,
) -> dict[str, object]:
    campaign = load_campaign()
    report = inspect_base(research_root, campaign_root, campaign=campaign)
    errors = report["errors"]
    assert isinstance(errors, list)

    arms = (
        [arm for arm in campaign["arms"] if arm["name"] == arm_name]
        if arm_name is not None
        else campaign["arms"]
    )
    if arm_name is not None and not arms:
        errors.append(f"unknown campaign arm {arm_name!r}")

    observed_arms: list[dict[str, object]] = []
    for arm in arms:
        variant = arm["env_variant"]
        observed: dict[str, object] = {"arm": arm["name"], "variant": variant}
        try:
            if variant == "default":
                expected_config, expected_class = DEFAULT_CONFIG, FoodTradingEnv
            else:
                expected_config, expected_class = SCALED_ENVIRONMENTS[variant]
            adapter = NashHungerMultiAgentEnv(
                {
                    "seed": 123,
                    "reward_kind": "terminal_only",
                    "env_variant": variant,
                    "include_survival_features": True,
                    "include_trade_history": True,
                }
            )
            adapter.reset(seed=123)
            inner = adapter._env
            if not isinstance(inner, expected_class):
                errors.append(
                    f"adapter built {type(inner).__name__} for {variant!r}, "
                    f"expected {expected_class.__name__}"
                )
            if inner.config != expected_config:
                errors.append(f"{variant} did not use its frozen default config")
            duration = inner.config.game_duration_seconds
            if duration != int(arm["duration_seconds"]):
                errors.append(
                    f"{arm['name']} duration is {duration}, expected "
                    f"{arm['duration_seconds']}"
                )
            # The v4 observation contract must be live end to end.
            local_dim = observation_dim(include_survival_features=True)
            global_dim = global_observation_dim(include_trade_history=True)
            space = adapter.observation_spaces["seat_0"]
            if space["observations"].shape != (local_dim,):
                errors.append(
                    f"{arm['name']} local obs is {space['observations'].shape}, "
                    f"expected ({local_dim},) with survival features"
                )
            if space["critic_observations"].shape != (global_dim,):
                errors.append(
                    f"{arm['name']} critic obs is "
                    f"{space['critic_observations'].shape}, expected "
                    f"({global_dim},) with trade history"
                )
            train_batch_size, minibatch_size = episode_matched_batch(
                duration,
                games_per_update=int(campaign["training"]["games_per_update"]),
                num_minibatches=int(campaign["training"]["num_minibatches"]),
            )
            if train_batch_size != int(arm["train_batch_size"]):
                errors.append(f"{arm['name']} train batch is not episode-matched")
            if minibatch_size != int(arm["minibatch_size"]):
                errors.append(f"{arm['name']} minibatch does not match campaign")
            observed.update(
                {
                    "inner_environment": type(inner).__name__,
                    "torso": arm["torso"],
                    "duration_seconds": duration,
                    "train_batch_size": train_batch_size,
                    "minibatch_size": minibatch_size,
                    "local_observation_dim": local_dim,
                    "global_observation_dim": global_dim,
                    "walltime": arm["walltime"],
                }
            )
        except Exception as exc:
            errors.append(f"cannot validate {arm['name']}: {exc}")
        observed_arms.append(observed)

    report["best_config_arms"] = observed_arms
    report["ok"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the best-config bundled HPC campaign"
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
    parser.add_argument("--arm", type=str, default=os.environ.get("HUNGER_ARM"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = inspect(
        args.research_root.resolve(),
        args.campaign_root.resolve() if args.campaign_root else None,
        arm_name=args.arm,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
