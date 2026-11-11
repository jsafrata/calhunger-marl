"""Run checkpoint-vs-scripted diagnostics for one configured game horizon.

The shared diagnostic CLI owns all ordinary arguments.  This wrapper selects
the standalone rollout environment through ``HUNGER_ENVIRONMENT_VARIANT`` so
Slurm can evaluate checkpoints in the same horizon in which they were trained.
"""
from __future__ import annotations

import os

from env.env import FoodTradingEnv
from env.small_env import (
    FoodTradingEnv60,
    FoodTradingEnv90,
    FoodTradingEnv120,
    FoodTradingEnv150,
)
from eval.checkpoint_scripted_diagnostics import main as run_diagnostics


ENVIRONMENT_SPECS = {
    "small_60": (FoodTradingEnv60, 60),
    "small_90": (FoodTradingEnv90, 90),
    "small_120": (FoodTradingEnv120, 120),
    "small_150": (FoodTradingEnv150, 150),
    "default": (FoodTradingEnv, 180),
}


def resolve_environment(variant: str) -> tuple[type[FoodTradingEnv], int]:
    try:
        return ENVIRONMENT_SPECS[variant]
    except KeyError as exc:
        choices = ", ".join(ENVIRONMENT_SPECS)
        raise ValueError(
            f"HUNGER_ENVIRONMENT_VARIANT must be one of {choices}; got {variant!r}"
        ) from exc


def main() -> int:
    variant = os.environ.get("HUNGER_ENVIRONMENT_VARIANT", "")
    environment_class, duration = resolve_environment(variant)
    return run_diagnostics(
        environment_class=environment_class,
        environment_variant=variant,
        required_duration_seconds=duration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
