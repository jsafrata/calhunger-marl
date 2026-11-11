"""Evaluate simplified-game checkpoints for exactly the 30-second horizon.

This is the safe entry point for checkpoints trained with
``env_variant=small``.  It deliberately fixes the standalone rollout
environment to ``SmallFoodTradingEnv`` and refuses to start if that class no
longer advertises the expected 30-second duration.
"""
from __future__ import annotations

from env.small_env import SMALL_GAME_DURATION_SECONDS, SmallFoodTradingEnv
from eval.checkpoint_scripted_diagnostics import main as run_checkpoint_diagnostics


def main() -> int:
    return run_checkpoint_diagnostics(
        environment_class=SmallFoodTradingEnv,
        environment_variant="small",
        required_duration_seconds=SMALL_GAME_DURATION_SECONDS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
