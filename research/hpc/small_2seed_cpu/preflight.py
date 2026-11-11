from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from env.rl.rllib_env import NashHungerMultiAgentEnv
from env.small_env import (
    SMALL_GAME_DURATION_SECONDS,
    SMALL_INITIAL_UNITS_PER_PLAYER,
    SMALL_REQUIRED_MAX,
    SMALL_REQUIRED_MIN,
    SMALL_REQUIRED_SUM,
    SmallFoodTradingEnv,
)
from hpc.v2_5seed_cpu.preflight import inspect as inspect_base

from .common import load_campaign


def inspect(research_root: Path, campaign_root: Path | None) -> dict[str, object]:
    campaign = load_campaign()
    report = inspect_base(research_root, campaign_root, campaign=campaign)
    errors = report["errors"]
    assert isinstance(errors, list)

    observed: dict[str, object] = {}
    try:
        adapter = NashHungerMultiAgentEnv(
            {"seed": 123, "reward_kind": "terminal_only", "env_variant": "small"}
        )
        adapter.reset(seed=123)
        inner = adapter._env
        if not isinstance(inner, SmallFoodTradingEnv):
            errors.append(
                "RLlib adapter did not construct SmallFoodTradingEnv for "
                "env_variant='small'"
            )
        elif inner.config.game_duration_seconds != SMALL_GAME_DURATION_SECONDS:
            errors.append("small environment horizon is not 30 seconds")
        else:
            players = []
            for player in inner.state.players:
                required = inner.config.required_foods(player.produces)
                required_values = [player.inventory[food] for food in required]
                players.append(
                    {
                        "seat": player.idx,
                        "total_inventory": sum(player.inventory.values()),
                        "required_inventory": required_values,
                    }
                )
                if sum(player.inventory.values()) != SMALL_INITIAL_UNITS_PER_PLAYER:
                    errors.append(f"seat {player.idx} does not start with 16 units")
                if sum(required_values) != SMALL_REQUIRED_SUM:
                    errors.append(f"seat {player.idx} required-food sum is not 8")
                if not all(
                    SMALL_REQUIRED_MIN <= units <= SMALL_REQUIRED_MAX
                    for units in required_values
                ):
                    errors.append(
                        f"seat {player.idx} required-food units are outside 3..5"
                    )
            observed = {
                "adapter": type(adapter).__name__,
                "inner_environment": type(inner).__name__,
                "duration_seconds": inner.config.game_duration_seconds,
                "players": players,
            }
    except Exception as exc:
        errors.append(f"cannot validate small RLlib environment: {exc}")

    report["small_environment"] = observed
    report["ok"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the small HPC campaign")
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
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = inspect(args.research_root.resolve(), args.campaign_root)
    if args.strict and report["git_dirty"]:
        errors = report["errors"]
        assert isinstance(errors, list)
        errors.append("strict mode rejects a dirty git worktree")
        report["ok"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
