from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from env.rl.rllib_env import NashHungerMultiAgentEnv
from env.small_env import SCALED_ENVIRONMENTS
from hpc.v2_5seed_cpu.preflight import inspect as inspect_base

from .common import load_campaign


def inspect(research_root: Path, campaign_root: Path | None) -> dict[str, object]:
    campaign = load_campaign()
    report = inspect_base(research_root, campaign_root, campaign=campaign)
    errors = report["errors"]
    assert isinstance(errors, list)

    observed_environments: list[dict[str, object]] = []
    for environment_variant in campaign["environment_variants"]:
        default_config, expected_class = SCALED_ENVIRONMENTS[environment_variant]
        spec = expected_class.FOOD_SPEC
        observed: dict[str, object] = {"variant": environment_variant}
        try:
            adapter = NashHungerMultiAgentEnv(
                {
                    "seed": 123,
                    "reward_kind": "terminal_only",
                    "env_variant": environment_variant,
                }
            )
            adapter.reset(seed=123)
            inner = adapter._env
            if not isinstance(inner, expected_class):
                errors.append(
                    f"RLlib adapter constructed {type(inner).__name__} for "
                    f"env_variant={environment_variant!r}, expected "
                    f"{expected_class.__name__}"
                )
                observed_environments.append(observed)
                continue

            if inner.config != default_config:
                errors.append(
                    f"{environment_variant} did not use its frozen default config"
                )

            players = []
            for player in inner.state.players:
                required = inner.config.required_foods(player.produces)
                required_values = [player.inventory[food] for food in required]
                total_inventory = sum(player.inventory.values())
                players.append(
                    {
                        "seat": player.idx,
                        "total_inventory": total_inventory,
                        "required_inventory": required_values,
                    }
                )
                if total_inventory != spec.initial_units_per_player:
                    errors.append(
                        f"{environment_variant} seat {player.idx} starts with "
                        f"{total_inventory}, expected {spec.initial_units_per_player}"
                    )
                if sum(required_values) != spec.required_sum:
                    errors.append(
                        f"{environment_variant} seat {player.idx} required-food "
                        f"sum is {sum(required_values)}, expected {spec.required_sum}"
                    )
                if not all(
                    spec.required_min <= units <= spec.required_max
                    for units in required_values
                ):
                    errors.append(
                        f"{environment_variant} seat {player.idx} required-food "
                        f"units are outside {spec.required_min}..{spec.required_max}"
                    )

            observed.update(
                {
                    "inner_environment": type(inner).__name__,
                    "duration_seconds": inner.config.game_duration_seconds,
                    "initial_units_per_player": spec.initial_units_per_player,
                    "required_sum": spec.required_sum,
                    "required_bounds": [spec.required_min, spec.required_max],
                    "players": players,
                }
            )
        except Exception as exc:
            errors.append(f"cannot validate {environment_variant}: {exc}")
        observed_environments.append(observed)

    report["scaled_environments"] = observed_environments
    report["ok"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the four-environment scaled HPC campaign"
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
