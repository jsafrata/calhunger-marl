"""Invariants for the 60/90/120/150-second scaled environments."""
from __future__ import annotations

import pytest

from env.config import DEFAULT_CONFIG
from env.rl.rllib_env import NashHungerMultiAgentEnv
from env.small_env import (
    CONFIG_60,
    CONFIG_90,
    CONFIG_120,
    CONFIG_150,
    FOOD_SPEC_60,
    FOOD_SPEC_90,
    FOOD_SPEC_120,
    FOOD_SPEC_150,
    FoodTradingEnv60,
    FoodTradingEnv90,
    FoodTradingEnv120,
    FoodTradingEnv150,
)


CASES = (
    ("small_60", FoodTradingEnv60, CONFIG_60, FOOD_SPEC_60, 60, 32, 16, 6, 10),
    ("small_90", FoodTradingEnv90, CONFIG_90, FOOD_SPEC_90, 90, 48, 24, 9, 15),
    (
        "small_120",
        FoodTradingEnv120,
        CONFIG_120,
        FOOD_SPEC_120,
        120,
        64,
        32,
        12,
        20,
    ),
    (
        "small_150",
        FoodTradingEnv150,
        CONFIG_150,
        FOOD_SPEC_150,
        150,
        80,
        40,
        15,
        25,
    ),
)


@pytest.mark.parametrize(
    "variant,env_class,config,spec,duration,total,required_sum,required_min,required_max",
    CASES,
)
def test_scaled_config_values(
    variant,
    env_class,
    config,
    spec,
    duration,
    total,
    required_sum,
    required_min,
    required_max,
):
    del variant, env_class
    assert config.game_duration_seconds == duration
    assert config.initial_units_per_player == total
    assert spec.game_duration_seconds == duration
    assert spec.initial_units_per_player == total
    assert spec.required_sum == required_sum
    assert spec.required_min == required_min
    assert spec.required_max == required_max
    assert 2 * spec.required_sum == spec.initial_units_per_player
    assert spec.produced_sum == required_sum

    # Rates, market rules, cash, player count, and action constraints are
    # identical to the original environment.
    for field in (
        "player_count",
        "food_types",
        "initial_cash",
        "production_per_second",
        "consumption_per_required_food_per_second",
        "consumption_interval_seconds",
        "min_price",
        "min_order_quantity",
        "max_open_orders_per_player",
        "max_new_orders_per_player_per_second",
        "max_cancels_per_player_per_second",
        "simultaneous_order_clearing",
    ):
        assert getattr(config, field) == getattr(DEFAULT_CONFIG, field), field


@pytest.mark.parametrize(
    "variant,env_class,config,spec,duration,total,required_sum,required_min,required_max",
    CASES,
)
@pytest.mark.parametrize("seed", range(25))
def test_scaled_dealer_invariants(
    variant,
    env_class,
    config,
    spec,
    duration,
    total,
    required_sum,
    required_min,
    required_max,
    seed,
):
    del variant, config, spec, duration
    env = env_class(seed=seed)
    env.reset()
    for player in env.state.players:
        required = env.config.required_foods(player.produces)
        required_values = [player.inventory[food] for food in required]
        produced_values = [player.inventory[food] for food in player.produces]

        assert sum(player.inventory.values()) == total
        assert sum(required_values) == required_sum == total // 2
        assert all(required_min <= value <= required_max for value in required_values)
        assert sum(produced_values) == total - required_sum == total // 2
        assert all(0 <= value <= total // 2 for value in produced_values)


@pytest.mark.parametrize(
    "variant,env_class,config,spec,duration,total,required_sum,required_min,required_max",
    CASES,
)
def test_scaled_environments_are_deterministic(
    variant,
    env_class,
    config,
    spec,
    duration,
    total,
    required_sum,
    required_min,
    required_max,
):
    del variant, config, spec, duration, total, required_sum, required_min, required_max
    first = env_class(seed=42)
    second = env_class(seed=42)
    first.reset()
    second.reset()
    assert [player.produces for player in first.state.players] == [
        player.produces for player in second.state.players
    ]
    assert [player.inventory for player in first.state.players] == [
        player.inventory for player in second.state.players
    ]


@pytest.mark.parametrize(
    "variant,env_class,config,spec,duration,total,required_sum,required_min,required_max",
    CASES,
)
def test_rllib_adapter_selects_scaled_environment(
    variant,
    env_class,
    config,
    spec,
    duration,
    total,
    required_sum,
    required_min,
    required_max,
):
    del config, spec, required_sum, required_min, required_max
    adapter = NashHungerMultiAgentEnv(
        {"seed": 17, "reward_kind": "terminal_only", "env_variant": variant}
    )
    adapter.reset(seed=17)
    assert isinstance(adapter._env, env_class)
    assert adapter.config.game_duration_seconds == duration
    assert adapter.config.initial_units_per_player == total


@pytest.mark.parametrize(
    "variant,env_class,config,spec,duration,total,required_sum,required_min,required_max",
    CASES,
)
def test_all_wait_ends_no_later_than_scaled_horizon(
    variant,
    env_class,
    config,
    spec,
    duration,
    total,
    required_sum,
    required_min,
    required_max,
):
    del variant, config, spec, total, required_sum, required_min, required_max
    env = env_class(seed=0)
    env.reset()
    while not env.is_done():
        env.step({player_idx: 0 for player_idx in range(4)})
    assert env.state.elapsed_seconds <= duration
    assert env.state.end_reason == "no_survivors"
