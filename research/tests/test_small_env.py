"""Invariants for the small (30 s, 16 initial units, required-sum 8) env.

Every check runs against a fresh ``SmallFoodTradingEnv`` for many seeds to
catch dealer-level bugs and confirm the base env is not implicitly affected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.config import DEFAULT_CONFIG
from env.env import FoodTradingEnv
from env.small_env import (
    SMALL_CONFIG,
    SMALL_GAME_DURATION_SECONDS,
    SMALL_INITIAL_UNITS_PER_PLAYER,
    SMALL_REQUIRED_MAX,
    SMALL_REQUIRED_MIN,
    SMALL_REQUIRED_SUM,
    SmallFoodTradingEnv,
    deal_small_initial_food,
)


# ---------------------------------------------------------------------------
# Config invariants
# ---------------------------------------------------------------------------


def test_small_config_values():
    assert SMALL_CONFIG.game_duration_seconds == 30
    assert SMALL_CONFIG.initial_units_per_player == 16
    # Everything else mirrors DEFAULT_CONFIG.
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
        assert getattr(SMALL_CONFIG, field) == getattr(DEFAULT_CONFIG, field), field


def test_default_config_unchanged():
    """Sanity: importing small_env has not perturbed DEFAULT_CONFIG."""
    assert DEFAULT_CONFIG.game_duration_seconds == 180
    assert DEFAULT_CONFIG.initial_units_per_player == 100
    assert DEFAULT_CONFIG.initial_units_per_food_type == 100
    assert DEFAULT_CONFIG.min_initial_units_per_food_per_player == 20


# ---------------------------------------------------------------------------
# Initial-food invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(50)))
def test_initial_food_totals_and_required_split(seed):
    env = SmallFoodTradingEnv(seed=seed)
    env.reset()
    for player in env.state.players:
        inv = player.inventory
        total = sum(inv.values())
        assert total == SMALL_INITIAL_UNITS_PER_PLAYER, (
            f"seed={seed} player={player.idx} inv={inv} total={total}"
        )
        required = env.config.required_foods(player.produces)
        req_amounts = [inv[f] for f in required]
        assert sum(req_amounts) == SMALL_REQUIRED_SUM, (
            f"seed={seed} player={player.idx} produces={player.produces} "
            f"required={required} amounts={req_amounts}"
        )
        for f, units in zip(required, req_amounts):
            assert SMALL_REQUIRED_MIN <= units <= SMALL_REQUIRED_MAX, (
                f"seed={seed} player={player.idx} required food {f}={units} "
                f"outside [{SMALL_REQUIRED_MIN},{SMALL_REQUIRED_MAX}]"
            )
        # Produced foods sum to 16 - 8 = 8, each in [0, 8].
        produced = tuple(player.produces)
        prod_amounts = [inv[f] for f in produced]
        assert sum(prod_amounts) == SMALL_INITIAL_UNITS_PER_PLAYER - SMALL_REQUIRED_SUM
        for units in prod_amounts:
            assert 0 <= units <= (SMALL_INITIAL_UNITS_PER_PLAYER - SMALL_REQUIRED_SUM)


def test_initial_cash_matches_default():
    env = SmallFoodTradingEnv(seed=7)
    env.reset()
    for player in env.state.players:
        assert player.cash == SMALL_CONFIG.initial_cash


def test_required_food_split_covers_all_three_pairs():
    """Across enough seeds, all three legal required splits appear at
    least once — confirms the dealer is not silently pinned to (4,4)."""
    seen = set()
    for seed in range(200):
        env = SmallFoodTradingEnv(seed=seed)
        env.reset()
        for player in env.state.players:
            required = env.config.required_foods(player.produces)
            pair = tuple(player.inventory[f] for f in required)
            seen.add(pair)
    # Store as unordered pairs since the dealer randomizes which required
    # food gets which unit count.
    seen_frozen = {frozenset(p) for p in seen}
    assert {3, 5} in [set(p) for p in seen] or (3, 5) in seen or (5, 3) in seen
    assert (4, 4) in seen
    # Every observed split must be one of the three legal ones.
    for a, b in seen:
        assert (a, b) in {(3, 5), (4, 4), (5, 3)}


# ---------------------------------------------------------------------------
# Producer assignment invariants (inherited from base setup)
# ---------------------------------------------------------------------------


def test_each_player_still_produces_two_foods_and_two_producers_per_food():
    env = SmallFoodTradingEnv(seed=3)
    env.reset()
    producers_by_food: dict[str, int] = {f: 0 for f in env.config.food_types}
    for player in env.state.players:
        assert len(player.produces) == 2, player.produces
        assert len(set(player.produces)) == 2, player.produces
        for f in player.produces:
            producers_by_food[f] += 1
    for f, n in producers_by_food.items():
        assert n == 2, (f, n)


# ---------------------------------------------------------------------------
# End-to-end: a 30s game runs and terminates cleanly
# ---------------------------------------------------------------------------


def test_all_wait_game_ends_at_30s_or_zero_alive():
    env = SmallFoodTradingEnv(seed=0)
    env.reset()
    max_steps = env.config.game_duration_seconds + 5
    steps = 0
    while not env.is_done():
        env.step({i: 0 for i in range(4)})  # WAIT
        steps += 1
        assert steps <= max_steps
    assert env.state.elapsed_seconds <= SMALL_GAME_DURATION_SECONDS
    assert env.state.end_reason in ("time_limit", "no_survivors")
    # Under the small config's tight required-food budget (8 units total,
    # 2 required foods × 2 units eaten per cycle × 3 cycles = 12 needed
    # to survive to 30 s without trading), every all-WAIT game must
    # exhaust its buffer by tick 30 and end in no_survivors.
    assert env.state.end_reason == "no_survivors"


# ---------------------------------------------------------------------------
# Isolation from the default env
# ---------------------------------------------------------------------------


def test_default_env_still_deals_100_units_after_small_env_import():
    """Instantiating the default env after small_env import must still
    deal the default 100 units per player."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    for player in env.state.players:
        assert sum(player.inventory.values()) == DEFAULT_CONFIG.initial_units_per_player


def test_default_env_horizon_still_180():
    env = FoodTradingEnv(seed=0)
    env.reset()
    assert env.config.game_duration_seconds == 180


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_initial_inventory():
    a = SmallFoodTradingEnv(seed=42)
    a.reset()
    b = SmallFoodTradingEnv(seed=42)
    b.reset()
    for pa, pb in zip(a.state.players, b.state.players):
        assert pa.produces == pb.produces
        assert pa.inventory == pb.inventory
