"""Known environment bugs captured as executable MREs.

These tests are intentionally marked xfail. They document currently observed
environment issues without forcing the suite red before the environment fixes
are approved.
"""
from __future__ import annotations

import importlib.util

import pytest

from env.actions import ACTION_NOOP, PRICE_BUCKETS, QTY_BUCKETS, encode_trade
from env.config import GameConfig
from env.env import FoodTradingEnv
from env.market import post_order


def _trade(side: str, food: str, qty: int, price: int) -> int:
    return encode_trade(
        side,
        food,
        QTY_BUCKETS.index(qty),
        PRICE_BUCKETS.index(price),
    )


def _fund_everyone(env: FoodTradingEnv, *, cash: int = 1000, inventory: int = 50) -> None:
    for player in env.state.players:
        player.cash = cash
        player.reserved_cash = 0
        player.inventory = {food: inventory for food in env.config.food_types}
        player.reserved_inventory = {food: 0 for food in env.config.food_types}


@pytest.mark.xfail(
    strict=True,
    reason="legal_action_mask ignores reusable reserves from the order being replaced",
)
def test_known_bug_replacement_post_mask_reuses_reserved_resources():
    env = FoodTradingEnv(seed=0)
    _fund_everyone(env, cash=10, inventory=1)
    player_idx = 0
    post_order(
        env.state,
        player_idx=player_idx,
        side="bid",
        food_type="A",
        quantity=1,
        price_per_unit=10,
    )

    replace_same_bid = _trade("bid", "A", 1, 10)

    assert env.legal_actions(player_idx)[replace_same_bid]


@pytest.mark.xfail(
    strict=True,
    reason="legacy immediate post_order validates before locating the replaceable prior order",
)
def test_known_bug_legacy_post_replacement_reuses_reserved_resources():
    env = FoodTradingEnv(
        seed=0,
        config=GameConfig(simultaneous_order_clearing=False),
    )
    _fund_everyone(env, cash=10, inventory=1)
    player_idx = 0
    first_order, _ = post_order(
        env.state,
        player_idx=player_idx,
        side="bid",
        food_type="A",
        quantity=1,
        price_per_unit=10,
    )

    replacement, _ = post_order(
        env.state,
        player_idx=player_idx,
        side="bid",
        food_type="A",
        quantity=1,
        price_per_unit=10,
    )

    assert replacement.id != first_order.id
    assert env.state.players[player_idx].reserved_cash == 10


@pytest.mark.xfail(
    strict=True,
    reason="simultaneous post can become invalid after another same-tick action consumes its prior order",
)
def test_known_bug_simultaneous_stale_replacement_post_does_not_crash():
    action_config = GameConfig(simultaneous_order_clearing=True)

    for seed in range(100):
        env = FoodTradingEnv(seed=seed, config=action_config)
        _fund_everyone(env, cash=1000, inventory=0)
        env.state.players[0].inventory["A"] = 1
        env.state.players[1].cash = 10
        post_order(
            env.state,
            player_idx=0,
            side="ask",
            food_type="A",
            quantity=1,
            price_per_unit=10,
        )

        actions = {
            0: _trade("ask", "A", 1, 10),
            1: _trade("bid", "A", 1, 10),
            2: ACTION_NOOP,
            3: ACTION_NOOP,
        }
        try:
            _, _, _, info = env.step(actions)
        except Exception as exc:  # current bug path
            pytest.fail(f"seed={seed} crashed with {exc!r}")
        if not info.actions_applied[0]["applied"]:
            assert info.actions_applied[0]["code"] == "insufficient_inventory"
            return

    pytest.fail("stress loop did not hit a stale-replacement ordering")


if importlib.util.find_spec("gymnasium") is not None:
    from env.mappo_env import NashHungerGym


@pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None,
    reason="gymnasium not installed",
)
@pytest.mark.xfail(
    strict=True,
    reason="NashHungerGym.reset(seed=None) always rebuilds seed 0",
)
def test_known_bug_gym_reset_without_seed_advances_episode_seeds():
    env = NashHungerGym()

    seen = []
    for _ in range(4):
        env.reset()
        assert env._env is not None
        seen.append(env._env.seed)

    assert len(set(seen)) > 1
