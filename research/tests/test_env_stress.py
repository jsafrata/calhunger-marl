"""One-command stress coverage for Nash Hunger environment invariants.

Run with:

    python -m pytest research/tests/test_env_stress.py

The tests are deterministic and bounded. They favor clear failure context over
huge random searches, but still exercise many seeds, wrapper resets, masked
actions, market replacement/cancel/take interactions, reward timing, and
terminal transitions.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from env.actions import ACTION_NOOP, NUM_ACTIONS, PRICE_BUCKETS, QTY_BUCKETS, encode_trade
from env.config import GameConfig
from env.env import FoodTradingEnv
from env.market import post_order
from env.rl.rewards import ShapedRewardTracker


def _trade(side: str, food: str, qty: int, price: int) -> int:
    return encode_trade(
        side,
        food,
        QTY_BUCKETS.index(qty),
        PRICE_BUCKETS.index(price),
    )


def _fund_everyone(
    env: FoodTradingEnv,
    *,
    cash: int = 1000,
    inventory: int = 50,
) -> None:
    for player in env.state.players:
        player.cash = cash
        player.reserved_cash = 0
        player.inventory = {food: inventory for food in env.config.food_types}
        player.reserved_inventory = {food: 0 for food in env.config.food_types}


def _assert_resource_invariants(env: FoodTradingEnv, *, label: str) -> None:
    expected_reserved_cash = {player.idx: 0 for player in env.state.players}
    expected_reserved_inventory = {
        player.idx: {food: 0 for food in env.config.food_types}
        for player in env.state.players
    }

    for food in env.config.food_types:
        book = env.state.order_books[food]
        for order in book.bids:
            expected_reserved_cash[order.player_idx] += (
                order.price_per_unit * order.remaining_quantity
            )
        for order in book.asks:
            expected_reserved_inventory[order.player_idx][food] += (
                order.remaining_quantity
            )

    for player in env.state.players:
        assert player.cash >= 0, f"{label}: P{player.idx} negative cash"
        assert player.reserved_cash >= 0, (
            f"{label}: P{player.idx} negative reserved cash"
        )
        assert player.reserved_cash <= player.cash, (
            f"{label}: P{player.idx} reserved cash exceeds cash"
        )
        assert player.reserved_cash == expected_reserved_cash[player.idx], (
            f"{label}: P{player.idx} reserved cash does not match open bids"
        )
        for food in env.config.food_types:
            assert player.inventory[food] >= 0, (
                f"{label}: P{player.idx} negative inventory {food}"
            )
            assert player.reserved_inventory[food] >= 0, (
                f"{label}: P{player.idx} negative reserved inventory {food}"
            )
            assert player.reserved_inventory[food] <= player.inventory[food], (
                f"{label}: P{player.idx} reserved inventory exceeds inventory {food}"
            )
            assert (
                player.reserved_inventory[food]
                == expected_reserved_inventory[player.idx][food]
            ), (
                f"{label}: P{player.idx} reserved inventory {food} does not "
                "match open asks"
            )


def _assert_order_book_invariants(env: FoodTradingEnv, *, label: str) -> None:
    seen_ids: set[int] = set()
    open_counts = {player.idx: 0 for player in env.state.players}

    for food in env.config.food_types:
        book = env.state.order_books[food]
        bid_prices = [order.price_per_unit for order in book.bids]
        ask_prices = [order.price_per_unit for order in book.asks]
        assert bid_prices == sorted(bid_prices, reverse=True), (
            f"{label}: bids unsorted for {food}"
        )
        assert ask_prices == sorted(ask_prices), f"{label}: asks unsorted for {food}"

        for side, orders in (("bid", book.bids), ("ask", book.asks)):
            for order in orders:
                assert order.id not in seen_ids, f"{label}: duplicate order {order.id}"
                seen_ids.add(order.id)
                open_counts[order.player_idx] += 1
                assert order.food_type == food, (
                    f"{label}: order {order.id} in wrong food book"
                )
                assert order.side == side, (
                    f"{label}: order {order.id} in wrong side book"
                )
                assert order.remaining_quantity > 0, (
                    f"{label}: non-positive remaining qty on order {order.id}"
                )
                assert order.status in {"open", "partially_filled"}, (
                    f"{label}: finalized order {order.id} remains on book"
                )

    for player_idx, count in open_counts.items():
        assert count <= env.config.max_open_orders_per_player, (
            f"{label}: P{player_idx} exceeds max open orders"
        )


def _assert_mask_invariants(env: FoodTradingEnv, *, label: str) -> None:
    for player in env.state.players:
        mask = env.legal_actions(player.idx)
        assert mask.shape == (NUM_ACTIONS,), (
            f"{label}: P{player.idx} wrong mask shape {mask.shape}"
        )
        assert mask.dtype == np.bool_, f"{label}: P{player.idx} non-bool mask"
        assert mask[ACTION_NOOP], f"{label}: P{player.idx} cannot wait"
        if player.status != "alive" or env.state.phase != "active":
            only_wait = mask.copy()
            only_wait[ACTION_NOOP] = False
            assert not only_wait.any(), (
                f"{label}: P{player.idx} inactive mask has non-wait actions"
            )


def _assert_all_invariants(env: FoodTradingEnv, *, label: str) -> None:
    _assert_order_book_invariants(env, label=label)
    _assert_resource_invariants(env, label=label)
    _assert_mask_invariants(env, label=label)


def _sample_legal_action(
    env: FoodTradingEnv,
    player_idx: int,
    rng: np.random.Generator,
) -> int:
    legal = np.flatnonzero(env.legal_actions(player_idx))
    assert legal.size > 0
    return int(rng.choice(legal))


@pytest.mark.parametrize("simultaneous", [False, True])
def test_stress_random_masked_actions_preserve_market_invariants(simultaneous: bool):
    """Random legal actions should not corrupt cash, inventory, reserves, or books."""
    config = GameConfig(simultaneous_order_clearing=simultaneous)
    for seed in range(10):
        env = FoodTradingEnv(seed=seed, config=config)
        env.reset()
        _fund_everyone(env)
        rng = np.random.default_rng(seed + 10_000)
        _assert_all_invariants(env, label=f"seed={seed} initial")

        for step in range(60):
            if env.is_done():
                break
            actions = {
                player.idx: _sample_legal_action(env, player.idx, rng)
                for player in env.state.players
            }
            env.step(actions)
            _assert_all_invariants(
                env,
                label=f"simultaneous={simultaneous} seed={seed} step={step}",
            )


@pytest.mark.parametrize("simultaneous", [False, True])
def test_stress_adversarial_replacements_cancels_and_takes(simultaneous: bool):
    """Repeated replace/cancel/take cycles should keep reservations exact."""
    config = GameConfig(simultaneous_order_clearing=simultaneous)
    for seed in range(20):
        env = FoodTradingEnv(seed=seed, config=config)
        env.reset()
        _fund_everyone(env, cash=500, inventory=30)

        for step in range(30):
            bid_price = 8 + (step % 5)
            ask_price = 6 + (step % 5)
            actions = {
                0: _trade("bid", "A", 1 + (step % 3), bid_price),
                1: _trade("ask", "A", 1 + ((step + 1) % 3), ask_price),
                2: _trade("bid", "B", 1 + ((step + 2) % 3), bid_price),
                3: _trade("ask", "B", 1 + (step % 3), ask_price),
            }
            env.step(actions)

            if step % 4 == 3:
                env.step(
                    {
                        0: _trade("bid", "A", 1, bid_price),
                        1: ACTION_NOOP,
                        2: _trade("ask", "B", 1, ask_price),
                        3: ACTION_NOOP,
                    }
                )
            _assert_all_invariants(
                env,
                label=f"simultaneous={simultaneous} seed={seed} step={step}",
            )


def test_stress_reward_timing_and_terminal_rewards_are_well_formed():
    """Dense reward should only fire on consumption ticks; terminal once at end."""
    for seed in range(20):
        env = FoodTradingEnv(seed=seed)
        observations = env.reset()
        tracker = ShapedRewardTracker(env.config, dense_reward=True)
        tracker.reset(env)

        saw_terminal = False
        for step in range(env.config.game_duration_seconds + 5):
            observations, _, done, info = env.step(
                {idx: ACTION_NOOP for idx in range(env.config.player_count)}
            )
            rewards = tracker.compute(env, observations, done=done, info=info)
            breakdown = tracker.last_breakdown

            is_consumption = (
                info.tick > 0
                and info.tick % env.config.consumption_interval_seconds == 0
            )
            for idx, reward in rewards.items():
                assert reward == pytest.approx(breakdown[idx]["total"])
                assert breakdown[idx]["dense"] >= 0.0
                if not is_consumption:
                    assert breakdown[idx]["dense"] == 0.0, (
                        f"seed={seed} step={step}: dense reward off tick"
                    )
                if not done:
                    assert breakdown[idx]["terminal"] == 0.0, (
                        f"seed={seed} step={step}: early terminal reward"
                    )

            if done:
                assert not saw_terminal, f"seed={seed}: terminal emitted twice"
                saw_terminal = True
                assert env.state.phase == "ended"
                assert info.end_reason in {
                    "time_limit",
                    "single_survivor",
                    "no_survivors",
                }
                break

        assert saw_terminal, f"seed={seed}: episode did not terminate"


def test_stress_death_terminal_state_and_masks():
    """Players killed at consumption should become wait-only after one terminal edge."""
    for seed in range(30):
        env = FoodTradingEnv(seed=seed)
        env.reset()
        victim = env.state.players[0]
        missing = env.config.required_foods(victim.produces)[0]
        victim.inventory[missing] = 0
        victim.reserved_inventory[missing] = 0
        env.state.elapsed_seconds = env.config.consumption_interval_seconds - 1

        _, _, done, info = env.step(
            {idx: ACTION_NOOP for idx in range(env.config.player_count)}
        )

        assert 0 in info.deaths_this_tick, f"seed={seed}: victim did not die"
        assert env.state.players[0].status == "dead"
        mask = env.legal_actions(0)
        assert mask[ACTION_NOOP]
        mask_without_wait = mask.copy()
        mask_without_wait[ACTION_NOOP] = False
        assert not mask_without_wait.any()
        if not done:
            env.step({idx: ACTION_NOOP for idx in range(env.config.player_count)})
            assert env.state.players[0].status == "dead"


def test_stress_explicit_seed_resets_are_reproducible():
    """Core env and RLlib wrapper should repeat exact observations for explicit seeds."""
    for seed in range(20):
        env_a = FoodTradingEnv(seed=seed)
        env_b = FoodTradingEnv(seed=seed)
        assert env_a.reset() == env_b.reset()

    if importlib.util.find_spec("ray") is None:
        pytest.skip("ray[rllib] is not installed")

    from env.rl.rllib_env import NashHungerMultiAgentEnv

    for seed in range(5):
        wrapper = NashHungerMultiAgentEnv()
        obs_a, _ = wrapper.reset(seed=seed)
        obs_b, _ = wrapper.reset(seed=seed)
        for agent_id in wrapper.possible_agents:
            for key in ("observations", "action_mask", "critic_observations"):
                np.testing.assert_array_equal(obs_a[agent_id][key], obs_b[agent_id][key])


@pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None,
    reason="gymnasium not installed",
)
@pytest.mark.xfail(
    strict=True,
    reason="NashHungerGym.reset(seed=None) currently repeats seed 0",
)
def test_stress_gym_no_seed_resets_do_not_repeat_same_episode():
    from env.mappo_env import NashHungerGym

    env = NashHungerGym()
    seeds = []
    first_observation_signatures = []
    for _ in range(8):
        obs, _ = env.reset()
        assert env._env is not None
        seeds.append(env._env.seed)
        first_observation_signatures.append(tuple(obs[0].tolist()))

    assert len(set(seeds)) > 1
    assert len(set(first_observation_signatures)) > 1


@pytest.mark.xfail(
    strict=True,
    reason="legal_action_mask currently ignores reusable reserves from replacement orders",
)
def test_stress_replacement_posts_are_legal_when_prior_reserves_cover_them():
    """Known bug: replacement masks should match replacement-aware post validation."""
    for side in ("bid", "ask"):
        for food in ("A", "B", "C", "D"):
            env = FoodTradingEnv(seed=0)
            env.reset()
            _fund_everyone(env, cash=10, inventory=1)
            if side == "bid":
                post_order(
                    env.state,
                    player_idx=0,
                    side=side,
                    food_type=food,
                    quantity=1,
                    price_per_unit=10,
                )
                action = _trade(side, food, 1, 10)
            else:
                post_order(
                    env.state,
                    player_idx=0,
                    side=side,
                    food_type=food,
                    quantity=1,
                    price_per_unit=5,
                )
                env.state.players[0].inventory[food] = 1
                action = _trade(side, food, 1, 5)

            assert env.legal_actions(0)[action], f"side={side} food={food}"


@pytest.mark.xfail(
    strict=True,
    reason="legacy immediate post_order validates before locating replaceable prior order",
)
def test_stress_legacy_replacements_reuse_prior_reserves():
    config = GameConfig(simultaneous_order_clearing=False)
    for side in ("bid", "ask"):
        for food in ("A", "B", "C", "D"):
            env = FoodTradingEnv(seed=0, config=config)
            env.reset()
            _fund_everyone(env, cash=10, inventory=1)
            price = 10 if side == "bid" else 5
            first, _ = post_order(
                env.state,
                player_idx=0,
                side=side,
                food_type=food,
                quantity=1,
                price_per_unit=price,
            )
            second, _ = post_order(
                env.state,
                player_idx=0,
                side=side,
                food_type=food,
                quantity=1,
                price_per_unit=price,
            )

            assert second.id != first.id
            _assert_all_invariants(env, label=f"side={side} food={food}")


@pytest.mark.xfail(
    strict=True,
    reason="simultaneous stale replacement post can raise after same-tick take",
)
def test_stress_simultaneous_stale_replacements_are_rejected_not_crashes():
    config = GameConfig(simultaneous_order_clearing=True)
    saw_rejected_replacement = False

    for seed in range(100):
        env = FoodTradingEnv(seed=seed, config=config)
        env.reset()
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

        try:
            _, _, _, info = env.step(
                {
                    0: _trade("ask", "A", 1, 10),
                    1: _trade("bid", "A", 1, 10),
                    2: ACTION_NOOP,
                    3: ACTION_NOOP,
                }
            )
        except Exception as exc:
            pytest.fail(f"seed={seed}: stale replacement crashed: {exc!r}")

        if not info.actions_applied[0]["applied"]:
            assert info.actions_applied[0]["code"] == "insufficient_inventory"
            saw_rejected_replacement = True

    assert saw_rejected_replacement
