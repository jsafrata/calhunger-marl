"""ScriptedV2 — inventory-skewed market maker."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from agents.scripted_common import CONSUME_PER_CYCLE, horizon_need
from agents.scripted_v2 import PRODUCED_WORKING_STOCK, SKEW_GAIN, TARGET_BUFFER, ScriptedV2Agent
from env.actions import PostAction, TakeAction, decode
from env.env import FoodTradingEnv
from env.market import post_order

from env.small_env import SMALL_GAME_DURATION_SECONDS, SmallFoodTradingEnv

from tests.scripted_harness import CADENCES, ENV_CLASSES, obs_with_inventory, play_game


def _agent_with_history(obs: Dict[str, Any], price: int, seed: int = 0) -> ScriptedV2Agent:
    """A V2 whose fair value has settled at `price` for every food.

    Without trade history the estimate falls back to the book midpoint, which
    makes any resting order look correctly priced by construction — so a
    crossing test needs an independently-established fair value.
    """
    agent = ScriptedV2Agent(seed=seed)
    feed = dict(obs)
    for tick in range(1, 15):
        feed["recent_trades"] = [
            {"food": f, "price": price, "qty": 1, "tick": tick} for f in "ABCD"
        ]
        agent._fv.observe(feed)
    return agent


# ---------------------------------------------------------------------------
# Targets and skew
# ---------------------------------------------------------------------------


def test_required_target_is_the_remaining_horizon():
    agent = ScriptedV2Agent(seed=0)
    obs = obs_with_inventory(tick=0)
    for food in obs["required_foods"]:
        assert agent._target(obs, food) == horizon_need(obs, food) + TARGET_BUFFER
    for food in obs["produces"]:
        assert agent._target(obs, food) == PRODUCED_WORKING_STOCK


def test_required_food_reads_scarce_early_and_ample_late():
    """The whole point of the horizon target.

    A short-horizon target (e.g. two consumption ticks = 4 units) would read a
    20-40 unit opening position as ~10x overstocked and skew the bot into
    selling the food it needs to survive. The horizon target inverts that.
    """
    agent = ScriptedV2Agent(seed=0)

    early = obs_with_inventory(tick=0)
    for food in early["required_foods"]:
        assert early["available_inventory"][food] < agent._target(early, food)

    late = obs_with_inventory(tick=170, inventory={f: 10 for f in "ABCD"})
    for food in late["required_foods"]:
        assert late["available_inventory"][food] > agent._target(late, food)


def test_skew_raises_reservation_when_short_and_lowers_it_when_long():
    obs_short = obs_with_inventory(tick=0, inventory={f: 1 for f in "ABCD"})
    obs_long = obs_with_inventory(tick=0, inventory={f: 90 for f in "ABCD"})
    short_agent = _agent_with_history(obs_short, price=10)
    long_agent = _agent_with_history(obs_long, price=10)

    food = obs_short["required_foods"][0]
    assert short_agent._reservation_price(obs_short, food) > 10
    assert long_agent._reservation_price(obs_long, food) < 10


def test_skew_is_bounded():
    """Normalised deviation keeps the skew from saturating as the horizon shrinks."""
    agent = ScriptedV2Agent(seed=0)
    for tick in (0, 90, 170):
        for units in (0, 5, 50, 200):
            obs = obs_with_inventory(tick=tick, inventory={f: units for f in "ABCD"})
            local = _agent_with_history(obs, price=10)
            for food in "ABCD":
                r = local._reservation_price(obs, food)
                assert 10 * (1 - SKEW_GAIN) - 1e-9 <= r <= 10 * (1 + SKEW_GAIN) + 1e-9


def test_quotes_straddle_the_reservation_price():
    obs = obs_with_inventory(tick=0)
    agent = _agent_with_history(obs, price=12)
    for food in "ABCD":
        bid, ask = agent._quotes(obs, food)
        assert bid < ask


# ---------------------------------------------------------------------------
# Sizing and the survival floor
# ---------------------------------------------------------------------------


def test_ask_size_never_breaches_the_consumption_floor():
    agent = ScriptedV2Agent(seed=0)
    for units in range(0, 12):
        obs = obs_with_inventory(inventory={f: units for f in "ABCD"})
        for food in obs["required_foods"]:
            qty = agent._ask_qty(obs, food)
            assert units - qty >= CONSUME_PER_CYCLE or qty == 0


def test_bid_size_scales_with_near_term_deficit():
    agent = ScriptedV2Agent(seed=0)
    stocked = obs_with_inventory(inventory={f: 40 for f in "ABCD"})
    thin = obs_with_inventory(inventory={f: 1 for f in "ABCD"})
    food = stocked["required_foods"][0]
    assert agent._bid_qty(stocked, food) == 1
    assert agent._bid_qty(thin, food) > 1


# ---------------------------------------------------------------------------
# Crossing — the clearest behavioural break from ScriptedV1, which only posts
# ---------------------------------------------------------------------------


def test_takes_an_ask_priced_below_its_reservation():
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["required_foods"][0]
    seller = next(p for p in env.state.players if p.idx != 0 and food in p.produces)
    post_order(
        env.state,
        player_idx=seller.idx,
        side="ask",
        food_type=food,
        quantity=3,
        price_per_unit=2,
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=20)

    action = agent.act(obs, env.legal_actions(0))
    decoded = decode(action)
    assert isinstance(decoded, TakeAction)
    assert decoded.food_type == food
    assert decoded.direction == "buy"


def test_does_not_take_an_ask_priced_above_its_reservation():
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["required_foods"][0]
    seller = next(p for p in env.state.players if p.idx != 0 and food in p.produces)
    post_order(
        env.state,
        player_idx=seller.idx,
        side="ask",
        food_type=food,
        quantity=3,
        price_per_unit=40,
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=5)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert not isinstance(decoded, TakeAction)


def test_emergency_crosses_rather_than_resting_an_order():
    """`take` reserves nothing; an aggressive post would lock cash instead."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["required_foods"][0]
    env.state.players[0].inventory[food] = 0
    seller = next(p for p in env.state.players if p.idx != 0 and food in p.produces)
    post_order(
        env.state,
        player_idx=seller.idx,
        side="ask",
        food_type=food,
        quantity=2,
        price_per_unit=9,
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=5)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, TakeAction)
    assert decoded.food_type == food


# ---------------------------------------------------------------------------
# Whole-game behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_class", ENV_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("cadence", CADENCES)
def test_self_play_is_legal_and_never_self_starves(cadence, env_class):
    """`play_game` asserts action legality and the reserved-inventory invariant."""
    for seed in range(3):
        play_game(ScriptedV2Agent, seed=seed, cadence=cadence, env_class=env_class)


def test_quotes_both_sides_of_a_required_food():
    """A maker shows both sides; ScriptedV1 only ever quotes one.

    Both sides have to be observed *resting on the book*, not merely posted —
    an order that crosses on arrival never provided any liquidity. Six quote
    slots share one action per step and the crossing layer outranks requoting
    when there is edge available, so reaching every slot takes a while.
    """
    env = FoodTradingEnv(seed=0)
    obs = env.reset()
    agents = [ScriptedV2Agent(seed=i) for i in range(4)]
    rested = set()
    for _ in range(60):
        for order in obs[0]["own_orders"]:
            if order["food"] in obs[0]["required_foods"]:
                rested.add(order["side"])
        actions = {i: agents[i].act(obs[i], env.legal_actions(i)) for i in range(4)}
        obs, _, done, _ = env.step(actions)
        if done:
            break
    assert rested == {"bid", "ask"}, f"expected two-sided quoting, saw {rested}"


def test_uses_action_types_scripted_v1_never_emits():
    result = play_game(ScriptedV2Agent, seed=0, cadence="fast")
    kinds = {type(decode(a)).__name__ for step in result["action_log"] for a in step}
    assert "TakeAction" in kinds
    assert "PostAction" in kinds


def test_self_play_is_deterministic():
    first = play_game(ScriptedV2Agent, seed=5, cadence="medium")
    second = play_game(ScriptedV2Agent, seed=5, cadence="medium")
    assert first["action_log"] == second["action_log"]


@pytest.mark.parametrize("cadence", CADENCES)
def test_self_play_circulates_goods_and_sustains_survivors(cadence):
    """The core claim of the design: four identical bots still trade.

    Producers hold surplus and mark down, consumers run short and mark up, so
    the reservation ranges overlap and goods move even in pure self-play.
    """
    results = [play_game(ScriptedV2Agent, seed=s, cadence=cadence) for s in range(5)]
    assert all(r["n_trades"] > 0 for r in results)
    mean_survivors = sum(r["survivors"] for r in results) / len(results)
    assert mean_survivors >= 2.5, f"{cadence}: mean survivors {mean_survivors}"


def test_targets_track_the_short_horizon_env():
    """Nothing is hard-coded to the 180s game — the horizon comes from the obs.

    In the 30-second variant the whole game is three consumption cycles, so the
    opening target is 3 x 2 + buffer rather than 36 + buffer.
    """
    agent = ScriptedV2Agent(seed=0)
    obs = obs_with_inventory(tick=0, env_class=SmallFoodTradingEnv)
    assert obs["remaining_seconds"] == SMALL_GAME_DURATION_SECONDS
    for food in obs["required_foods"]:
        assert agent._target(obs, food) == 3 * CONSUME_PER_CYCLE + TARGET_BUFFER
        # Still scarce: the small dealer hands out 3-5 units of each required food.
        assert obs["available_inventory"][food] < agent._target(obs, food)


def test_small_env_self_play_survives():
    """Four V2s must still clear the tighter market at the evaluation cadence."""
    results = [
        play_game(ScriptedV2Agent, seed=s, cadence="fast", env_class=SmallFoodTradingEnv)
        for s in range(10)
    ]
    assert all(r["n_trades"] > 0 for r in results)
    mean_survivors = sum(r["survivors"] for r in results) / len(results)
    assert mean_survivors >= 3.0, f"small-env mean survivors {mean_survivors}"


def test_price_attempt_telemetry_matches_the_v1_contract():
    """`eval/game_runner.py` reads `_inner.attempted_prices` off the wrapper."""
    agent = ScriptedV2Agent(seed=0)
    env = FoodTradingEnv(seed=0)
    obs = env.reset()
    for _ in range(10):
        actions = {0: agent.act(obs[0], env.legal_actions(0))}
        obs, _, _, _ = env.step(actions)
    assert agent.attempted_prices
    record = agent.attempted_prices[0]
    for key in ("tick", "player_idx", "food", "side", "attempted_price", "submitted"):
        assert key in record


def test_reset_clears_all_state():
    agent = ScriptedV2Agent(seed=3)
    env = FoodTradingEnv(seed=0)
    obs = env.reset()
    for _ in range(10):
        obs, _, _, _ = env.step({0: agent.act(obs[0], env.legal_actions(0))})
    assert agent.attempted_prices
    agent.reset()
    assert agent.attempted_prices == []
    assert agent._fv.value(obs[0], "A") == pytest.approx(
        ScriptedV2Agent(seed=3)._fv.value(obs[0], "A")
    )
