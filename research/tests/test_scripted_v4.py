"""ScriptedV4 — precautionary stockpiler / storage arbitrageur.

Fast cadence only: the eval harnesses run every scripted baseline through
`CadencedScripted("fast", ...)`, and V4's calibration targets that regime.
"""
from __future__ import annotations

import pytest

from agents.scripted_common import CONSUME_PER_CYCLE
from agents.scripted_v4 import (
    LOW_WATER_CYCLES,
    STOCK_BUFFER,
    STOCK_CYCLES,
    ScriptedV4Agent,
)
from env.actions import PostAction, TakeAction, decode
from env.env import FoodTradingEnv
from env.market import post_order
from env.small_env import FoodTradingEnv90, SmallFoodTradingEnv

from tests.scripted_harness import ENV_CLASSES, obs_with_inventory, play_game

FORMAT_ENVS = (SmallFoodTradingEnv, FoodTradingEnv90, FoodTradingEnv)


def _agent_with_history(obs, price: int, seed: int = 0) -> ScriptedV4Agent:
    agent = ScriptedV4Agent(seed=seed)
    feed = dict(obs)
    for tick in range(1, 15):
        feed["recent_trades"] = [
            {"food": f, "price": price, "qty": 1, "tick": tick} for f in "ABCD"
        ]
        agent._fv.observe(feed)
    return agent


# ---------------------------------------------------------------------------
# Stock accounting — the intertemporal shape of the archetype
# ---------------------------------------------------------------------------


def test_buy_target_is_capped_but_sale_floor_is_not():
    """Buy toward a bounded pile; sell only above the full horizon need.

    The asymmetry IS the storage trade: were sales floored at the (small)
    buy target, a bot endowed with more than the target would dump food in
    the opening at the cheapest prices of the game and re-buy it later at
    scarcity prices.
    """
    agent = ScriptedV4Agent(seed=0)
    obs = obs_with_inventory(tick=0, inventory={f: 30 for f in "ABCD"})
    food = obs["required_foods"][0]

    cap = STOCK_CYCLES * CONSUME_PER_CYCLE + STOCK_BUFFER
    assert agent._target_stock(obs, food) == cap
    # 180s game at t=0: horizon need is 36, so 30 held units are NOT surplus.
    assert agent._surplus(obs, food) < 0


def test_pile_becomes_sellable_as_the_horizon_shrinks():
    """The same holding flips from hoard to merchandise on schedule."""
    agent = ScriptedV4Agent(seed=0)
    early = obs_with_inventory(tick=0, inventory={f: 20 for f in "ABCD"})
    late = obs_with_inventory(tick=160, inventory={f: 20 for f in "ABCD"})
    food = early["required_foods"][0]
    assert agent._surplus(early, food) <= 0
    assert agent._surplus(late, food) > 0


def test_ask_premium_rises_with_game_progress():
    obs_early = obs_with_inventory(tick=0)
    obs_late = obs_with_inventory(tick=170)
    agent = _agent_with_history(obs_early, price=10)
    food = obs_early["produces"][0]
    assert agent._ask_price(obs_late, food) > agent._ask_price(obs_early, food)


def test_ask_matches_a_standing_competitor_instead_of_ratcheting():
    """Quoting formula-above-FV into a market that trades at your price
    ratchets the fair-value EMA — and your own next quote — upward forever."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["produces"][0]
    competitor = next(p for p in env.state.players if p.idx != 0)
    post_order(
        env.state, player_idx=competitor.idx, side="ask", food_type=food,
        quantity=2, price_per_unit=11,
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)
    # Formula price would be ~11.5+; the standing 11 is at/above FV, so match.
    assert agent._ask_price(obs, food) == 11


# ---------------------------------------------------------------------------
# Restocking
# ---------------------------------------------------------------------------


def test_below_low_water_takes_an_acceptable_ask():
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["required_foods"][0]
    env.state.players[0].inventory[food] = LOW_WATER_CYCLES * CONSUME_PER_CYCLE - 1
    seller = next(p for p in env.state.players if p.idx != 0 and food in p.produces)
    post_order(
        env.state, player_idx=seller.idx, side="ask", food_type=food,
        quantity=5, price_per_unit=12,
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, TakeAction)
    assert decoded.side == "bid" and decoded.food_type == food
    assert decoded.quantity > 1


def test_building_the_pile_lifts_cheap_asks_only():
    """Above low water the bot still buys toward the pile, but not at a premium."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["required_foods"][0]
    env.state.players[0].inventory[food] = LOW_WATER_CYCLES * CONSUME_PER_CYCLE + 1
    seller = next(p for p in env.state.players if p.idx != 0 and food in p.produces)
    post_order(
        env.state, player_idx=seller.idx, side="ask", food_type=food,
        quantity=5, price_per_unit=9,  # below FV*BUILD_ACCEPT = 12
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, TakeAction)
    assert decoded.side == "bid" and decoded.food_type == food


def test_rich_bid_gets_hit_when_surplus_exists():
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["produces"][0]
    env.state.players[0].inventory[food] = 20
    buyer = next(p for p in env.state.players if p.idx != 0)
    post_order(
        env.state, player_idx=buyer.idx, side="bid", food_type=food,
        quantity=3, price_per_unit=14,  # >= FV(10) * RICH_BID_EDGE
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, TakeAction)
    assert decoded.side == "ask" and decoded.food_type == food


def test_starving_with_bait_resting_falls_through_to_selling():
    """The all-V4 deadlock guard: several players can starve at once, and
    each is the only seller of what the others need."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    hungry = obs["required_foods"][0]
    produced = obs["produces"][0]
    env.state.players[0].inventory[hungry] = 0
    env.state.players[0].inventory[produced] = 20
    # Own panic bid already resting at a high price.
    post_order(
        env.state, player_idx=0, side="bid", food_type=hungry,
        quantity=2, price_per_unit=30,
    )
    # Someone else's rich bid on our produced food.
    buyer = next(p for p in env.state.players if p.idx != 0)
    post_order(
        env.state, player_idx=buyer.idx, side="bid", food_type=produced,
        quantity=3, price_per_unit=14,
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, TakeAction)
    assert decoded.side == "ask" and decoded.food_type == produced


# ---------------------------------------------------------------------------
# Whole-game behaviour (fast cadence — the eval harness regime)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_class", ENV_CLASSES, ids=lambda c: c.__name__)
def test_self_play_is_legal_and_never_self_starves(env_class):
    for seed in range(3):
        play_game(ScriptedV4Agent, seed=seed, cadence="fast", env_class=env_class)


@pytest.mark.parametrize("env_class", FORMAT_ENVS, ids=lambda c: c.__name__)
def test_self_play_survives_in_every_format(env_class):
    results = [
        play_game(ScriptedV4Agent, seed=s, cadence="fast", env_class=env_class)
        for s in range(10)
    ]
    assert all(r["n_trades"] > 0 for r in results)
    mean_survivors = sum(r["survivors"] for r in results) / len(results)
    assert mean_survivors >= 3.8, f"{env_class.__name__}: {mean_survivors}"


def test_uses_action_types_scripted_v1_never_emits():
    kinds = set()
    for seed in range(4):
        result = play_game(ScriptedV4Agent, seed=seed, cadence="fast")
        kinds |= {type(decode(a)).__name__ for step in result["action_log"] for a in step}
    assert "TakeAction" in kinds
    assert "PostAction" in kinds


def test_self_play_is_deterministic():
    first = play_game(ScriptedV4Agent, seed=5, cadence="fast")
    second = play_game(ScriptedV4Agent, seed=5, cadence="fast")
    assert first["action_log"] == second["action_log"]


def test_price_attempt_telemetry_matches_the_v1_contract():
    agent = ScriptedV4Agent(seed=0)
    env = FoodTradingEnv(seed=0)
    obs = env.reset()
    for _ in range(60):
        obs, _, _, _ = env.step({0: agent.act(obs[0], env.legal_actions(0))})
    assert agent.attempted_prices
    for key in ("tick", "player_idx", "food", "side", "attempted_price", "submitted"):
        assert key in agent.attempted_prices[0]
