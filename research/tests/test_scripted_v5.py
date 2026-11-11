"""ScriptedV5 — opportunistic sniper / edge-only scalper.

Fast cadence only: the eval harnesses run every scripted baseline through
`CadencedScripted("fast", ...)`, and V5's calibration targets that regime.
"""
from __future__ import annotations

import pytest

from agents.scripted_common import CONSUME_PER_CYCLE, FALLBACK_VALUE
from agents.scripted_v5 import KEEP_CYCLES, ScriptedV5Agent
from env.actions import ACTION_WAIT, PostAction, TakeAction, decode
from env.env import FoodTradingEnv
from env.market import post_order
from env.small_env import FoodTradingEnv90, SmallFoodTradingEnv

from tests.scripted_harness import ENV_CLASSES, obs_with_inventory, play_game

FORMAT_ENVS = (SmallFoodTradingEnv, FoodTradingEnv90, FoodTradingEnv)


def _agent_with_history(obs, price: int, seed: int = 0) -> ScriptedV5Agent:
    agent = ScriptedV5Agent(seed=seed)
    feed = dict(obs)
    for tick in range(1, 15):
        feed["recent_trades"] = [
            {"food": f, "price": price, "qty": 1, "tick": tick} for f in "ABCD"
        ]
        agent._fv.observe(feed)
    return agent


# ---------------------------------------------------------------------------
# Valuation — mark to prints, never to quotes
# ---------------------------------------------------------------------------


def test_anchor_ignores_a_one_sided_book():
    """In a bid-only book the mid IS the quote under judgement; an edge test
    against it is a tautology and deadlocks an all-sniper table."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["required_foods"][0]
    other = next(p for p in env.state.players if p.idx != 0)
    post_order(
        env.state, player_idx=other.idx, side="bid", food_type=food,
        quantity=1, price_per_unit=30,
    )
    obs = env.observe(0)
    agent = ScriptedV5Agent(seed=0)
    assert agent._value(obs, food) == FALLBACK_VALUE

    traded = _agent_with_history(obs, price=8)
    assert traded._value(obs, food) == 8


# ---------------------------------------------------------------------------
# The sniper's defining trait: it waits
# ---------------------------------------------------------------------------


def test_stocked_sniper_with_a_quiet_book_does_nothing():
    """Every other baseline would be posting or requoting here."""
    agent = ScriptedV5Agent(seed=0)
    env = FoodTradingEnv(seed=0)
    env.reset()
    for f in "ABCD":
        env.state.players[0].inventory[f] = 40
    obs = env.observe(0)
    assert agent.act(obs, env.legal_actions(0)) == ACTION_WAIT


def test_lifts_an_ask_resting_below_value():
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["required_foods"][0]
    env.state.players[0].inventory[food] = CONSUME_PER_CYCLE + 1  # room to keep-level
    seller = next(p for p in env.state.players if p.idx != 0 and food in p.produces)
    post_order(
        env.state, player_idx=seller.idx, side="ask", food_type=food,
        quantity=3, price_per_unit=7,  # anchor 10 -> threshold 9.4
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, TakeAction)
    assert decoded.side == "bid" and decoded.food_type == food


def test_hits_a_bid_resting_above_value():
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["produces"][0]
    env.state.players[0].inventory[food] = 20
    buyer = next(p for p in env.state.players if p.idx != 0)
    post_order(
        env.state, player_idx=buyer.idx, side="bid", food_type=food,
        quantity=3, price_per_unit=12,  # anchor 10 -> threshold 11
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, TakeAction)
    assert decoded.side == "ask" and decoded.food_type == food


def test_ignores_a_bid_without_edge():
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["produces"][0]
    env.state.players[0].inventory[food] = 20
    buyer = next(p for p in env.state.players if p.idx != 0)
    post_order(
        env.state, player_idx=buyer.idx, side="bid", food_type=food,
        quantity=3, price_per_unit=10,  # exactly at anchor: no edge
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)
    assert agent.act(obs, env.legal_actions(0)) == ACTION_WAIT


def test_required_food_below_keep_level_is_never_sold():
    agent = ScriptedV5Agent(seed=0)
    obs = obs_with_inventory(tick=0, inventory={f: 6 for f in "ABCD"})
    food = obs["required_foods"][0]
    assert agent._keep_level(obs, food) == KEEP_CYCLES * CONSUME_PER_CYCLE
    assert agent._sellable(obs, food) < 0


# ---------------------------------------------------------------------------
# Survival: deadline escalation and the deadlock guard
# ---------------------------------------------------------------------------


def test_escalation_posts_bait_when_the_deadline_is_close():
    """Short on cover, empty book, tick late in the cycle: the burst fires."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    env.state.elapsed_seconds = 8  # 2 steps to the consumption tick
    for food in env.observe(0)["required_foods"]:
        env.state.players[0].inventory[food] = CONSUME_PER_CYCLE  # survives 1 tick
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, PostAction)
    assert decoded.side == "bid"
    assert decoded.food_type in obs["required_foods"]
    assert decoded.price_per_unit > 10  # bait must be rich enough to pull sellers


def test_starving_with_bait_resting_falls_through_to_selling():
    """The all-sniper deadlock guard: a bot locked into re-posting its own
    panic bid never reaches the sale layer, and on a table where everyone is
    short of something, nobody would ever sell."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    hungry = obs["required_foods"][0]
    produced = obs["produces"][0]
    env.state.players[0].inventory[hungry] = 0
    env.state.players[0].inventory[produced] = 20
    post_order(
        env.state, player_idx=0, side="bid", food_type=hungry,
        quantity=2, price_per_unit=30,  # own panic bid, already rich
    )
    buyer = next(p for p in env.state.players if p.idx != 0)
    post_order(
        env.state, player_idx=buyer.idx, side="bid", food_type=produced,
        quantity=3, price_per_unit=12,
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, TakeAction)
    assert decoded.side == "ask" and decoded.food_type == produced


def test_bait_is_retired_once_cover_is_restored():
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["required_foods"][0]
    env.state.players[0].inventory[food] = 40  # fully covered
    post_order(
        env.state, player_idx=0, side="bid", food_type=food,
        quantity=2, price_per_unit=20,  # stale bait: cash lock + overbuy invite
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)

    action = agent.act(obs, env.legal_actions(0))
    from env.actions import CancelAction

    decoded = decode(action)
    assert isinstance(decoded, CancelAction)
    assert decoded.side == "bid" and decoded.food_type == food


# ---------------------------------------------------------------------------
# Whole-game behaviour (fast cadence — the eval harness regime)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_class", ENV_CLASSES, ids=lambda c: c.__name__)
def test_self_play_is_legal_and_never_self_starves(env_class):
    for seed in range(3):
        play_game(ScriptedV5Agent, seed=seed, cadence="fast", env_class=env_class)


@pytest.mark.parametrize("env_class", FORMAT_ENVS, ids=lambda c: c.__name__)
def test_self_play_survives_in_every_format(env_class):
    results = [
        play_game(ScriptedV5Agent, seed=s, cadence="fast", env_class=env_class)
        for s in range(10)
    ]
    assert all(r["n_trades"] > 0 for r in results)
    mean_survivors = sum(r["survivors"] for r in results) / len(results)
    assert mean_survivors >= 3.8, f"{env_class.__name__}: {mean_survivors}"


def test_order_footprint_is_survival_only():
    """The behavioural fingerprint that separates V5 from every other bot.

    V1-V4 are all defined by the quotes they rest; V5 rests nothing except
    survival orders. In self-play that means: the modal action is WAIT, every
    sale is a `take` (the ask side of a post never appears outside the
    cash-raise path), and takes are a substantial share of all trading
    actions rather than the garnish they are for V2/V3.
    """
    counts = {"wait": 0, "post_bid": 0, "post_ask": 0, "take": 0, "cancel": 0}
    for seed in range(4):
        result = play_game(ScriptedV5Agent, seed=seed, cadence="fast")
        for step in result["action_log"]:
            for a in step:
                d = decode(a)
                kind = type(d).__name__
                if kind == "WaitAction":
                    counts["wait"] += 1
                elif kind == "PostAction":
                    counts["post_bid" if d.side == "bid" else "post_ask"] += 1
                elif kind == "TakeAction":
                    counts["take"] += 1
                else:
                    counts["cancel"] += 1
    assert counts["wait"] > sum(v for k, v in counts.items() if k != "wait"), counts
    assert counts["post_ask"] == 0, f"quoted an ask outside cash-raise: {counts}"
    assert counts["take"] >= counts["post_bid"] * 0.4, counts


def test_self_play_is_deterministic():
    first = play_game(ScriptedV5Agent, seed=5, cadence="fast")
    second = play_game(ScriptedV5Agent, seed=5, cadence="fast")
    assert first["action_log"] == second["action_log"]


def test_price_attempt_telemetry_matches_the_v1_contract():
    agent = ScriptedV5Agent(seed=0)
    env = FoodTradingEnv(seed=0)
    obs = env.reset()
    # A stocked sniper posts nothing at all — telemetry only exists on the
    # survival paths, so start it short of both required foods.
    for food in obs[0]["required_foods"]:
        env.state.players[0].inventory[food] = CONSUME_PER_CYCLE
    obs = {0: env.observe(0)}
    for _ in range(60):
        obs, _, _, _ = env.step({0: agent.act(obs[0], env.legal_actions(0))})
    assert agent.attempted_prices
    for key in ("tick", "player_idx", "food", "side", "attempted_price", "submitted"):
        assert key in agent.attempted_prices[0]
