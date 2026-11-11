"""ScriptedV3 — deadline-aware procurement trader."""
from __future__ import annotations

import pytest

from agents.scripted_common import CONSUME_PER_CYCLE
from agents.scripted_v3 import (
    COVERAGE_CYCLES,
    MAX_COVERAGE_CYCLES,
    ScriptedV3Agent,
    concession_stage,
)
from env.actions import CancelAction, PostAction, TakeAction, decode
from env.env import FoodTradingEnv
from env.market import post_order

from env.small_env import SmallFoodTradingEnv

from tests.scripted_harness import CADENCES, ENV_CLASSES, obs_with_inventory, play_game


def _agent_with_history(obs, price: int, seed: int = 0) -> ScriptedV3Agent:
    agent = ScriptedV3Agent(seed=seed)
    feed = dict(obs)
    for tick in range(1, 15):
        feed["recent_trades"] = [
            {"food": f, "price": price, "qty": 1, "tick": tick} for f in "ABCD"
        ]
        agent._fv.observe(feed)
    return agent


# ---------------------------------------------------------------------------
# Concession schedule
# ---------------------------------------------------------------------------


def test_concession_stages_partition_the_cycle():
    assert concession_stage(0.0) == "early"
    assert concession_stage(0.39) == "early"
    assert concession_stage(0.4) == "mid"
    assert concession_stage(0.69) == "mid"
    assert concession_stage(0.7) == "late"
    assert concession_stage(0.89) == "late"
    assert concession_stage(0.9) == "final"
    assert concession_stage(1.0) == "final"


def test_buyers_concede_upward_and_sellers_downward():
    """The whole archetype: reservation prices converge as the deadline nears.

    Buyers must raise and sellers must lower monotonically, or the acceptable
    ranges never overlap and four identical bots deadlock.
    """
    obs = obs_with_inventory(tick=0)
    agent = _agent_with_history(obs, price=10)
    food = obs["required_foods"][0]

    bids = [agent._bid_price(obs, food, s) for s in ("early", "mid", "late", "final")]
    asks = [agent._ask_price(obs, food, s) for s in ("early", "mid", "late", "final")]

    assert bids == sorted(bids), f"bid path must be non-decreasing, got {bids}"
    assert asks == sorted(asks, reverse=True), f"ask path must be non-increasing, got {asks}"
    assert bids[0] < 10 < asks[0], "early stage should straddle fair value"
    assert bids[-1] > asks[-1], "final stage must cross"


# ---------------------------------------------------------------------------
# Cycle planning
# ---------------------------------------------------------------------------


def test_plan_is_recomputed_once_per_cycle():
    agent = ScriptedV3Agent(seed=0)
    first = agent._ensure_plan(obs_with_inventory(tick=0))
    assert agent._ensure_plan(obs_with_inventory(tick=5)) is first
    assert agent._ensure_plan(obs_with_inventory(tick=9)) is first
    second = agent._ensure_plan(obs_with_inventory(tick=10))
    assert second is not first
    assert second.cycle == first.cycle + 1


def test_plan_buys_the_shortfall_and_sells_the_surplus():
    agent = ScriptedV3Agent(seed=0)
    obs = obs_with_inventory(inventory={f: 1 for f in "ABCD"})
    plan = agent._ensure_plan(obs)
    for food in obs["required_foods"]:
        assert plan.buy[food] == COVERAGE_CYCLES * CONSUME_PER_CYCLE - 1
    for food in obs["produces"]:
        assert food not in plan.sell  # 1 unit is below the working stock


def test_stocked_bot_plans_no_purchases():
    """Just-in-time means buying nothing while cover is ample."""
    agent = ScriptedV3Agent(seed=0)
    obs = obs_with_inventory(inventory={f: 40 for f in "ABCD"})
    plan = agent._ensure_plan(obs)
    assert plan.buy == {}
    assert all(plan.sell[f] > 0 for f in obs["produces"])


def test_coverage_stretches_when_decisions_are_scarce():
    """Safety stock rises as order frequency falls — standard inventory policy.

    A bot that only acts twice per cycle cannot service two required foods and
    its own liquidation on a two-cycle plan, so it has to buy further ahead.
    """
    obs = obs_with_inventory(tick=0)

    frequent = ScriptedV3Agent(seed=0)
    for tick in range(0, 12):
        frequent._rate.observe({**obs, "tick": tick})
    assert frequent._coverage_cycles(obs) == COVERAGE_CYCLES

    infrequent = ScriptedV3Agent(seed=0)
    for tick in range(0, 40, 5):
        infrequent._rate.observe({**obs, "tick": tick})
    assert infrequent._coverage_cycles(obs) > COVERAGE_CYCLES
    assert infrequent._coverage_cycles(obs) <= MAX_COVERAGE_CYCLES


def test_escalation_counts_decisions_not_env_steps():
    """The bug this rule exists to prevent.

    A bot firing every 5th step that waits until `steps_remaining == 2` has
    already missed its last turn. Two bots with identical inventory and clock
    must disagree about urgency purely because of their firing rate.
    """
    obs = obs_with_inventory(tick=5, inventory={f: 1 for f in "ABCD"})

    fast = ScriptedV3Agent(seed=0)
    for tick in range(0, 6):
        fast._rate.observe({**obs, "tick": tick})
    slow = ScriptedV3Agent(seed=0)
    for tick in range(0, 30, 5):
        slow._rate.observe({**obs, "tick": tick})

    plan_fast = fast._ensure_plan(obs)
    plan_slow = slow._ensure_plan(obs)
    assert not fast._escalate(obs, plan_fast), "5 steps of turns left is not urgent"
    assert slow._escalate(obs, plan_slow), "1 turn left must be urgent"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_batches_larger_than_one_unit():
    """ScriptedV1 always trades exactly 1 unit; V3 orders the planned deficit."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    for food in obs["required_foods"]:
        env.state.players[0].inventory[food] = 0
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=3)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, (PostAction, TakeAction))
    assert decoded.quantity > 1


def test_final_stage_crosses_a_takeable_ask():
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    food = obs["required_foods"][0]
    env.state.players[0].inventory[food] = 0
    seller = next(p for p in env.state.players if p.idx != 0 and food in p.produces)
    post_order(
        env.state, player_idx=seller.idx, side="ask", food_type=food,
        quantity=4, price_per_unit=6,
    )
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=5)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, TakeAction)
    assert decoded.food_type == food


def test_early_stage_rests_a_patient_bid_below_fair_value():
    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    # Short of the coverage plan but not starving: 3 units survives the next
    # tick, so this exercises the concession schedule rather than the panic path.
    for food in obs["required_foods"]:
        env.state.players[0].inventory[food] = CONSUME_PER_CYCLE + 1
    obs = env.observe(0)
    agent = _agent_with_history(obs, price=10)

    decoded = decode(agent.act(obs, env.legal_actions(0)))
    assert isinstance(decoded, PostAction)
    assert decoded.side == "bid"
    assert decoded.price_per_unit < 10


def test_cancels_a_bid_that_is_starving_the_survival_budget():
    """A resting bid locks `price * qty` of cash the moment it is posted."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    player = env.state.players[0]
    obs = env.observe(0)
    hungry, spare = obs["required_foods"][0], obs["produces"][0]
    player.inventory[hungry] = 1          # a real, unfunded shortfall next cycle
    player.cash = 30
    post_order(
        env.state, player_idx=0, side="bid", food_type=spare,
        quantity=5, price_per_unit=6,     # locks 30 -> available_cash 0
    )
    obs = env.observe(0)
    assert obs["available_cash"] == 0

    agent = _agent_with_history(obs, price=8)
    action = agent._free_cash_action(obs, env.legal_actions(0))
    assert action is not None
    decoded = decode(action)
    assert isinstance(decoded, CancelAction)
    assert decoded.side == "bid" and decoded.food_type == spare


# ---------------------------------------------------------------------------
# Whole-game behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_class", ENV_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("cadence", CADENCES)
def test_self_play_is_legal_and_never_self_starves(cadence, env_class):
    for seed in range(3):
        play_game(ScriptedV3Agent, seed=seed, cadence=cadence, env_class=env_class)


def test_planning_window_is_clamped_to_the_remaining_game():
    """Never procure food the game has no time to consume.

    The plan looks `COVERAGE_CYCLES` ahead, but each unit ordered costs cash and
    locks `price * qty` in `reserved_cash` while the bid rests. In the 30-second
    variant the entire game is three cycles, so an unclamped window would order
    up to five cycles of food and strand the budget the final tick needs.
    """
    agent = ScriptedV3Agent(seed=0)
    late = obs_with_inventory(
        tick=20, inventory={f: 0 for f in "ABCD"}, env_class=SmallFoodTradingEnv
    )
    plan = agent._ensure_plan(late)
    for food in late["required_foods"]:
        # One cycle left, holding nothing: exactly one cycle of food, no more.
        assert plan.buy[food] == CONSUME_PER_CYCLE


def test_small_env_self_play_survives():
    results = [
        play_game(ScriptedV3Agent, seed=s, cadence="fast", env_class=SmallFoodTradingEnv)
        for s in range(10)
    ]
    assert all(r["n_trades"] > 0 for r in results)
    mean_survivors = sum(r["survivors"] for r in results) / len(results)
    assert mean_survivors >= 3.0, f"small-env mean survivors {mean_survivors}"


def test_uses_action_types_scripted_v1_never_emits():
    kinds = set()
    for seed in range(4):
        result = play_game(ScriptedV3Agent, seed=seed, cadence="fast")
        kinds |= {type(decode(a)).__name__ for step in result["action_log"] for a in step}
    assert "TakeAction" in kinds
    assert "PostAction" in kinds


def test_self_play_is_deterministic():
    first = play_game(ScriptedV3Agent, seed=5, cadence="medium")
    second = play_game(ScriptedV3Agent, seed=5, cadence="medium")
    assert first["action_log"] == second["action_log"]


@pytest.mark.parametrize("cadence", CADENCES)
def test_self_play_circulates_goods_and_sustains_survivors(cadence):
    """Producers offer, consumers concede, and the ranges overlap before the tick."""
    results = [play_game(ScriptedV3Agent, seed=s, cadence=cadence) for s in range(5)]
    assert all(r["n_trades"] > 0 for r in results)
    mean_survivors = sum(r["survivors"] for r in results) / len(results)
    assert mean_survivors >= 2.5, f"{cadence}: mean survivors {mean_survivors}"


def test_price_attempt_telemetry_matches_the_v1_contract():
    agent = ScriptedV3Agent(seed=0)
    env = FoodTradingEnv(seed=0)
    obs = env.reset()
    for _ in range(40):
        obs, _, _, _ = env.step({0: agent.act(obs[0], env.legal_actions(0))})
    assert agent.attempted_prices
    for key in ("tick", "player_idx", "food", "side", "attempted_price", "submitted"):
        assert key in agent.attempted_prices[0]
