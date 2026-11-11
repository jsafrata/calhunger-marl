"""Cycle arithmetic and valuation shared by ScriptedV2 / ScriptedV3.

Everything both bots do is scheduled off `steps_to_next_consumption` and
`cycles_left`, so those are pinned against the real tick engine rather than
against a re-derivation of the same formula.
"""
from __future__ import annotations

import pytest

from agents.scripted_common import (
    ASK_COVER_CYCLES,
    CONSUME_PER_CYCLE,
    DecisionRateEstimator,
    FairValueTracker,
    clamp_price,
    clamp_qty,
    cycle_progress,
    cycles_left,
    deficit_for_cycles,
    horizon_need,
    safe_ask_qty,
    steps_to_next_consumption,
)
from env.actions import MAX_PRICE, MAX_QTY, encode_post
from env.config import DEFAULT_CONFIG
from env.env import FoodTradingEnv
from env.market import post_order

from tests.scripted_harness import obs_with_inventory


def test_steps_to_next_consumption_matches_tick_engine():
    """The countdown must hit exactly 1 on the step that actually consumes.

    `step_tick` increments the clock and *then* consumes, while the agent acts
    before the clock advances — an off-by-one here would have both bots
    conceding one step too late.
    """
    env = FoodTradingEnv(seed=1)
    obs = env.reset()
    interval = DEFAULT_CONFIG.consumption_interval_seconds

    for _ in range(3 * interval):
        predicted = steps_to_next_consumption(obs[0])
        before = dict(env.state.players[0].inventory)
        obs, _, _, _ = env.step({i: 0 for i in range(4)})
        consumed = env.state.elapsed_seconds % interval == 0
        assert (predicted == 1) == consumed, (
            f"predicted {predicted} steps to consumption but consumption "
            f"{'did' if consumed else 'did not'} fire at t={env.state.elapsed_seconds}"
        )
        if consumed:
            required = DEFAULT_CONFIG.required_foods(env.state.players[0].produces)
            for food in required:
                assert env.state.players[0].inventory[food] == before[food] - CONSUME_PER_CYCLE


def test_cycles_left_counts_the_final_consumption():
    """The tick at `game_duration_seconds` consumes before the game ends."""
    obs = obs_with_inventory(tick=0)
    interval = obs["consumption_interval_seconds"]
    assert cycles_left(obs) == DEFAULT_CONFIG.game_duration_seconds // interval

    # Mid-cycle must not round the pending consumption away.
    assert cycles_left(obs_with_inventory(tick=9)) == cycles_left(obs)
    assert cycles_left(obs_with_inventory(tick=10)) == cycles_left(obs) - 1
    assert cycles_left(obs_with_inventory(tick=179)) == 1


def test_cycle_progress_spans_the_cycle():
    assert cycle_progress(obs_with_inventory(tick=0)) == 0.0
    assert cycle_progress(obs_with_inventory(tick=5)) == pytest.approx(0.5)
    assert cycle_progress(obs_with_inventory(tick=9)) == pytest.approx(0.9)


def test_horizon_need_exceeds_starting_inventory():
    """The structural shortage that drives ScriptedV2's target choice.

    Each player needs `2 x 18 = 36` units of each required food over a full
    game but opens with 100 units spread across all four foods. Required food
    is scarce from t=0 — a short-horizon target would misread it as surplus.
    """
    obs = obs_with_inventory(tick=0)
    for food in obs["required_foods"]:
        assert horizon_need(obs, food) == 36
        assert obs["available_inventory"][food] < horizon_need(obs, food)
    for food in obs["produces"]:
        assert horizon_need(obs, food) == 0


def test_safe_ask_qty_reserves_the_consumption_floor():
    floor = ASK_COVER_CYCLES * CONSUME_PER_CYCLE
    obs = obs_with_inventory(inventory={f: 10 for f in "ABCD"})
    for food in obs["required_foods"]:
        assert safe_ask_qty(obs, food) == min(MAX_QTY, 10 - floor)
    for food in obs["produces"]:
        assert safe_ask_qty(obs, food) == min(MAX_QTY, 10)


def test_safe_ask_qty_is_zero_at_the_floor():
    floor = ASK_COVER_CYCLES * CONSUME_PER_CYCLE
    obs = obs_with_inventory(inventory={f: floor for f in "ABCD"})
    for food in obs["required_foods"]:
        assert safe_ask_qty(obs, food) == 0


def test_resting_ask_floor_survives_a_consumption_tick():
    """The floor exists because reservations outlive the tick that eats the stock.

    Post an ask sized against the floor, run a consumption tick, and the
    remaining inventory must still cover the *next* one.
    """
    env = FoodTradingEnv(seed=0)
    env.reset()
    player = env.state.players[0]
    food = DEFAULT_CONFIG.required_foods(player.produces)[0]
    player.inventory[food] = ASK_COVER_CYCLES * CONSUME_PER_CYCLE + 1

    qty = safe_ask_qty(env.observe(0), food)
    assert qty >= 1
    post_order(
        env.state, player_idx=0, side="ask", food_type=food,
        quantity=qty, price_per_unit=MAX_PRICE,
    )
    # Advance to the consumption tick with the ask still resting, unfilled.
    while env.state.elapsed_seconds % DEFAULT_CONFIG.consumption_interval_seconds != 0 \
            or env.state.elapsed_seconds == 0:
        env.step({i: 0 for i in range(4)})

    assert player.status == "alive"
    available = player.inventory[food] - player.reserved_inventory[food]
    assert available >= CONSUME_PER_CYCLE, (
        "the resting ask ate the next cycle's food"
    )


def test_safe_ask_qty_accounts_for_already_reserved_inventory():
    """A resting ask has already locked its units; they must not be offered twice."""
    env = FoodTradingEnv(seed=0)
    env.reset()
    player = env.state.players[0]
    food = DEFAULT_CONFIG.required_foods(player.produces)[0]
    player.inventory[food] = 10
    post_order(
        env.state, player_idx=0, side="ask", food_type=food, quantity=4, price_per_unit=MAX_PRICE
    )
    obs = env.observe(0)
    assert player.reserved_inventory[food] == 4
    assert safe_ask_qty(obs, food) == 10 - 4 - ASK_COVER_CYCLES * CONSUME_PER_CYCLE


def test_deficit_for_cycles():
    obs = obs_with_inventory(inventory={f: 3 for f in "ABCD"})
    for food in obs["required_foods"]:
        assert deficit_for_cycles(obs, food, 1) == 0          # 3 >= 2
        assert deficit_for_cycles(obs, food, 2) == 1          # need 4, hold 3
        assert deficit_for_cycles(obs, food, 3) == 3          # need 6, hold 3
    for food in obs["produces"]:
        assert deficit_for_cycles(obs, food, 3) == 0


def test_fair_value_is_stable_under_an_outlier_trade():
    """The EMA must not chase a single print — both bots quote off it."""
    fv = FairValueTracker(alpha=0.3)
    obs = obs_with_inventory()
    food = "A"
    for tick in range(1, 11):
        obs["recent_trades"] = [{"food": food, "price": 5, "qty": 1, "tick": tick}]
        fv.observe(obs)
    settled = fv.value(obs, food)
    assert settled == pytest.approx(5.0, abs=0.2)

    obs["recent_trades"] = [{"food": food, "price": 50, "qty": 1, "tick": 11}]
    fv.observe(obs)
    after = fv.value(obs, food)
    assert after < settled + 0.35 * 50, "EMA moved more than one alpha-step"
    assert after == pytest.approx(0.3 * 50 + 0.7 * settled)


def test_fair_value_ignores_already_seen_trades():
    """`recent_trades` is a rolling window; re-folding it would double-count."""
    fv = FairValueTracker(alpha=0.5)
    obs = obs_with_inventory()
    obs["recent_trades"] = [{"food": "A", "price": 10, "qty": 1, "tick": 1}]
    fv.observe(obs)
    first = fv.value(obs, "A")
    fv.observe(obs)
    fv.observe(obs)
    assert fv.value(obs, "A") == first


def test_fair_value_falls_back_without_history():
    fv = FairValueTracker()
    obs = obs_with_inventory()
    obs["recent_trades"] = []
    for food in "ABCD":
        obs["order_books"][food]["bid_orders"] = []
        obs["order_books"][food]["ask_orders"] = []
    assert fv.value(obs, "A") == pytest.approx(5.0)


def test_decision_rate_estimator_tracks_gap():
    """Cadence gating means env steps != decisions; the schedule needs the gap."""
    est = DecisionRateEstimator(alpha=0.5)
    obs = obs_with_inventory()
    for tick in range(0, 20, 4):  # fires every 4th step, like "slow"
        obs["tick"] = tick
        est.observe(obs)
    assert est.gap == pytest.approx(4.0)
    assert est.decisions_within(10) == 2

    fast = DecisionRateEstimator(alpha=0.5)
    for tick in range(0, 10):
        obs["tick"] = tick
        fast.observe(obs)
    assert fast.gap == pytest.approx(1.0)
    assert fast.decisions_within(10) == 10


def test_clamps_keep_actions_encodable():
    """`encode_post` raises outside 1..MAX; the clamps are what stop that."""
    assert clamp_price(-5) == 1
    assert clamp_price(10_000) == MAX_PRICE
    assert clamp_qty(0) == 1
    assert clamp_qty(999) == MAX_QTY
    encode_post("bid", "A", clamp_qty(999), clamp_price(10_000))
