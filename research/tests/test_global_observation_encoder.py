from __future__ import annotations

import numpy as np
import pytest

from env.config import DEFAULT_CONFIG, GameConfig
from env.env import FoodTradingEnv
from env.market import cancel_order, post_order
from env.rl.global_observation_encoder import (
    GLOBAL_OBS_DIM,
    ORDER_FEATURES,
    ORDER_STATE_DIM,
    PLAYER_FEATURES,
    PLAYER_STATE_DIM,
    encode_global_observation,
)


def test_global_encoder_shape_dtype_and_contiguity():
    env = FoodTradingEnv(seed=9)
    env.reset()

    encoded = encode_global_observation(
        env.state,
        target_player_idx=2,
        config=DEFAULT_CONFIG,
    )

    assert encoded.shape == (GLOBAL_OBS_DIM,)
    assert GLOBAL_OBS_DIM == 305
    assert encoded.dtype == np.float32
    assert encoded.flags.c_contiguous
    assert np.isfinite(encoded).all()


def test_target_player_rotates_into_first_player_slot():
    env = FoodTradingEnv(seed=4)
    env.reset()
    for idx, player in enumerate(env.state.players):
        player.cash = 20 + idx * 10

    for target_idx in range(4):
        encoded = encode_global_observation(
            env.state,
            target_idx,
            DEFAULT_CONFIG,
        )
        assert encoded[5] == env.state.players[target_idx].cash / 200.0
        next_idx = (target_idx + 1) % 4
        assert (
            encoded[PLAYER_FEATURES + 5]
            == env.state.players[next_idx].cash / 200.0
        )


def test_player_slots_include_total_reserved_and_available_resources():
    env = FoodTradingEnv(seed=3)
    env.reset()
    player = env.state.players[0]
    player.cash = 80
    player.reserved_cash = 30
    player.inventory["A"] = 25
    player.reserved_inventory["A"] = 7

    encoded = encode_global_observation(env.state, 0, DEFAULT_CONFIG)

    np.testing.assert_array_equal(
        encoded[5:8],
        np.asarray([80 / 200, 30 / 200, 50 / 200], dtype=np.float32),
    )
    assert encoded[8] == 25 / 50
    assert encoded[12] == 7 / 50
    assert encoded[16] == 18 / 50


def test_order_slots_encode_create_partial_fill_and_cancel():
    env = FoodTradingEnv(seed=8)
    env.reset()
    state = env.state
    state.elapsed_seconds = 12

    order, _ = post_order(
        state,
        player_idx=0,
        side="ask",
        food_type="A",
        quantity=8,
        price_per_unit=7,
    )
    order.remaining_quantity = 3
    order.status = "partially_filled"

    encoded = encode_global_observation(state, 0, DEFAULT_CONFIG)
    ask_a_start = PLAYER_STATE_DIM + ORDER_FEATURES
    slot = encoded[ask_a_start : ask_a_start + ORDER_FEATURES]
    assert slot[0] == 1.0
    assert slot[1] == 0.7
    assert slot[2] == 8 / 20
    assert slot[3] == 3 / 20
    assert slot[4] == 1.0
    assert slot[5] == 12 / DEFAULT_CONFIG.game_duration_seconds

    cancel_order(state, player_idx=0, order_id=order.id)
    after_cancel = encode_global_observation(state, 0, DEFAULT_CONFIG)
    np.testing.assert_array_equal(
        after_cancel[ask_a_start : ask_a_start + ORDER_FEATURES],
        np.zeros(ORDER_FEATURES, dtype=np.float32),
    )


def test_orders_rotate_with_their_owner():
    env = FoodTradingEnv(seed=6)
    env.reset()
    order, _ = post_order(
        env.state,
        player_idx=2,
        side="bid",
        food_type="B",
        quantity=2,
        price_per_unit=3,
    )
    assert order.remaining_quantity == 2

    target_two = encode_global_observation(env.state, 2, DEFAULT_CONFIG)
    target_zero = encode_global_observation(env.state, 0, DEFAULT_CONFIG)
    bid_b_offset = PLAYER_STATE_DIM + 2 * ORDER_FEATURES
    owner_two_offset = (
        PLAYER_STATE_DIM
        + 2 * 8 * ORDER_FEATURES
        + 2 * ORDER_FEATURES
    )

    assert target_two[bid_b_offset] == 1.0
    assert target_zero[owner_two_offset] == 1.0


def test_final_dimension_is_elapsed_time_only():
    env = FoodTradingEnv(seed=2)
    env.reset()
    env.state.elapsed_seconds = 45

    encoded = encode_global_observation(env.state, 0, DEFAULT_CONFIG)

    assert PLAYER_STATE_DIM == 80
    assert ORDER_STATE_DIM == 224
    assert encoded[-1] == 45 / DEFAULT_CONFIG.game_duration_seconds


def test_global_encoder_rejects_non_four_player_config():
    env = FoodTradingEnv(seed=1)
    env.reset()

    with pytest.raises(ValueError, match="requires 4 players"):
        encode_global_observation(
            env.state,
            target_player_idx=0,
            config=GameConfig(player_count=3),
        )


# ---------------------------------------------------------------------------
# v4 trade history (include_trade_history=True)
# ---------------------------------------------------------------------------


def _state_with_previous_tick_trade(seed: int = 0):
    """A state whose `elapsed_seconds - 1` tick contains exactly one trade."""
    env = FoodTradingEnv(seed=seed)
    env.reset()
    state = env.state
    seller = state.players[0]
    food = seller.produces[0]
    buyer = next(p for p in state.players if food not in p.produces)
    post_order(
        state, player_idx=seller.idx, side="ask", food_type=food,
        quantity=2, price_per_unit=7,
    )
    _, trades = post_order(
        state, player_idx=buyer.idx, side="bid", food_type=food,
        quantity=2, price_per_unit=7,
    )
    assert len(trades) == 1
    # The trade printed at the current elapsed_seconds; advance the clock so
    # it becomes "the previous tick" from the encoder's point of view.
    state.elapsed_seconds += 1
    return state, trades[0]


def test_trade_history_extends_the_contract_only_when_requested():
    from env.rl.global_observation_encoder import (
        TRADE_HISTORY_DIM,
        global_observation_dim,
    )

    state, _ = _state_with_previous_tick_trade()
    base = encode_global_observation(state, 0, DEFAULT_CONFIG)
    extended = encode_global_observation(
        state, 0, DEFAULT_CONFIG, include_trade_history=True
    )

    assert global_observation_dim() == GLOBAL_OBS_DIM
    assert global_observation_dim(include_trade_history=True) == (
        GLOBAL_OBS_DIM + TRADE_HISTORY_DIM
    )
    assert TRADE_HISTORY_DIM == 96
    assert extended.shape == (GLOBAL_OBS_DIM + TRADE_HISTORY_DIM,)
    np.testing.assert_array_equal(extended[:GLOBAL_OBS_DIM], base)


def test_trade_history_rotates_buyer_and_seller_with_the_target():
    state, trade = _state_with_previous_tick_trade()
    foods = DEFAULT_CONFIG.food_types
    food_idx = foods.index(trade.food_type)

    for target_idx in range(4):
        encoded = encode_global_observation(
            state, target_idx, DEFAULT_CONFIG, include_trade_history=True
        )
        block = encoded[GLOBAL_OBS_DIM:]
        per_food = 24  # 12 off-diagonal qty + 12 off-diagonal avg price
        food_block = block[food_idx * per_food:(food_idx + 1) * per_food]

        slot_of = {(target_idx + o) % 4: o for o in range(4)}
        buyer_slot = slot_of[trade.buyer_idx]
        seller_slot = slot_of[trade.seller_idx]
        # Off-diagonal flattening: row-major over the 4x4 matrix minus the
        # diagonal, same convention as the local encoder.
        offdiag = [
            (b, s) for b in range(4) for s in range(4) if b != s
        ]
        cell = offdiag.index((buyer_slot, seller_slot))

        expected_qty = trade.quantity / 20.0
        expected_price = trade.price_per_unit / 10.0
        assert food_block[cell] == np.float32(expected_qty)
        assert food_block[12 + cell] == np.float32(expected_price)
        # Exactly one populated cell on each matrix.
        assert np.count_nonzero(food_block[:12]) == 1
        assert np.count_nonzero(food_block[12:]) == 1

    # Every other food's block is all zeros.
    all_blocks = encode_global_observation(
        state, 0, DEFAULT_CONFIG, include_trade_history=True
    )[GLOBAL_OBS_DIM:]
    for other_idx in range(4):
        if other_idx == food_idx:
            continue
        assert not np.any(all_blocks[other_idx * 24:(other_idx + 1) * 24])


def test_trade_history_ignores_older_trades():
    state, _ = _state_with_previous_tick_trade()
    # Advance one more tick: the trade is now two ticks old and must vanish.
    state.elapsed_seconds += 1
    encoded = encode_global_observation(
        state, 0, DEFAULT_CONFIG, include_trade_history=True
    )
    assert not np.any(encoded[GLOBAL_OBS_DIM:])


def test_agent_id_shifts_after_trade_history():
    from env.rl.global_observation_encoder import (
        AGENT_ID_DIM,
        TRADE_HISTORY_DIM,
        global_observation_dim,
    )

    state, _ = _state_with_previous_tick_trade()
    both = encode_global_observation(
        state, 3, DEFAULT_CONFIG,
        include_agent_id=True, include_trade_history=True,
    )
    assert both.shape == (
        global_observation_dim(
            include_agent_id=True, include_trade_history=True
        ),
    )
    one_hot = both[GLOBAL_OBS_DIM + TRADE_HISTORY_DIM:]
    assert one_hot.shape == (AGENT_ID_DIM,)
    np.testing.assert_array_equal(
        one_hot, np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
