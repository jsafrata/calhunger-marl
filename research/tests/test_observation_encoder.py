from __future__ import annotations

import numpy as np

from env.actions import ACTION_NOOP
from env.config import DEFAULT_CONFIG, GameConfig
from env.env import FoodTradingEnv
from env.market import post_order
from env.rl.observation_encoder import (
    OBS_DIM,
    OPPONENTS_SLICE,
    OWN_ORDERS_SLICE,
    OWN_PRIVATE_SLICE,
    PUBLIC_MARKET_SLICE,
    PUBLIC_TRADES_SLICE,
    TIME_SLICE,
    encode_observation,
)


def _raw_observation(seed: int = 0, player_idx: int = 0) -> dict:
    env = FoodTradingEnv(seed=seed)
    return env.reset()[player_idx]


def test_encoder_shape_dtype_contiguity_and_determinism():
    obs = _raw_observation()

    first = encode_observation(obs, DEFAULT_CONFIG)
    second = encode_observation(obs, DEFAULT_CONFIG)

    assert first.shape == (OBS_DIM,)
    assert OBS_DIM == 171
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_private_resources_keep_the_documented_indices():
    obs = _raw_observation()
    encoded = encode_observation(obs, DEFAULT_CONFIG)
    foods = DEFAULT_CONFIG.food_types

    expected = [
        obs["cash"] / 200.0,
        obs["reserved_cash"] / 200.0,
        obs["available_cash"] / 200.0,
        *[obs["inventory"][food] / 50.0 for food in foods],
        *[obs["reserved_inventory"][food] / 50.0 for food in foods],
        *[obs["available_inventory"][food] / 50.0 for food in foods],
        *[float(food in obs["produces"]) for food in foods],
    ]

    np.testing.assert_array_equal(
        encoded[OWN_PRIVATE_SLICE],
        np.asarray(expected, dtype=np.float32),
    )


def test_public_market_summary_has_best_quote_levels_only():
    env = FoodTradingEnv(seed=0)
    env.reset()
    for player in env.state.players:
        player.cash = 1000
        player.inventory = {food: 50 for food in env.config.food_types}

    post_order(
        env.state,
        player_idx=0,
        side="bid",
        food_type="A",
        quantity=4,
        price_per_unit=7,
    )
    post_order(
        env.state,
        player_idx=1,
        side="ask",
        food_type="A",
        quantity=2,
        price_per_unit=9,
    )

    obs = env.observe(0)
    encoded = encode_observation(obs, env.config)

    expected_a = np.asarray([0.7, 0.9, 0.2, 0.1], dtype=np.float32)
    np.testing.assert_array_equal(encoded[19:23], expected_a)
    np.testing.assert_array_equal(
        encoded[23:35],
        np.zeros(12, dtype=np.float32),
    )


def test_current_two_food_production_is_encoded_for_self_and_opponents():
    obs = _raw_observation()
    encoded = encode_observation(obs, DEFAULT_CONFIG)

    own_production = encoded[15:19]
    expected_own = np.asarray(
        [float(food in obs["produces"]) for food in DEFAULT_CONFIG.food_types],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(own_production, expected_own)
    assert own_production.sum() == 2.0

    opponents = [p for p in obs["players"] if p["idx"] != obs["player_idx"]]
    first_opponent = encoded[35:40]
    expected_opponent = np.asarray(
        [
            1.0 if opponents[0]["status"] == "alive" else 0.0,
            *[
                float(food in opponents[0]["produces"])
                for food in DEFAULT_CONFIG.food_types
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(first_opponent, expected_opponent)
    assert first_opponent[1:].sum() == 2.0


def test_legacy_single_food_production_remains_supported():
    obs = _raw_observation()
    obs["produces"] = "C"
    opponents = [p for p in obs["players"] if p["idx"] != obs["player_idx"]]
    opponents[0]["produces"] = "A"

    encoded = encode_observation(obs, DEFAULT_CONFIG)

    np.testing.assert_array_equal(
        encoded[15:19], np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        encoded[36:40], np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )


def test_own_order_slots_encode_exists_remaining_quantity_and_price():
    env = FoodTradingEnv(seed=0)
    env.reset()
    player = env.state.players[0]
    player.cash = 1000
    player.inventory = {food: 50 for food in env.config.food_types}

    post_order(
        env.state,
        player_idx=0,
        side="bid",
        food_type="B",
        quantity=4,
        price_per_unit=6,
    )

    encoded = encode_observation(env.observe(0), env.config)

    assert encoded[OWN_ORDERS_SLICE].shape == (24,)
    b_bid_start = 50 + 2 * 3
    np.testing.assert_array_equal(
        encoded[b_bid_start : b_bid_start + 3],
        np.asarray([1.0, 4 / 20.0, 6 / 10.0], dtype=np.float32),
    )


def test_previous_tick_trade_matrix_uses_rotated_buyer_seller_order():
    config = GameConfig(simultaneous_order_clearing=False)
    env = FoodTradingEnv(seed=0, config=config)
    env.reset()
    for player in env.state.players:
        player.cash = 1000
        player.inventory = {food: 50 for food in env.config.food_types}

    post_order(
        env.state,
        player_idx=0,
        side="bid",
        food_type="A",
        quantity=2,
        price_per_unit=8,
    )
    post_order(
        env.state,
        player_idx=1,
        side="ask",
        food_type="A",
        quantity=2,
        price_per_unit=7,
    )
    env.step({idx: ACTION_NOOP for idx in range(4)})

    obs = env.observe(0)
    encoded = encode_observation(obs, env.config)

    assert len(obs["previous_tick_trades"]) == 1
    trade_slice = encoded[PUBLIC_TRADES_SLICE]
    assert trade_slice.shape == (96,)
    assert trade_slice[0] == 2 / 20.0
    assert trade_slice[12] == 8 / 10.0
    assert np.isclose(trade_slice.sum(), (2 / 20.0) + (8 / 10.0))


def test_optional_public_sections_zero_fill():
    obs = _raw_observation()
    obs["order_books"] = {}
    obs["own_orders"] = []
    obs["players"] = [
        p for p in obs["players"] if p["idx"] == obs["player_idx"]
    ]
    obs["previous_tick_trades"] = []

    encoded = encode_observation(obs, DEFAULT_CONFIG)

    np.testing.assert_array_equal(
        encoded[PUBLIC_MARKET_SLICE], np.zeros(16, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        encoded[OPPONENTS_SLICE], np.zeros(15, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        encoded[PUBLIC_TRADES_SLICE], np.zeros(96, dtype=np.float32)
    )


def test_major_sections_keep_the_documented_indices():
    obs = _raw_observation()
    encoded = encode_observation(obs, DEFAULT_CONFIG)

    assert encoded[OWN_PRIVATE_SLICE].shape == (19,)
    assert encoded[PUBLIC_MARKET_SLICE].shape == (16,)
    assert encoded[OPPONENTS_SLICE].shape == (15,)
    assert encoded[OWN_ORDERS_SLICE].shape == (24,)
    assert encoded[PUBLIC_TRADES_SLICE].shape == (96,)
    assert encoded[TIME_SLICE].shape == (1,)
    assert encoded[170] == obs["tick"] / DEFAULT_CONFIG.game_duration_seconds


# ---------------------------------------------------------------------------
# v4 survival features (include_survival_features=True)
# ---------------------------------------------------------------------------


def test_survival_features_extend_the_contract_only_when_requested():
    from env.rl.observation_encoder import (
        SURVIVAL_FEATURES_DIM,
        SURVIVAL_FEATURES_SLICE,
        observation_dim,
    )

    obs = _raw_observation()
    base = encode_observation(obs, DEFAULT_CONFIG)
    extended = encode_observation(
        obs, DEFAULT_CONFIG, include_survival_features=True
    )

    assert observation_dim() == OBS_DIM
    assert observation_dim(include_survival_features=True) == (
        OBS_DIM + SURVIVAL_FEATURES_DIM
    )
    assert extended.shape == (OBS_DIM + SURVIVAL_FEATURES_DIM,)
    # The base prefix is bit-identical — the flag only appends.
    np.testing.assert_array_equal(extended[:OBS_DIM], base)
    assert SURVIVAL_FEATURES_SLICE == slice(171, 176)


def test_survival_cover_counts_consumption_cycles_per_food():
    from env.rl.observation_encoder import COVER_CYCLES_CAP

    env = FoodTradingEnv(seed=0)
    env.reset()
    obs = env.observe(0)
    required = obs["required_foods"]
    consume = DEFAULT_CONFIG.consumption_per_required_food_per_second
    # Force known inventories: 3 cycles of cover on the first required
    # food, starving (0 cycles) on the second.
    env.state.players[0].inventory[required[0]] = 3 * consume
    env.state.players[0].inventory[required[1]] = consume - 1
    obs = env.observe(0)

    encoded = encode_observation(
        obs, DEFAULT_CONFIG, include_survival_features=True
    )
    cover = {
        food: encoded[171 + i]
        for i, food in enumerate(DEFAULT_CONFIG.food_types)
    }
    assert cover[required[0]] == np.float32(3.0 / COVER_CYCLES_CAP)
    assert cover[required[1]] == 0.0
    # Produced foods are never consumed by their producer: cover pegs at max.
    for food in obs["produces"]:
        assert cover[food] == 1.0


def test_cycle_phase_wraps_each_consumption_interval():
    interval = DEFAULT_CONFIG.consumption_interval_seconds

    def phase_at(tick: int) -> float:
        env = FoodTradingEnv(seed=0)
        env.reset()
        env.state.elapsed_seconds = tick
        obs = env.observe(0)
        encoded = encode_observation(
            obs, DEFAULT_CONFIG, include_survival_features=True
        )
        return float(encoded[175])

    # Just after a consumption tick the whole interval remains; one step
    # before the next tick only 1/interval remains.
    assert phase_at(0) == 1.0
    assert phase_at(interval - 1) == np.float32(1.0 / interval)
    assert phase_at(interval) == 1.0


def test_agent_id_shifts_after_survival_features():
    from env.rl.observation_encoder import (
        AGENT_ID_DIM,
        SURVIVAL_FEATURES_DIM,
        observation_dim,
    )

    obs = _raw_observation(player_idx=2)
    both = encode_observation(
        obs,
        DEFAULT_CONFIG,
        include_agent_id=True,
        include_survival_features=True,
    )
    assert both.shape == (
        observation_dim(include_agent_id=True, include_survival_features=True),
    )
    assert both.shape == (OBS_DIM + SURVIVAL_FEATURES_DIM + AGENT_ID_DIM,)
    one_hot = both[OBS_DIM + SURVIVAL_FEATURES_DIM:]
    np.testing.assert_array_equal(
        one_hot, np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    )
