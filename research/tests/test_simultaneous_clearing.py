from env.actions import ACTION_NOOP, PRICE_BUCKETS, QTY_BUCKETS, encode_trade
from env.config import GameConfig
from env.env import FoodTradingEnv
from env.market import post_order, post_order_without_matching


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


def _sim_config() -> GameConfig:
    return GameConfig(simultaneous_order_clearing=True)


def _legacy_config() -> GameConfig:
    return GameConfig(simultaneous_order_clearing=False)


def test_default_config_uses_simultaneous_clearing():
    env = FoodTradingEnv(seed=0)
    _fund_everyone(env)

    actions = {
        0: _trade("bid", "A", 1, 10),
        1: _trade("ask", "A", 1, 8),
    }
    _, _, _, info = env.step(actions)

    assert env.config.simultaneous_order_clearing is True
    assert len(env.state.trades) == 1
    assert info.actions_applied[0]["n_trades"] == 1
    assert info.actions_applied[1]["n_trades"] == 1


def test_explicit_legacy_config_preserves_immediate_matching():
    env = FoodTradingEnv(seed=0, config=_legacy_config())
    _fund_everyone(env)

    actions = {
        0: _trade("bid", "A", 1, 10),
        1: _trade("ask", "A", 1, 8),
    }
    _, _, _, info = env.step(actions)

    assert env.config.simultaneous_order_clearing is False
    assert len(env.state.trades) == 1
    assert env.state.trades[0].price_per_unit == 10
    assert info.actions_applied[0]["n_trades"] == 0
    assert info.actions_applied[1]["n_trades"] == 1


def test_simultaneous_mode_does_not_make_lower_index_order_the_maker():
    found_seed = None
    for seed in range(100):
        env = FoodTradingEnv(seed=seed, config=_sim_config())
        _fund_everyone(env)

        env.step(
            {
                0: _trade("bid", "A", 1, 10),
                1: _trade("ask", "A", 1, 8),
            }
        )
        if env.state.trades[0].price_per_unit == 8:
            found_seed = seed
            break

    assert found_seed is not None

    legacy = FoodTradingEnv(seed=found_seed, config=_legacy_config())
    _fund_everyone(legacy)
    legacy.step(
        {
            0: _trade("bid", "A", 1, 10),
            1: _trade("ask", "A", 1, 8),
        }
    )

    assert legacy.state.trades[0].price_per_unit == 10


def test_simultaneous_replacement_reuses_replaced_order_reserve():
    env = FoodTradingEnv(seed=0, config=_sim_config())
    _fund_everyone(env, cash=10)
    old_order, _ = post_order(
        env.state,
        player_idx=0,
        side="bid",
        food_type="A",
        quantity=1,
        price_per_unit=10,
    )

    _, _, _, info = env.step(
        {
            0: _trade("bid", "A", 1, 10),
            1: ACTION_NOOP,
            2: ACTION_NOOP,
            3: ACTION_NOOP,
        }
    )

    assert info.actions_applied[0]["applied"] is True
    assert info.actions_applied[0]["order_id"] != old_order.id
    assert env.state.players[0].cash == 10
    assert env.state.players[0].reserved_cash == 10


def test_same_tick_priority_is_seeded_and_deterministic():
    def run(seed: int) -> int:
        env = FoodTradingEnv(seed=seed, config=_sim_config())
        _fund_everyone(env)
        env.step(
            {
                0: _trade("bid", "A", 1, 10),
                1: _trade("ask", "A", 1, 8),
            }
        )
        return env.state.trades[0].price_per_unit

    assert run(7) == run(7)
    assert {run(seed) for seed in range(40)} == {8, 10}


def test_crossed_book_clears_price_time_avoids_self_trade_and_leaves_residual():
    env = FoodTradingEnv(seed=0, config=_sim_config())
    _fund_everyone(env)
    state = env.state

    own_bid = post_order_without_matching(
        state,
        player_idx=0,
        side="bid",
        food_type="A",
        quantity=5,
        price_per_unit=10,
    )
    own_ask = post_order_without_matching(
        state,
        player_idx=0,
        side="ask",
        food_type="A",
        quantity=5,
        price_per_unit=1,
    )
    post_order_without_matching(
        state,
        player_idx=1,
        side="ask",
        food_type="A",
        quantity=2,
        price_per_unit=8,
    )

    env.step({idx: ACTION_NOOP for idx in range(4)})

    assert len(state.trades) == 1
    trade = state.trades[0]
    assert trade.buyer_idx == 0
    assert trade.seller_idx == 1
    assert trade.price_per_unit == 10
    assert trade.maker_order_id == own_bid.id
    assert own_bid.remaining_quantity == 3
    assert own_ask.remaining_quantity == 5
    assert state.players[0].reserved_cash == 30
    assert state.players[0].reserved_inventory["A"] == 5
