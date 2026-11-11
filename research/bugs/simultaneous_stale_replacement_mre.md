# Simultaneous Stale Replacement MRE

## Summary

In simultaneous clearing, a post can pass prevalidation because it plans to replace an existing order, but another same-tick action can fill that prior order before the replacement is posted. The second validation can then raise instead of returning a normal rejected action result.

## Minimal Repro

```python
from env.actions import ACTION_NOOP, PRICE_BUCKETS, QTY_BUCKETS, encode_trade
from env.config import GameConfig
from env.env import FoodTradingEnv
from env.market import post_order

def trade(side, food, qty, price):
    return encode_trade(side, food, QTY_BUCKETS.index(qty), PRICE_BUCKETS.index(price))

env = FoodTradingEnv(seed=0, config=GameConfig(simultaneous_order_clearing=True))
env.reset()
for player in env.state.players:
    player.cash = 1000
    player.inventory = {food: 0 for food in env.config.food_types}
    player.reserved_inventory = {food: 0 for food in env.config.food_types}

env.state.players[0].inventory["A"] = 1
post_order(env.state, player_idx=0, side="ask", food_type="A", quantity=1, price_per_unit=10)

env.step({
    0: trade("ask", "A", 1, 10),
    1: trade("bid", "A", 1, 10),
    2: ACTION_NOOP,
    3: ACTION_NOOP,
})
```

## Current Behavior

For seeds whose shuffled player order applies player 1's take before player 0's replacement post, `env.step()` can raise `OrderError("insufficient_inventory", ...)`.

## Expected Behavior

The environment should not crash. The stale replacement should be represented in `info.actions_applied[0]` as a normal rejected action.

## Executable Coverage

See `research/tests/test_known_env_bugs.py::test_known_bug_simultaneous_stale_replacement_post_does_not_crash`.
