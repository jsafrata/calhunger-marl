# Replacement Post Mask MRE

## Summary

`legal_action_mask()` excludes a replacement post that should be legal because it ignores resources already reserved by the order being replaced.

## Minimal Repro

```python
from env.actions import PRICE_BUCKETS, QTY_BUCKETS, encode_trade
from env.env import FoodTradingEnv
from env.market import post_order

def trade(side, food, qty, price):
    return encode_trade(side, food, QTY_BUCKETS.index(qty), PRICE_BUCKETS.index(price))

env = FoodTradingEnv(seed=0)
env.reset()
player = env.state.players[0]
player.cash = 10
player.reserved_cash = 0

post_order(env.state, player_idx=0, side="bid", food_type="A", quantity=1, price_per_unit=10)
assert player.available_cash() == 0

replace_same_bid = trade("bid", "A", 1, 10)
assert env.legal_actions(0)[replace_same_bid]
```

## Current Behavior

The final assertion fails. The mask only considers available cash/food, so it treats the replacement as unaffordable.

## Expected Behavior

The replacement action should be legal because the prior bid reserves exactly the cash needed by the replacement.

## Executable Coverage

See `research/tests/test_known_env_bugs.py::test_known_bug_replacement_post_mask_reuses_reserved_resources`.
