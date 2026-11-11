# Legacy Immediate Replacement MRE

## Summary

With `GameConfig(simultaneous_order_clearing=False)`, `post_order()` rejects a same-side same-food replacement before it reuses the prior order's reserved resources.

## Minimal Repro

```python
from env.config import GameConfig
from env.env import FoodTradingEnv
from env.market import post_order

env = FoodTradingEnv(seed=0, config=GameConfig(simultaneous_order_clearing=False))
env.reset()
player = env.state.players[0]
player.cash = 10
player.reserved_cash = 0

first, _ = post_order(env.state, player_idx=0, side="bid", food_type="A", quantity=1, price_per_unit=10)
replacement, _ = post_order(env.state, player_idx=0, side="bid", food_type="A", quantity=1, price_per_unit=10)

assert replacement.id != first.id
assert player.reserved_cash == 10
```

## Current Behavior

The second `post_order()` raises `OrderError("insufficient_cash", ...)`.

## Expected Behavior

The replacement should cancel the prior bid, reuse its reserved cash, and leave one open bid with `reserved_cash == 10`.

## Executable Coverage

See `research/tests/test_known_env_bugs.py::test_known_bug_legacy_post_replacement_reuses_reserved_resources`.
