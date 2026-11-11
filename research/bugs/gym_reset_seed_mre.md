# Gym Reset Seed MRE

## Summary

`NashHungerGym.reset(seed=None)` rebuilds `FoodTradingEnv(seed=0)` every episode, so MAPPO/EPyMARL-style runners that do not pass a seed can train on the same initial game repeatedly.

## Minimal Repro

```python
from env.mappo_env import NashHungerGym

env = NashHungerGym()
seeds = []
for _ in range(4):
    env.reset()
    seeds.append(env._env.seed)

assert len(set(seeds)) > 1
```

## Current Behavior

`seeds == [0, 0, 0, 0]`.

## Expected Behavior

No-seed resets should advance through a reproducible internal seed stream, matching the RLlib adapter's behavior.

## Executable Coverage

See `research/tests/test_known_env_bugs.py::test_known_bug_gym_reset_without_seed_advances_episode_seeds`.
