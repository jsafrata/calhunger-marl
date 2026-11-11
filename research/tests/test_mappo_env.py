"""Sanity tests for the gymnasium adapter that feeds FoodTradingEnv into EPyMARL.

Skipped if gymnasium isn't installed (which is the case in the default research
venv — these tests are intended to run in the .venv-mappo venv).
"""
from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("gymnasium") is None:
    pytest.skip("gymnasium not installed in this venv", allow_module_level=True)

import gymnasium as gym
import numpy as np

from env.actions import ACTION_NOOP, NUM_ACTIONS
from env.mappo_env import NashHungerGym, OBS_DIM


def test_registered():
    env = gym.make("NashHunger-v0")
    assert env is not None


def test_obs_dim_matches_constant():
    env = NashHungerGym()
    obs, _ = env.reset(seed=0)
    assert len(obs) == env.n_agents
    for o in obs:
        assert o.shape == (OBS_DIM,)
        assert o.dtype == np.float32


def test_step_returns_correct_shapes():
    env = NashHungerGym()
    env.reset(seed=0)
    actions = [ACTION_NOOP] * env.n_agents
    obs, rewards, done, truncated, info = env.step(actions)
    assert len(obs) == env.n_agents
    assert len(rewards) == env.n_agents
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)


def test_action_space_is_correct():
    env = NashHungerGym()
    assert len(env.action_space) == env.n_agents
    for sp in env.action_space:
        assert sp.n == NUM_ACTIONS


def test_full_episode_runs():
    """All-noop game should end (no_survivors / single_survivor) within game duration."""
    env = NashHungerGym()
    env.reset(seed=7)
    for _ in range(env.config.game_duration_seconds + 1):
        _, _, done, _, _ = env.step([ACTION_NOOP] * env.n_agents)
        if done:
            break
    else:
        pytest.fail("episode did not terminate within game_duration_seconds")


def test_dense_reward_pays_positive_survival_position_while_alive():
    """Dense reward should pay for current survival position, not penalties.
    (Legacy shaped reward path — opts in explicitly under v2.)"""
    from env.config import GameConfig

    env = NashHungerGym(
        config=GameConfig(game_duration_seconds=400),
        dense_reward=True,
        reward_kind="shaped",
    )
    env.reset(seed=11)
    cumulative = np.zeros(env.n_agents)
    for _ in range(env.config.game_duration_seconds + 1):
        _, rewards, done, _, _ = env.step([ACTION_NOOP] * env.n_agents)
        cumulative += np.asarray(rewards)
        if done:
            break
    assert (cumulative >= 0.0).all(), f"cumulative={cumulative}"
    assert (cumulative > 0.0).any(), f"cumulative={cumulative}"


def test_dense_reward_uses_minimum_available_required_food():
    """Dense shaping only fires at consumption ticks (every 10s) and clips
    the buffer at BUFFER_CAP. We step to the first consumption tick, then
    verify the reward equals `SCALE × min(clipped_inventory) = 0.01 × 3`.
    """
    env = NashHungerGym(dense_reward=True, reward_kind="shaped")
    env.reset(seed=5)
    assert env._env is not None
    interval = env.config.consumption_interval_seconds
    # Advance to the tick JUST BEFORE the first consumption tick so the
    # next step triggers it. Consumption fires when elapsed_seconds is a
    # positive multiple of interval; the reward is emitted on the same step.
    for _ in range(interval - 1):
        env.step([ACTION_NOOP] * env.n_agents)
    # Set up the target player's inventory right before the consumption step.
    player = env._env.state.players[0]
    required = env.config.required_foods(player.produces)
    player.cash = 0
    for food in required:
        player.inventory[food] = 100
        player.reserved_inventory[food] = 0
    # Consumption fires DURING the step (order: actions → consumption →
    # observations → reward). The reward reads post-consumption inventory,
    # so we set pre-consumption = target + consume_per_tick(=2) to land the
    # target min buffer of 3 in the reward computation.
    consume = env.config.consumption_per_required_food_per_second
    player.inventory[required[0]] = 3 + consume  # post-consumption → 3
    player.inventory[required[1]] = 7 + consume  # post-consumption → 7

    _, rewards, _, _, _ = env.step([ACTION_NOOP] * env.n_agents)

    # min(clip(3, 20), clip(7, 20)) = 3 → 0.01 × 3 = 0.03
    assert rewards[0] == pytest.approx(0.01 * 3)


def test_dense_reward_only_fires_at_consumption_ticks():
    """No shaping on non-consumption ticks, positive shaping on the tick
    that fires consumption (ticks 10, 20, 30, ..., every `interval` steps)."""
    env = NashHungerGym(dense_reward=True, reward_kind="shaped")
    env.reset(seed=17)
    interval = env.config.consumption_interval_seconds
    for step_idx in range(1, 2 * interval + 1):
        _, rewards, _, _, _ = env.step([ACTION_NOOP] * env.n_agents)
        total = float(sum(rewards))
        if step_idx % interval == 0:
            assert total > 0.0, (
                f"consumption tick {step_idx} should have paid dense reward, got 0"
            )
        else:
            assert total == 0.0, (
                f"non-consumption tick {step_idx} paid dense reward {total}"
            )


def test_sparse_eval_mode_only_pays_at_terminal():
    """With dense_reward=False, non-terminal steps should receive no reward.
    With dense_reward=True, only consumption ticks pay before the terminal."""
    env_sparse = NashHungerGym(dense_reward=False, reward_kind="shaped")
    env_dense = NashHungerGym(dense_reward=True, reward_kind="shaped")
    env_sparse.reset(seed=11)
    env_dense.reset(seed=11)
    interval = env_dense.config.consumption_interval_seconds
    # Step to the first consumption tick — sparse still pays nothing, dense pays.
    for step_idx in range(1, interval + 1):
        _, rs, _, _, _ = env_sparse.step([ACTION_NOOP] * env_sparse.n_agents)
        _, rd, _, _, _ = env_dense.step([ACTION_NOOP] * env_dense.n_agents)
        assert sum(rs) == 0.0, f"sparse mode paid reward on step {step_idx}"
        if step_idx % interval == 0:
            assert sum(rd) > 0.0, (
                f"dense mode should pay on consumption tick {step_idx}"
            )
        else:
            assert sum(rd) == 0.0, (
                f"dense mode paid reward on non-consumption tick {step_idx}"
            )
