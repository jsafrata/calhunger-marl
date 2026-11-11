from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("ray") is None:
    pytest.skip("ray[rllib] is not installed", allow_module_level=True)

from env.actions import ACTION_NOOP, NUM_ACTIONS
from env.config import GameConfig
from env.env import FoodTradingEnv
from env.mappo_env import NashHungerGym
from env.rl.global_observation_encoder import GLOBAL_OBS_DIM
from env.rl.observation_encoder import OBS_DIM
from env.rl.rllib_env import NashHungerMultiAgentEnv


def _noop_actions(env: NashHungerMultiAgentEnv) -> dict[str, int]:
    return {agent_id: ACTION_NOOP for agent_id in env.agents}


def test_reset_returns_four_masked_observations_in_declared_spaces():
    env = NashHungerMultiAgentEnv()
    observations, infos = env.reset(seed=7)

    assert set(observations) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    for agent_id, observation in observations.items():
        assert observation["observations"].shape == (OBS_DIM,)
        assert observation["observations"].dtype == np.float32
        assert observation["action_mask"].shape == (NUM_ACTIONS,)
        assert observation["action_mask"].dtype == np.float32
        assert observation["critic_observations"].shape == (GLOBAL_OBS_DIM,)
        assert observation["critic_observations"].dtype == np.float32
        assert env.observation_spaces[agent_id].contains(observation)
        assert env.action_spaces[agent_id].contains(ACTION_NOOP)


def test_action_masks_match_core_environment():
    env = NashHungerMultiAgentEnv()
    observations, _ = env.reset(seed=3)

    assert env._env is not None
    for idx, agent_id in enumerate(env.possible_agents):
        np.testing.assert_array_equal(
            observations[agent_id]["action_mask"],
            env._env.legal_actions(idx).astype(np.float32),
        )


def test_one_step_matches_direct_core_environment_state():
    wrapped = NashHungerMultiAgentEnv()
    wrapped.reset(seed=11)
    assert wrapped._env is not None
    direct = FoodTradingEnv(seed=wrapped._env.seed)
    direct.reset()

    actions = _noop_actions(wrapped)
    wrapped.step(actions)
    direct.step({idx: ACTION_NOOP for idx in range(4)})

    assert wrapped._env.get_result() == direct.get_result()
    assert wrapped._env.render_text() == direct.render_text()


def test_dead_agent_gets_terminal_reward_final_observation_and_disappears():
    """v2: a seat that dies mid-episode gets its terminal reward on the
    death step. Under the default TerminalOnlyRewardTracker this is -1.
    v1 delivered 0 here because the terminal reward was gated on
    game_done, which is why the archived RL runs never learned that
    dying is bad."""
    env = NashHungerMultiAgentEnv()
    env.reset(seed=5)
    assert env._env is not None
    state = env._env.state
    player = state.players[0]
    missing_food = env.config.required_foods(player.produces)[0]
    player.inventory[missing_food] = 0
    player.reserved_inventory[missing_food] = 0
    state.elapsed_seconds = env.config.consumption_interval_seconds - 1

    observations, rewards, terminated, _, infos = env.step(
        _noop_actions(env)
    )

    assert rewards["seat_0"] == pytest.approx(-1.0)
    assert terminated["seat_0"] is True
    assert infos["seat_0"]["died"] is True
    assert "seat_0" in observations
    assert env.observation_spaces["seat_0"].contains(observations["seat_0"])
    assert "seat_0" not in env.agents


def test_global_termination_marks_all_remaining_agents():
    config = GameConfig(game_duration_seconds=1)
    env = NashHungerMultiAgentEnv({"game_config": config})
    env.reset(seed=2)

    observations, _, terminated, truncated, _ = env.step(
        _noop_actions(env)
    )

    assert observations == {}
    assert terminated["__all__"] is True
    assert truncated["__all__"] is False
    assert all(terminated[agent_id] for agent_id in env.possible_agents)
    assert env.agents == []


def test_reset_restores_dead_agents_and_seed_is_deterministic():
    env = NashHungerMultiAgentEnv()
    first, _ = env.reset(seed=13)
    assert env._env is not None
    env._env.state.players[0].status = "dead"
    env.agents.remove("seat_0")

    second, _ = env.reset(seed=13)

    assert env.agents == env.possible_agents
    for agent_id in env.possible_agents:
        np.testing.assert_array_equal(
            first[agent_id]["observations"],
            second[agent_id]["observations"],
        )
        np.testing.assert_array_equal(
            first[agent_id]["action_mask"],
            second[agent_id]["action_mask"],
        )
        np.testing.assert_array_equal(
            first[agent_id]["critic_observations"],
            second[agent_id]["critic_observations"],
        )


def test_gym_and_rllib_wrappers_match_observations_and_rewards():
    rllib_env = NashHungerMultiAgentEnv()
    rllib_observations, _ = rllib_env.reset(seed=17)
    assert rllib_env._env is not None
    gym_env = NashHungerGym()
    gym_observations, _ = gym_env.reset(seed=rllib_env._env.seed)

    for idx in range(4):
        np.testing.assert_array_equal(
            gym_observations[idx],
            rllib_observations[f"seat_{idx}"]["observations"],
        )

    _, gym_rewards, _, _, _ = gym_env.step([ACTION_NOOP] * 4)
    _, rllib_rewards, _, _, _ = rllib_env.step(
        _noop_actions(rllib_env)
    )

    assert gym_rewards == [rllib_rewards[f"seat_{idx}"] for idx in range(4)]


def test_v4_observation_flags_extend_both_observation_streams():
    from env.rl.global_observation_encoder import global_observation_dim
    from env.rl.observation_encoder import observation_dim

    env = NashHungerMultiAgentEnv(
        {
            "seed": 0,
            "include_survival_features": True,
            "include_trade_history": True,
        }
    )
    local_dim = observation_dim(include_survival_features=True)
    global_dim = global_observation_dim(include_trade_history=True)
    space = env.observation_spaces["seat_0"]
    assert space["observations"].shape == (local_dim,)
    assert space["critic_observations"].shape == (global_dim,)

    observations, _ = env.reset(seed=0)
    for agent_id, obs in observations.items():
        assert obs["observations"].shape == (local_dim,)
        assert obs["critic_observations"].shape == (global_dim,)
        assert space.contains(obs), agent_id

    observations, _, _, _, _ = env.step(_noop_actions(env))
    for obs in observations.values():
        assert obs["observations"].shape == (local_dim,)
        assert obs["critic_observations"].shape == (global_dim,)
