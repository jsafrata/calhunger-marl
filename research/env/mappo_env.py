"""Gymnasium adapter for FoodTradingEnv → EPyMARL compatibility.

EPyMARL's gymma wrapper consumes any registered gym.Env with:
  - tuple action_space  (one Discrete per agent)
  - tuple observation_space (one Box per agent)
  - step(actions: list) → (obs_list, reward_list, done, truncated, info)
  - reset(seed=, options=) → (obs_list, info)
  - .unwrapped.n_agents

This module also handles obs flattening + dense reward shaping for training.
Constructing the env with `dense_reward=False` keeps only terminal rewards.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple as GymTuple

from .actions import NUM_ACTIONS
from .config import DEFAULT_CONFIG, GameConfig
from .env import FoodTradingEnv
from .rl.observation_encoder import OBS_DIM, encode_observation
from .rl.rewards import ShapedRewardTracker, TerminalOnlyRewardTracker


class NashHungerGym(gym.Env):
    """gymnasium.Env exposing FoodTradingEnv via tuple action/observation spaces.

    Reward shaping when dense_reward=True:
        + 0.01 * min available required food while alive
        + 5.0/k for a k-way win
        + 1.0 for surviving to game end without winning

    The dense survival score does not reward cash or total food.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        dense_reward: bool = True,
        reward_kind: str = "terminal_only",
    ) -> None:
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        self.n_agents = self.config.player_count
        self.dense_reward = dense_reward
        self.reward_kind = reward_kind

        self.action_space = GymTuple(
            tuple(Discrete(NUM_ACTIONS) for _ in range(self.n_agents))
        )
        self.observation_space = GymTuple(
            tuple(
                Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)
                for _ in range(self.n_agents)
            )
        )

        self._env: Optional[FoodTradingEnv] = None
        if reward_kind == "terminal_only":
            self._reward_tracker = TerminalOnlyRewardTracker(self.config)
        elif reward_kind == "shaped":
            self._reward_tracker = ShapedRewardTracker(
                self.config,
                dense_reward=self.dense_reward,
            )
        else:
            raise ValueError(
                f"reward_kind must be 'terminal_only' or 'shaped', got "
                f"{reward_kind!r}"
            )

    # ------------------------------------------------------------------

    def _build_obs(self, obs_dict: dict) -> tuple:
        return tuple(
            encode_observation(obs_dict[i], self.config)
            for i in range(self.n_agents)
        )

    def reset(self, seed: Optional[int] = None, options=None):
        s = int(seed) if seed is not None else 0
        self._env = FoodTradingEnv(seed=s, config=self.config)
        obs_dict = self._env.reset()
        self._reward_tracker.reset(self._env)
        return self._build_obs(obs_dict), {}

    def step(self, actions):
        assert self._env is not None, "must reset() before step()"
        actions_dict = {i: int(actions[i]) for i in range(self.n_agents)}
        obs_dict, _terminal_rewards, done, info = self._env.step(actions_dict)

        rewards_by_idx = self._reward_tracker.compute(
            self._env,
            obs_dict,
            done=done,
            info=info,
        )
        rewards = [rewards_by_idx[i] for i in range(self.n_agents)]

        return self._build_obs(obs_dict), rewards, done, False, {}

    def render(self):
        return self._env.render_text() if self._env is not None else ""

    def close(self):
        pass

    # ---------------------------------------------------------------
    # Action-mask interface (consumed by patched EPyMARL gymma wrapper).
    # ---------------------------------------------------------------

    def legal_actions(self, agent_id: int) -> np.ndarray:
        """Return a bool mask of length NUM_ACTIONS indicating legal actions for agent_id.

        Used by EPyMARL's action-mask path so MAPPO never samples illegal actions
        (which would silently no-op in the underlying env and waste exploration).
        """
        if self._env is None:
            # Before reset() — be permissive.
            return np.ones(NUM_ACTIONS, dtype=bool)
        return self._env.legal_actions(int(agent_id))

    def get_avail_actions(self) -> list:
        """All-agent convenience used by some MARL frameworks."""
        return [self.legal_actions(i).astype(int).tolist() for i in range(self.n_agents)]


# Register on import so EPyMARL can `gym.make("NashHunger-v0")`.
_REGISTERED = False


def _register():
    global _REGISTERED
    if _REGISTERED:
        return
    gym.register(
        id="NashHunger-v0",
        entry_point="env.mappo_env:NashHungerGym",
        kwargs={"dense_reward": True},
    )
    gym.register(
        id="NashHunger-eval-v0",
        entry_point="env.mappo_env:NashHungerGym",
        kwargs={"dense_reward": False},
    )
    _REGISTERED = True


_register()
