"""RLlib multi-agent adapter for the Nash Hunger simulator."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from gymnasium.spaces import Box, Dict as SpaceDict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from ..actions import ACTION_NOOP, NUM_ACTIONS
from ..config import DEFAULT_CONFIG, GameConfig
from ..env import FoodTradingEnv
from ..small_env import SCALED_ENVIRONMENTS
from .global_observation_encoder import (
    encode_global_observation,
    global_observation_dim,
)
from .observation_encoder import encode_observation, observation_dim
from .rewards import ShapedRewardTracker, TerminalOnlyRewardTracker

# True repeats the same game setup every episode. False generates a
# reproducible sequence of different episode seeds from the configured seed.
USE_FIXED_EPISODE_SEED = False


class NashHungerMultiAgentEnv(MultiAgentEnv):
    """Expose each game seat as an RLlib agent with a shared space contract."""

    def __init__(self, env_config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        options = dict(env_config or {})
        self.env_variant = str(options.get("env_variant", "default"))
        explicit_config = options.get("game_config")
        self.config: GameConfig
        if self.env_variant == "default":
            self.config = explicit_config or DEFAULT_CONFIG
            self._env_class = FoodTradingEnv
        elif self.env_variant in SCALED_ENVIRONMENTS:
            default_config, environment_class = SCALED_ENVIRONMENTS[self.env_variant]
            self.config = explicit_config or default_config
            self._env_class = environment_class
        else:
            choices = ", ".join(repr(name) for name in (
                "default", *SCALED_ENVIRONMENTS.keys()
            ))
            raise ValueError(
                f"env_variant must be one of {choices}, got "
                f"{self.env_variant!r}"
            )
        if self.config.player_count != 4:
            raise ValueError(
                "NashHungerMultiAgentEnv centralized observations require "
                "exactly 4 players"
            )
        # v2 default: "terminal_only" tracker (dead=-1, alive-nonwinner=0,
        # winner=+1, per-tie-winner=+1). Set `reward_kind="shaped"` in
        # env_config to fall back to the legacy dense+terminal tracker
        # used by the v1 archive baselines.
        self.reward_kind: str = str(
            options.get("reward_kind", "terminal_only")
        )
        self.dense_reward = bool(options.get("dense_reward", True))
        # Optional per-instance terminal magnitudes for the LEGACY shaped
        # tracker's A/B experiments. Ignored when reward_kind="terminal_only".
        self._terminal_reward_config: dict = dict(
            options.get("terminal_reward_config", {}) or {}
        )
        self.default_seed = int(options.get("seed", 0))
        self.include_agent_id = bool(options.get("include_agent_id", False))
        # v4 observation extensions — both default False so existing
        # checkpoints keep their exact contracts.
        self.include_survival_features = bool(
            options.get("include_survival_features", False)
        )
        self.include_trade_history = bool(
            options.get("include_trade_history", False)
        )
        self._episode_rng = np.random.default_rng(self.default_seed)

        self.possible_agents = [
            self._agent_id(idx) for idx in range(self.config.player_count)
        ]
        self.agents = list(self.possible_agents)

        observation_space = SpaceDict(
            {
                "observations": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(observation_dim(
                        include_agent_id=self.include_agent_id,
                        include_survival_features=self.include_survival_features,
                    ),),
                    dtype=np.float32,
                ),
                "action_mask": Box(
                    low=0.0,
                    high=1.0,
                    shape=(NUM_ACTIONS,),
                    dtype=np.float32,
                ),
                "critic_observations": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(global_observation_dim(
                        include_agent_id=self.include_agent_id,
                        include_trade_history=self.include_trade_history,
                    ),),
                    dtype=np.float32,
                ),
            }
        )
        action_space = Discrete(NUM_ACTIONS)
        self.observation_spaces = {
            agent_id: observation_space for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: action_space for agent_id in self.possible_agents
        }

        self._env: Optional[FoodTradingEnv] = None
        if self.reward_kind == "terminal_only":
            self._reward_tracker = TerminalOnlyRewardTracker(self.config)
        elif self.reward_kind == "shaped":
            self._reward_tracker = ShapedRewardTracker(
                self.config,
                dense_reward=self.dense_reward,
                **{
                    k: v for k, v in self._terminal_reward_config.items()
                    if k in (
                        "win_reward",
                        "survived_non_win_reward",
                        "death_penalty",
                        "win_split_across_ties",
                    )
                },
            )
        else:
            raise ValueError(
                f"reward_kind must be 'terminal_only' or 'shaped', got "
                f"{self.reward_kind!r}"
            )

    @staticmethod
    def _agent_id(player_idx: int) -> str:
        return f"seat_{player_idx}"

    @staticmethod
    def _player_idx(agent_id: str) -> int:
        if not agent_id.startswith("seat_"):
            raise ValueError(f"invalid Nash Hunger agent id: {agent_id!r}")
        try:
            player_idx = int(agent_id.removeprefix("seat_"))
        except ValueError as exc:
            raise ValueError(
                f"invalid Nash Hunger agent id: {agent_id!r}"
            ) from exc
        return player_idx

    def _build_observations(
        self,
        raw_observations: Mapping[int, dict],
        agent_ids: Optional[list[str]] = None,
    ) -> dict[str, dict]:
        assert self._env is not None
        if agent_ids is None:
            agent_ids = self.agents
        encoded_local = [
            encode_observation(
                raw_observations[idx],
                self.config,
                include_agent_id=self.include_agent_id,
                include_survival_features=self.include_survival_features,
            )
            for idx in range(self.config.player_count)
        ]
        return {
            agent_id: {
                "observations": encoded_local[self._player_idx(agent_id)],
                "action_mask": np.ascontiguousarray(
                    self._env.legal_actions(self._player_idx(agent_id)),
                    dtype=np.float32,
                ),
                "critic_observations": encode_global_observation(
                    self._env.state,
                    self._player_idx(agent_id),
                    self.config,
                    include_agent_id=self.include_agent_id,
                    include_trade_history=self.include_trade_history,
                ),
            }
            for agent_id in agent_ids
        }

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.default_seed = int(seed)
            self._episode_rng = np.random.default_rng(self.default_seed)
        if USE_FIXED_EPISODE_SEED:
            episode_seed = self.default_seed
        else:
            episode_seed = int(
                self._episode_rng.integers(0, np.iinfo(np.int32).max)
            )
        self._env = self._env_class(seed=episode_seed, config=self.config)
        raw_observations = self._env.reset()
        self._reward_tracker.reset(self._env)
        self.agents = list(self.possible_agents)

        observations = self._build_observations(raw_observations)
        infos = {agent_id: {} for agent_id in self.agents}
        return observations, infos

    def step(self, action_dict: Mapping[str, int]):
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")

        active_before_step = list(self.agents)
        actions = {
            idx: ACTION_NOOP for idx in range(self.config.player_count)
        }
        for agent_id, action in action_dict.items():
            if agent_id not in active_before_step:
                raise ValueError(f"action supplied for inactive agent {agent_id!r}")
            actions[self._player_idx(agent_id)] = int(action)

        raw_observations, _, game_done, step_info = self._env.step(actions)
        rewards_by_idx = self._reward_tracker.compute(
            self._env,
            raw_observations,
            done=game_done,
            info=step_info,
        )
        deaths = set(step_info.deaths_this_tick)

        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}
        rewards: dict[str, float] = {}
        infos: dict[str, dict] = {}

        for agent_id in active_before_step:
            player_idx = self._player_idx(agent_id)
            died = player_idx in deaths
            rewards[agent_id] = rewards_by_idx[player_idx]
            terminated[agent_id] = bool(died or game_done)
            truncated[agent_id] = False
            # Reward split — the callback aggregates dense vs terminal
            # separately so we can spot if either channel disappears or
            # dominates unexpectedly. Applied every step, even if 0.
            reward_split = self._reward_tracker.last_breakdown.get(
                player_idx, {"dense": 0.0, "terminal": 0.0, "total": 0.0}
            )
            action_result = step_info.actions_applied.get(player_idx) or {}
            applied_ok = action_result.get("applied") is not False
            # "Rejection code" is the specific reason the env couldn't
            # apply the action. Kept separately from the boolean flag so
            # the callback can tell mask violations (would-be code:
            # "insufficient_cash" / "insufficient_food" — indicating the
            # policy sampled an action outside the legal mask) apart from
            # simultaneous-clearing edge cases ("no_order_to_cancel"
            # after another seat's cross ate our order, etc.).
            rejection_code = None if applied_ok else action_result.get("code")
            infos[agent_id] = {
                "tick": step_info.tick,
                "action_applied": action_result,
                "died": died,
                "end_reason": step_info.end_reason,
                "reward_dense": reward_split["dense"],
                "reward_terminal": reward_split["terminal"],
                # `rejected_action` = env couldn't apply what the policy
                # asked. Not necessarily a mask violation — see the
                # callback's per-code breakdown.
                "rejected_action": not applied_ok,
                "rejection_code": rejection_code,
                # Preserved for back-compat with older tests / dashboards.
                "invalid_action": not applied_ok,
            }
        if game_done:
            result = self._env.get_result()
            winners = set(result["winner_idxs"])
            horizon = self.config.game_duration_seconds
            elapsed = result["elapsed_seconds"]

            # v2 terminal rank: 1 = winner (all tied winners share rank 1),
            # then remaining alive-nonwinners sorted by (cash desc, total
            # inventory desc) then dead players (later-death outranks
            # earlier-death). Never used as a training signal — logged for
            # diagnostics.
            def _sort_key(idx):
                p = result["players"][idx]
                alive = p["status"] == "alive"
                inv = sum(p["final_inventory"].values())
                death_time = p["died_at"] if p["died_at"] is not None else horizon
                if idx in winners:
                    tier = 0
                elif alive:
                    tier = 1
                else:
                    tier = 2
                return (tier, -p["final_cash"], -inv, -death_time)

            player_order = sorted(range(self.config.player_count), key=_sort_key)
            terminal_rank: dict[int, int] = {}
            next_rank = 1
            for pos, idx in enumerate(player_order):
                if idx in winners:
                    terminal_rank[idx] = 1
                else:
                    if pos == 0 or player_order[pos - 1] in winners:
                        next_rank = max(next_rank, len(winners) + 1)
                    terminal_rank[idx] = next_rank
                    next_rank += 1

            episode_metrics = {
                "end_reason": result["end_reason"],
                "elapsed_seconds": elapsed,
                "n_trades": result["n_trades"],
                "reached_horizon": elapsed >= horizon,
                "players": [
                    {
                        "win": idx in winners,
                        "survived_to_180": (
                            player["status"] == "alive" and elapsed >= horizon
                        ),
                        "died_before_180": player["status"] == "dead",
                        "final_cash": player["final_cash"],
                        "final_total_inventory": sum(
                            player["final_inventory"].values()
                        ),
                        "death_time": (
                            player["died_at"]
                            if player["died_at"] is not None
                            else elapsed
                        ),
                        "terminal_rank": terminal_rank[idx],
                        "terminal_reward": rewards_by_idx[idx],
                        # Kept for back-compat with dashboards from the v1
                        # archive; equivalent to `survived_to_180` under
                        # v2 termination (game only ends at 180 or 0).
                        "survived": player["status"] == "alive",
                    }
                    for idx, player in enumerate(result["players"])
                ],
            }
            for agent_id in active_before_step:
                infos[agent_id]["episode_metrics"] = episode_metrics
                pidx = self._player_idx(agent_id)
                p = result["players"][pidx]
                infos[agent_id]["died_before_180"] = p["status"] == "dead"
                infos[agent_id]["survived_to_180"] = (
                    p["status"] == "alive" and elapsed >= horizon
                )
                infos[agent_id]["winner"] = pidx in winners
                infos[agent_id]["terminal_rank"] = terminal_rank[pidx]
                infos[agent_id]["terminal_reward"] = rewards_by_idx[pidx]

        self.agents = [
            agent_id
            for agent_id in active_before_step
            if not terminated[agent_id]
        ]
        # Per-agent terminations need a final observation for RLlib's new
        # multi-agent connector, even though the seat should not act again.
        observation_agent_ids = [] if game_done else active_before_step
        observations = self._build_observations(
            raw_observations,
            agent_ids=observation_agent_ids,
        )

        terminated["__all__"] = bool(game_done)
        truncated["__all__"] = False
        return observations, rewards, terminated, truncated, infos

    def render(self):
        return self._env.render_text() if self._env is not None else ""

    def close(self) -> None:
        self._env = None
        self.agents = []
