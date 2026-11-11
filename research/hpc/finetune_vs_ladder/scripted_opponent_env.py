"""RLlib env: one learned seat vs three cadenced scripted bots.

Capacity-control experiment env. Each episode:

- the *focal* seat (the only RLlib agent) is drawn uniformly from
  {0..3} (or rotates deterministically with ``focal_selection="rotate"``),
- the other three seats are driven internally by ``CadencedScripted``
  ("fast" cadence — the standard eval cadence from
  ``eval/checkpoint_scripted_diagnostics.py``), all three the SAME
  archetype, drawn uniformly per episode from {v2, v3, v4, v5},
- scripted seeds are ``episode_seed * 17 + seat`` — the exact per-seat
  convention used by the eval harnesses (checkpoint_policy_crossplay,
  scripted_version_tournament).

Implementation is by COMPOSITION around ``NashHungerMultiAgentEnv``: the
wrapped env computes observations, action masks, rewards (terminal-only,
-1 on the death step / 0 / +1 at game end) and infos exactly as during
self-play training; this wrapper only injects the three scripted seats'
actions into the inner ``step()`` and filters every output down to the
focal agent. The MAPPO centralized-critic observation therefore has the
identical shape and contract as at training time (the global encoder
reads the full game state, scripted seats included).

Episode termination: the RLlib episode ends when the focal seat
terminates (death step, or game end at the full horizon). When the focal
seat dies before the horizon we do not simulate the remaining scripted
seats further — the focal seat's terminal reward (-1) has already been
delivered on its death step, matching training-time semantics exactly.
As in the training env, a focal death before game end still emits a
final observation; a game-end termination does not.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from agents.cadenced_scripted import CadencedScripted
from agents.scripted_v2 import ScriptedV2Agent
from agents.scripted_v3 import ScriptedV3Agent
from agents.scripted_v4 import ScriptedV4Agent
from agents.scripted_v5 import ScriptedV5Agent
from env.rl.callbacks import NashHungerCallbacks
from env.rl.rllib_env import NashHungerMultiAgentEnv

# The eval ladder: same archetypes checkpoint_scripted_diagnostics scores
# against (v1 is the frozen live-game port and not part of this control).
SCRIPTED_CLASSES = {
    "v2": ScriptedV2Agent,
    "v3": ScriptedV3Agent,
    "v4": ScriptedV4Agent,
    "v5": ScriptedV5Agent,
}
ARCHETYPES = tuple(sorted(SCRIPTED_CLASSES))

FINETUNE_ENV_NAME = "nash-hunger-scripted-opponent-v0"

# Wrapper-only env_config keys, stripped before constructing the inner env.
_WRAPPER_KEYS = ("focal_selection", "cadence")


class ScriptedOpponentEnv(MultiAgentEnv):
    """One RLlib seat per episode; three same-archetype scripted seats.

    ``env_config`` accepts everything ``NashHungerMultiAgentEnv`` accepts
    (``seed``, ``reward_kind``, ``env_variant``, ``include_survival_features``,
    ``include_trade_history``, ...) plus:

    - ``focal_selection``: ``"random"`` (default, uniform over seats) or
      ``"rotate"`` (deterministic cycle 0,1,2,3,...).
    - ``cadence``: scripted decision cadence, default ``"fast"``.
    """

    def __init__(self, env_config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        options = dict(env_config or {})
        self.focal_selection = str(options.get("focal_selection", "random"))
        if self.focal_selection not in ("random", "rotate"):
            raise ValueError(
                "focal_selection must be 'random' or 'rotate', got "
                f"{self.focal_selection!r}"
            )
        self.cadence = str(options.get("cadence", "fast"))
        self.default_seed = int(options.get("seed", 0))

        inner_options = {
            key: value
            for key, value in options.items()
            if key not in _WRAPPER_KEYS
        }
        self._inner = NashHungerMultiAgentEnv(inner_options)
        self.config = self._inner.config

        # Same per-seat space contract as the training env; RLlib supports
        # a per-episode subset of possible_agents being active.
        self.possible_agents = list(self._inner.possible_agents)
        self.agents: list[str] = []
        self.observation_spaces = dict(self._inner.observation_spaces)
        self.action_spaces = dict(self._inner.action_spaces)

        # Lineup RNG drives (episode_seed, focal seat, archetype); the
        # inner env is re-seeded per episode with the drawn episode_seed,
        # so the whole episode stream is a pure function of `seed`.
        self._lineup_rng = np.random.default_rng(self.default_seed)
        self._episode_index = -1
        self._focal_idx: Optional[int] = None
        self._focal_agent: Optional[str] = None
        self._archetype: Optional[str] = None
        self._scripted: dict[int, CadencedScripted] = {}

    # ------------------------------------------------------------------

    @property
    def focal_seat(self) -> Optional[int]:
        return self._focal_idx

    @property
    def opponent_archetype(self) -> Optional[str]:
        return self._archetype

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.default_seed = int(seed)
            self._lineup_rng = np.random.default_rng(self.default_seed)
        self._episode_index += 1

        episode_seed = int(
            self._lineup_rng.integers(0, np.iinfo(np.int32).max)
        )
        if self.focal_selection == "rotate":
            focal_idx = self._episode_index % self.config.player_count
        else:
            focal_idx = int(
                self._lineup_rng.integers(0, self.config.player_count)
            )
        archetype = ARCHETYPES[
            int(self._lineup_rng.integers(0, len(ARCHETYPES)))
        ]

        inner_obs, inner_infos = self._inner.reset(seed=episode_seed)

        self._focal_idx = focal_idx
        self._focal_agent = f"seat_{focal_idx}"
        self._archetype = archetype
        self._scripted = {}
        for seat in range(self.config.player_count):
            if seat == focal_idx:
                continue
            bot = CadencedScripted(
                self.cadence,
                seed=episode_seed * 17 + seat,
                inner_factory=SCRIPTED_CLASSES[archetype],
            )
            bot.reset()
            self._scripted[seat] = bot

        self.agents = [self._focal_agent]
        observations = {self._focal_agent: inner_obs[self._focal_agent]}
        info = dict(inner_infos.get(self._focal_agent, {}))
        info["focal_seat"] = focal_idx
        info["opponent_archetype"] = archetype
        return observations, {self._focal_agent: info}

    def step(self, action_dict: Mapping[str, int]):
        if self._focal_idx is None or self._focal_agent is None:
            raise RuntimeError("reset() must be called before step()")
        extraneous = set(action_dict) - {self._focal_agent}
        if extraneous:
            raise ValueError(
                "only the focal seat is an RLlib agent; got actions for "
                f"{sorted(extraneous)}"
            )

        game_env = self._inner._env
        assert game_env is not None
        inner_actions: dict[str, int] = {
            agent_id: int(action)
            for agent_id, action in action_dict.items()
        }
        for seat, bot in self._scripted.items():
            agent_id = f"seat_{seat}"
            if agent_id not in self._inner.agents:
                continue  # scripted seat died earlier; inner NOOPs it
            inner_actions[agent_id] = int(
                bot.act(game_env.observe(seat), game_env.legal_actions(seat))
            )

        (
            inner_obs,
            inner_rewards,
            inner_terminated,
            inner_truncated,
            inner_infos,
        ) = self._inner.step(inner_actions)

        focal_done = bool(inner_terminated.get(self._focal_agent, False))
        game_done = bool(inner_terminated.get("__all__", False))

        observations = (
            {self._focal_agent: inner_obs[self._focal_agent]}
            if self._focal_agent in inner_obs
            else {}
        )
        rewards = {self._focal_agent: inner_rewards[self._focal_agent]}
        info = dict(inner_infos.get(self._focal_agent, {}))
        info["focal_seat"] = self._focal_idx
        info["opponent_archetype"] = self._archetype
        if focal_done:
            # game_done with the focal seat alive implies the full horizon
            # was reached (v2 termination: the game only ends at the
            # horizon or with zero players alive).
            info["focal_survived"] = bool(
                info.get("survived_to_180", False) if game_done else False
            )
        infos = {self._focal_agent: info}

        terminated = {self._focal_agent: focal_done, "__all__": focal_done}
        truncated = {
            self._focal_agent: bool(
                inner_truncated.get(self._focal_agent, False)
            ),
            "__all__": False,
        }
        if focal_done:
            self.agents = []
        return observations, rewards, terminated, truncated, infos

    def render(self):
        return self._inner.render()

    def close(self) -> None:
        self._inner.close()
        self.agents = []


class FinetuneVsLadderCallbacks(NashHungerCallbacks):
    """Standard training telemetry + focal-seat / opponent-mix metrics.

    Adds, on top of NashHungerCallbacks:

    - ``focal/survival_rate``   — fraction of episodes the focal seat
      survived to the horizon (the primary fine-tuning signal),
    - ``focal/win_rate``        — fraction of episodes it won,
    - ``focal/terminal_reward`` — mean terminal reward (+1/0/-1),
    - ``opponents/<arch>_episodes`` and ``focal/seat_<i>_episodes`` —
      per-iteration episode counts (sum, cleared each reduce) proving the
      archetype and seat draws stay uniform,
    - ``focal/survival_rate_<arch>`` — per-archetype survival.

    Outcomes are derived from the episode RETURN, not the terminal info:
    the env follows the training env's convention of emitting no final
    observation on the game_done step, so RLlib drops that step's info
    dict; and with terminal-only rewards any episode(-chunk) return equals
    the terminal reward exactly (-1 died / 0 survived / +1 won).
    """

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        super().on_episode_end(
            episode=episode, metrics_logger=metrics_logger, **kwargs
        )
        if metrics_logger is None:
            return
        terminal_reward = float(episode.get_return())
        survived = terminal_reward >= 0.0
        won = terminal_reward > 0.0
        # Lineup identity: the focal seat is the only agent with infos, and
        # every step info (including reset's) carries these keys — robust
        # to the terminal info being dropped and to episode chunking.
        archetype = None
        focal_seat = None
        for agent_id in self._AGENT_IDS:
            for info in self._all_infos_for_agent(episode, agent_id):
                if "opponent_archetype" in info:
                    archetype = info["opponent_archetype"]
                    focal_seat = info.get("focal_seat")
                    break
            if archetype is not None:
                break

        metrics_logger.log_value(
            "focal/survival_rate", float(survived), reduce="mean"
        )
        metrics_logger.log_value("focal/win_rate", float(won), reduce="mean")
        metrics_logger.log_value(
            "focal/terminal_reward", terminal_reward, reduce="mean"
        )
        if archetype is not None:
            metrics_logger.log_value(
                f"opponents/{archetype}_episodes",
                1,
                reduce="sum",
                clear_on_reduce=True,
            )
            metrics_logger.log_value(
                f"focal/survival_rate_{archetype}",
                float(survived),
                reduce="mean",
            )
        if focal_seat is not None:
            metrics_logger.log_value(
                f"focal/seat_{int(focal_seat)}_episodes",
                1,
                reduce="sum",
                clear_on_reduce=True,
            )
