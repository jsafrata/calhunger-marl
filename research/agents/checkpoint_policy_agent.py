from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns

from env.config import DEFAULT_CONFIG
from env.rl.global_observation_encoder import GLOBAL_OBS_DIM
from env.rl.observation_encoder import encode_observation
from env.rl.rllib_config import SHARED_POLICY_ID, register_nash_hunger_env

from .base import Agent


_ALGORITHM_CACHE: dict[str, Algorithm] = {}
_MODULE_CACHE: dict[str, Any] = {}
_RAY_STARTED = False


def _ensure_ray_started() -> None:
    global _RAY_STARTED
    if _RAY_STARTED:
        return
    import ray

    register_nash_hunger_env()
    ray.init(ignore_reinit_error=True)
    _RAY_STARTED = True


def _load_module(checkpoint_path: Path):
    key = str(checkpoint_path.expanduser().resolve())
    if key not in _MODULE_CACHE:
        _ensure_ray_started()
        algorithm = Algorithm.from_checkpoint(key)
        _ALGORITHM_CACHE[key] = algorithm
        _MODULE_CACHE[key] = algorithm.get_module(SHARED_POLICY_ID)
    return _MODULE_CACHE[key]


def cleanup_checkpoint_policy_cache() -> None:
    global _RAY_STARTED
    for algorithm in _ALGORITHM_CACHE.values():
        algorithm.stop()
    _ALGORITHM_CACHE.clear()
    _MODULE_CACHE.clear()
    if _RAY_STARTED:
        import ray

        ray.shutdown()
        _RAY_STARTED = False


atexit.register(cleanup_checkpoint_policy_cache)


class CheckpointPolicyAgent(Agent):
    """Agent wrapper around an RLlib checkpoint's shared Nash Hunger policy."""

    def __init__(
        self,
        *,
        label: str,
        checkpoint_path: Path,
        stochastic: bool = False,
        seed: int = 0,
    ) -> None:
        self.name = f"policy:{label}"
        self.module = _load_module(checkpoint_path)
        self.module.eval()
        self.stochastic = stochastic
        self.device = next(self.module.parameters()).device
        self.generator = torch.Generator(
            device=self.device.type,
        ).manual_seed(seed)
        self.states: dict[int, dict[str, torch.Tensor]] = {}

    def reset(self) -> None:
        self.states = {}

    def act(self, obs: Dict[str, Any], legal_mask: np.ndarray) -> int:
        player_idx = int(obs["player_idx"])
        observation = {
            "observations": torch.from_numpy(
                encode_observation(obs, DEFAULT_CONFIG)
            ).unsqueeze(0).to(self.device),
            "action_mask": torch.from_numpy(
                legal_mask.astype(np.float32)
            ).unsqueeze(0).to(self.device),
            "critic_observations": torch.zeros(
                (1, GLOBAL_OBS_DIM),
                dtype=torch.float32,
                device=self.device,
            ),
        }
        batch = {Columns.OBS: observation}
        initial = (
            self.module.get_initial_state()
            if hasattr(self.module, "get_initial_state")
            else {}
        )
        if initial:
            if player_idx not in self.states:
                self.states[player_idx] = {
                    key: torch.from_numpy(value).unsqueeze(0).to(self.device)
                    for key, value in initial.items()
                }
            batch[Columns.STATE_IN] = self.states[player_idx]

        with torch.no_grad():
            output = self.module.forward_inference(batch)
        logits = output[Columns.ACTION_DIST_INPUTS][0]
        if self.stochastic:
            probabilities = torch.softmax(logits, dim=-1)
            action = torch.multinomial(
                probabilities,
                1,
                generator=self.generator,
            ).item()
        else:
            action = torch.argmax(logits).item()
        if Columns.STATE_OUT in output:
            self.states[player_idx] = {
                key: value.detach()
                for key, value in output[Columns.STATE_OUT].items()
            }
        return int(action)
