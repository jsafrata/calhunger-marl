"""Shared reinforcement-learning adapters for the Nash Hunger simulator."""

from .global_observation_encoder import (
    GLOBAL_OBS_DIM,
    encode_global_observation,
)
from .observation_encoder import OBS_DIM, encode_observation
from .rewards import ShapedRewardTracker

__all__ = [
    "GLOBAL_OBS_DIM",
    "OBS_DIM",
    "NashHungerMultiAgentEnv",
    "NashHungerPPOTorchRLModule",
    "ShapedRewardTracker",
    "build_ppo_config",
    "encode_global_observation",
    "encode_observation",
]


def __getattr__(name: str):
    """Keep the encoder importable when optional RLlib dependencies are absent."""
    if name == "NashHungerMultiAgentEnv":
        from .rllib_env import NashHungerMultiAgentEnv

        return NashHungerMultiAgentEnv
    if name == "NashHungerPPOTorchRLModule":
        from .rllib_module import NashHungerPPOTorchRLModule

        return NashHungerPPOTorchRLModule
    if name == "build_ppo_config":
        from .rllib_config import build_ppo_config

        return build_ppo_config
    raise AttributeError(name)
