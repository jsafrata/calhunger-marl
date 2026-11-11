"""Structural tests for `NashHungerPPOTorchRLModule` (post-factored-actor).

Covers:
  - MLP torso: forward + no state
  - LSTM torso: forward + state passthrough
  - `_forward_exploration/_forward_inference/_forward_train` return the
    keys PPO needs (ACTIONS, ACTION_LOGP, ACTION_DIST_INPUTS)
  - Value function uses the right observation stream per algo
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("ray") is None:
    pytest.skip("ray[rllib] is not installed", allow_module_level=True)
if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("torch is not installed", allow_module_level=True)

import gymnasium as gym
import torch
from ray.rllib.core.columns import Columns

from env.actions import NUM_ACTIONS
from env.rl.global_observation_encoder import GLOBAL_OBS_DIM
from env.rl.observation_encoder import OBS_DIM
from env.rl.rllib_module import (
    LSTMTorso,
    MLPTorso,
    NashHungerPPOTorchRLModule,
)


def _spaces():
    observation_space = gym.spaces.Dict({
        "observations": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
        "action_mask": gym.spaces.Box(0.0, 1.0, (NUM_ACTIONS,), np.float32),
        "critic_observations": gym.spaces.Box(
            -np.inf, np.inf, (GLOBAL_OBS_DIM,), np.float32
        ),
    })
    return observation_space, gym.spaces.Discrete(NUM_ACTIONS)


def _module(torso: str = "mlp", algo: str = "mappo"):
    observation_space, action_space = _spaces()
    return NashHungerPPOTorchRLModule(
        observation_space=observation_space,
        action_space=action_space,
        model_config={"torso": torso, "algo": algo},
    )


def _batch(batch_size: int):
    return {Columns.OBS: {
        "observations": torch.randn(batch_size, OBS_DIM),
        "action_mask": torch.ones(batch_size, NUM_ACTIONS),
        "critic_observations": torch.randn(batch_size, GLOBAL_OBS_DIM),
    }}


def _batched_state(module, batch_size: int):
    state = module.get_initial_state()
    return {
        k: torch.from_numpy(v).unsqueeze(0).repeat(batch_size, *([1] * v.ndim))
        for k, v in state.items()
    }


# ---------------------------------------------------------------------------
# Forward-mode shape / key contracts
# ---------------------------------------------------------------------------


def test_mlp_forward_shapes_and_no_state():
    module = _module("mlp")
    batch = _batch(5)
    for forward in (
        module.forward_inference,
        module.forward_exploration,
    ):
        out = forward(batch)
        assert out[Columns.ACTIONS].shape == (5,)
        assert out[Columns.ACTION_LOGP].shape == (5,)
        assert Columns.STATE_OUT not in out
    assert module.compute_values(batch).shape == (5,)


def test_lstm_forward_shapes_and_passes_state():
    module = _module("lstm")
    batch = _batch(3)
    batch[Columns.STATE_IN] = _batched_state(module, 3)
    out = module.forward_inference(batch)
    assert out[Columns.ACTIONS].shape == (3,)
    assert out[Columns.ACTION_LOGP].shape == (3,)
    assert Columns.STATE_OUT in out
    assert set(out[Columns.STATE_OUT]) == {"h", "c"}


def test_forward_train_returns_action_dist_inputs_dict():
    """`_forward_train` computes logp + entropy from previously-sampled
    actions in the batch and hands them to `FactoredActionDistribution`
    via `ACTION_DIST_INPUTS`."""
    module = _module("mlp")
    batch = _batch(5)
    # Sample real actions for the batch first.
    exp_out = module.forward_exploration(batch)
    batch[Columns.ACTIONS] = exp_out[Columns.ACTIONS]

    train_out = module.forward_train(batch)
    dist_inputs = train_out[Columns.ACTION_DIST_INPUTS]
    assert isinstance(dist_inputs, dict)
    assert set(dist_inputs) >= {"logp", "entropy"}
    assert dist_inputs["logp"].shape == (5,)
    assert dist_inputs["entropy"].shape == (5,)
    assert torch.isfinite(dist_inputs["logp"]).all()
    assert torch.isfinite(dist_inputs["entropy"]).all()


# ---------------------------------------------------------------------------
# Value-function stream (regression: MAPPO reads global, IPPO reads local)
# ---------------------------------------------------------------------------


def test_mappo_critic_uses_global_not_local_observations():
    module = _module("mlp", algo="mappo")
    first = _batch(2)
    second = {Columns.OBS: {k: v.clone() for k, v in first[Columns.OBS].items()}}
    second[Columns.OBS]["observations"] += 100.0  # perturb local
    torch.testing.assert_close(
        module.compute_values(first), module.compute_values(second)
    )
    second[Columns.OBS]["critic_observations"] += 100.0  # perturb global
    assert not torch.allclose(
        module.compute_values(first), module.compute_values(second)
    )


def test_ippo_critic_uses_local_not_global_observations():
    module = _module("mlp", algo="ippo")
    first = _batch(2)
    second = {Columns.OBS: {k: v.clone() for k, v in first[Columns.OBS].items()}}
    second[Columns.OBS]["critic_observations"] += 100.0
    torch.testing.assert_close(
        module.compute_values(first), module.compute_values(second)
    )
    second[Columns.OBS]["observations"] += 100.0
    assert not torch.allclose(
        module.compute_values(first), module.compute_values(second)
    )


def test_missing_observation_key_fails_cleanly():
    module = _module("mlp")
    batch = _batch(1)
    del batch[Columns.OBS]["action_mask"]
    with pytest.raises(ValueError, match="missing required keys"):
        module.forward_exploration(batch)


def test_v4_observation_flags_resize_actor_and_critic_inputs():
    from env.rl.global_observation_encoder import global_observation_dim
    from env.rl.observation_encoder import observation_dim

    local_dim = observation_dim(include_survival_features=True)
    global_dim = global_observation_dim(include_trade_history=True)
    observation_space = gym.spaces.Dict({
        "observations": gym.spaces.Box(-np.inf, np.inf, (local_dim,), np.float32),
        "action_mask": gym.spaces.Box(0.0, 1.0, (NUM_ACTIONS,), np.float32),
        "critic_observations": gym.spaces.Box(
            -np.inf, np.inf, (global_dim,), np.float32
        ),
    })
    module = NashHungerPPOTorchRLModule(
        observation_space=observation_space,
        action_space=gym.spaces.Discrete(NUM_ACTIONS),
        model_config={
            "torso": "mlp",
            "algo": "mappo",
            "include_survival_features": True,
            "include_trade_history": True,
        },
    )
    batch = {Columns.OBS: {
        "observations": torch.randn(5, local_dim),
        "action_mask": torch.ones(5, NUM_ACTIONS),
        "critic_observations": torch.randn(5, global_dim),
    }}
    out = module._forward_exploration(batch)
    assert out[Columns.ACTIONS].shape == (5,)
    values = module.compute_values(batch)
    assert values.shape == (5,)
    # A module built WITHOUT the flags must reject the extended spaces.
    with pytest.raises(ValueError, match="observations must have shape"):
        NashHungerPPOTorchRLModule(
            observation_space=observation_space,
            action_space=gym.spaces.Discrete(NUM_ACTIONS),
            model_config={"torso": "mlp", "algo": "mappo"},
        )
