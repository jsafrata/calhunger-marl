"""Structural tests for the IPPO vs MAPPO algorithm switch and the MLP torso.

These tests verify the invariants the training team committed to before
launching the baseline comparison:

  1. `torso="mlp"` builds a stateless feed-forward torso; `torso="lstm"`
     builds a recurrent one. Default is `mlp`.
  2. `algo="ippo"` wires the critic to the local observation (OBS_DIM=171).
  3. `algo="mappo"` wires the critic to the centralized observation
     (GLOBAL_OBS_DIM=305).
  4. Actor forward path is IDENTICAL for both algorithms on the same
     inputs (with identical initial weights) — the ONLY difference must
     be in the critic.
  5. Critic value output changes when the critic's input changes;
     specifically, altering `critic_observations` moves MAPPO's value
     but not IPPO's, and altering `observations` moves IPPO's value but
     not MAPPO's.
"""
from __future__ import annotations

import importlib.util
import pytest

if importlib.util.find_spec("ray") is None:
    pytest.skip("ray[rllib] is not installed", allow_module_level=True)

import gymnasium as gym
import numpy as np
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


def _obs_space():
    return gym.spaces.Dict({
        "observations": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
        "action_mask": gym.spaces.Box(0.0, 1.0, (NUM_ACTIONS,), np.float32),
        "critic_observations": gym.spaces.Box(
            -np.inf, np.inf, (GLOBAL_OBS_DIM,), np.float32
        ),
    })


def _build(algo: str, torso: str = "mlp", seed: int = 0):
    torch.manual_seed(seed)
    return NashHungerPPOTorchRLModule(
        observation_space=_obs_space(),
        action_space=gym.spaces.Discrete(NUM_ACTIONS),
        model_config={"algo": algo, "torso": torso},
    )


def _random_batch(B: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {Columns.OBS: {
        "observations": torch.tensor(
            rng.standard_normal((B, OBS_DIM)).astype(np.float32)
        ),
        "action_mask": torch.ones(B, NUM_ACTIONS),  # all legal for a clean test
        "critic_observations": torch.tensor(
            rng.standard_normal((B, GLOBAL_OBS_DIM)).astype(np.float32)
        ),
    }}


# ---------------------------------------------------------------------------
# Torso construction
# ---------------------------------------------------------------------------


def test_default_torso_is_mlp():
    m = NashHungerPPOTorchRLModule(
        observation_space=_obs_space(),
        action_space=gym.spaces.Discrete(NUM_ACTIONS),
        model_config={},
    )
    assert isinstance(m.torso, MLPTorso), (
        "default torso must be MLP for the first IPPO/MAPPO baselines"
    )
    assert m._is_recurrent is False
    assert m.get_initial_state() == {}


def test_torso_lstm_selectable():
    m = NashHungerPPOTorchRLModule(
        observation_space=_obs_space(),
        action_space=gym.spaces.Discrete(NUM_ACTIONS),
        model_config={"torso": "lstm"},
    )
    assert isinstance(m.torso, LSTMTorso)
    assert m._is_recurrent is True
    state = m.get_initial_state()
    assert set(state) == {"h", "c"}


def test_unknown_torso_raises():
    with pytest.raises(ValueError, match="torso"):
        NashHungerPPOTorchRLModule(
            observation_space=_obs_space(),
            action_space=gym.spaces.Discrete(NUM_ACTIONS),
            model_config={"torso": "gru"},
        )


# ---------------------------------------------------------------------------
# Algorithm switch
# ---------------------------------------------------------------------------


def test_ippo_critic_reads_local_observation():
    m = _build("ippo")
    assert m._critic_obs_key == "observations"
    assert m.critic.input_dim == OBS_DIM


def test_mappo_critic_reads_centralized_observation():
    m = _build("mappo")
    assert m._critic_obs_key == "critic_observations"
    assert m.critic.input_dim == GLOBAL_OBS_DIM


def test_unknown_algo_raises():
    with pytest.raises(ValueError, match="algo"):
        NashHungerPPOTorchRLModule(
            observation_space=_obs_space(),
            action_space=gym.spaces.Discrete(NUM_ACTIONS),
            model_config={"algo": "mixed"},
        )


# ---------------------------------------------------------------------------
# Actor observations are identical for IPPO and MAPPO
# ---------------------------------------------------------------------------


def test_actor_forward_path_identical_across_algos():
    """With the same torso + actor weights, the sampled action + logp under
    IPPO and MAPPO must match bit-for-bit on the same input. Any drift
    would mean the algo switch accidentally touched the actor path."""
    torch.manual_seed(42)
    ippo = _build("ippo", seed=42)
    torch.manual_seed(42)
    mappo = _build("mappo", seed=42)

    # Sanity: torso + actor weights identical (same seed, same class shapes).
    for (n_a, p_a), (n_b, p_b) in zip(
        ippo.torso.named_parameters(), mappo.torso.named_parameters()
    ):
        assert n_a == n_b
        assert torch.equal(p_a, p_b), f"torso param {n_a} drifted between algos"
    for (n_a, p_a), (n_b, p_b) in zip(
        ippo.actor.named_parameters(), mappo.actor.named_parameters()
    ):
        assert n_a == n_b
        assert torch.equal(p_a, p_b), f"actor param {n_a} drifted between algos"

    batch = _random_batch(B=4, seed=7)
    torch.manual_seed(0)  # sync any downstream sampling RNG
    out_ippo = ippo._forward_inference(batch)  # deterministic path
    torch.manual_seed(0)
    out_mappo = mappo._forward_inference(batch)

    assert torch.equal(out_ippo[Columns.ACTIONS], out_mappo[Columns.ACTIONS])
    assert torch.allclose(
        out_ippo[Columns.ACTION_LOGP], out_mappo[Columns.ACTION_LOGP]
    )


# ---------------------------------------------------------------------------
# Critic INPUT differs — verified via a "perturb only one obs stream" test
# ---------------------------------------------------------------------------


def test_ippo_value_reacts_to_local_obs_only():
    """Perturbing `observations` must change IPPO's value; perturbing
    `critic_observations` must NOT change it (IPPO ignores the centralized
    stream by construction)."""
    m = _build("ippo", seed=99)
    batch = _random_batch(B=4, seed=1)
    v0 = m.compute_values(batch)

    # Perturb centralized obs — IPPO must ignore.
    perturbed = {Columns.OBS: {**batch[Columns.OBS]}}
    perturbed[Columns.OBS]["critic_observations"] = (
        batch[Columns.OBS]["critic_observations"] + 100.0
    )
    v1 = m.compute_values(perturbed)
    assert torch.allclose(v0, v1), "IPPO critic must ignore critic_observations"

    # Perturb local obs — IPPO must respond.
    perturbed = {Columns.OBS: {**batch[Columns.OBS]}}
    perturbed[Columns.OBS]["observations"] = (
        batch[Columns.OBS]["observations"] + 100.0
    )
    v2 = m.compute_values(perturbed)
    assert not torch.allclose(v0, v2), (
        "IPPO critic didn't respond to a change in local observations"
    )


def test_mappo_value_reacts_to_centralized_obs_only():
    m = _build("mappo", seed=101)
    batch = _random_batch(B=4, seed=2)
    v0 = m.compute_values(batch)

    # Perturb local obs — MAPPO must ignore.
    perturbed = {Columns.OBS: {**batch[Columns.OBS]}}
    perturbed[Columns.OBS]["observations"] = (
        batch[Columns.OBS]["observations"] + 100.0
    )
    v1 = m.compute_values(perturbed)
    assert torch.allclose(v0, v1), (
        "MAPPO critic must ignore local observations"
    )

    # Perturb centralized obs — MAPPO must respond.
    perturbed = {Columns.OBS: {**batch[Columns.OBS]}}
    perturbed[Columns.OBS]["critic_observations"] = (
        batch[Columns.OBS]["critic_observations"] + 100.0
    )
    v2 = m.compute_values(perturbed)
    assert not torch.allclose(v0, v2), (
        "MAPPO critic didn't respond to a change in critic_observations"
    )
