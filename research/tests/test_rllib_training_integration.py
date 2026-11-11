from __future__ import annotations

import importlib.util
import os

import pytest

if importlib.util.find_spec("ray") is None:
    pytest.skip("ray[rllib] is not installed", allow_module_level=True)
if os.environ.get("NASH_HUNGER_RUN_RLLIB_INTEGRATION") != "1":
    pytest.skip(
        "set NASH_HUNGER_RUN_RLLIB_INTEGRATION=1 for trainer integration",
        allow_module_level=True,
    )

import ray
from ray.rllib.algorithms.algorithm import Algorithm

from env.rl.rllib_config import SHARED_POLICY_ID, build_ppo_config


@pytest.mark.parametrize("torso", ["mlp", "lstm"])
def test_minimal_cpu_iteration_checkpoint_and_restore(tmp_path, torso):
    ray.init(ignore_reinit_error=True)
    algorithm = None
    restored = None
    try:
        # Recurrent batches are minibatched along the sequence axis with
        # n_seqs * minibatch/len(batch) sequences per slice — a tiny batch
        # with a small minibatch fraction rounds that to ZERO sequences and
        # the module receives an empty batch (Ray 2.53, see the TODO in
        # ray.rllib.utils.minibatch_utils). Campaign-sized batches are far
        # from the cliff; the test sizes below stay above it too.
        train_batch_size, minibatch_size = (360, 180) if torso == "lstm" else (64, 32)
        config = build_ppo_config(
            torso=torso,
            num_env_runners=0,
            num_gpus=0,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=1,
            evaluation_interval=None,
        )
        if hasattr(config, "build_algo"):
            algorithm = config.build_algo()
        else:
            algorithm = config.build()
        result = algorithm.train()
        assert result

        saved = algorithm.save(str(tmp_path))
        candidate = getattr(saved, "checkpoint", None) or saved
        checkpoint = (
            getattr(candidate, "path", None)
            or getattr(saved, "path", None)
            or str(candidate)
        )
        restored = Algorithm.from_checkpoint(checkpoint)
        module = restored.get_module(SHARED_POLICY_ID)

        # Torso is now an nn.Module, not a string. Assert by type instead.
        from env.rl.rllib_module import MLPTorso, LSTMTorso
        expected_cls = MLPTorso if torso == "mlp" else LSTMTorso
        assert isinstance(module.torso, expected_cls)
    finally:
        if restored is not None:
            restored.stop()
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


def test_v4_flags_schedule_and_scaled_variant_train_end_to_end(tmp_path):
    """One real PPO iteration with every v4 knob on: survival features,
    critic trade history, an entropy-coefficient schedule, and a scaled
    environment variant — the exact combination a new campaign would run."""
    from env.rl.rllib_config import episode_matched_batch, linear_anneal_schedule

    ray.init(ignore_reinit_error=True)
    algorithm = None
    try:
        train_batch_size, minibatch_size = episode_matched_batch(
            90, games_per_update=1
        )
        config = build_ppo_config(
            num_env_runners=0,
            num_gpus=0,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=1,
            env_config={"env_variant": "small_90", "reward_kind": "terminal_only"},
            entropy_coeff=linear_anneal_schedule(0.01, 0.001, 10 * train_batch_size),
            include_survival_features=True,
            include_trade_history=True,
        )
        algorithm = config.build_algo()
        result = algorithm.train()
        assert result
        module = algorithm.get_module(SHARED_POLICY_ID)
        from env.rl.global_observation_encoder import global_observation_dim
        from env.rl.observation_encoder import observation_dim
        assert module._actor_obs_dim == observation_dim(
            include_survival_features=True
        )
        assert module.critic.input_dim == global_observation_dim(
            include_trade_history=True
        )
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
