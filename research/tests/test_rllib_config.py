from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("ray") is None:
    pytest.skip("ray[rllib] is not installed", allow_module_level=True)

from env.rl.rllib_config import (
    SHARED_POLICY_ID,
    build_ppo_config,
    shared_policy_mapping_fn,
    validate_gpu_request,
)


@pytest.mark.parametrize("torso", ["mlp", "lstm"])
def test_ppo_config_builds_with_shared_module(torso):
    config = build_ppo_config(
        torso=torso,
        num_env_runners=0,
        num_gpus=0,
        train_batch_size=128,
        minibatch_size=64,
        num_epochs=1,
    )

    assert set(config.policies) == {SHARED_POLICY_ID}
    assert shared_policy_mapping_fn("seat_0") == SHARED_POLICY_ID
    assert shared_policy_mapping_fn("seat_3") == SHARED_POLICY_ID
    assert (
        config.rl_module_spec.model_config["torso"]
        == torso
    )


def test_gpu_request_fails_cleanly_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="cannot access CUDA"):
        validate_gpu_request(1)


def test_episode_matched_batch_holds_games_per_update_constant():
    from env.rl.rllib_config import episode_matched_batch

    for duration in (30, 60, 90, 120, 150, 180):
        batch, minibatch = episode_matched_batch(
            duration, games_per_update=45
        )
        assert batch == 45 * duration
        assert minibatch == batch // 4
    with pytest.raises(ValueError):
        episode_matched_batch(0, games_per_update=45)
    with pytest.raises(ValueError):
        episode_matched_batch(90, games_per_update=0)


def test_linear_anneal_schedule_format():
    from env.rl.rllib_config import linear_anneal_schedule

    schedule = linear_anneal_schedule(0.01, 0.001, 4_915_200)
    assert schedule == [[0, 0.01], [4_915_200, 0.001]]
    with pytest.raises(ValueError):
        linear_anneal_schedule(0.01, 0.001, 0)


def test_ppo_config_accepts_entropy_schedule_and_v4_observation_flags():
    from env.rl.rllib_config import linear_anneal_schedule

    schedule = linear_anneal_schedule(0.01, 0.001, 100_000)
    config = build_ppo_config(
        num_env_runners=0,
        num_gpus=0,
        train_batch_size=128,
        minibatch_size=64,
        num_epochs=1,
        entropy_coeff=schedule,
        include_survival_features=True,
        include_trade_history=True,
    )

    assert config.entropy_coeff == schedule
    model_config = config.rl_module_spec.model_config
    assert model_config["include_survival_features"] is True
    assert model_config["include_trade_history"] is True
    assert config.env_config["include_survival_features"] is True
    assert config.env_config["include_trade_history"] is True
