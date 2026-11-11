"""v3 tests — Shared / Shared+ID / Independent parameterization invariants.

Nine required checks per the experiment spec:
  1. Independent policies are separate parameter objects (no shared storage).
  2. Updating pi_i leaves pi_j untouched (j != i).
  3. Trajectories/rewards route only to the correct seat policy.
  4. MAPPO critics get centralized obs, actors get local obs (independent
     preserves this).
  5. Checkpoints save+restore all four policies and their optimizer state.
  6. Policy-to-seat mapping is permutable at eval time.
  7. Shared baseline is behaviorally and configurationally unchanged.
  8. Shared+ID: agent id one-hot is correctly plumbed into obs and grows
     OBS_DIM by exactly 4.
  9. All three conditions complete a short smoke train + deep eval without
     crashing.
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helper: build a config and algo for a given parameterization.
# ---------------------------------------------------------------------------


def _build_algo(*, parameterization: str, algo: str = "ippo",
                include_agent_id: bool = False):
    """Build the smallest workable PPO algo for the given condition.

    ``include_agent_id`` is a condition-local environment/module setting;
    it must not alter the process-global v2 observation contract.
    """
    import ray
    from env.rl.rllib_config import build_ppo_config, register_nash_hunger_env
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False,
             log_to_driver=False)
    register_nash_hunger_env()
    cfg = build_ppo_config(
        seed=0,
        num_env_runners=0,
        algo=algo,
        head_kind="mlp",
        entropy_coeff=0.01,
        gamma=1.0,
        lambda_=1.0,
        train_batch_size=64,
        minibatch_size=32,
        num_epochs=1,
        env_config={"reward_kind": "terminal_only"},
        parameterization=parameterization,
        include_agent_id=include_agent_id,
    )
    return cfg.build_algo()


# ---------------------------------------------------------------------------
# 1 · Independent policies have separate parameter storage
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_independent_policies_have_separate_parameter_storage():
    algo_obj = _build_algo(parameterization="independent")
    try:
        from env.rl.rllib_config import INDEPENDENT_POLICY_IDS
        modules = {pid: algo_obj.get_module(pid) for pid in INDEPENDENT_POLICY_IDS}
        # All parameter tensors distinct across policies.
        for i, pi_i in enumerate(INDEPENDENT_POLICY_IDS):
            for j, pi_j in enumerate(INDEPENDENT_POLICY_IDS):
                if i >= j:
                    continue
                params_i = list(modules[pi_i].parameters())
                params_j = list(modules[pi_j].parameters())
                assert len(params_i) == len(params_j)
                for a, b in zip(params_i, params_j):
                    # Different Parameter objects, and different underlying
                    # storage (no aliasing).
                    assert a is not b
                    assert a.data_ptr() != b.data_ptr(), (
                        f"parameter storage shared between {pi_i} and {pi_j}"
                    )
    finally:
        algo_obj.stop()
        import ray; ray.shutdown()


# ---------------------------------------------------------------------------
# 2 · Updating one policy does not touch others
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_updating_one_independent_policy_does_not_touch_others():
    algo_obj = _build_algo(parameterization="independent")
    try:
        from env.rl.rllib_config import INDEPENDENT_POLICY_IDS
        learner = algo_obj.learner_group._learner
        mods = {pid: learner.module[pid] for pid in INDEPENDENT_POLICY_IDS}
        # Snapshot pi_1..pi_3 params.
        snap = {
            pid: [p.detach().clone() for p in mods[pid].parameters()]
            for pid in INDEPENDENT_POLICY_IDS[1:]
        }
        # Perform a genuine optimizer update through pi_0's registered Adam
        # optimizer. This exercises optimizer-to-module ownership, rather
        # than merely mutating one parameter tensor by hand.
        optimizer = learner.get_optimizer("pi_0")
        optimizer.zero_grad()
        loss = sum((p.square().mean() for p in mods["pi_0"].parameters()))
        loss.backward()
        optimizer.step()
        # pi_1..pi_3 must be byte-identical to their snapshots.
        for pid, snap_params in snap.items():
            new_params = list(mods[pid].parameters())
            for a, b in zip(snap_params, new_params):
                assert torch.equal(a, b), (
                    f"perturbing pi_0 changed {pid} parameters — sharing bug"
                )
    finally:
        algo_obj.stop()
        import ray; ray.shutdown()


# ---------------------------------------------------------------------------
# 3 · Trajectory routing — each seat's data goes only to its own policy
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_trajectory_routing_seat_i_to_pi_i_only():
    from env.rl.rllib_config import independent_policy_mapping_fn
    # The mapping fn itself is the routing rule; verify it directly.
    for seat in range(4):
        aid = f"seat_{seat}"
        assert independent_policy_mapping_fn(aid) == f"pi_{seat}"
    with pytest.raises(ValueError):
        independent_policy_mapping_fn("bogus")

    # Sample a real multi-agent batch and check RLlib's accounting agrees
    # with the mapping: every module receives exactly its own seat's steps.
    algo_obj = _build_algo(parameterization="independent")
    try:
        result = algo_obj.train()
        env_metrics = result["env_runners"]
        by_seat = env_metrics["num_agent_steps_sampled_lifetime"]
        by_module = env_metrics["num_module_steps_sampled_lifetime"]
        for seat in range(4):
            assert by_module[f"pi_{seat}"] == by_seat[f"seat_{seat}"]
    finally:
        algo_obj.stop()
        import ray; ray.shutdown()


# ---------------------------------------------------------------------------
# 4 · MAPPO independent: actors local, critics centralized
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mappo_independent_actors_local_critics_centralized():
    algo_obj = _build_algo(parameterization="independent", algo="mappo")
    try:
        from env.rl.rllib_config import INDEPENDENT_POLICY_IDS
        from env.rl.global_observation_encoder import GLOBAL_OBS_DIM
        from env.rl.observation_encoder import OBS_DIM
        for pid in INDEPENDENT_POLICY_IDS:
            m = algo_obj.get_module(pid)
            # Actor torso reads OBS_DIM-shaped local obs.
            assert m.torso.net[0].in_features == OBS_DIM, (
                f"{pid} actor torso in={m.torso.net[0].in_features}, expected OBS_DIM={OBS_DIM}"
            )
            # Critic reads GLOBAL_OBS_DIM centralized state.
            assert m.critic.input_dim == GLOBAL_OBS_DIM, (
                f"{pid} critic in={m.critic.input_dim}, expected GLOBAL_OBS_DIM={GLOBAL_OBS_DIM}"
            )
            # And by algo attribute:
            assert m._algo == "mappo"
    finally:
        algo_obj.stop()
        import ray; ray.shutdown()


# ---------------------------------------------------------------------------
# 5 · Checkpoint save + restore for all four policies
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_checkpoint_save_and_restore_all_four_policies(tmp_path):
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from env.rl.rllib_config import INDEPENDENT_POLICY_IDS
    algo_obj = _build_algo(parameterization="independent")
    ck_dir = tmp_path / "ck"
    try:
        # Populate Adam moments so optimizer restoration is substantive.
        algo_obj.train()
        # Save & capture pi_0/pi_1 parameter tensors before restore.
        pre = {
            pid: [p.detach().clone() for p in algo_obj.get_module(pid).parameters()]
            for pid in INDEPENDENT_POLICY_IDS
        }
        pre_optimizer = algo_obj.learner_group._learner.get_state()["optimizer"]
        assert set(pre_optimizer) == {
            f"{pid}_default_optimizer" for pid in INDEPENDENT_POLICY_IDS
        }
        assert all(state["state"] for state in pre_optimizer.values())
        ck_dir.mkdir(parents=True, exist_ok=True)
        algo_obj.save(str(ck_dir))
    finally:
        algo_obj.stop()
        ray.shutdown()

    # Fresh restore.
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False,
             log_to_driver=False)
    from env.rl.rllib_config import register_nash_hunger_env
    register_nash_hunger_env()
    restored = Algorithm.from_checkpoint(str(ck_dir.resolve()))
    try:
        for pid in INDEPENDENT_POLICY_IDS:
            m = restored.get_module(pid)
            post = list(m.parameters())
            assert len(post) == len(pre[pid]), pid
            for a, b in zip(pre[pid], post):
                assert torch.allclose(a, b, atol=0), (
                    f"{pid} parameters differ post-restore"
                )
        post_optimizer = restored.learner_group._learner.get_state()["optimizer"]
        assert set(post_optimizer) == set(pre_optimizer)

        def assert_nested_equal(a, b):
            if isinstance(a, dict):
                assert set(a) == set(b)
                for key in a:
                    assert_nested_equal(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    assert_nested_equal(x, y)
            elif torch.is_tensor(a):
                assert torch.equal(a, b)
            elif isinstance(a, __import__("numpy").ndarray):
                assert __import__("numpy").array_equal(a, b)
            elif isinstance(a, float):
                assert __import__("math").isclose(a, b, rel_tol=1e-7, abs_tol=1e-12)
            else:
                assert a == b

        assert_nested_equal(pre_optimizer, post_optimizer)
    finally:
        restored.stop()
        ray.shutdown()


# ---------------------------------------------------------------------------
# 6 · Policy-to-seat mapping is permutable at eval time
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_policy_to_seat_mapping_is_permutable_at_eval_time():
    """The evaluator does not use RLlib's policy_mapping_fn — it drives
    per-seat modules directly. Verify the module registry supports
    permutations by fetching modules for arbitrary seat assignments."""
    algo_obj = _build_algo(parameterization="independent")
    try:
        from env.rl.rllib_config import INDEPENDENT_POLICY_IDS
        # Every permutation of the 4 policy ids must map to valid modules.
        from itertools import permutations
        n_perms_checked = 0
        for perm in permutations(range(4)):
            for seat_i, policy_j in enumerate(perm):
                m = algo_obj.get_module(INDEPENDENT_POLICY_IDS[policy_j])
                assert m is not None
            n_perms_checked += 1
        assert n_perms_checked == 24
    finally:
        algo_obj.stop()
        import ray; ray.shutdown()


# ---------------------------------------------------------------------------
# 7 · Shared baseline is behaviorally + configurationally unchanged
# ---------------------------------------------------------------------------


def test_shared_baseline_configurationally_unchanged():
    """Building a shared config should still produce a SINGLE shared_policy
    with the v2 defaults (gamma=1.0, lambda_=1.0, head_kind=mlp)."""
    from env.rl.rllib_config import (SHARED_POLICY_ID, build_ppo_config,
                                     shared_policy_mapping_fn)
    cfg = build_ppo_config(
        seed=0, num_env_runners=0, algo="ippo",
        train_batch_size=64, minibatch_size=32, num_epochs=1,
        parameterization="shared",
    )
    # Only one policy, shared name, v2 hyperparameters.
    assert cfg.gamma == 1.0
    assert cfg.lambda_ == 1.0
    assert cfg.entropy_coeff == 0.01
    spec = cfg.rl_module_spec
    # In shared mode this is a single RLModuleSpec, not a MultiRLModuleSpec.
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    assert isinstance(spec, RLModuleSpec)
    assert not isinstance(spec, MultiRLModuleSpec)
    model_cfg = getattr(spec, "model_config", {}) or {}
    assert model_cfg.get("head_kind") == "mlp"
    assert model_cfg.get("include_agent_id") is False
    assert cfg.env_config["include_agent_id"] is False


def test_shared_baseline_behavior_unchanged_when_id_switch_is_explicitly_off():
    """The new condition-local ID switch must be a strict no-op for v2.

    Compare both initialized weights and deterministic inference for the
    pre-v3 model config (key absent) against the explicit Shared condition.
    """
    import numpy as np
    from ray.rllib.core.columns import Columns
    from env.rl.rllib_env import NashHungerMultiAgentEnv
    from env.rl.rllib_module import NashHungerPPOTorchRLModule

    wrapped = NashHungerMultiAgentEnv({
        "seed": 7,
        "reward_kind": "terminal_only",
        "include_agent_id": False,
    })
    observations, _ = wrapped.reset(seed=91)
    obs_space = wrapped.observation_spaces["seat_0"]
    action_space = wrapped.action_spaces["seat_0"]
    legacy_model_config = {
        "algo": "ippo", "torso": "mlp", "head_kind": "mlp"
    }
    explicit_model_config = {**legacy_model_config, "include_agent_id": False}

    torch.manual_seed(1234)
    legacy = NashHungerPPOTorchRLModule(
        observation_space=obs_space,
        action_space=action_space,
        model_config=legacy_model_config,
    )
    torch.manual_seed(1234)
    explicit = NashHungerPPOTorchRLModule(
        observation_space=obs_space,
        action_space=action_space,
        model_config=explicit_model_config,
    )
    for legacy_value, explicit_value in zip(
        legacy.state_dict().values(), explicit.state_dict().values()
    ):
        assert torch.equal(legacy_value, explicit_value)

    batch = {
        Columns.OBS: {
            key: torch.from_numpy(np.expand_dims(value, axis=0)).float()
            for key, value in observations["seat_0"].items()
        }
    }
    with torch.no_grad():
        legacy_out = legacy._forward_inference(batch)
        explicit_out = explicit._forward_inference(batch)
    assert torch.equal(
        legacy_out[Columns.ACTIONS], explicit_out[Columns.ACTIONS]
    )


# ---------------------------------------------------------------------------
# 8 · Shared+ID: agent id one-hot plumbed correctly into observation
# ---------------------------------------------------------------------------


def test_shared_baseline_no_agent_id():
    from env.rl.global_observation_encoder import (
        GLOBAL_OBS_DIM,
        global_observation_dim,
    )
    from env.rl.observation_encoder import OBS_DIM, observation_dim
    assert OBS_DIM == observation_dim() == 171
    assert GLOBAL_OBS_DIM == global_observation_dim() == 305


def test_shared_plus_id_adds_four_dims_to_both_obs():
    from env.rl.global_observation_encoder import global_observation_dim
    from env.rl.observation_encoder import observation_dim
    assert observation_dim(include_agent_id=True) == 175
    assert global_observation_dim(include_agent_id=True) == 309


def test_shared_plus_id_appends_correct_one_hot_bits():
    """The last four local/global features identify the absolute seat."""
    from env.env import FoodTradingEnv
    from env.rl.global_observation_encoder import encode_global_observation
    from env.rl.observation_encoder import AGENT_ID_DIM, encode_observation
    env = FoodTradingEnv(seed=0)
    env.reset()
    for i in range(4):
        local = encode_observation(
            env.observe(i), env.config, include_agent_id=True
        )
        global_obs = encode_global_observation(
            env.state, i, env.config, include_agent_id=True
        )
        for vector in (local, global_obs):
            tail = vector[-AGENT_ID_DIM:].tolist()
            assert tail[i] == 1.0 and sum(tail) == 1.0, (i, tail)


def test_shared_plus_id_is_stored_in_config_and_module():
    from env.rl.rllib_config import build_ppo_config
    cfg = build_ppo_config(
        seed=0,
        num_env_runners=0,
        algo="ippo",
        train_batch_size=64,
        minibatch_size=32,
        num_epochs=1,
        include_agent_id=True,
    )
    assert cfg.env_config["include_agent_id"] is True
    assert cfg.rl_module_spec.model_config["include_agent_id"] is True


# ---------------------------------------------------------------------------
# 9 · Smoke: all three conditions complete train + checkpoint + deep eval.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_all_three_conditions_train_and_deep_eval(tmp_path):
    """Each condition (A shared, B shared+ID, C independent) must complete
    2 training iters, checkpoint, and complete native deterministic and
    stochastic deep evaluation without crashing."""
    import ray
    conditions = (
        ("shared", "shared", False),
        ("shared_id", "shared", True),
        ("independent", "independent", False),
    )
    for condition, parameterization, include_id in conditions:
        algo_obj = _build_algo(
            parameterization=parameterization,
            algo="ippo",
            include_agent_id=include_id,
        )
        checkpoint = tmp_path / condition / "checkpoint"
        try:
            for _ in range(2):
                algo_obj.train()
            checkpoint.mkdir(parents=True)
            algo_obj.save(str(checkpoint))
        finally:
            algo_obj.stop()
            ray.shutdown()
        out = tmp_path / condition / "native_eval.json"
        subprocess.check_call([
            sys.executable,
            "-m", "eval.evaluate_shared_vs_independent",
            "--checkpoint", str(checkpoint),
            "--algo", "ippo",
            "--condition", condition,
            "--seed", "0",
            "--out", str(out),
            "--n_games", "2",
            "--sections", "native",
            "--num_cpus", "2",
        ], cwd=REPO_ROOT, timeout=300)
        report = __import__("json").loads(out.read_text())
        assert report["sections"]["native"]["deterministic"]["n_games"] == 2
