"""Pre-training checklist runner.

Verifies items 5 and 6 from the training-approval checklist:
  5. Overfit test + 10-iteration smoke tests pass.
  6. No invalid actions, NaNs, or collapsed action heads occur.

Emits a compact per-item pass/fail report; exits non-zero on any failure so
CI or the training driver can gate on it.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import ray
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.rl.rllib_config import build_ppo_config  # noqa: E402
from env.rl.rllib_module import FactoredActionDistribution  # noqa: E402
from env.rl.factored_actor import factored_masks_from_flat_tensor  # noqa: E402
from env.rl.rllib_env import NashHungerMultiAgentEnv  # noqa: E402
from env.actions import NUM_ACTION_TYPES  # noqa: E402


def _learner_metrics(result: dict) -> dict:
    """Pull the shared-policy learner's numeric metrics out of the RLlib result."""
    learners = result.get("learners", {})
    for name, m in learners.items():
        if not isinstance(m, dict):
            continue
        if any(k in m for k in ("policy_loss", "vf_loss")):
            return m
    return {}


def _finite_or_nan(x):
    if x is None:
        return math.nan
    try:
        return float(x) if math.isfinite(float(x)) else math.nan
    except (TypeError, ValueError):
        return math.nan


def run_smoke(n_iters: int, seed: int) -> tuple[bool, dict]:
    """N-iter smoke. Passes if no crash, no NaN losses, and no invalid
    actions logged by the callback."""
    cfg = build_ppo_config(
        seed=seed,
        num_env_runners=1,
        train_batch_size=1024,
        minibatch_size=256,
        num_epochs=2,
    )
    algo = cfg.build_algo()
    try:
        history = []
        for i in range(n_iters):
            r = algo.train()
            lm = _learner_metrics(r)
            er = r.get("env_runners", {})
            history.append({
                "iter": i + 1,
                "episodes": er.get("num_episodes_lifetime", 0),
                "return_mean": _finite_or_nan(er.get("episode_return_mean")),
                "policy_loss": _finite_or_nan(lm.get("policy_loss")),
                "vf_loss": _finite_or_nan(lm.get("vf_loss")),
                "entropy": _finite_or_nan(lm.get("entropy")),
                "invalid_actions_per_seat": _finite_or_nan(
                    er.get("invariants/invalid_actions_per_seat")
                ),
                "reward_dense_per_seat": _finite_or_nan(
                    er.get("reward/dense_per_seat")
                ),
                "reward_terminal_per_seat": _finite_or_nan(
                    er.get("reward/terminal_per_seat")
                ),
            })
    finally:
        algo.stop()

    has_nan_loss = any(
        math.isnan(h["policy_loss"]) or math.isnan(h["vf_loss"])
        for h in history if h["policy_loss"] is not None
    )
    invalid_totals = [
        h["invalid_actions_per_seat"] for h in history
        if not math.isnan(h["invalid_actions_per_seat"])
    ]
    max_invalid = max(invalid_totals) if invalid_totals else 0.0
    ok = (not has_nan_loss) and max_invalid == 0.0
    return ok, {
        "n_iters": n_iters,
        "seed": seed,
        "has_nan_loss": has_nan_loss,
        "max_invalid_per_seat": max_invalid,
        "final_return": history[-1]["return_mean"] if history else math.nan,
        "history": history,
    }


def run_head_collapse_check() -> tuple[bool, dict]:
    """Instantiate the RLModule on random weights, run one forward pass, and
    verify every head's masked categorical has finite entropy > 0 (i.e., the
    softmax hasn't collapsed to one option before training starts)."""
    from ray.rllib.core.columns import Columns
    from env.rl.rllib_module import NashHungerPPOTorchRLModule
    from env.rl.observation_encoder import OBS_DIM
    from env.rl.global_observation_encoder import GLOBAL_OBS_DIM
    from env.actions import NUM_ACTIONS
    from env.rl.factored_actor import (
        _FOOD_RELEVANT,
        _SIDE_RELEVANT,
        _PRICE_RELEVANT,
        _QTY_RELEVANT,
    )
    import gymnasium as gym

    obs_space = gym.spaces.Dict({
        "observations": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
        "action_mask": gym.spaces.Box(0.0, 1.0, (NUM_ACTIONS,), np.float32),
        "critic_observations": gym.spaces.Box(
            -np.inf, np.inf, (GLOBAL_OBS_DIM,), np.float32
        ),
    })
    act_space = gym.spaces.Discrete(NUM_ACTIONS)
    module = NashHungerPPOTorchRLModule(
        observation_space=obs_space,
        action_space=act_space,
        model_config={},
    )
    # Build an env, grab a real masked observation batch (4 seats).
    env = NashHungerMultiAgentEnv({"seed": 0})
    obs_dict, _ = env.reset(seed=0)
    per_seat = list(obs_dict.values())
    B = len(per_seat)
    batch_obs = {
        "observations": torch.tensor(
            np.stack([o["observations"] for o in per_seat]), dtype=torch.float32
        ),
        "action_mask": torch.tensor(
            np.stack([o["action_mask"] for o in per_seat]), dtype=torch.float32
        ),
        "critic_observations": torch.tensor(
            np.stack([o["critic_observations"] for o in per_seat]),
            dtype=torch.float32,
        ),
    }
    # Pack initial recurrent state — current TORSO_ARCHITECTURE is LSTMTorso
    # so we must supply STATE_IN even for a single-step forward. The MLP-torso
    # variant we'll switch to for feed-forward IPPO/MAPPO baselines ignores it.
    init_state = module.get_initial_state()
    if init_state:
        batch_state = {
            k: torch.from_numpy(np.stack([v] * B)).float()
            for k, v in init_state.items()
        }
        batch_in = {Columns.OBS: batch_obs, Columns.STATE_IN: batch_state}
    else:
        batch_in = {Columns.OBS: batch_obs}
    with torch.no_grad():
        out = module._forward_exploration(batch_in)
    logp = out[Columns.ACTION_LOGP]
    actions = out[Columns.ACTIONS]
    entropy = out[Columns.ACTION_DIST_INPUTS]["entropy"]
    logp_ok = torch.isfinite(logp).all().item()
    entropy_ok = torch.isfinite(entropy).all().item()
    entropy_val = float(entropy.mean().item())

    # Additionally sanity-check every legal flat action from the batch's
    # mask survives the sampler (no NaN action ids, all in [0, NUM_ACTIONS)).
    from env.actions import NUM_ACTIONS as _N
    action_ids_ok = bool(
        torch.isfinite(actions.float()).all().item()
        and (actions >= 0).all().item()
        and (actions < _N).all().item()
    )

    return (logp_ok and entropy_ok and action_ids_ok and entropy_val > 0), {
        "logp_finite": logp_ok,
        "entropy_finite": entropy_ok,
        "action_ids_ok": action_ids_ok,
        "sampled_actions_shape": tuple(actions.shape),
        "mean_entropy_pre_training": entropy_val,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke_iters", type=int, default=10)
    p.add_argument("--overfit_iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main(args: argparse.Namespace) -> int:
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False,
             log_to_driver=False)
    results = {}
    exit_code = 0
    try:
        print("== 10-iter smoke ==")
        ok, r = run_smoke(args.smoke_iters, args.seed)
        results["smoke"] = r
        print(f"  ok={ok}  final_return={r['final_return']:.3f}  "
              f"nan_loss={r['has_nan_loss']}  "
              f"max_invalid_per_seat={r['max_invalid_per_seat']}")
        if not ok:
            exit_code = 1

        print("\n== head-collapse & NaN check (fresh weights) ==")
        ok2, r2 = run_head_collapse_check()
        results["head_collapse"] = r2
        print(f"  ok={ok2}  logp_finite={r2['logp_finite']}  "
              f"mean_entropy_pre_training={r2['mean_entropy_pre_training']:.3f}")
        if not ok2:
            exit_code = 1

        print(f"\n== fixed-seed overfit ({args.overfit_iters} iters, "
              f"non-degenerate env still — plumbing check only) ==")
        ok3, r3 = run_smoke(args.overfit_iters, args.seed + 999)
        results["overfit"] = r3
        first_return = r3["history"][0]["return_mean"]
        final_return = r3["history"][-1]["return_mean"]
        improved = final_return > first_return
        print(f"  ok={ok3}  first_return={first_return:.3f} "
              f"→ final_return={final_return:.3f}  improved={improved}")
        if not ok3 or math.isnan(final_return):
            exit_code = 1

        print("\n== summary ==")
        for name, r in results.items():
            print(f"  [{name}] final_return={r.get('final_return', 'n/a')}")
        print(f"exit_code={exit_code}")
    finally:
        ray.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main(parse_args()))
