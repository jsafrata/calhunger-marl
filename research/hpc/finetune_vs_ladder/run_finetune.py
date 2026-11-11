"""Fine-tune a best-config PPO checkpoint against the scripted eval ladder.

Warm-starts an RLlib PPO algorithm from an existing checkpoint's RLModule
weights and trains it in ``ScriptedOpponentEnv`` (one learned seat per
episode, three same-archetype cadenced scripted opponents drawn uniformly
from {v2..v5}).

Usage (best-config warm starts, on HPC — see run_array.sbatch):

    python -m hpc.finetune_vs_ladder.run_finetune \
        --checkpoint <...>/ippo_seed0_headmlp/checkpoints/iter_1200 \
        --algo ippo --seed 0 --iterations 300 --checkpoint-every 50 \
        --output-root $HUNGER_CAMPAIGN_ROOT \
        --include-survival-features --include-trade-history

Local smoke (v2-era checkpoint trained WITHOUT the v4 observation flags —
omit both flags and the load-time contract check enforces the match):

    PYTHONPATH=. python -m hpc.finetune_vs_ladder.run_finetune \
        --checkpoint eval/results/hpc_v2_5seed_ippo_mappo_1500/ippo_seed0_headmlp/checkpoints/iter_1500 \
        --algo ippo --seed 0 --iterations 2 --checkpoint-every 1 \
        --train-batch-size 360 --minibatch-size 90 --num-env-runners 1 \
        --output-root /tmp/finetune_smoke

Design notes
------------
- Batching mirrors ``episode_matched_batch``: 45 complete games per PPO
  update at 180 s = 8100 env steps (train batch), minibatch = batch/4,
  4 epochs. With one learned seat per episode an env step now carries ONE
  agent transition (self-play carried four), so 8100 env steps ≈ 8100
  agent steps ≈ 45+ complete games (more when the focal seat dies early,
  since the RLlib episode ends at focal death).
- Entropy/lr default to the TERMINAL values of the best-config anneals
  (0.001 / 1e-4) held constant: that is what the checkpoint was training
  with at iter 1200. Pass --entropy-anneal-to / --lr-anneal-to to anneal
  over the fine-tune budget instead.
- Checkpoints continue the warm start's numbering: iter_1200 -> iter_1250,
  ..., iter_1500 under <output-root>/<algo>_seed<N>_headmlp/checkpoints/,
  so eval/checkpoint_scripted_diagnostics.py works unchanged with
  --checkpoint-start 1250 --checkpoint-stop 1500 --checkpoint-every 50.
- Warm-start integrity: the checkpoint's RLModule is loaded standalone
  (no Ray needed), its declared observation/architecture contract is
  asserted against the CLI flags (loud failure on mismatch), its weights
  are pushed into the fresh algorithm's learner and synced to env
  runners, and a logits-identity check on a fixed observation batch is
  run before (must differ) and after (must match) the load.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval._warnings import silence_ray_deprecations  # noqa: E402

silence_ray_deprecations()

import ray  # noqa: E402
from ray.rllib.core.columns import Columns  # noqa: E402
from ray.rllib.core.rl_module.rl_module import RLModule  # noqa: E402
from ray.tune.registry import register_env  # noqa: E402

from env.rl.rllib_config import (  # noqa: E402
    SHARED_POLICY_ID,
    build_ppo_config,
    episode_matched_batch,
    linear_anneal_schedule,
    seed_everything,
)
from hpc.finetune_vs_ladder.campaign import (  # noqa: E402
    GAME_DURATION_SECONDS,
    GAMES_PER_UPDATE,
)
from hpc.finetune_vs_ladder.scripted_opponent_env import (  # noqa: E402
    ARCHETYPES,
    FINETUNE_ENV_NAME,
    FinetuneVsLadderCallbacks,
    ScriptedOpponentEnv,
)


# ---------------------------------------------------------------------------
# Checkpoint module loading + contract check
# ---------------------------------------------------------------------------


def load_source_module(checkpoint_dir: Path) -> RLModule:
    """Load the checkpoint's shared-policy RLModule without starting Ray."""
    if not (checkpoint_dir / "rllib_checkpoint.json").is_file():
        raise FileNotFoundError(
            f"not an RLlib checkpoint directory: {checkpoint_dir}"
        )
    module_dir = (
        checkpoint_dir
        / "learner_group"
        / "learner"
        / "rl_module"
        / SHARED_POLICY_ID
    )
    if not module_dir.is_dir():
        raise FileNotFoundError(
            f"checkpoint has no shared-policy module dir: {module_dir}"
        )
    module = RLModule.from_checkpoint(str(module_dir))
    module.eval()
    return module


def module_contract(module: RLModule) -> dict[str, Any]:
    """The architecture/observation contract the module was built with.

    Read from attributes set in ``NashHungerPPOTorchRLModule.setup()`` —
    they reflect the actually-constructed network, independent of any
    model_config key drift across repo eras.
    """
    return {
        "algo": getattr(module, "_algo"),
        "torso": "lstm" if getattr(module, "_is_recurrent") else "mlp",
        "head_kind": dict(module.model_config).get("head_kind", "mlp"),
        "include_agent_id": bool(module.include_agent_id),
        "include_survival_features": bool(module.include_survival_features),
        "include_trade_history": bool(module.include_trade_history),
    }


def assert_contract_matches(module: RLModule, args: argparse.Namespace) -> None:
    actual = module_contract(module)
    expected = {
        "algo": args.algo,
        "torso": args.torso,
        "head_kind": args.head_kind,
        "include_agent_id": False,
        "include_survival_features": bool(args.include_survival_features),
        "include_trade_history": bool(args.include_trade_history),
    }
    mismatches = {
        key: (actual[key], expected[key])
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        lines = "\n".join(
            f"  {key}: checkpoint={got!r} cli={want!r}"
            for key, (got, want) in mismatches.items()
        )
        raise SystemExit(
            "FATAL: checkpoint module config does not match the CLI flags "
            "(warm-starting with a mismatched contract would silently "
            f"corrupt the run):\n{lines}"
        )
    print(f"checkpoint contract verified: {actual}")


# ---------------------------------------------------------------------------
# Warm-start weight injection + logits-identity check
# ---------------------------------------------------------------------------


def fixed_probe_batch(
    env_config: dict[str, Any], n_obs: int = 16, probe_seed: int = 123_456
) -> dict[str, Any]:
    """A deterministic batch of real observations + actions for the
    logits-identity check: roll the scripted-opponent env with random
    legal focal actions and stack the focal observations."""
    env = ScriptedOpponentEnv({**env_config, "seed": probe_seed})
    rng = np.random.default_rng(probe_seed)
    collected: list[dict[str, np.ndarray]] = []
    actions: list[int] = []
    obs, _ = env.reset()
    while len(collected) < n_obs:
        (agent_id, agent_obs), = obs.items()
        legal = np.flatnonzero(agent_obs["action_mask"] > 0)
        action = int(rng.choice(legal))
        collected.append(agent_obs)
        actions.append(action)
        obs, _, terminated, _, _ = env.step({agent_id: action})
        if terminated["__all__"]:
            obs, _ = env.reset()
    batch_obs = {
        key: torch.tensor(
            np.stack([entry[key] for entry in collected]),
            dtype=torch.float32,
        )
        for key in ("observations", "action_mask", "critic_observations")
    }
    return {
        Columns.OBS: batch_obs,
        Columns.ACTIONS: torch.tensor(actions, dtype=torch.long),
    }


def module_fingerprint(
    module: RLModule, probe: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """(logp-of-fixed-actions, value estimates) — a deterministic function
    of the module weights (teacher-forced ``_forward_train``, no sampling)."""
    with torch.no_grad():
        out = module._forward_train(dict(probe))
        logp = out[Columns.ACTION_LOGP].detach().clone()
        values = module.compute_values(dict(probe)).detach().clone()
    return logp, values


def inject_weights(algo_obj, state: dict[str, Any]) -> None:
    """Push module weights into every learner, then sync env runners."""
    algo_obj.learner_group.foreach_learner(
        lambda learner: learner.module[SHARED_POLICY_ID].set_state(state)
    )
    algo_obj.env_runner_group.sync_weights(
        from_worker_or_learner_group=algo_obj.learner_group,
        inference_only=False,
    )


def verify_warm_start(
    algo_obj,
    source_module: RLModule,
    probe: dict[str, Any],
    *,
    stage: str,
    expect_match: bool,
) -> None:
    src_logp, src_values = module_fingerprint(source_module, probe)
    runner_module = algo_obj.get_module(SHARED_POLICY_ID)
    new_logp, new_values = module_fingerprint(runner_module, probe)
    logp_match = bool(torch.allclose(src_logp, new_logp, atol=1e-6))
    values_match = bool(torch.allclose(src_values, new_values, atol=1e-6))
    max_dlogp = float((src_logp - new_logp).abs().max())
    max_dv = float((src_values - new_values).abs().max())
    print(
        f"warm-start check [{stage}]: logp_match={logp_match} "
        f"(max|Δ|={max_dlogp:.3e})  values_match={values_match} "
        f"(max|Δ|={max_dv:.3e})"
    )
    if expect_match and not (logp_match and values_match):
        raise SystemExit(
            f"FATAL: warm-start verification failed at stage {stage!r}: "
            "the env-runner module does not reproduce the checkpoint's "
            "logp/values on the fixed probe batch."
        )
    if not expect_match and logp_match and values_match:
        # A fresh random init matching the checkpoint exactly would mean
        # the probe has no discriminative power — treat as fatal too.
        raise SystemExit(
            "FATAL: pre-load fingerprint already matches the checkpoint; "
            "the logits-identity check is not discriminative."
        )


# ---------------------------------------------------------------------------
# Telemetry (mirrors eval/long_baseline.py conventions)
# ---------------------------------------------------------------------------


def _flt(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def snapshot(iter_num: int, result: dict, wall_time_s: float) -> dict:
    er = result.get("env_runners", {})
    learners = result.get("learners", {})
    lm = {}
    for m in learners.values():
        if isinstance(m, dict) and any(k in m for k in ("policy_loss", "vf_loss")):
            lm = m
            break
    snap = {
        "iter": iter_num,
        "episodes_lifetime": _flt(er.get("num_episodes_lifetime")),
        "return_mean": _flt(er.get("episode_return_mean")),
        "return_min": _flt(er.get("episode_return_min")),
        "return_max": _flt(er.get("episode_return_max")),
        "policy_loss": _flt(lm.get("policy_loss")),
        "vf_loss": _flt(lm.get("vf_loss")),
        "entropy": _flt(lm.get("entropy")),
        "vf_explained_var": _flt(lm.get("vf_explained_var")),
        "grad_norm": _flt(lm.get("gradients_default_optimizer_global_norm")),
        # In this env game/n_survivors comes from the standard callback but
        # only the focal seat is an RLlib agent — focal_survival_rate below
        # is the meaningful survival signal.
        "game_n_survivors": _flt(er.get("game/n_survivors")),
        "game_episode_length": _flt(er.get("game/episode_length")),
        "action_rejections_total_per_seat": _flt(
            er.get("action_rejections/per_seat_total")
        ),
        # ---- finetune_vs_ladder extensions ----
        "focal_survival_rate": _flt(er.get("focal/survival_rate")),
        "focal_win_rate": _flt(er.get("focal/win_rate")),
        "focal_terminal_reward": _flt(er.get("focal/terminal_reward")),
        "opponent_episodes": {
            arch: _flt(er.get(f"opponents/{arch}_episodes"))
            for arch in ARCHETYPES
        },
        "focal_survival_by_opponent": {
            arch: _flt(er.get(f"focal/survival_rate_{arch}"))
            for arch in ARCHETYPES
        },
        "focal_seat_episodes": {
            f"seat_{idx}": _flt(er.get(f"focal/seat_{idx}_episodes"))
            for idx in range(4)
        },
        "wall_time_s": round(wall_time_s, 2),
    }
    return snap


def _should_save_checkpoint(
    *, local_iter: int, total_local_iters: int, checkpoint_every: int
) -> bool:
    if local_iter == total_local_iters:
        return True
    return checkpoint_every > 0 and local_iter % checkpoint_every == 0


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def parse_start_iter(checkpoint_dir: Path) -> int | None:
    match = re.search(r"iter_(\d+)$", checkpoint_dir.name)
    return int(match.group(1)) if match else None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a PPO checkpoint vs the scripted eval ladder"
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="RLlib checkpoint dir (e.g. .../checkpoints/iter_1200)")
    p.add_argument("--algo", choices=("ippo", "mappo"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--env-seed-base", type=int, default=10_000,
        help="Env lineup/game seed = env_seed_base + seed (per-task streams).",
    )
    p.add_argument(
        "--start-iter", type=int, default=None,
        help="Absolute iteration of the warm start (default: parsed from "
             "the checkpoint dir name, e.g. iter_1200 -> 1200).",
    )
    # Architecture contract — must match the checkpoint (asserted at load).
    p.add_argument("--torso", choices=("mlp", "lstm"), default="mlp")
    p.add_argument("--head-kind", choices=("mlp", "linear"), default="mlp")
    p.add_argument("--include-survival-features", action="store_true")
    p.add_argument("--include-trade-history", action="store_true")
    # Batching (episode-matched defaults; overrides for smoke tests).
    p.add_argument("--games-per-update", type=int, default=GAMES_PER_UPDATE)
    p.add_argument("--train-batch-size", type=int, default=None,
                   help="Override the episode-matched train batch (smoke).")
    p.add_argument("--minibatch-size", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=4)
    p.add_argument("--num-env-runners", type=int, default=2)
    # Optimization — defaults hold the best-config anneals' TERMINAL values.
    p.add_argument("--entropy-coeff", type=float, default=0.001)
    p.add_argument("--entropy-anneal-to", type=float, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-anneal-to", type=float, default=None)
    p.add_argument("--focal-selection", choices=("random", "rotate"),
                   default="random")
    p.add_argument("--num-cpus", type=int, default=6)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    checkpoint_dir = args.checkpoint.expanduser().resolve()

    start_iter = (
        args.start_iter
        if args.start_iter is not None
        else parse_start_iter(checkpoint_dir)
    )
    if start_iter is None:
        raise SystemExit(
            "FATAL: cannot parse the warm-start iteration from "
            f"{checkpoint_dir.name!r}; pass --start-iter explicitly."
        )

    # Episode-matched batch unless explicitly overridden (smoke tests).
    train_batch_size, minibatch_size = episode_matched_batch(
        GAME_DURATION_SECONDS, games_per_update=args.games_per_update
    )
    if args.train_batch_size is not None:
        train_batch_size = int(args.train_batch_size)
        minibatch_size = (
            int(args.minibatch_size)
            if args.minibatch_size is not None
            else max(1, train_batch_size // 4)
        )

    entropy_setting: float | list = args.entropy_coeff
    if args.entropy_anneal_to is not None:
        entropy_setting = linear_anneal_schedule(
            args.entropy_coeff,
            args.entropy_anneal_to,
            args.iterations * train_batch_size,
        )
    lr_setting: float | list = args.lr
    if args.lr_anneal_to is not None:
        lr_setting = linear_anneal_schedule(
            args.lr, args.lr_anneal_to, args.iterations * train_batch_size
        )

    env_seed = args.env_seed_base + args.seed
    env_config: dict[str, Any] = {
        "seed": env_seed,
        "reward_kind": "terminal_only",
        "env_variant": "default",
        "focal_selection": args.focal_selection,
        "cadence": "fast",
        "include_agent_id": False,
        "include_survival_features": bool(args.include_survival_features),
        "include_trade_history": bool(args.include_trade_history),
    }

    out_dir = (
        args.output_root.expanduser().resolve()
        / f"{args.algo}_seed{args.seed}_headmlp"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_root = out_dir / "checkpoints"
    ck_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "iters.jsonl"

    print(
        f"=== finetune_vs_ladder: {args.algo} seed{args.seed} | warm start "
        f"iter_{start_iter} | +{args.iterations} iters | ckpt every "
        f"{args.checkpoint_every} ==="
    )
    print(
        f"    train_batch={train_batch_size} minibatch={minibatch_size} "
        f"epochs={args.num_epochs} entropy={entropy_setting} lr={lr_setting}"
    )
    print(f"    env_config={env_config}")
    print(f"    checkpoint={checkpoint_dir}")
    print(f"    output={out_dir}")

    # ---- load + verify the source module (no Ray needed) ----
    source_module = load_source_module(checkpoint_dir)
    assert_contract_matches(source_module, args)
    source_state = source_module.get_state()

    # Fixed probe batch for the logits-identity check, built from the
    # actual fine-tuning env so the observation contract is exercised.
    probe = fixed_probe_batch(env_config)

    ray.init(
        ignore_reinit_error=True,
        num_cpus=args.num_cpus,
        include_dashboard=False,
        log_to_driver=False,
    )
    seed_everything(args.seed)
    register_env(
        FINETUNE_ENV_NAME, lambda cfg: ScriptedOpponentEnv(cfg)
    )

    cfg = build_ppo_config(
        seed=args.seed,
        num_env_runners=args.num_env_runners,
        train_batch_size=train_batch_size,
        minibatch_size=minibatch_size,
        num_epochs=args.num_epochs,
        algo=args.algo,
        torso=args.torso,
        head_kind=args.head_kind,
        env_config=env_config,
        entropy_coeff=entropy_setting,
        lr=lr_setting,
        gamma=1.0,
        lambda_=1.0,
        parameterization="shared",
        include_survival_features=bool(args.include_survival_features),
        include_trade_history=bool(args.include_trade_history),
    )
    # Swap in the scripted-opponent env (build_ppo_config registers and
    # points at the self-play env). env_config must be passed complete —
    # .environment() replaces the whole dict.
    cfg = cfg.environment(env=FINETUNE_ENV_NAME, env_config=env_config)
    cfg = cfg.callbacks(FinetuneVsLadderCallbacks)

    algo_obj = cfg.build_algo()
    try:
        # Pre-load check: a fresh init must NOT match the checkpoint —
        # proves the probe discriminates.
        verify_warm_start(
            algo_obj, source_module, probe, stage="before-load",
            expect_match=False,
        )
        inject_weights(algo_obj, source_state)
        verify_warm_start(
            algo_obj, source_module, probe, stage="after-load",
            expect_match=True,
        )

        t0 = time.time()
        with open(jsonl_path, "a") as fh:
            for local_iter in range(1, args.iterations + 1):
                iter_start = time.time()
                result = algo_obj.train()
                absolute_iter = start_iter + local_iter
                snap = snapshot(
                    absolute_iter, result, time.time() - iter_start
                )
                fh.write(json.dumps(snap) + "\n")
                fh.flush()
                if (
                    local_iter % max(1, args.iterations // 20) == 0
                    or local_iter == args.iterations
                ):
                    print(
                        f"  iter {local_iter}/{args.iterations} "
                        f"(abs {absolute_iter}) ret={snap['return_mean']} "
                        f"focal_surv={snap['focal_survival_rate']} "
                        f"ent={snap['entropy']} "
                        f"elapsed={time.time() - t0:.0f}s",
                        flush=True,
                    )
                if _should_save_checkpoint(
                    local_iter=local_iter,
                    total_local_iters=args.iterations,
                    checkpoint_every=args.checkpoint_every,
                ):
                    ck_dir = ck_root / f"iter_{absolute_iter:04d}"
                    if ck_dir.exists():
                        print(f"    skip save (exists): {ck_dir}")
                    else:
                        ck_dir.mkdir(parents=True, exist_ok=True)
                        saved = algo_obj.save(str(ck_dir))
                        candidate = getattr(saved, "checkpoint", None) or saved
                        ck_path = (
                            getattr(candidate, "path", None)
                            or getattr(saved, "path", None)
                            or str(candidate)
                        )
                        (out_dir / "latest_checkpoint.txt").write_text(
                            str(ck_path)
                        )
        total = time.time() - t0
        print(
            f"[{args.algo} seed{args.seed}] done in {total:.0f}s "
            f"({total / args.iterations:.2f}s/iter)"
        )
    finally:
        algo_obj.stop()
        ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
