"""Run one unchanged-config baseline per algorithm for the requested number
of iterations, checkpointing every N iters and logging all training-side
metrics to JSONL. No policy or reward changes; this is the "give it enough
compute and see what happens" run.

Usage:
    python -m eval.long_baseline --iters 500 --checkpoint_every 25 --seed 0
    python -m eval.long_baseline --iters 1500 --checkpoint_every 100 \
        --checkpoint_start 100 --seed 0 --algos ippo

Writes:
    eval/results/long_baseline/<algo>/iters.jsonl        — per-iter metrics
    eval/results/long_baseline/<algo>/checkpoints/iter_N — RLlib checkpoint
    eval/results/long_baseline/<algo>/latest_checkpoint.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import ray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval._warnings import silence_ray_deprecations  # noqa: E402
silence_ray_deprecations()

from ray.rllib.algorithms.algorithm import Algorithm  # noqa: E402

from env.config import DEFAULT_CONFIG  # noqa: E402
from env.rl.rllib_config import (  # noqa: E402
    build_ppo_config,
    episode_matched_batch,
    linear_anneal_schedule,
    register_nash_hunger_env,
    seed_everything,
)
from env.small_env import SCALED_ENVIRONMENTS  # noqa: E402


def _variant_game_duration(env_variant: str) -> int:
    if env_variant == "default":
        return DEFAULT_CONFIG.game_duration_seconds
    return SCALED_ENVIRONMENTS[env_variant][0].game_duration_seconds


def _flt(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _snapshot(iter_num: int, result: dict) -> dict:
    er = result.get("env_runners", {})
    learners = result.get("learners", {})
    # Shared policy learner metrics
    lm = {}
    for m in learners.values():
        if isinstance(m, dict) and any(k in m for k in ("policy_loss", "vf_loss")):
            lm = m
            break
    return {
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
        # env-runner side custom metrics
        "reward_dense_per_seat": _flt(er.get("reward/dense_per_seat")),
        "reward_terminal_per_seat": _flt(er.get("reward/terminal_per_seat")),
        "reward_dense_share": _flt(er.get("reward/dense_share")),
        "game_n_survivors": _flt(er.get("game/n_survivors")),
        "game_episode_length": _flt(er.get("game/episode_length")),
        "action_rejections_total_per_seat": _flt(
            er.get("action_rejections/per_seat_total")
        ),
        "action_rejections_no_order_to_cancel_per_seat": _flt(
            er.get("action_rejections/no_order_to_cancel_per_seat")
        ),
    }


def _should_save_checkpoint(
    *,
    absolute_iter: int,
    local_iter: int,
    total_local_iters: int,
    checkpoint_every: int,
    checkpoint_start: int,
) -> bool:
    """Return whether this iteration should produce a full checkpoint.

    The final local iteration is always saved.  ``checkpoint_start`` only
    gates periodic checkpoints, which lets campaigns request an exact series
    such as 100, 200, ..., 1500 without changing resume semantics.
    """
    if local_iter == total_local_iters:
        return True
    return (
        checkpoint_every > 0
        and absolute_iter >= checkpoint_start
        and absolute_iter % checkpoint_every == 0
    )


def run(algo: str, iters: int, ckpt_every: int, seed: int,
        train_batch_size: int, minibatch_size: int, num_epochs: int,
        num_env_runners: int, out_dir: Path,
        terminal_reward_config: dict | None = None,
        entropy_coeff: float = 0.01,
        head_kind: str = "mlp",
        torso: str = "mlp",
        gamma: float = 1.0,
        lambda_: float = 1.0,
        reward_kind: str = "terminal_only",
        env_variant: str = "default",
        resume_from_checkpoint: Path | None = None,
        checkpoint_start: int = 1,
        fail_if_output_exists: bool = False,
        games_per_update: int | None = None,
        entropy_anneal_to: float | None = None,
        lr_anneal_to: float | None = None,
        include_survival_features: bool = False,
        include_trade_history: bool = False) -> int:
    if iters <= 0:
        raise ValueError("iters must be positive")
    if ckpt_every < 0:
        raise ValueError("ckpt_every must be non-negative")
    if checkpoint_start < 1:
        raise ValueError("checkpoint_start must be at least 1")
    if fail_if_output_exists and out_dir.exists():
        raise FileExistsError(
            f"refusing to append to existing run directory: {out_dir}"
        )

    # Episode-matched batching: hold the number of games per PPO update
    # constant across durations instead of the number of env steps.
    if games_per_update is not None:
        duration = _variant_game_duration(env_variant)
        train_batch_size, minibatch_size = episode_matched_batch(
            duration, games_per_update=games_per_update
        )

    # Entropy annealing: linear from `entropy_coeff` at step 0 down to
    # `entropy_anneal_to` at the end of this run's sampling budget.
    entropy_setting: float | list = entropy_coeff
    if entropy_anneal_to is not None:
        entropy_setting = linear_anneal_schedule(
            entropy_coeff, entropy_anneal_to, iters * train_batch_size
        )

    # Learning-rate annealing on the same env-step clock (base lr is the
    # v2 constant 3e-4).
    lr_setting: float | list = 3e-4
    if lr_anneal_to is not None:
        lr_setting = linear_anneal_schedule(
            3e-4, lr_anneal_to, iters * train_batch_size
        )

    print(f"\n=== long baseline: {algo} | {iters} iters | ckpt every {ckpt_every} ===")
    print(f"    head_kind={head_kind}  torso={torso}  gamma={gamma}  "
          f"lambda={lambda_}  reward_kind={reward_kind}  env_variant={env_variant}")
    print(f"    train_batch_size={train_batch_size}  minibatch_size={minibatch_size}"
          + (f"  (episode-matched: {games_per_update} games/update)"
             if games_per_update is not None else ""))
    print(f"    periodic checkpoints start at iter {checkpoint_start}")
    if include_survival_features or include_trade_history:
        print(f"    include_survival_features={include_survival_features}  "
              f"include_trade_history={include_trade_history}")
    if terminal_reward_config:
        print(f"    terminal_reward_config = {terminal_reward_config}")
    if entropy_anneal_to is not None:
        print(f"    entropy_coeff schedule = {entropy_setting}")
    elif entropy_coeff != 0.01:
        print(f"    entropy_coeff = {entropy_coeff} (baseline default is 0.01)")
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_root = (out_dir / "checkpoints").resolve()
    ck_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "iters.jsonl"

    env_config: dict = {
        "reward_kind": reward_kind,
        "env_variant": env_variant,
    }
    if terminal_reward_config:
        env_config["terminal_reward_config"] = terminal_reward_config

    if resume_from_checkpoint is not None:
        # Full RLlib checkpoint restore: actor+critic weights, optimizer
        # state, iter/timestep counters, RNG state (where supported). Config
        # kwargs above are ignored — the checkpoint's baked-in config wins.
        # This is the whole point of "resume": the config MUST match what
        # trained the checkpoint or the restore will fail or silently drift.
        print(f"    resuming from checkpoint: {resume_from_checkpoint}")
        register_nash_hunger_env()
        algo_obj = Algorithm.from_checkpoint(str(Path(resume_from_checkpoint).resolve()))
        start_iter = int(getattr(algo_obj, "iteration", 0))
        print(f"    resumed at algo iteration {start_iter}")
    else:
        cfg = build_ppo_config(
            seed=seed, num_env_runners=num_env_runners,
            train_batch_size=train_batch_size, minibatch_size=minibatch_size,
            num_epochs=num_epochs, algo=algo, torso=torso,
            head_kind=head_kind,
            env_config=env_config,
            entropy_coeff=entropy_setting,
            lr=lr_setting,
            gamma=gamma,
            lambda_=lambda_,
            include_survival_features=include_survival_features,
            include_trade_history=include_trade_history,
        )
        algo_obj = cfg.build_algo()
        start_iter = 0

    t0 = time.time()
    with open(jsonl_path, "a") as fh:
        for i in range(1, iters + 1):
            result = algo_obj.train()
            absolute_iter = start_iter + i
            snap = _snapshot(absolute_iter, result)
            fh.write(json.dumps(snap) + "\n")
            fh.flush()
            if i % max(1, iters // 20) == 0 or i == iters:
                dt = time.time() - t0
                print(f"  iter {i}/{iters} (abs {absolute_iter})  ret={snap['return_mean']} "
                      f"surv={snap['game_n_survivors']} ent={snap['entropy']} "
                      f"ev={snap['vf_explained_var']} elapsed={dt:.0f}s")
            # Checkpoint naming uses the ABSOLUTE iter counter so resumed
            # runs slot in after the run's existing iter_0500 without
            # colliding with iter_0100..iter_0500 already on disk.
            if _should_save_checkpoint(
                absolute_iter=absolute_iter,
                local_iter=i,
                total_local_iters=iters,
                checkpoint_every=ckpt_every,
                checkpoint_start=checkpoint_start,
            ):
                ck_dir = ck_root / f"iter_{absolute_iter:04d}"
                if ck_dir.exists():
                    # Never overwrite an existing pre-resume checkpoint —
                    # skip cleanly with a warning.
                    print(f"    skip save (exists): {ck_dir}")
                else:
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    saved = algo_obj.save(str(ck_dir))
                    candidate = getattr(saved, "checkpoint", None) or saved
                    ck_path = getattr(candidate, "path", None) or getattr(saved, "path", None) or str(candidate)
                    (out_dir / "latest_checkpoint.txt").write_text(str(ck_path))
    algo_obj.stop()
    total = time.time() - t0
    print(f"[{algo}] done in {total:.0f}s ({total/iters:.2f}s/iter)")
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--checkpoint_every", type=int, default=25)
    p.add_argument(
        "--checkpoint_start",
        type=int,
        default=1,
        help=("First absolute iteration eligible for periodic checkpointing. "
              "The final iteration is always checkpointed."),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algos", type=str, default="ippo,mappo")
    p.add_argument("--train_batch_size", type=int, default=4096)
    p.add_argument("--minibatch_size", type=int, default=1024)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--num_env_runners", type=int, default=2)
    p.add_argument("--out_root", type=Path,
                   default=Path(
                       "eval/results/"
                       "v2_full_horizon_terminal_only_qty5_price50_deep_heads_gamma1_lambda1/"
                       "long_baseline"
                   ))
    p.add_argument("--head_kind", type=str, default="mlp",
                   choices=("mlp", "linear"))
    p.add_argument("--torso", type=str, default="mlp",
                   choices=("mlp", "lstm"),
                   help="Actor torso: feed-forward baseline or recurrent LSTM.")
    p.add_argument(
        "--games_per_update", type=int, default=None,
        help=("If set, override --train_batch_size/--minibatch_size with an "
              "episode-matched batch: train_batch = games x game_duration "
              "env steps, minibatch = train_batch/4. Holds the number of "
              "independent Monte-Carlo returns per PPO update constant "
              "across duration formats."),
    )
    p.add_argument(
        "--entropy_anneal_to", type=float, default=None,
        help=("If set, anneal the entropy coefficient linearly from "
              "--entropy_coeff at step 0 to this value at the end of the "
              "run's sampling budget (iters x train_batch env steps)."),
    )
    p.add_argument(
        "--lr_anneal_to", type=float, default=None,
        help=("If set, anneal the learning rate linearly from 3e-4 at step 0 "
              "to this value over the run's sampling budget."),
    )
    p.add_argument(
        "--include_survival_features", action="store_true",
        help=("Append per-food starvation cover + consumption-cycle phase "
              "to the local observation (v4 contract; new checkpoints only)."),
    )
    p.add_argument(
        "--include_trade_history", action="store_true",
        help=("Append previous-tick executed-trade matrices to the "
              "centralized critic observation (v4 contract; makes the MAPPO "
              "critic input a superset of the actor's market information)."),
    )
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lambda_", type=float, default=1.0, dest="lambda_")
    p.add_argument("--reward_kind", type=str, default="terminal_only",
                   choices=("terminal_only", "shaped"))
    p.add_argument(
        "--env_variant",
        type=str,
        default="default",
        choices=(
            "default",
            "small",
            "small_60",
            "small_90",
            "small_120",
            "small_150",
        ),
        help=("Environment implementation to train in. 'small' is the "
              "30-second baseline; small_60 through small_150 select its "
              "duration- and initial-food-scaled variants."),
    )
    p.add_argument("--resume_from_checkpoint", type=Path, default=None,
                   help="If set, restore the full RLlib checkpoint (weights, "
                        "optimizer, iter/timestep counters, RNG state) and "
                        "train for --iters MORE iterations. Config kwargs "
                        "above are ignored — checkpoint's baked-in config wins.")
    p.add_argument(
        "--fail_if_output_exists",
        action="store_true",
        help=("Fail before training if this algorithm's output directory "
              "already exists. Recommended for independent seed jobs."),
    )
    p.add_argument(
        "--reward_variant", type=str, default="baseline",
        choices=("baseline", "survival"),
        help=("baseline = original terminal rewards (win=5/k, survive=1, dead=0). "
              "survival = A/B experiment (win=5, survive=2, dead=-5, no split)."),
    )
    p.add_argument(
        "--entropy_coeff", type=float, default=0.01,
        help="PPO entropy bonus coefficient. Baseline default is 0.01.",
    )
    p.add_argument(
        "--variant_suffix", type=str, default="",
        help="Optional extra suffix for the output subdir (e.g. 'ent005').",
    )
    return p.parse_args()


REWARD_VARIANTS = {
    # Original baseline — do NOT override anything; tracker uses module
    # defaults (win=5.0, split k-way; survived_non_win=1.0; dead=0.0).
    "baseline": None,
    # Survival-shaping A/B: dead=-5, survived-non-winner=+2, winner=+5
    # (no split across ties, so a k-way tie gives each winner +5).
    # Every other reward, hyperparameter, obs and mask stays identical.
    "survival": {
        "win_reward": 5.0,
        "survived_non_win_reward": 2.0,
        "death_penalty": -5.0,
        "win_split_across_ties": False,
    },
}


def main(args) -> int:
    ray.init(ignore_reinit_error=True, num_cpus=6, include_dashboard=False,
             log_to_driver=False)
    seed_everything(args.seed)
    terminal_cfg = REWARD_VARIANTS.get(args.reward_variant)
    tags = []
    if args.reward_variant != "baseline":
        tags.append(args.reward_variant)
    if args.variant_suffix:
        tags.append(args.variant_suffix)
    variant_tag = "_".join(tags)
    try:
        for algo in [a.strip() for a in args.algos.split(",") if a.strip()]:
            sub = f"{algo}_{variant_tag}" if variant_tag else algo
            out_dir = (args.out_root / sub).resolve()
            run(algo, args.iters, args.checkpoint_every, args.seed,
                args.train_batch_size, args.minibatch_size, args.num_epochs,
                args.num_env_runners, out_dir,
                terminal_reward_config=terminal_cfg,
                entropy_coeff=args.entropy_coeff,
                head_kind=args.head_kind,
                torso=args.torso,
                gamma=args.gamma,
                lambda_=args.lambda_,
                reward_kind=args.reward_kind,
                env_variant=args.env_variant,
                resume_from_checkpoint=args.resume_from_checkpoint,
                checkpoint_start=args.checkpoint_start,
                fail_if_output_exists=args.fail_if_output_exists,
                games_per_update=args.games_per_update,
                entropy_anneal_to=args.entropy_anneal_to,
                lr_anneal_to=args.lr_anneal_to,
                include_survival_features=args.include_survival_features,
                include_trade_history=args.include_trade_history)
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
