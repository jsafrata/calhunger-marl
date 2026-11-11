"""Train one v3 shared-vs-independent condition.

Wraps eval.long_baseline for the v3 experiment matrix. Conditions:

    A: shared          — one shared_policy, current v2 baseline.
    B: shared_id       — one shared_policy + 4-dim agent-id one-hot
                          appended to actor + critic observations.
    C: independent     — 4 separate policies pi_0..pi_3, no parameter or
                          gradient sharing.

Everything else is the v2 default:
    terminal_only reward, MAX_QTY=5, MAX_PRICE=50, head_kind=mlp,
    gamma=1.0, lambda_=1.0, entropy_coeff=0.01, lr=3e-4,
    train_batch=4096, minibatch=1024, num_epochs=4.

The output directory layout is:
    eval/results/v3_shared_vs_independent/
        <algo>_<condition>_seed<N>/
            iters.jsonl
            checkpoints/iter_XXXX/...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval._warnings import silence_ray_deprecations  # noqa: E402
silence_ray_deprecations()

from env.actions import MAX_PRICE, MAX_QTY, NUM_ACTIONS  # noqa: E402
from env.rl.global_observation_encoder import global_observation_dim  # noqa: E402
from env.rl.observation_encoder import observation_dim  # noqa: E402
from env.rl.rllib_config import build_ppo_config, register_nash_hunger_env, seed_everything  # noqa: E402


CONDITION_ALIASES = {
    "A": "shared", "shared": "shared",
    "B": "shared_id", "shared_id": "shared_id",
    "C": "independent", "independent": "independent",
}


def _flt(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _number_dict(value) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(k): float(v)
        for k, v in value.items()
        if _flt(v) is not None
    }


def _snapshot(
    iter_num: int,
    result: dict,
    *,
    elapsed_seconds: float,
    minibatch_size: int,
) -> dict:
    er = result.get("env_runners", {})
    learners = result.get("learners", {})
    # For shared: pick the sole policy metric block.
    # For independent: aggregate per-policy losses and log each.
    per_policy_losses: dict[str, dict] = {}
    for pid, m in learners.items():
        if isinstance(m, dict) and any(k in m for k in ("policy_loss", "vf_loss")):
            per_policy_losses[pid] = m
    # Pick a representative lm for the top-level snapshot fields.
    lm = next(iter(per_policy_losses.values()), {})
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
        "reward_dense_per_seat": _flt(er.get("reward/dense_per_seat")),
        "reward_terminal_per_seat": _flt(er.get("reward/terminal_per_seat")),
        "game_n_survivors": _flt(er.get("game/n_survivors")),
        "game_n_survived_to_180": _flt(er.get("game/n_survived_to_180")),
        "game_episode_length": _flt(er.get("game/episode_length")),
        "wall_clock_seconds": float(elapsed_seconds),
        "env_steps_sampled_lifetime": _flt(
            er.get("num_env_steps_sampled_lifetime")
        ),
        "agent_steps_sampled_lifetime_by_seat": _number_dict(
            er.get("num_agent_steps_sampled_lifetime")
        ),
        "module_steps_sampled_lifetime": _number_dict(
            er.get("num_module_steps_sampled_lifetime")
        ),
    }
    # Per-policy training-side stats (populated for Condition C).
    for pid, m in per_policy_losses.items():
        for key in ("policy_loss", "vf_loss", "entropy", "vf_explained_var"):
            snap[f"per_policy/{pid}/{key}"] = _flt(m.get(key))
        trained = _flt(m.get("num_module_steps_trained_lifetime"))
        snap[f"per_policy/{pid}/module_steps_trained_lifetime"] = trained
        snap[f"per_policy/{pid}/learner_updates_lifetime"] = _flt(
            m.get("weights_seq_no")
        )
        snap[f"per_policy/{pid}/optimizer_steps_lifetime"] = (
            trained / minibatch_size if trained is not None else None
        )
        snap[f"per_policy/{pid}/train_batch_size_mean"] = _flt(
            m.get("module_train_batch_size_mean")
        )
    snap["total_optimizer_steps_lifetime"] = float(sum(
        value
        for key, value in snap.items()
        if key.endswith("/optimizer_steps_lifetime") and value is not None
    ))
    snap["total_agent_steps_sampled_lifetime"] = float(sum(
        snap["module_steps_sampled_lifetime"].values()
    ))
    return snap


def _policy_manifest(algo_obj, policy_ids: list[str]) -> dict:
    policies = {}
    prior_storage: set[int] = set()
    for pid in policy_ids:
        module = algo_obj.get_module(pid)
        params = list(module.parameters())
        ptrs = [int(p.data_ptr()) for p in params]
        policies[pid] = {
            "trainable_parameters": int(sum(p.numel() for p in params)),
            "actor_parameters": (
                int(sum(p.numel() for p in module.torso.parameters()))
                + int(sum(p.numel() for p in module.actor.parameters()))
            ),
            "critic_parameters": int(
                sum(p.numel() for p in module.critic.parameters())
            ),
            "actor_observation_dim": int(module._actor_obs_dim),
            "critic_observation_dim": int(module.critic.input_dim),
            "parameter_tensor_count": len(params),
            "storage_is_disjoint_from_previous_policies": not any(
                ptr in prior_storage for ptr in ptrs
            ),
        }
        prior_storage.update(ptrs)
    return policies


def train_condition(
    algo: str,
    condition: str,
    seed: int,
    iters: int,
    ckpt_every: int,
    out_dir: Path,
    *,
    num_env_runners: int = 2,
    train_batch_size: int = 4096,
    minibatch_size: int = 1024,
    num_epochs: int = 4,
    resume_from_checkpoint: Path | None = None,
) -> int:
    cond = CONDITION_ALIASES[condition]
    parameterization = "independent" if cond == "independent" else "shared"
    include_id = cond == "shared_id"

    print(f"\n=== v3 train: algo={algo} condition={cond} seed={seed} iters={iters} ===")
    print(f"    parameterization={parameterization} include_agent_id={include_id}")
    print(f"    output dir={out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ck_root = (out_dir / "checkpoints").resolve()
    ck_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "iters.jsonl"
    manifest_path = out_dir / "run_manifest.json"
    if jsonl_path.exists() and resume_from_checkpoint is None:
        raise FileExistsError(
            f"refusing to append a fresh run to existing {jsonl_path}; "
            "use a new output directory or --resume_from_checkpoint"
        )

    env_config = {"reward_kind": "terminal_only"}

    previous_manifest = None
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            previous_manifest = None

    prior_elapsed = 0.0
    if resume_from_checkpoint is not None:
        from ray.rllib.algorithms.algorithm import Algorithm
        register_nash_hunger_env()
        algo_obj = Algorithm.from_checkpoint(str(Path(resume_from_checkpoint).resolve()))
        start_iter = int(getattr(algo_obj, "iteration", 0))
        print(f"    resumed at algo iteration {start_iter}")
        if jsonl_path.exists():
            rows = [
                json.loads(line)
                for line in jsonl_path.read_text().splitlines()
                if line.strip()
            ]
            checkpointed_rows = [
                row for row in rows if int(row["iter"]) <= start_iter
            ]
            if checkpointed_rows:
                prior_elapsed = float(
                    checkpointed_rows[-1].get("wall_clock_seconds", 0.0)
                )
            if len(checkpointed_rows) != len(rows):
                jsonl_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in checkpointed_rows)
                )
                print(
                    f"    discarded {len(rows) - len(checkpointed_rows)} "
                    "uncheckpointed metric row(s) before exact resume"
                )
    else:
        cfg = build_ppo_config(
            seed=seed,
            num_env_runners=num_env_runners,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            algo=algo,
            torso="mlp",
            head_kind="mlp",
            env_config=env_config,
            entropy_coeff=0.01,
            gamma=1.0,
            lambda_=1.0,
            parameterization=parameterization,
            include_agent_id=include_id,
        )
        algo_obj = cfg.build_algo()
        start_iter = 0

    policy_ids = (
        ["pi_0", "pi_1", "pi_2", "pi_3"]
        if parameterization == "independent"
        else ["shared_policy"]
    )
    manifest = {
        "experiment": "v3_shared_vs_independent",
        "status": "running",
        "started_at_utc": (
            previous_manifest.get("started_at_utc")
            if previous_manifest else datetime.now(timezone.utc).isoformat()
        ),
        "resumed_at_utc": (
            datetime.now(timezone.utc).isoformat()
            if resume_from_checkpoint is not None else None
        ),
        "algo": algo,
        "condition": cond,
        "seed": seed,
        "parameterization": parameterization,
        "include_agent_id": include_id,
        "policy_ids": policy_ids,
        "fixed_configuration": {
            "horizon": 180,
            "reward_kind": "terminal_only",
            "death_reward": -1,
            "surviving_nonwinner_reward": 0,
            "winner_reward": 1,
            "max_qty": MAX_QTY,
            "max_price": MAX_PRICE,
            "num_actions": NUM_ACTIONS,
            "head_kind": "mlp",
            "torso": "mlp",
            "gamma": 1.0,
            "lambda": 1.0,
            "lr": 3e-4,
            "entropy_coeff": 0.01,
            "clip_param": 0.2,
            "train_batch_size_environment_steps": train_batch_size,
            "minibatch_size": minibatch_size,
            "num_epochs": num_epochs,
            "num_env_runners": num_env_runners,
            "local_observation_dim": observation_dim(
                include_agent_id=include_id
            ),
            "global_observation_dim": global_observation_dim(
                include_agent_id=include_id
            ),
        },
        "requested_iterations": start_iter + iters,
        "requested_environment_steps": (start_iter + iters) * train_batch_size,
        "requested_agent_steps": (start_iter + iters) * train_batch_size * 4,
        "resume_from_checkpoint": (
            str(resume_from_checkpoint.resolve())
            if resume_from_checkpoint is not None else None
        ),
        "network_manifest": _policy_manifest(algo_obj, policy_ids),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    t0 = time.time()
    with open(jsonl_path, "a") as fh:
        for i in range(1, iters + 1):
            result = algo_obj.train()
            absolute_iter = start_iter + i
            snap = _snapshot(
                absolute_iter,
                result,
                elapsed_seconds=prior_elapsed + time.time() - t0,
                minibatch_size=minibatch_size,
            )
            fh.write(json.dumps(snap) + "\n")
            fh.flush()
            if i % max(1, iters // 20) == 0 or i == iters:
                dt = time.time() - t0
                print(f"  iter {i}/{iters} (abs {absolute_iter})  ret={snap['return_mean']} "
                      f"surv={snap['game_n_survivors']} elapsed={dt:.0f}s")
            if absolute_iter % ckpt_every == 0 or i == iters:
                ck_dir = ck_root / f"iter_{absolute_iter:04d}"
                if ck_dir.exists():
                    print(f"    skip save (exists): {ck_dir}")
                else:
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    saved = algo_obj.save(str(ck_dir))
                    candidate = getattr(saved, "checkpoint", None) or saved
                    ck_path = getattr(candidate, "path", None) or getattr(saved, "path", None) or str(candidate)
                    (out_dir / "latest_checkpoint.txt").write_text(str(ck_path))
    algo_obj.stop()
    total = prior_elapsed + time.time() - t0
    manifest.update({
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": total,
        "final_accounting": snap,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"[{algo}/{cond}/seed{seed}] done in {total:.0f}s cumulative "
        f"({total/(start_iter + iters):.2f}s/iter)"
    )
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, required=True, choices=("ippo", "mappo"))
    p.add_argument("--condition", type=str, required=True,
                   choices=("A", "B", "C", "shared", "shared_id", "independent"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--num_env_runners", type=int, default=2)
    p.add_argument("--num_cpus", type=int, default=6)
    p.add_argument("--object_store_memory_mb", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4096)
    p.add_argument("--minibatch_size", type=int, default=1024)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--out_root", type=Path,
                   default=Path("eval/results/v3_shared_vs_independent"))
    p.add_argument("--resume_from_checkpoint", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ray.init(
        ignore_reinit_error=True,
        num_cpus=args.num_cpus,
        object_store_memory=args.object_store_memory_mb * 1024 * 1024,
        include_dashboard=False,
        log_to_driver=False,
    )
    seed_everything(args.seed)
    cond_slug = CONDITION_ALIASES[args.condition]
    sub = f"{args.algo}_{cond_slug}_seed{args.seed}"
    out_dir = (args.out_root / sub).resolve()
    try:
        train_condition(
            algo=args.algo,
            condition=cond_slug,
            seed=args.seed,
            iters=args.iters,
            ckpt_every=args.checkpoint_every,
            out_dir=out_dir,
            num_env_runners=args.num_env_runners,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
