"""Orchestrate the complete v3 shared-vs-independent campaign.

Runs are resumable at the job level: completed manifests/evaluations are
skipped, failed jobs retain their logs, and no v2 artifact is modified.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "eval/results/v3_shared_vs_independent"
V2_ROOT = REPO_ROOT / (
    "eval/results/"
    "v2_full_horizon_terminal_only_qty5_price50_deep_heads_gamma1_lambda1"
)
CONDITIONS = ("shared", "shared_id", "independent")
ALGOS = ("ippo", "mappo")
SEEDS = (0, 1, 2)


def run_dir(root: Path, algo: str, condition: str, seed: int) -> Path:
    return root / f"{algo}_{condition}_seed{seed}"


def v2_checkpoint(algo: str) -> Path:
    return (
        V2_ROOT
        / f"long_baseline/{algo}_seed0_headmlp/checkpoints/iter_1100"
    )


def _checkpoint_metrics(checkpoint: Path, algo: str) -> dict:
    state = pickle.loads((checkpoint / "algorithm_state.pkl").read_bytes())
    stats = state["metrics_logger"]["stats"]

    def lifetime(key: str) -> float:
        return float(stats[key]["lifetime_sum"])

    sampled = lifetime(
        "env_runners--num_module_steps_sampled_lifetime--shared_policy"
    )
    trained = lifetime(
        "learners--shared_policy--num_module_steps_trained_lifetime"
    )
    wall_clock = 0.0
    for name in (
        f"{algo}_train.log",
        f"{algo}_resume_train.log",
        f"{algo}_resume2_train.log",
    ):
        path = V2_ROOT / name
        if path.exists():
            matches = re.findall(r"done in ([0-9]+)s", path.read_text())
            wall_clock += sum(float(value) for value in matches)
    return {
        "iter": 1100,
        "env_steps_sampled_lifetime": lifetime(
            "env_runners--num_env_steps_sampled_lifetime"
        ),
        "module_steps_sampled_lifetime": {"shared_policy": sampled},
        "total_agent_steps_sampled_lifetime": sampled,
        "per_policy/shared_policy/module_steps_trained_lifetime": trained,
        "per_policy/shared_policy/optimizer_steps_lifetime": trained / 1024,
        "per_policy/shared_policy/learner_updates_lifetime": 1100,
        "total_optimizer_steps_lifetime": trained / 1024,
        "wall_clock_seconds": wall_clock,
        "accounting_source": f"{checkpoint}/algorithm_state.pkl",
        "algo": algo,
    }


def import_v2_seed0_reference(root: Path, algo: str) -> None:
    """Reference the exact already-trained v2 seed-0 shared run read-only."""
    destination = run_dir(root, algo, "shared", 0)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = v2_checkpoint(algo).resolve()
    source_iters = (
        V2_ROOT / f"long_baseline/{algo}_seed0_headmlp/iters.jsonl"
    ).resolve()
    manifest = {
        "experiment": "v3_shared_vs_independent",
        "status": "complete",
        "imported_exact_v2_reference": True,
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "algo": algo,
        "condition": "shared",
        "seed": 0,
        "parameterization": "shared",
        "include_agent_id": False,
        "source_checkpoint": str(checkpoint),
        "source_iters_jsonl": str(source_iters),
        "fixed_configuration": {
            "horizon": 180,
            "reward_kind": "terminal_only",
            "max_qty": 5,
            "max_price": 50,
            "head_kind": "mlp",
            "gamma": 1.0,
            "lambda": 1.0,
            "lr": 3e-4,
            "entropy_coeff": 0.01,
            "clip_param": 0.2,
            "train_batch_size_environment_steps": 4096,
            "minibatch_size": 1024,
            "num_epochs": 4,
        },
        "final_accounting": _checkpoint_metrics(checkpoint, algo),
        "note": (
            "This is the exact completed v2 seed-0 Condition-A run. The v2 "
            "directory is read-only; v3 evaluations are written here."
        ),
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (destination / "latest_checkpoint.txt").write_text(str(checkpoint))


def _is_complete(path: Path) -> bool:
    manifest = path / "run_manifest.json"
    if not manifest.exists():
        return False
    try:
        return json.loads(manifest.read_text()).get("status") == "complete"
    except (json.JSONDecodeError, OSError):
        return False


def _launch_jobs(
    jobs: list[tuple[str, list[str], Path]],
    *,
    parallel: int,
) -> None:
    queue = deque(jobs)
    active: list[tuple[str, subprocess.Popen, object, Path]] = []
    failures: list[str] = []
    while queue or active:
        while queue and len(active) < parallel:
            label, command, log_path = queue.popleft()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("a")
            log.write(
                f"\n[{datetime.now(timezone.utc).isoformat()}] START {' '.join(command)}\n"
            )
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append((label, process, log, log_path))
            print(f"[campaign] started {label} pid={process.pid}", flush=True)

        time.sleep(2)
        still_active = []
        for label, process, log, log_path in active:
            code = process.poll()
            if code is None:
                still_active.append((label, process, log, log_path))
                continue
            log.write(
                f"[{datetime.now(timezone.utc).isoformat()}] END exit={code}\n"
            )
            log.close()
            if code != 0:
                failures.append(f"{label} (exit {code}; {log_path})")
            print(f"[campaign] finished {label} exit={code}", flush=True)
        active = still_active

        if failures:
            for _, process, log, _ in active:
                process.terminate()
                log.close()
            raise RuntimeError("campaign job failure(s): " + "; ".join(failures))


def training_jobs(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    jobs = []
    for algo in ALGOS:
        import_v2_seed0_reference(args.root, algo)
        for condition in CONDITIONS:
            for seed in SEEDS:
                destination = run_dir(args.root, algo, condition, seed)
                if _is_complete(destination):
                    print(f"[campaign] skip complete training {destination.name}")
                    continue
                remaining_iters = args.iters
                resume_args: list[str] = []
                jsonl = destination / "iters.jsonl"
                latest = destination / "latest_checkpoint.txt"
                if jsonl.exists():
                    lines = [line for line in jsonl.read_text().splitlines() if line.strip()]
                    last_iter = int(json.loads(lines[-1])["iter"]) if lines else 0
                    if not latest.exists():
                        raise RuntimeError(
                            f"cannot resume {destination}: missing latest_checkpoint.txt"
                        )
                    checkpoint_path = Path(latest.read_text().strip())
                    match = re.search(r"iter_(\d+)", checkpoint_path.name)
                    if match is None:
                        raise RuntimeError(
                            f"cannot infer iteration from checkpoint {checkpoint_path}"
                        )
                    checkpoint_iter = int(match.group(1))
                    if checkpoint_iter > last_iter:
                        raise RuntimeError(
                            f"checkpoint iteration {checkpoint_iter} exceeds final metric "
                            f"iteration {last_iter} in {destination}"
                        )
                    remaining_iters = args.iters - checkpoint_iter
                    if remaining_iters <= 0:
                        raise RuntimeError(
                            f"{destination} reached checkpoint iteration {checkpoint_iter} "
                            "but has no complete manifest"
                        )
                    resume_args = [
                        "--resume_from_checkpoint", str(checkpoint_path)
                    ]
                command = [
                    sys.executable,
                    "-m", "eval.train_shared_vs_independent",
                    "--algo", algo,
                    "--condition", condition,
                    "--seed", str(seed),
                    "--iters", str(remaining_iters),
                    "--checkpoint_every", str(args.checkpoint_every),
                    "--num_env_runners", str(args.num_env_runners),
                    "--num_cpus", str(args.train_cpus_per_job),
                    "--train_batch_size", "4096",
                    "--minibatch_size", "1024",
                    "--num_epochs", "4",
                    "--out_root", str(args.root),
                    *resume_args,
                ]
                jobs.append((
                    f"train:{algo}:{condition}:seed{seed}",
                    command,
                    destination / "train.log",
                ))
    return jobs


def evaluation_jobs(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    jobs = []
    for algo in ALGOS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                destination = run_dir(args.root, algo, condition, seed)
                if not _is_complete(destination):
                    raise RuntimeError(f"training is not complete: {destination}")
                out = destination / "evaluation.json"
                if out.exists():
                    print(f"[campaign] skip existing evaluation {destination.name}")
                    continue
                checkpoint = Path(
                    (destination / "latest_checkpoint.txt").read_text().strip()
                )
                sections = ["native", "crossplay", "differentiation"]
                if condition == "independent":
                    sections.extend(["permutations", "clones"])
                command = [
                    sys.executable,
                    "-m", "eval.evaluate_shared_vs_independent",
                    "--checkpoint", str(checkpoint),
                    "--algo", algo,
                    "--condition", condition,
                    "--seed", str(seed),
                    "--out", str(out),
                    "--n_games", str(args.eval_games),
                    "--permutation_games", str(args.permutation_games),
                    "--num_cpus", str(args.eval_cpus_per_job),
                    "--sections", *sections,
                ]
                jobs.append((
                    f"eval:{algo}:{condition}:seed{seed}",
                    command,
                    destination / "eval.log",
                ))
    return jobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--stage", choices=("train", "eval", "all"), default="all")
    p.add_argument("--iters", type=int, default=1100)
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--num_env_runners", type=int, default=2)
    p.add_argument("--train_parallel", type=int, default=2)
    p.add_argument("--train_cpus_per_job", type=int, default=4)
    p.add_argument("--eval_parallel", type=int, default=2)
    p.add_argument("--eval_cpus_per_job", type=int, default=2)
    p.add_argument("--eval_games", type=int, default=100)
    p.add_argument("--permutation_games", type=int, default=10)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    campaign = {
        "experiment": "v3_shared_vs_independent",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "algorithms": list(ALGOS),
        "conditions": list(CONDITIONS),
        "training_iterations": args.iters,
        "environment_steps_per_run": args.iters * 4096,
        "evaluation_games_per_primary_cell": args.eval_games,
        "permutation_games_per_cell": args.permutation_games,
        "shared_seed0_policy": "exact completed v2 reference, not retrained",
    }
    (args.root / "campaign_manifest.json").write_text(
        json.dumps(campaign, indent=2)
    )

    if args.stage in ("train", "all"):
        _launch_jobs(training_jobs(args), parallel=args.train_parallel)
    if args.stage in ("eval", "all"):
        _launch_jobs(evaluation_jobs(args), parallel=args.eval_parallel)

    campaign["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    campaign["status"] = "complete"
    (args.root / "campaign_manifest.json").write_text(
        json.dumps(campaign, indent=2)
    )
    print(f"[campaign] complete: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
