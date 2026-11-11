"""Train the shared Nash Hunger PPO policy with RLlib."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from torch.utils.tensorboard import SummaryWriter

from env.rl.rllib_config import (
    MAX_SEQ_LEN,
    build_ppo_config,
    register_nash_hunger_env,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=16_384)
    parser.add_argument("--minibatch-size", type=int, default=2_048)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument(
        "--algo", type=str, default="mappo", choices=("ippo", "mappo"),
        help="Critic input: IPPO = local obs; MAPPO = centralized global obs.",
    )
    parser.add_argument(
        "--torso", type=str, default="mlp", choices=("mlp", "lstm"),
        help="Actor+critic torso — MLP feed-forward (default) or LSTM.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _save_checkpoint(algorithm: Algorithm, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    result = algorithm.save(str(directory))
    candidate = getattr(result, "checkpoint", None) or result
    path = getattr(candidate, "path", None) or getattr(result, "path", None)
    checkpoint = Path(path or str(candidate)).resolve()
    return checkpoint


def _build_algorithm(config):
    if hasattr(config, "build_algo"):
        return config.build_algo()
    return config.build()


def _numeric_metrics(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            yield from _numeric_metrics(item, child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def main(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.output / f"ppo-{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = vars(args).copy()
    resolved["output"] = str(args.output)
    resolved["resume"] = str(args.resume) if args.resume else None
    (run_dir / "run_config.json").write_text(
        json.dumps(_jsonable(resolved), indent=2)
    )

    ray.init(ignore_reinit_error=True)
    algorithm = None
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    completed = False
    try:
        if args.resume:
            register_nash_hunger_env()
            algorithm = Algorithm.from_checkpoint(str(args.resume))
        else:
            config = build_ppo_config(
                seed=args.seed,
                num_env_runners=args.num_env_runners,
                num_gpus=args.num_gpus,
                evaluation_interval=args.evaluation_interval or None,
                evaluation_episodes=args.evaluation_episodes,
                max_seq_len=args.max_seq_len,
                train_batch_size=args.train_batch_size,
                minibatch_size=args.minibatch_size,
                num_epochs=args.num_epochs,
                algo=args.algo,
                torso=args.torso,
            )
            algorithm = _build_algorithm(config)

        metrics_path = run_dir / "iterations.jsonl"
        with metrics_path.open("a") as metrics_file:
            for iteration in range(1, args.iterations + 1):
                result = algorithm.train()
                metrics_file.write(json.dumps(_jsonable(result)) + "\n")
                metrics_file.flush()
                for key, value in _numeric_metrics(result):
                    if key:
                        writer.add_scalar(key, value, iteration)
                writer.flush()
                reward = result.get("env_runners", {}).get(
                    "episode_return_mean",
                    result.get("episode_reward_mean"),
                )
                print(f"iteration={iteration} mean_return={reward}")
                if (
                    args.checkpoint_interval > 0
                    and iteration % args.checkpoint_interval == 0
                ):
                    checkpoint = _save_checkpoint(
                        algorithm, run_dir / "checkpoints"
                    )
                    (run_dir / "latest_checkpoint.txt").write_text(
                        str(checkpoint)
                    )
        completed = True
        checkpoint = _save_checkpoint(algorithm, run_dir / "checkpoints")
        (run_dir / "latest_checkpoint.txt").write_text(str(checkpoint))
        print(f"final checkpoint: {checkpoint}")
        return 0
    finally:
        if algorithm is not None:
            algorithm.stop()
        writer.close()
        ray.shutdown()
        if not completed:
            print(f"training stopped before completion; logs: {run_dir}")


if __name__ == "__main__":
    sys.exit(main(parse_args()))
