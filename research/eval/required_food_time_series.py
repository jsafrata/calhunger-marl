"""Evaluate learned-vs-scripted required-food inventory through time.

This intentionally records only the learned seat's two required-food inventory
means, once over all games and once over games the learned seat won.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval._warnings import silence_ray_deprecations  # noqa: E402

silence_ray_deprecations()

import ray  # noqa: E402
from ray.rllib.algorithms.algorithm import Algorithm  # noqa: E402

from agents.cadenced_scripted import CadencedScripted  # noqa: E402
from env.env import FoodTradingEnv  # noqa: E402
from env.rl.rllib_config import SHARED_POLICY_ID, register_nash_hunger_env  # noqa: E402
from eval.evaluate_shared_vs_independent import ModuleSource  # noqa: E402


def _load_source(checkpoint: Path) -> ModuleSource:
    algorithm = Algorithm.from_checkpoint(str(checkpoint.resolve()))
    try:
        module = copy.deepcopy(algorithm.get_module(SHARED_POLICY_ID))
    finally:
        algorithm.stop()
    # Use one common label so paired games use the same Torch sampling seed.
    return ModuleSource(module, "learned")


def _summarize(rows: list[np.ndarray], mask: np.ndarray, horizon: int) -> dict:
    selected = np.stack([row for row, keep in zip(rows, mask) if keep]) if mask.any() else None
    if selected is None:
        return {
            "n_games": 0,
            "food_1_mean": [None] * (horizon + 1),
            "food_2_mean": [None] * (horizon + 1),
            "n_observed_at_time": [0] * (horizon + 1),
        }
    valid = np.isfinite(selected[:, :, 0])
    counts = valid.sum(axis=0)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(selected, axis=0)
    return {
        "n_games": int(selected.shape[0]),
        "food_1_mean": [None if not np.isfinite(v) else round(float(v), 6) for v in means[:, 0]],
        "food_2_mean": [None if not np.isfinite(v) else round(float(v), 6) for v in means[:, 1]],
        "n_observed_at_time": [int(v) for v in counts],
    }


def evaluate(
    source: ModuleSource,
    *,
    n_games: int,
    seed_base: int,
) -> dict:
    horizon = FoodTradingEnv().config.game_duration_seconds
    trajectories: list[np.ndarray] = []
    wins: list[bool] = []

    for game_idx in range(n_games):
        game_seed = seed_base + game_idx
        env = FoodTradingEnv(seed=game_seed)
        observations = env.reset()
        source.reset(game_seed, [0])

        scripted = {}
        for seat in (1, 2, 3):
            agent = CadencedScripted("fast", seed=game_seed * 17 + seat)
            agent.reset()
            scripted[seat] = agent

        required = env.config.required_foods(env.state.players[0].produces)
        if len(required) != 2:
            raise RuntimeError(f"expected two required foods, got {required}")

        trajectory = np.full((horizon + 1, 2), np.nan, dtype=float)

        def record() -> None:
            tick = env.state.elapsed_seconds
            if tick <= horizon:
                trajectory[tick] = [
                    env.state.players[0].inventory[required[0]],
                    env.state.players[0].inventory[required[1]],
                ]

        record()
        while not env.is_done():
            actions = source.act_batch(
                env, observations, [0], deterministic=False
            )
            for seat, agent in scripted.items():
                actions[seat] = agent.act(
                    observations[seat], env.legal_actions(seat)
                )
            observations, _, _, _ = env.step(actions)
            record()

        result = env.get_result()
        trajectories.append(trajectory)
        wins.append(0 in set(result["winner_idxs"]))

        if (game_idx + 1) % 10 == 0:
            print(
                f"  {game_idx + 1}/{n_games} games, "
                f"wins={sum(wins)}",
                flush=True,
            )

    win_mask = np.asarray(wins, dtype=bool)
    return {
        "time_seconds": list(range(horizon + 1)),
        "all_games": _summarize(
            trajectories, np.ones(n_games, dtype=bool), horizon
        ),
        "wins_only": _summarize(trajectories, win_mask, horizon),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("LABEL", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=700_000)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ray.init(
        ignore_reinit_error=True,
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
    )
    register_nash_hunger_env()
    output = {
        "n_games_per_model": args.n_games,
        "evaluation_seed_base": args.seed_base,
        "inventory_kind": "total inventory (reserved units included)",
        "required_food_order": "Food 1 and Food 2 follow config order A, B, C, D after excluding the seat's two produced foods.",
        "models": {},
    }
    try:
        for label, checkpoint_text in args.model:
            checkpoint = Path(checkpoint_text)
            print(f"[{label}] loading {checkpoint}", flush=True)
            source = _load_source(checkpoint)
            output["models"][label] = evaluate(
                source,
                n_games=args.n_games,
                seed_base=args.seed_base,
            )
    finally:
        ray.shutdown()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
