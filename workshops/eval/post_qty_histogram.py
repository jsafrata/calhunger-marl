"""One-off diagnostic: for a trained checkpoint, play N deterministic
self-play games and record every learned-seat POST action's (side, quantity).
Emit a per-side histogram of posted quantities.

Usage:
    python -m eval.post_qty_histogram \
        --checkpoint eval/results/long_baseline/ippo_ent005/checkpoints/iter_0200 \
        --label ippo_ent005 \
        --n_games 100 \
        --mode deterministic \
        --out eval/results/deep_eval/ippo_ent005_qty_hist.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval._warnings import silence_ray_deprecations  # noqa: E402
silence_ray_deprecations()

import ray  # noqa: E402
from ray.rllib.algorithms.algorithm import Algorithm  # noqa: E402
from ray.rllib.core.columns import Columns  # noqa: E402

from env.actions import MAX_QTY, PostAction, decode  # noqa: E402
from env.env import FoodTradingEnv  # noqa: E402
from env.rl.global_observation_encoder import encode_global_observation  # noqa: E402
from env.rl.observation_encoder import encode_observation  # noqa: E402
from env.rl.rllib_config import SHARED_POLICY_ID, register_nash_hunger_env  # noqa: E402


def collect(module, n_games: int, seed_base: int, deterministic: bool) -> dict:
    bid_qtys = Counter()  # qty (1..MAX_QTY) -> count
    ask_qtys = Counter()
    total_posts = 0
    total_actions = 0
    for g in range(n_games):
        seed = seed_base + g
        env = FoodTradingEnv(seed=seed)
        obs_raw = env.reset()
        cfg = env.config
        init_state = module.get_initial_state()
        while not env.is_done():
            per_seat_obs = [obs_raw[i] for i in range(4)]
            enc_obs = [
                {
                    "observations": encode_observation(o, cfg),
                    "action_mask": env.legal_actions(i).astype(np.float32),
                    "critic_observations": encode_global_observation(env.state, i, cfg),
                }
                for i, o in enumerate(per_seat_obs)
            ]
            batch_obs = {
                "observations": torch.tensor(
                    np.stack([e["observations"] for e in enc_obs]), dtype=torch.float32),
                "action_mask": torch.tensor(
                    np.stack([e["action_mask"] for e in enc_obs])),
                "critic_observations": torch.tensor(
                    np.stack([e["critic_observations"] for e in enc_obs]), dtype=torch.float32),
            }
            batch = {Columns.OBS: batch_obs}
            if init_state:
                B = 4
                batch[Columns.STATE_IN] = {
                    k: torch.from_numpy(np.stack([v] * B)).float()
                    for k, v in init_state.items()
                }
            with torch.no_grad():
                if deterministic:
                    out = module._forward_inference(batch)
                else:
                    out = module._forward_exploration(batch)
            actions = {}
            for i in range(4):
                aid = int(out[Columns.ACTIONS][i].item())
                actions[i] = aid
                total_actions += 1
                d = decode(aid)
                if isinstance(d, PostAction):
                    total_posts += 1
                    q = int(d.quantity)
                    if d.side == "bid":
                        bid_qtys[q] += 1
                    else:
                        ask_qtys[q] += 1
            obs_raw, _, _, _ = env.step(actions)
    return {
        "n_games": n_games,
        "mode": "deterministic" if deterministic else "stochastic",
        "total_actions": total_actions,
        "total_posts": total_posts,
        "post_frac_of_actions": total_posts / max(1, total_actions),
        "bid_qty_histogram": {str(k): bid_qtys[k] for k in sorted(bid_qtys)},
        "ask_qty_histogram": {str(k): ask_qtys[k] for k in sorted(ask_qtys)},
        "n_bids": sum(bid_qtys.values()),
        "n_asks": sum(ask_qtys.values()),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--n_games", type=int, default=100)
    p.add_argument("--seed_base", type=int, default=200_000)
    p.add_argument("--mode", choices=("deterministic","stochastic","both"), default="both")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False)
    register_nash_hunger_env()
    algo = Algorithm.from_checkpoint(str(args.checkpoint.resolve()))
    module = algo.get_module(SHARED_POLICY_ID)
    report = {"checkpoint": str(args.checkpoint), "label": args.label, "runs": {}}
    modes = ("deterministic","stochastic") if args.mode == "both" else (args.mode,)
    try:
        for mode in modes:
            print(f"[{args.label}] {mode} — {args.n_games} self-play games")
            r = collect(module, args.n_games, args.seed_base + (0 if mode == "deterministic" else 500_000),
                        deterministic=(mode == "deterministic"))
            report["runs"][mode] = r
            print(f"    n_bids={r['n_bids']}  n_asks={r['n_asks']}  post_frac={r['post_frac_of_actions']:.3f}")
            print(f"    bid qtys: {r['bid_qty_histogram']}")
            print(f"    ask qtys: {r['ask_qty_histogram']}")
    finally:
        algo.stop()
        ray.shutdown()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
