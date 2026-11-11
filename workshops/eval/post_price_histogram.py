"""One-off: for a checkpoint, play N deterministic (and stochastic) self-play
games and record every learned-seat POST's (side, price). Emit ASCII histograms.

Usage:
    python -m eval.post_price_histogram \
        --checkpoint <path> --label ippo_ent005 --n_games 100 \
        --out eval/results/v2_.../ippo_ent005_price_hist.json
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval._warnings import silence_ray_deprecations  # noqa
silence_ray_deprecations()

import ray  # noqa
from ray.rllib.algorithms.algorithm import Algorithm  # noqa
from ray.rllib.core.columns import Columns  # noqa

from env.actions import MAX_PRICE, PostAction, decode  # noqa
from env.env import FoodTradingEnv  # noqa
from env.rl.global_observation_encoder import encode_global_observation  # noqa
from env.rl.observation_encoder import encode_observation  # noqa
from env.rl.rllib_config import SHARED_POLICY_ID, register_nash_hunger_env  # noqa


def collect(module, n_games, seed_base, deterministic):
    bid_prices = Counter()
    ask_prices = Counter()
    for g in range(n_games):
        seed = seed_base + g
        env = FoodTradingEnv(seed=seed)
        obs_raw = env.reset()
        cfg = env.config
        init_state = module.get_initial_state()
        while not env.is_done():
            enc = [
                {
                    "observations": encode_observation(obs_raw[i], cfg),
                    "action_mask": env.legal_actions(i).astype(np.float32),
                    "critic_observations": encode_global_observation(env.state, i, cfg),
                }
                for i in range(4)
            ]
            batch = {Columns.OBS: {
                "observations": torch.tensor(np.stack([e["observations"] for e in enc]), dtype=torch.float32),
                "action_mask": torch.tensor(np.stack([e["action_mask"] for e in enc])),
                "critic_observations": torch.tensor(np.stack([e["critic_observations"] for e in enc]), dtype=torch.float32),
            }}
            if init_state:
                batch[Columns.STATE_IN] = {
                    k: torch.from_numpy(np.stack([v]*4)).float() for k,v in init_state.items()
                }
            with torch.no_grad():
                out = module._forward_inference(batch) if deterministic else module._forward_exploration(batch)
            actions = {}
            for i in range(4):
                aid = int(out[Columns.ACTIONS][i].item())
                actions[i] = aid
                d = decode(aid)
                if isinstance(d, PostAction):
                    if d.side == "bid":
                        bid_prices[int(d.price_per_unit)] += 1
                    else:
                        ask_prices[int(d.price_per_unit)] += 1
            obs_raw, _, _, _ = env.step(actions)
    return {
        "n_games": n_games,
        "mode": "deterministic" if deterministic else "stochastic",
        "n_bids": sum(bid_prices.values()),
        "n_asks": sum(ask_prices.values()),
        "bid_price_histogram": {str(k): bid_prices[k] for k in sorted(bid_prices)},
        "ask_price_histogram": {str(k): ask_prices[k] for k in sorted(ask_prices)},
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--n_games", type=int, default=100)
    p.add_argument("--seed_base", type=int, default=200_000)
    p.add_argument("--mode", choices=("deterministic","stochastic","both"), default="both")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
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
            r = collect(module, args.n_games,
                        args.seed_base + (0 if mode=="deterministic" else 500_000),
                        deterministic=(mode=="deterministic"))
            report["runs"][mode] = r
            print(f"    n_bids={r['n_bids']}  n_asks={r['n_asks']}")
    finally:
        algo.stop()
        ray.shutdown()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
