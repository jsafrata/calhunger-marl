"""BC control vs the scripted ladder (v2-v5), seat-balanced, 100 games per opponent.

Purpose: the papers' BC control currently cites self-play survival (98.4%) -- the very
metric the papers argue is unreliable. This evaluates the same BC actor (mlp heads,
v2-era config) with 1 BC seat + 3 cadenced copies of each scripted archetype, so the
capacity-control claim rests on cross-play numbers.

Design: for each opponent version, 4 seat rotations x 25 games (BC occupies each seat
exactly 25 times); game g in 0..99 always uses seed GAME_SEED_BASE+g, so all opponents
see identical games. Reports BC-seat survival %, win %, mean terminal cash, and the same
for a 4x-BC self-play reference on the same seeds.

Usage (from research root, PYTHONPATH=.):
  python hpc/bc_vs_ladder/run_bc_vs_ladder.py \
      --bc-actor eval/results/v2_full_horizon_.../bc_diagnostic/mlp/bc_actor.pt \
      --output-dir <dir> [--n-games 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from agents.cadenced_scripted import CadencedScripted  # noqa: E402
from agents.scripted_v2 import ScriptedV2Agent  # noqa: E402
from agents.scripted_v3 import ScriptedV3Agent  # noqa: E402
from agents.scripted_v4 import ScriptedV4Agent  # noqa: E402
from agents.scripted_v5 import ScriptedV5Agent  # noqa: E402
from env.env import FoodTradingEnv  # noqa: E402
from eval.bc_diagnostic import BCActor, ClonedBCAgent  # noqa: E402

GAME_SEED_BASE = 700_000
OPPONENTS = {"v2": ScriptedV2Agent, "v3": ScriptedV3Agent,
             "v4": ScriptedV4Agent, "v5": ScriptedV5Agent}


def play(agents, seed: int) -> dict:
    env = FoodTradingEnv(seed=seed)
    obs = env.reset()
    for a in agents:
        a.reset()
    while not env.is_done():
        actions = {i: agents[i].act(obs[i], env.legal_actions(i)) for i in range(4)}
        obs, _, _, _ = env.step(actions)
    return env.get_result()


def eval_lineup(model, factory, n_games: int) -> dict:
    """factory=None -> 4x BC self-play; else 1 BC + 3x cadenced scripted."""
    surv = wins = 0
    cash = []
    per_rotation = n_games // 4
    for g in range(n_games):
        rotation = g // per_rotation if factory is not None else 0
        agents = []
        for seat in range(4):
            if factory is None or seat == rotation:
                agents.append(ClonedBCAgent(model, device="cpu"))
            else:
                agents.append(CadencedScripted("fast", seed=1_000_000 + seat,
                                               inner_factory=factory))
        res = play(agents, GAME_SEED_BASE + g)
        focal = rotation if factory is not None else 0
        p = res["players"][focal]
        surv += p["status"] == "alive"
        wins += focal in res["winner_idxs"]
        cash.append(p["final_cash"])
    return {"n_games": n_games, "survival_pct": 100.0 * surv / n_games,
            "win_pct": 100.0 * wins / n_games,
            "mean_terminal_cash": sum(cash) / len(cash)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-actor", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-games", type=int, default=100)
    args = ap.parse_args()

    model = BCActor(head_kind="mlp")
    model.load_state_dict(torch.load(args.bc_actor, map_location="cpu"))
    model.eval()

    report = {"bc_actor": str(args.bc_actor), "game_seed_base": GAME_SEED_BASE,
              "cadence": "fast", "lineups": {}}
    for name, factory in [("bc_selfplay", None)] + list(OPPONENTS.items()):
        t0 = time.time()
        r = eval_lineup(model, factory, args.n_games)
        r["elapsed_seconds"] = round(time.time() - t0, 1)
        report["lineups"][name] = r
        print(f"{name:12} surv {r['survival_pct']:5.1f}%  win {r['win_pct']:5.1f}%  "
              f"cash {r['mean_terminal_cash']:7.2f}  ({r['elapsed_seconds']:.0f}s)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "bc_vs_ladder_report.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
