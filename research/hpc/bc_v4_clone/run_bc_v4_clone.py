"""BC-of-v4 capacity control: clone the strongest archetype from MIXED-lineup demos,
then measure (a) clone fidelity vs real v4 on identical seeds and (b) the clone vs the
scripted ladder.

Demo design (records ONLY v4 seats): per-game lineup schedule cycles through
  four-way [v4,v2,v3,v5]  (40% of games, 1 v4 seat),
  2xV4+2xv2 / 2xV4+2xv3 / 2xV4+2xv5  (15% each, 2 v4 seats),
  4xV4  (15%, 4 seats),
with v4 seat positions rotating by game index, so the clone sees v4's rule responding to
diverse market conditions rather than one echo chamber.

Stages: generate -> train (reuses eval.bc_diagnostic.train, mlp heads) -> fidelity
(real-v4 vs clone in the same seats, same held-out seeds, per lineup type) -> ladder
(clone 1 seat + 3x each of v2-v5, seat-balanced, + clone self-play).

Usage (from research root):
  python hpc/bc_v4_clone/run_bc_v4_clone.py --root <workdir> \
      [--n-train 3000 --n-val 300 --n-test 300 --epochs 8 --eval-games 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from agents.cadenced_scripted import CadencedScripted  # noqa: E402
from agents.scripted_v2 import ScriptedV2Agent  # noqa: E402
from agents.scripted_v3 import ScriptedV3Agent  # noqa: E402
from agents.scripted_v4 import ScriptedV4Agent  # noqa: E402
from agents.scripted_v5 import ScriptedV5Agent  # noqa: E402
from env.actions import decode  # noqa: E402
from env.env import FoodTradingEnv  # noqa: E402
from eval import bc_diagnostic as bc  # noqa: E402

FACTORIES = {"v2": ScriptedV2Agent, "v3": ScriptedV3Agent,
             "v4": ScriptedV4Agent, "v5": ScriptedV5Agent}
# (name, seat lineup as version labels) — v4 seats rotate by game index.
LINEUPS = [
    ("fourway", ["v4", "v2", "v3", "v5"]),
    ("fourway", ["v4", "v2", "v3", "v5"]),
    ("fourway", ["v4", "v2", "v3", "v5"]),
    ("fourway", ["v4", "v2", "v3", "v5"]),
    ("pair_v2", ["v4", "v4", "v2", "v2"]),
    ("pair_v3", ["v4", "v4", "v3", "v3"]),
    ("pair_v5", ["v4", "v4", "v5", "v5"]),
    ("all_v4",  ["v4", "v4", "v4", "v4"]),
    ("pair_v2", ["v4", "v4", "v2", "v2"]),
    ("pair_v3", ["v4", "v4", "v3", "v3"]),
]  # 10-game cycle: 40% fourway, 20/20/10 pairwise, 10% all-v4  (v4 seats/cycle = 18)


def lineup_for_game(g: int) -> tuple[str, list[str]]:
    name, versions = LINEUPS[g % len(LINEUPS)]
    rot = (g // len(LINEUPS)) % 4
    return name, [versions[(i - rot) % 4] for i in range(4)]


def make_agents(versions: list[str], game_seed: int) -> list:
    return [CadencedScripted("fast", seed=game_seed * 977 + i,
                             inner_factory=FACTORIES[v]) for i, v in enumerate(versions)]


def factored_ids(action: int, cfg) -> tuple[int, int, int, int, int]:
    d = decode(action)
    if isinstance(d, bc.WaitAction):
        return bc.TYPE_WAIT, 0, 0, 0, 0
    if isinstance(d, bc.CancelAction):
        return (bc.TYPE_CANCEL, cfg.food_types.index(d.food_type),
                0 if d.side == "bid" else 1, 0, 0)
    if isinstance(d, bc.TakeAction):
        return (bc.TYPE_TAKE, cfg.food_types.index(d.food_type),
                0 if d.side == "bid" else 1, 0, d.quantity - 1)
    return (bc.TYPE_POST, cfg.food_types.index(d.food_type),
            0 if d.side == "bid" else 1, d.price_per_unit - 1, d.quantity - 1)


def collect_game(game_seed: int, g: int) -> dict:
    _, versions = lineup_for_game(g)
    v4_seats = [i for i, v in enumerate(versions) if v == "v4"]
    env = FoodTradingEnv(seed=game_seed)
    obs = env.reset()
    cfg = env.config
    agents = make_agents(versions, game_seed)
    for a in agents:
        a.reset()
    rows = {k: [] for k in ("obs", "mask_packed", "action",
                            "type_id", "food_id", "side_id", "price_id", "qty_id")}
    while not env.is_done():
        actions = {}
        for i in range(4):
            mask_np = env.legal_actions(i).astype(np.float32)
            action = agents[i].act(obs[i], mask_np)
            actions[i] = action
            if i in v4_seats:
                rows["obs"].append(bc.encode_observation(obs[i], cfg).astype(np.float32))
                rows["mask_packed"].append(bc._pack_mask(mask_np > 0.5))
                rows["action"].append(np.int32(action))
                t, f, s, p, q = factored_ids(action, cfg)
                for k, v in zip(("type_id", "food_id", "side_id", "price_id", "qty_id"),
                                (t, f, s, p, q)):
                    rows[k].append(np.int32(v))
        obs, _, _, _ = env.step(actions)
    return {k: np.stack(v, axis=0) for k, v in rows.items()}


def generate(root: Path, n_train: int, n_val: int, n_test: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for split, n_games, seed_base in (("train", n_train, 100_000),
                                      ("val", n_val, 500_000),
                                      ("test", n_test, 900_000)):
        t0 = time.time()
        acc = {k: [] for k in ("obs", "mask_packed", "action",
                               "type_id", "food_id", "side_id", "price_id", "qty_id")}
        for g in range(n_games):
            r = collect_game(seed_base + g, g)
            for k in acc:
                acc[k].append(r[k])
            if (g + 1) % 200 == 0 or g == n_games - 1:
                print(f"[generate:{split}] {g+1}/{n_games} rows="
                      f"{sum(a.shape[0] for a in acc['action'])} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(root / f"{split}.npz",
                            **{k: np.concatenate(v, axis=0) for k, v in acc.items()},
                            seed_range=np.array([seed_base, seed_base + n_games],
                                                dtype=np.int64))


def seat_result(res: dict, seat: int) -> tuple[int, int, float]:
    p = res["players"][seat]
    return (1 if p["status"] == "alive" else 0,
            1 if seat in res["winner_idxs"] else 0, p["final_cash"])


def play(agents, seed: int) -> dict:
    env = FoodTradingEnv(seed=seed)
    obs = env.reset()
    for a in agents:
        a.reset()
    while not env.is_done():
        actions = {i: agents[i].act(obs[i], env.legal_actions(i)) for i in range(4)}
        obs, _, _, _ = env.step(actions)
    return env.get_result()


def eval_fidelity(model, n_games: int) -> dict:
    """Real v4 vs clone occupying the identical v4 seats, identical seeds."""
    out = {}
    fid_lineups = [("fourway", ["v4", "v2", "v3", "v5"]),
                   ("pair_v2", ["v4", "v4", "v2", "v2"]),
                   ("pair_v3", ["v4", "v4", "v3", "v3"]),
                   ("pair_v5", ["v4", "v4", "v5", "v5"])]
    for name, base in fid_lineups:
        stats = {"real": [0, 0, 0.0], "clone": [0, 0, 0.0]}
        n_seats = {"real": 0, "clone": 0}
        for g in range(n_games):
            rot = g % 4
            versions = [base[(i - rot) % 4] for i in range(4)]
            v4_seats = [i for i, v in enumerate(versions) if v == "v4"]
            seed = 950_000 + g
            for kind in ("real", "clone"):
                agents = []
                for i, v in enumerate(versions):
                    if v == "v4" and kind == "clone":
                        agents.append(bc.ClonedBCAgent(model, device="cpu"))
                    else:
                        agents.append(CadencedScripted("fast", seed=seed * 977 + i,
                                                       inner_factory=FACTORIES[v]))
                res = play(agents, seed)
                for s in v4_seats:
                    a, w, c = seat_result(res, s)
                    stats[kind][0] += a
                    stats[kind][1] += w
                    stats[kind][2] += c
                    n_seats[kind] += 1
        out[name] = {kind: {"survival_pct": 100.0 * stats[kind][0] / n_seats[kind],
                            "win_pct": 100.0 * stats[kind][1] / n_seats[kind],
                            "mean_cash": stats[kind][2] / n_seats[kind],
                            "n_seat_games": n_seats[kind]}
                     for kind in ("real", "clone")}
        r, c = out[name]["real"], out[name]["clone"]
        print(f"[fidelity:{name}] real surv {r['survival_pct']:5.1f} win {r['win_pct']:5.1f} "
              f"cash {r['mean_cash']:6.1f} | clone surv {c['survival_pct']:5.1f} "
              f"win {c['win_pct']:5.1f} cash {c['mean_cash']:6.1f}", flush=True)
    return out


def eval_ladder(model, n_games: int) -> dict:
    out = {}
    for name, factory in [("bc_selfplay", None)] + list(FACTORIES.items()):
        if name == "v4":
            pass  # clone vs its source archetype is informative too — keep
        surv = wins = 0
        cash = []
        per_rot = n_games // 4
        for g in range(n_games):
            rot = g // per_rot if factory is not None else 0
            agents = [bc.ClonedBCAgent(model, device="cpu") if (factory is None or i == rot)
                      else CadencedScripted("fast", seed=1_000_000 + i, inner_factory=factory)
                      for i in range(4)]
            res = play(agents, 700_000 + g)
            focal = rot if factory is not None else 0
            a, w, c = seat_result(res, focal)
            surv += a
            wins += w
            cash.append(c)
        out[name] = {"survival_pct": 100.0 * surv / n_games,
                     "win_pct": 100.0 * wins / n_games,
                     "mean_cash": sum(cash) / len(cash), "n_games": n_games}
        print(f"[ladder:{name}] surv {out[name]['survival_pct']:5.1f} "
              f"win {out[name]['win_pct']:5.1f} cash {out[name]['mean_cash']:7.2f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--stage", choices=("all", "generate", "train", "eval"), default="all")
    args = ap.parse_args()

    if args.stage in ("all", "generate"):
        generate(args.root, args.n_train, args.n_val, args.n_test)
    if args.stage in ("all", "train"):
        bc.train(args.root, args.epochs, args.batch_size, args.lr, "cpu",
                 head_kind="mlp", patience=5)
    if args.stage in ("all", "eval"):
        model = bc.BCActor(head_kind="mlp")
        model.load_state_dict(torch.load(args.root / "bc_actor.pt", map_location="cpu"))
        model.eval()
        report = {"fidelity": eval_fidelity(model, args.eval_games),
                  "ladder": eval_ladder(model, args.eval_games)}
        (args.root / "bc_v4_report.json").write_text(json.dumps(report, indent=2))
        print("wrote", args.root / "bc_v4_report.json")


if __name__ == "__main__":
    main()
