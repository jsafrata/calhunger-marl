"""Deep evaluation for the v3 shared-vs-independent experiment.

Unlike ``eval.deep_eval``, this evaluator can assign a different RLModule to
every seat.  It keeps policy identity in all aggregates and supports native
heterogeneous populations, all 24 policy/seat permutations, clones, and
rotated fixed-opponent cross-play.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import zlib
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval._warnings import silence_ray_deprecations  # noqa: E402
silence_ray_deprecations()

import ray  # noqa: E402
from ray.rllib.algorithms.algorithm import Algorithm  # noqa: E402
from ray.rllib.core.columns import Columns  # noqa: E402

from agents.cadenced_scripted import CadencedScripted  # noqa: E402
from env.actions import (  # noqa: E402
    CancelAction,
    PostAction,
    TakeAction,
    WaitAction,
    decode,
)
from env.env import FoodTradingEnv  # noqa: E402
from env.rl.factored_actor import factored_masks_from_flat_tensor  # noqa: E402
from env.rl.global_observation_encoder import encode_global_observation  # noqa: E402
from env.rl.observation_encoder import encode_observation  # noqa: E402
from env.rl.rllib_config import (  # noqa: E402
    INDEPENDENT_POLICY_IDS,
    SHARED_POLICY_ID,
    register_nash_hunger_env,
)


DEFAULT_V2_ROOT = Path(
    "eval/results/"
    "v2_full_horizon_terminal_only_qty5_price50_deep_heads_gamma1_lambda1"
)
DEFAULT_HISTORICAL_IPPO = (
    DEFAULT_V2_ROOT
    / "long_baseline/ippo_seed0_headmlp/checkpoints/iter_1100"
)
DEFAULT_HISTORICAL_MAPPO = (
    DEFAULT_V2_ROOT
    / "long_baseline/mappo_seed0_headmlp/checkpoints/iter_0700"
)
DEFAULT_BC = DEFAULT_V2_ROOT / "bc_diagnostic/mlp/bc_actor.pt"
DEFAULT_BC_DATASET = DEFAULT_V2_ROOT / "bc_diagnostic/shared_dataset/val.npz"


def _seed_for(label: str, mode: str, offset: int = 0) -> int:
    """Stable paired evaluation seeds, independent of Python hash state."""
    return 1_000_000 + zlib.crc32(f"{label}:{mode}".encode()) % 500_000 + offset


def _stats(values: Iterable[float]) -> dict:
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None}
    return {
        "n": int(xs.size),
        "mean": float(xs.mean()),
        "sd": float(xs.std(ddof=1)) if xs.size > 1 else 0.0,
        "median": float(np.median(xs)),
    }


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


class PolicySource:
    """A policy implementation reusable in one or more lineup seats."""

    kind = "unknown"

    def reset(self, game_seed: int, seats: list[int]) -> None:
        del game_seed, seats

    def act_batch(
        self,
        env: FoodTradingEnv,
        observations: dict[int, dict],
        seats: list[int],
        *,
        deterministic: bool,
    ) -> dict[int, int]:
        raise NotImplementedError


class ModuleSource(PolicySource):
    kind = "rllib"

    def __init__(self, module, label: str) -> None:
        self.module = module
        self.label = label
        self.module.eval()
        self.include_agent_id = bool(
            getattr(module, "include_agent_id", False)
        )

    def reset(self, game_seed: int, seats: list[int]) -> None:
        del seats
        torch.manual_seed(game_seed ^ zlib.crc32(self.label.encode()))

    def act_batch(
        self,
        env: FoodTradingEnv,
        observations: dict[int, dict],
        seats: list[int],
        *,
        deterministic: bool,
    ) -> dict[int, int]:
        if not seats:
            return {}
        cfg = env.config
        local = np.stack([
            encode_observation(
                observations[seat],
                cfg,
                include_agent_id=self.include_agent_id,
            )
            for seat in seats
        ])
        masks = np.stack([
            env.legal_actions(seat).astype(np.float32) for seat in seats
        ])
        global_obs = np.stack([
            encode_global_observation(
                env.state,
                seat,
                cfg,
                include_agent_id=self.include_agent_id,
            )
            for seat in seats
        ])
        batch = {
            Columns.OBS: {
                "observations": torch.from_numpy(local).float(),
                "action_mask": torch.from_numpy(masks).float(),
                "critic_observations": torch.from_numpy(global_obs).float(),
            }
        }
        with torch.no_grad():
            output = (
                self.module._forward_inference(batch)
                if deterministic
                else self.module._forward_exploration(batch)
            )
        return {
            seat: int(output[Columns.ACTIONS][pos].item())
            for pos, seat in enumerate(seats)
        }


class BCSource(PolicySource):
    kind = "mlp_bc"

    def __init__(self, checkpoint: Path) -> None:
        from eval.bc_diagnostic import BCActor

        self.label = "mlp_bc"
        self.model = BCActor(head_kind="mlp")
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def reset(self, game_seed: int, seats: list[int]) -> None:
        del seats
        torch.manual_seed(game_seed ^ 0xBCBC)

    def act_batch(
        self,
        env: FoodTradingEnv,
        observations: dict[int, dict],
        seats: list[int],
        *,
        deterministic: bool,
    ) -> dict[int, int]:
        if not seats:
            return {}
        obs = torch.from_numpy(np.stack([
            encode_observation(observations[s], env.config) for s in seats
        ])).float()
        masks_flat = torch.from_numpy(np.stack([
            env.legal_actions(s).astype(np.float32) for s in seats
        ])).float()
        with torch.no_grad():
            embedding, _ = self.model.torso(obs, None)
            masks = factored_masks_from_flat_tensor(masks_flat)
            factored, _, _ = self.model.actor.sample(
                embedding, masks, deterministic=deterministic
            )
            from env.rl.factored_actor import factored_to_flat_batch
            actions = factored_to_flat_batch(factored)
        return {s: int(actions[i].item()) for i, s in enumerate(seats)}


class ScriptedSource(PolicySource):
    kind = "scripted_v1_fast"

    def __init__(self, salt: int = 0) -> None:
        self.salt = salt
        self._agents: dict[int, CadencedScripted] = {}

    def reset(self, game_seed: int, seats: list[int]) -> None:
        self._agents = {}
        for seat in seats:
            agent = CadencedScripted(
                "fast", seed=game_seed * 977 + seat + self.salt
            )
            agent.reset()
            self._agents[seat] = agent

    def act_batch(
        self,
        env: FoodTradingEnv,
        observations: dict[int, dict],
        seats: list[int],
        *,
        deterministic: bool,
    ) -> dict[int, int]:
        del deterministic
        return {
            s: self._agents[s].act(observations[s], env.legal_actions(s))
            for s in seats
        }


def _new_policy_accumulator() -> dict[str, list]:
    fields = (
        "survived", "won", "died", "terminal_reward", "terminal_cash",
        "relative_cash", "cash_rank", "required_food_min",
        "required_food_total", "trades", "trade_volume", "buy_trades",
        "sell_trades", "units_bought", "units_sold", "maker_trades",
        "maker_volume", "posts", "post_bids", "post_asks", "takes",
        "waits", "cancels", "decisions", "terminal_cash_winner",
        "terminal_cash_surviving_nonwinner", "terminal_cash_dead",
    )
    out: dict[str, list] = {field: [] for field in fields}
    for prefix in ("action_food", "trade_food_bought", "trade_food_sold"):
        out[prefix] = []
    out["post_prices"] = []
    out["action_quantities"] = []
    out["trade_prices"] = []
    out["trade_quantities"] = []
    return out


def _summarize_policy(acc: dict[str, list]) -> dict:
    n = len(acc["survived"])
    wins = sum(acc["won"])
    deaths = sum(acc["died"])
    action_counts = {
        name: int(sum(acc[name]))
        for name in ("waits", "cancels", "takes", "posts")
    }
    total_actions = sum(action_counts.values())
    food_actions = Counter(acc["action_food"])
    bought_food = Counter(acc["trade_food_bought"])
    sold_food = Counter(acc["trade_food_sold"])
    return {
        "n_policy_games": n,
        "survival_probability": _ratio(sum(acc["survived"]), n),
        "win_probability": _ratio(wins, n),
        "win_probability_given_survival": _ratio(
            wins, sum(acc["survived"])
        ),
        "death_probability": _ratio(deaths, n),
        "expected_terminal_reward": _ratio(
            sum(acc["terminal_reward"]), n
        ),
        "p_win_minus_p_die": _ratio(wins - deaths, n),
        "terminal_cash": _stats(acc["terminal_cash"]),
        "terminal_cash_given_winner": _stats(acc["terminal_cash_winner"]),
        "terminal_cash_given_surviving_nonwinner": _stats(
            acc["terminal_cash_surviving_nonwinner"]
        ),
        "terminal_cash_given_death": _stats(acc["terminal_cash_dead"]),
        "relative_terminal_cash": _stats(acc["relative_cash"]),
        "cash_rank": _stats(acc["cash_rank"]),
        "minimum_required_food_inventory": _stats(acc["required_food_min"]),
        "total_required_food_inventory": _stats(acc["required_food_total"]),
        "action_type_counts": action_counts,
        "action_type_frequencies": {
            key: _ratio(value, total_actions)
            for key, value in action_counts.items()
        },
        "executed_trades": _stats(acc["trades"]),
        "trade_volume": _stats(acc["trade_volume"]),
        "buy_trades": _stats(acc["buy_trades"]),
        "sell_trades": _stats(acc["sell_trades"]),
        "units_bought": _stats(acc["units_bought"]),
        "units_sold": _stats(acc["units_sold"]),
        "posts": _stats(acc["posts"]),
        "post_bids": _stats(acc["post_bids"]),
        "post_asks": _stats(acc["post_asks"]),
        "takes": _stats(acc["takes"]),
        "maker_trades": _stats(acc["maker_trades"]),
        "maker_volume": _stats(acc["maker_volume"]),
        "maker_execution_per_post": _ratio(
            sum(acc["maker_trades"]), sum(acc["posts"])
        ),
        "action_food_counts": dict(sorted(food_actions.items())),
        "foods_bought_counts": dict(sorted(bought_food.items())),
        "foods_sold_counts": dict(sorted(sold_food.items())),
        "post_price_distribution": _stats(acc["post_prices"]),
        "action_quantity_distribution": _stats(acc["action_quantities"]),
        "trade_price_distribution": _stats(acc["trade_prices"]),
        "trade_quantity_distribution": _stats(acc["trade_quantities"]),
        "post_price_counts": dict(sorted(Counter(acc["post_prices"]).items())),
        "action_quantity_counts": dict(sorted(Counter(
            acc["action_quantities"]
        ).items())),
        "trade_price_counts": dict(sorted(Counter(acc["trade_prices"]).items())),
        "trade_quantity_counts": dict(sorted(Counter(
            acc["trade_quantities"]
        ).items())),
    }


AssignmentFactory = Callable[[int], list[tuple[str, PolicySource]]]


def evaluate_games(
    assignment_factory: AssignmentFactory,
    *,
    n_games: int,
    seed_base: int,
    deterministic: bool,
) -> dict:
    per_policy = defaultdict(_new_policy_accumulator)
    per_seat = defaultdict(_new_policy_accumulator)
    survivors, market_trades, market_volume, reached_horizon = [], [], [], []
    all_survive, no_survivors = [], []
    t0 = time.time()

    for game_idx in range(n_games):
        game_seed = seed_base + game_idx
        env = FoodTradingEnv(seed=game_seed)
        observations = env.reset()
        assignments = assignment_factory(game_idx)
        if len(assignments) != 4:
            raise ValueError("assignment factory must return exactly four seats")

        groups: dict[int, tuple[PolicySource, list[int]]] = {}
        for seat, (_, source) in enumerate(assignments):
            key = id(source)
            groups.setdefault(key, (source, []))[1].append(seat)
        for source, seats in groups.values():
            source.reset(game_seed, seats)

        game_actions = [Counter() for _ in range(4)]
        game_food_actions = [[] for _ in range(4)]
        game_post_prices = [[] for _ in range(4)]
        game_action_qty = [[] for _ in range(4)]
        order_owner: dict[int, int] = {}
        maker_trades = [0] * 4
        maker_volume = [0] * 4
        trades_seen = 0

        while not env.is_done():
            active = [p.idx for p in env.state.players if p.status == "alive"]
            actions: dict[int, int] = {}
            for source, group_seats in groups.values():
                source_seats = [s for s in group_seats if s in active]
                actions.update(source.act_batch(
                    env,
                    observations,
                    source_seats,
                    deterministic=deterministic,
                ))

            for seat, action in actions.items():
                decoded = decode(action)
                if isinstance(decoded, WaitAction):
                    game_actions[seat]["waits"] += 1
                elif isinstance(decoded, CancelAction):
                    game_actions[seat]["cancels"] += 1
                    game_food_actions[seat].append(decoded.food_type)
                elif isinstance(decoded, TakeAction):
                    game_actions[seat]["takes"] += 1
                    game_food_actions[seat].append(decoded.food_type)
                    game_action_qty[seat].append(decoded.quantity)
                elif isinstance(decoded, PostAction):
                    game_actions[seat]["posts"] += 1
                    game_actions[seat][
                        "post_bids" if decoded.side == "bid" else "post_asks"
                    ] += 1
                    game_food_actions[seat].append(decoded.food_type)
                    game_post_prices[seat].append(decoded.price_per_unit)
                    game_action_qty[seat].append(decoded.quantity)
                game_actions[seat]["decisions"] += 1

            observations, _, _, info = env.step(actions)
            for seat, applied in info.actions_applied.items():
                if (
                    applied.get("applied")
                    and applied.get("kind") == "post"
                    and applied.get("order_id") is not None
                ):
                    order_owner[int(applied["order_id"])] = seat
            new_trades = env.state.trades[trades_seen:]
            trades_seen = len(env.state.trades)
            for trade in new_trades:
                maker = order_owner.get(trade.maker_order_id)
                if maker is not None:
                    maker_trades[maker] += 1
                    maker_volume[maker] += trade.quantity

        result = env.get_result()
        winners = set(result["winner_idxs"])
        cash = [float(p["final_cash"]) for p in result["players"]]
        n_surv = sum(p["status"] == "alive" for p in result["players"])
        survivors.append(n_surv)
        all_survive.append(int(n_surv == 4))
        no_survivors.append(int(n_surv == 0))
        reached_horizon.append(int(result["elapsed_seconds"] >= 180))
        market_trades.append(len(env.state.trades))
        market_volume.append(sum(t.quantity for t in env.state.trades))

        for seat, (label, _) in enumerate(assignments):
            player = result["players"][seat]
            alive = player["status"] == "alive"
            won = seat in winners
            required = [
                f for f in env.config.food_types if f not in player["produces"]
            ]
            required_values = [player["final_inventory"][f] for f in required]
            own_trades = [
                t for t in env.state.trades
                if t.buyer_idx == seat or t.seller_idx == seat
            ]
            bought = [t for t in env.state.trades if t.buyer_idx == seat]
            sold = [t for t in env.state.trades if t.seller_idx == seat]
            row = {
                "survived": int(alive),
                "won": int(won),
                "died": int(not alive),
                "terminal_reward": 1 if won else (0 if alive else -1),
                "terminal_cash": cash[seat],
                "relative_cash": cash[seat] - float(np.mean(
                    [cash[j] for j in range(4) if j != seat]
                )),
                "cash_rank": 1 + sum(c > cash[seat] for c in cash),
                "required_food_min": min(required_values),
                "required_food_total": sum(required_values),
                "trades": len(own_trades),
                "trade_volume": sum(t.quantity for t in own_trades),
                "buy_trades": len(bought),
                "sell_trades": len(sold),
                "units_bought": sum(t.quantity for t in bought),
                "units_sold": sum(t.quantity for t in sold),
                "maker_trades": maker_trades[seat],
                "maker_volume": maker_volume[seat],
                "posts": game_actions[seat]["posts"],
                "post_bids": game_actions[seat]["post_bids"],
                "post_asks": game_actions[seat]["post_asks"],
                "takes": game_actions[seat]["takes"],
                "waits": game_actions[seat]["waits"],
                "cancels": game_actions[seat]["cancels"],
                "decisions": game_actions[seat]["decisions"],
                "terminal_cash_winner": cash[seat] if won else None,
                "terminal_cash_surviving_nonwinner": (
                    cash[seat] if alive and not won else None
                ),
                "terminal_cash_dead": cash[seat] if not alive else None,
            }
            for accumulator in (per_policy[label], per_seat[str(seat)]):
                for field, value in row.items():
                    if value is not None:
                        accumulator[field].append(value)
                accumulator["action_food"].extend(game_food_actions[seat])
                accumulator["post_prices"].extend(game_post_prices[seat])
                accumulator["action_quantities"].extend(game_action_qty[seat])
                accumulator["trade_food_bought"].extend(t.food_type for t in bought)
                accumulator["trade_food_sold"].extend(t.food_type for t in sold)
                accumulator["trade_prices"].extend(t.price_per_unit for t in own_trades)
                accumulator["trade_quantities"].extend(t.quantity for t in own_trades)

    return {
        "n_games": n_games,
        "mode": "deterministic" if deterministic else "stochastic",
        "seed_base": seed_base,
        "elapsed_seconds": time.time() - t0,
        "population": {
            "mean_survivors": float(np.mean(survivors)),
            "all_four_survive_probability": float(np.mean(all_survive)),
            "no_survivors_probability": float(np.mean(no_survivors)),
            "reach_horizon_probability": float(np.mean(reached_horizon)),
            "market_trades": _stats(market_trades),
            "market_trade_volume": _stats(market_volume),
        },
        "per_policy": {
            label: _summarize_policy(acc)
            for label, acc in sorted(per_policy.items())
        },
        "per_seat": {
            seat: _summarize_policy(acc)
            for seat, acc in sorted(per_seat.items())
        },
    }


def _load_algorithm(path: Path) -> Algorithm:
    if not path.exists():
        raise FileNotFoundError(path)
    return Algorithm.from_checkpoint(str(path.resolve()))


def _module_sources(
    algorithm: Algorithm,
    condition: str,
) -> dict[str, ModuleSource]:
    if condition == "independent":
        return {
            pid: ModuleSource(copy.deepcopy(algorithm.get_module(pid)), pid)
            for pid in INDEPENDENT_POLICY_IDS
        }
    return {
        SHARED_POLICY_ID: ModuleSource(
            copy.deepcopy(algorithm.get_module(SHARED_POLICY_ID)),
            SHARED_POLICY_ID,
        )
    }


def _shared_source_from_checkpoint(path: Path, label: str) -> ModuleSource:
    algorithm = _load_algorithm(path)
    try:
        module = copy.deepcopy(algorithm.get_module(SHARED_POLICY_ID))
    finally:
        algorithm.stop()
    return ModuleSource(module, label)


def _type_probabilities(module, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float()
        mask_t = torch.from_numpy(masks).float()
        embedding, _ = module.torso(obs_t, None)
        factored_masks = factored_masks_from_flat_tensor(mask_t)
        logits = module.actor.type_head(embedding)
        logits = logits.masked_fill(~factored_masks.type_mask, float("-inf"))
        return torch.softmax(logits, dim=-1).cpu().numpy()


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m), axis=-1)
    js += 0.5 * np.sum(q * np.log(q / m), axis=-1)
    return float(np.mean(js))


def _counter_js_divergence(a: dict, b: dict) -> float | None:
    keys = sorted(set(a) | set(b), key=str)
    if not keys:
        return None
    p = np.asarray([float(a.get(key, 0)) for key in keys], dtype=float)
    q = np.asarray([float(b.get(key, 0)) for key in keys], dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    return _js_divergence((p / p.sum())[None, :], (q / q.sum())[None, :])


def behavioral_differentiation(
    sources: dict[str, ModuleSource],
    dataset: Path,
    n_obs: int = 512,
    native_results: dict | None = None,
) -> dict:
    if len(sources) < 2:
        return {"applicable": False, "reason": "condition has one policy"}
    data = np.load(dataset)
    obs = data["obs"][:n_obs].astype(np.float32)
    packed = data["mask_packed"][:n_obs]
    n_actions = next(iter(sources.values())).module.observation_space[
        "action_mask"
    ].shape[0]
    masks = np.unpackbits(packed, axis=-1)[..., :n_actions].astype(np.float32)
    probs = {
        pid: _type_probabilities(source.module, obs, masks)
        for pid, source in sources.items()
    }
    pairs = {}
    for a, b in combinations_sorted(sources):
        pairs[f"{a}__{b}"] = {
            "type_output_js_divergence_nats": _js_divergence(probs[a], probs[b]),
            "type_argmax_disagreement": float(np.mean(
                probs[a].argmax(axis=-1) != probs[b].argmax(axis=-1)
            )),
        }
    empirical = {}
    for mode, result in (native_results or {}).items():
        per_policy = result["per_policy"]
        mode_pairs = {}
        for a, b in combinations_sorted(sources):
            left, right = per_policy[a], per_policy[b]
            mode_pairs[f"{a}__{b}"] = {
                "action_type_js_divergence_nats": _counter_js_divergence(
                    left["action_type_counts"], right["action_type_counts"]
                ),
                "action_food_js_divergence_nats": _counter_js_divergence(
                    left["action_food_counts"], right["action_food_counts"]
                ),
                "foods_bought_js_divergence_nats": _counter_js_divergence(
                    left["foods_bought_counts"], right["foods_bought_counts"]
                ),
                "foods_sold_js_divergence_nats": _counter_js_divergence(
                    left["foods_sold_counts"], right["foods_sold_counts"]
                ),
                "post_price_js_divergence_nats": _counter_js_divergence(
                    left["post_price_counts"], right["post_price_counts"]
                ),
                "action_quantity_js_divergence_nats": _counter_js_divergence(
                    left["action_quantity_counts"], right["action_quantity_counts"]
                ),
                "trade_price_js_divergence_nats": _counter_js_divergence(
                    left["trade_price_counts"], right["trade_price_counts"]
                ),
                "trade_quantity_js_divergence_nats": _counter_js_divergence(
                    left["trade_quantity_counts"], right["trade_quantity_counts"]
                ),
                "absolute_terminal_cash_gap": abs(
                    left["terminal_cash"]["mean"] - right["terminal_cash"]["mean"]
                ),
                "absolute_required_food_min_gap": abs(
                    left["minimum_required_food_inventory"]["mean"]
                    - right["minimum_required_food_inventory"]["mean"]
                ),
                "absolute_maker_volume_gap": abs(
                    left["maker_volume"]["mean"] - right["maker_volume"]["mean"]
                ),
            }
        empirical[mode] = mode_pairs
    return {
        "applicable": True,
        "dataset": str(dataset.resolve()),
        "n_observations": int(len(obs)),
        "pairwise": pairs,
        "empirical_native_pairwise": empirical,
    }


def combinations_sorted(mapping: dict) -> list[tuple[str, str]]:
    keys = sorted(mapping)
    return [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]


def _run_modes(
    assignment: AssignmentFactory,
    *,
    n_games: int,
    seed_label: str,
) -> dict:
    return {
        mode: evaluate_games(
            assignment,
            n_games=n_games,
            seed_base=_seed_for(seed_label, mode),
            deterministic=(mode == "deterministic"),
        )
        for mode in ("deterministic", "stochastic")
    }


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    register_nash_hunger_env()
    condition = args.condition
    target_algo = _load_algorithm(args.checkpoint)
    try:
        sources = _module_sources(target_algo, condition)
    finally:
        # Algorithms restored from training checkpoints recreate their env
        # runners. Evaluation needs only the copied modules; closing each
        # Algorithm immediately prevents multi-checkpoint cross-play from
        # exhausting Ray CPU resources.
        target_algo.stop()
    report = {
        "experiment": "v3_shared_vs_independent",
        "algo": args.algo,
        "condition": condition,
        "training_seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluation_convention": {
            "modes": ["deterministic", "stochastic"],
            "research_seed_unit": "training seed; evaluation games are repeated trials",
            "paired_evaluation_seeds_across_conditions": True,
        },
        "sections": {},
    }
    try:
        policy_ids = list(sources)
        if "native" in args.sections:
            if condition == "independent":
                native = [(pid, sources[pid]) for pid in INDEPENDENT_POLICY_IDS]
            else:
                source = sources[SHARED_POLICY_ID]
                native = [(SHARED_POLICY_ID, source)] * 4
            report["sections"]["native"] = _run_modes(
                lambda _: native,
                n_games=args.n_games,
                seed_label="native",
            )

        if condition == "independent" and "permutations" in args.sections:
            permutation_report = {}
            for perm in permutations(range(4)):
                name = "".join(str(x) for x in perm)
                assignment = [
                    (f"pi_{policy_idx}", sources[f"pi_{policy_idx}"])
                    for policy_idx in perm
                ]
                permutation_report[name] = _run_modes(
                    lambda _, assignment=assignment: assignment,
                    n_games=args.permutation_games,
                    # Every permutation sees the same games so differences
                    # isolate policy-to-seat reassignment.
                    seed_label="permutation",
                )
            report["sections"]["permutations"] = {
                "n_permutations": 24,
                "games_per_permutation_per_mode": args.permutation_games,
                "original_training_assignment": "0123",
                "results": permutation_report,
            }

        if condition == "independent" and "clones" in args.sections:
            clone_report = {}
            for pid in INDEPENDENT_POLICY_IDS:
                lineup = [(pid, sources[pid])] * 4
                clone_report[pid] = _run_modes(
                    lambda _, lineup=lineup: lineup,
                    n_games=args.n_games,
                    seed_label="clone",
                )
            report["sections"]["clones"] = clone_report

        if "crossplay" in args.sections:
            hist_ippo = _shared_source_from_checkpoint(
                args.historical_ippo, "historical_ippo"
            )
            hist_mappo = _shared_source_from_checkpoint(
                args.historical_mappo, "historical_mappo"
            )
            bc = BCSource(args.bc_checkpoint)
            scripted = ScriptedSource(salt=11)
            opponent_sets: dict[str, list[tuple[str, PolicySource]]] = {
                "scripted_v1_fast": [("scripted_v1_fast", scripted)] * 3,
                "mlp_bc": [("mlp_bc", bc)] * 3,
                "historical_shared_ippo": [("historical_ippo", hist_ippo)] * 3,
                "historical_shared_mappo": [("historical_mappo", hist_mappo)] * 3,
                "mixed": [
                    ("scripted_v1_fast", scripted),
                    ("mlp_bc", bc),
                    ("historical_ippo", hist_ippo),
                ],
            }
            cross = {
                "historical_checkpoints": {
                    "ippo": str(args.historical_ippo.resolve()),
                    "mappo": str(args.historical_mappo.resolve()),
                    "mlp_bc": str(args.bc_checkpoint.resolve()),
                },
                "games_per_target_opponent_per_mode": args.n_games,
                "seat_rotation": "game_index mod 4; balanced when n_games divisible by 4",
                "results": {},
            }
            for pid, target in sources.items():
                cross["results"][pid] = {}
                for opponent_name, opponents in opponent_sets.items():
                    def assignment_factory(
                        game_idx: int,
                        *,
                        target=target,
                        pid=pid,
                        opponents=opponents,
                    ) -> list[tuple[str, PolicySource]]:
                        target_seat = game_idx % 4
                        lineup: list[tuple[str, PolicySource] | None] = [None] * 4
                        lineup[target_seat] = (pid, target)
                        opponent_iter = iter(opponents)
                        for seat in range(4):
                            if lineup[seat] is None:
                                lineup[seat] = next(opponent_iter)
                        return list(lineup)  # type: ignore[arg-type]

                    cross["results"][pid][opponent_name] = _run_modes(
                        assignment_factory,
                        n_games=args.n_games,
                        seed_label=f"cross:{opponent_name}",
                    )
            report["sections"]["crossplay"] = cross

        if "differentiation" in args.sections:
            report["sections"]["behavioral_differentiation"] = (
                behavioral_differentiation(
                    sources,
                    args.bc_dataset,
                    n_obs=args.divergence_observations,
                    native_results=report["sections"].get("native"),
                )
            )
    finally:
        # Inference modules are ordinary local torch modules. There are no
        # live Algorithms left by this point.
        pass
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--algo", choices=("ippo", "mappo"), required=True)
    p.add_argument(
        "--condition",
        choices=("shared", "shared_id", "independent"),
        required=True,
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_games", type=int, default=500)
    p.add_argument("--permutation_games", type=int, default=25)
    p.add_argument(
        "--sections",
        nargs="+",
        choices=("native", "permutations", "clones", "crossplay", "differentiation"),
        default=("native", "permutations", "clones", "crossplay", "differentiation"),
    )
    p.add_argument("--historical_ippo", type=Path, default=DEFAULT_HISTORICAL_IPPO)
    p.add_argument("--historical_mappo", type=Path, default=DEFAULT_HISTORICAL_MAPPO)
    p.add_argument("--bc_checkpoint", type=Path, default=DEFAULT_BC)
    p.add_argument("--bc_dataset", type=Path, default=DEFAULT_BC_DATASET)
    p.add_argument("--divergence_observations", type=int, default=512)
    p.add_argument("--num_cpus", type=int, default=2)
    p.add_argument("--object_store_memory_mb", type=int, default=512)
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
    try:
        report = evaluate_checkpoint(args)
    finally:
        ray.shutdown()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"[v3 eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
