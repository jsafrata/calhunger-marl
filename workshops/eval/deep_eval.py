"""Deep diagnostic evaluation of a trained checkpoint.

For each of three lineups (self-play / mixed / 4-scripted-baseline) and
each sampling mode (deterministic argmax / stochastic training temperature),
plays N games and records:

Per learned seat:
  - survival rate, mean death tick
  - trades initiated as taker (post that crossed, or take)
  - trades completed as maker
  - posted bids / asks; posted orders that later executed
  - marketable take frequency
  - action-type frequency (wait / cancel / take / post)
  - conditional distributions: for POSTs, price and qty histograms
Per lineup:
  - mean survivors, reach-time-limit fraction, trades/game
Policy-head diagnostics (from a batched forward pass on real observations):
  - mean probability the type head assigns to each of {wait, cancel, take, post}
  - argmax frequency for each type (over evaluated observations)
  - entropy per head (type / food / side / price / qty), on the mask distribution
  - conditional price/qty entropies given POST-type

Reports a JSON summary; downstream tooling can produce plots.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

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
    ACTION_TYPES,
    ACTION_WAIT,
    NUM_ACTION_TYPES,
    NUM_ACTIONS,
    TYPE_CANCEL,
    TYPE_POST,
    TYPE_TAKE,
    TYPE_WAIT,
    CancelAction,
    PostAction,
    TakeAction,
    WaitAction,
    decode,
    flat_to_factored,
)
from env.env import FoodTradingEnv  # noqa: E402
from env.rl.factored_actor import factored_masks_from_flat_tensor  # noqa: E402
from env.rl.global_observation_encoder import encode_global_observation  # noqa: E402
from env.rl.observation_encoder import encode_observation  # noqa: E402
from env.rl.rllib_config import SHARED_POLICY_ID, register_nash_hunger_env  # noqa: E402


# ---------------------------------------------------------------------------
# Head-level probability diagnostics (uses the actor directly, not RLlib)
# ---------------------------------------------------------------------------


def sample_head_diagnostics(module, sample_obs: list[dict], n_samples: int = 64) -> dict:
    """Grab probability + entropy diagnostics from the type head + downstream
    conditional heads on a real batch of masked observations.

    For a recurrent torso each sample carries the `state_in` it was acted
    from (stashed by `play_games`), so the heads are probed at the same
    memory the policy actually had — probing at a zero state would measure
    a policy that never exists at play time.
    """
    B = len(sample_obs)
    if B == 0:
        return {}
    obs_batch = torch.tensor(
        np.stack([o["observations"] for o in sample_obs]), dtype=torch.float32
    )
    mask_batch = torch.tensor(
        np.stack([o["action_mask"] for o in sample_obs]), dtype=torch.float32
    )
    state_batch = None
    if sample_obs and "state_in" in sample_obs[0]:
        state_batch = {
            k: torch.tensor(
                np.stack([o["state_in"][k] for o in sample_obs]),
                dtype=torch.float32,
            )
            for k in sample_obs[0]["state_in"]
        }
    with torch.no_grad():
        emb, _ = module.torso(obs_batch, state_batch)
        masks = factored_masks_from_flat_tensor(mask_batch)
        # ---- type head ----
        type_logits = module.actor.type_head(emb)
        type_logits = type_logits.masked_fill(~masks.type_mask, float("-inf"))
        # Recover safe categorical to compute probs + entropy
        # (softmax masks out -inf naturally; nan-safe row = uniform)
        type_probs = torch.softmax(type_logits, dim=-1)
        type_probs = torch.nan_to_num(
            type_probs, nan=1.0 / NUM_ACTION_TYPES
        )
        type_entropy = -(type_probs * torch.log(type_probs + 1e-30)).sum(-1)
        type_argmax = type_probs.argmax(-1)

        results: dict = {
            "n_obs_sampled": int(B),
            "type/mean_prob_wait": float(type_probs[:, TYPE_WAIT].mean()),
            "type/mean_prob_cancel": float(type_probs[:, TYPE_CANCEL].mean()),
            "type/mean_prob_take": float(type_probs[:, TYPE_TAKE].mean()),
            "type/mean_prob_post": float(type_probs[:, TYPE_POST].mean()),
            "type/entropy_mean": float(type_entropy.mean()),
            "type/argmax_freq_wait": float((type_argmax == TYPE_WAIT).float().mean()),
            "type/argmax_freq_cancel": float((type_argmax == TYPE_CANCEL).float().mean()),
            "type/argmax_freq_take": float((type_argmax == TYPE_TAKE).float().mean()),
            "type/argmax_freq_post": float((type_argmax == TYPE_POST).float().mean()),
        }

        # ---- conditional heads at type=POST ----
        # Feed the "assume POST" branch through so we get its downstream
        # entropies without depending on what the type head sampled.
        post_type_id = torch.full((B,), TYPE_POST, dtype=torch.long)
        type_emb = module.actor.type_embed(post_type_id)
        food_logits = module.actor.food_head(torch.cat([emb, type_emb], dim=-1))
        food_mask = masks.food_mask[torch.arange(B), post_type_id]
        food_logits = food_logits.masked_fill(~food_mask, float("-inf"))
        food_probs = torch.softmax(food_logits, dim=-1)
        food_probs = torch.nan_to_num(food_probs, nan=0.25)
        food_entropy = -(food_probs * torch.log(food_probs + 1e-30)).sum(-1)

        food_id = food_probs.argmax(-1)
        food_emb = module.actor.food_embed(food_id)
        side_logits = module.actor.side_head(torch.cat([emb, type_emb, food_emb], dim=-1))
        side_mask = masks.side_mask[torch.arange(B), post_type_id, food_id]
        side_logits = side_logits.masked_fill(~side_mask, float("-inf"))
        side_probs = torch.softmax(side_logits, dim=-1)
        side_probs = torch.nan_to_num(side_probs, nan=0.5)
        side_entropy = -(side_probs * torch.log(side_probs + 1e-30)).sum(-1)

        side_id = side_probs.argmax(-1)
        side_emb = module.actor.side_embed(side_id)
        price_logits = module.actor.price_head(
            torch.cat([emb, type_emb, food_emb, side_emb], dim=-1)
        )
        price_mask = masks.price_mask[torch.arange(B), post_type_id, food_id, side_id]
        price_logits = price_logits.masked_fill(~price_mask, float("-inf"))
        price_probs = torch.softmax(price_logits, dim=-1)
        price_probs = torch.nan_to_num(
            price_probs, nan=1.0 / price_probs.shape[-1]
        )
        price_entropy = -(price_probs * torch.log(price_probs + 1e-30)).sum(-1)

        price_id = price_probs.argmax(-1)
        price_emb = module.actor.price_embed(price_id)
        qty_logits = module.actor.qty_head(
            torch.cat([emb, type_emb, food_emb, side_emb, price_emb], dim=-1)
        )
        qty_mask = masks.qty_mask[
            torch.arange(B), post_type_id, food_id, side_id, price_id
        ]
        qty_logits = qty_logits.masked_fill(~qty_mask, float("-inf"))
        qty_probs = torch.softmax(qty_logits, dim=-1)
        qty_probs = torch.nan_to_num(
            qty_probs, nan=1.0 / qty_probs.shape[-1]
        )
        qty_entropy = -(qty_probs * torch.log(qty_probs + 1e-30)).sum(-1)

        results.update({
            "food_given_post/entropy_mean": float(food_entropy.mean()),
            "side_given_post/entropy_mean": float(side_entropy.mean()),
            "price_given_post/entropy_mean": float(price_entropy.mean()),
            "qty_given_post/entropy_mean": float(qty_entropy.mean()),
            "price_given_post/argmax_median": float(
                (price_probs.argmax(-1) + 1).float().median()  # +1 → 1-indexed price
            ),
            "qty_given_post/argmax_median": float(
                (qty_probs.argmax(-1) + 1).float().median()
            ),
        })
    return results


# ---------------------------------------------------------------------------
# Play games — collect per-lineup and per-seat metrics
# ---------------------------------------------------------------------------


def play_games(
    module,
    is_learned_mask,
    n_games,
    seed_base,
    *,
    deterministic: bool,
    environment_class: type[FoodTradingEnv] = FoodTradingEnv,
    opponent_factory=None,
) -> dict:
    """Play diagnostic games in an explicitly selected environment.

    ``FoodTradingEnv`` remains the default so existing callers retain their
    original behavior.  Reduced-horizon experiments must pass their concrete
    environment class instead of relying on the training checkpoint to choose
    the standalone evaluation simulator.

    ``opponent_factory`` selects which scripted bot fills the non-learned
    seats; it defaults to ``ScriptedV1Agent`` so existing callers are
    unaffected.  Pass ``ScriptedV2Agent`` / ``ScriptedV3Agent`` to score a
    checkpoint against a different opponent archetype.  The cadence wrapper is
    always ``"fast"`` regardless of the bot, so the comparison isolates
    strategy rather than firing rate.
    """
    per_seat = defaultdict(list)  # per-seat per-game aggregates
    type_freq: Counter = Counter()
    n_actions_total = 0
    trades_hist = []
    survivors_hist = []
    time_limit_hist = []
    death_tick_hist = []
    return_hist = []
    sample_obs_for_head_diag: list[dict] = []
    game_duration_seconds: int | None = None

    for game_idx in range(n_games):
        game_seed = seed_base + game_idx
        env = environment_class(seed=game_seed)
        obs_raw = env.reset()
        cfg = env.config
        current_duration = int(cfg.game_duration_seconds)
        if game_duration_seconds is None:
            game_duration_seconds = current_duration
        elif game_duration_seconds != current_duration:
            raise ValueError(
                "evaluation environment changed duration between games: "
                f"{game_duration_seconds} then {current_duration}"
            )

        # Set up scripted opponents for scripted seats.
        scripts = {}
        for i, is_learned in enumerate(is_learned_mask):
            if not is_learned:
                a = CadencedScripted(
                    "fast",
                    seed=game_seed * 17 + i,
                    inner_factory=opponent_factory,
                )
                a.reset()
                scripts[i] = a

        # Per-game per-seat action-type counter
        per_seat_type_counts = [Counter() for _ in range(4)]
        per_seat_action_count = [0 for _ in range(4)]
        per_seat_completed_taker_trades = [0 for _ in range(4)]
        per_seat_completed_maker_trades = [0 for _ in range(4)]
        per_seat_posted_bids = [0 for _ in range(4)]
        per_seat_posted_asks = [0 for _ in range(4)]
        per_seat_takes = [0 for _ in range(4)]

        trades_seen = 0
        init_state = module.get_initial_state()
        # Encode with the checkpoint's own observation contract — a module
        # trained with the v4 features expects the wider vectors and will
        # reject (or worse, misread) the base encoding.
        include_survival = bool(getattr(module, "include_survival_features", False))
        include_trades = bool(getattr(module, "include_trade_history", False))
        # Per-seat recurrent state, carried across the whole game. Feeding
        # the initial zero state at every step instead would evaluate an
        # LSTM checkpoint with amnesia — it would still act, just without
        # the memory it was trained to use.
        seat_states: dict[int, dict] = {
            i: {
                k: torch.from_numpy(np.asarray(v)).float()
                for k, v in init_state.items()
            }
            for i, learned in enumerate(is_learned_mask)
            if learned and init_state
        }

        while not env.is_done():
            actions = {}
            learned_idxs = [i for i, learned in enumerate(is_learned_mask) if learned]
            if learned_idxs:
                per_seat_obs = [obs_raw[i] for i in learned_idxs]
                enc_obs = [
                    {
                        "observations": encode_observation(
                            o, cfg, include_survival_features=include_survival
                        ),
                        "action_mask": env.legal_actions(i).astype(np.float32),
                        "critic_observations": encode_global_observation(
                            env.state, i, cfg, include_trade_history=include_trades
                        ),
                        # Stashed so head diagnostics can replay a stateful
                        # module at the same memory it acted from.
                        **(
                            {
                                "state_in": {
                                    k: v.numpy().copy()
                                    for k, v in seat_states[i].items()
                                }
                            }
                            if i in seat_states
                            else {}
                        ),
                    }
                    for i, o in zip(learned_idxs, per_seat_obs)
                ]
                # Reservoir sample real observations for head diagnostics
                # (cap at 512 total across the whole eval to keep it fast).
                if len(sample_obs_for_head_diag) < 512:
                    for e in enc_obs:
                        if len(sample_obs_for_head_diag) < 512:
                            sample_obs_for_head_diag.append(e)

                batch_obs = {
                    "observations": torch.tensor(np.stack(
                        [e["observations"] for e in enc_obs]), dtype=torch.float32),
                    "action_mask": torch.tensor(np.stack(
                        [e["action_mask"] for e in enc_obs])),
                    "critic_observations": torch.tensor(np.stack(
                        [e["critic_observations"] for e in enc_obs]),
                        dtype=torch.float32),
                }
                batch = {Columns.OBS: batch_obs}
                if init_state:
                    batch[Columns.STATE_IN] = {
                        k: torch.stack(
                            [seat_states[i][k] for i in learned_idxs]
                        )
                        for k in init_state
                    }
                with torch.no_grad():
                    if deterministic:
                        out = module._forward_inference(batch)
                    else:
                        out = module._forward_exploration(batch)
                if init_state and Columns.STATE_OUT in out:
                    for pos, seat in enumerate(learned_idxs):
                        seat_states[seat] = {
                            k: v[pos].detach()
                            for k, v in out[Columns.STATE_OUT].items()
                        }
                for pos, seat in enumerate(learned_idxs):
                    aid = int(out[Columns.ACTIONS][pos].item())
                    actions[seat] = aid
                    n_actions_total += 1
                    decoded = decode(aid)
                    t = type(decoded).__name__
                    per_seat_type_counts[seat][t] += 1
                    type_freq[t] += 1
                    per_seat_action_count[seat] += 1
                    if isinstance(decoded, PostAction):
                        if decoded.side == "bid":
                            per_seat_posted_bids[seat] += 1
                        else:
                            per_seat_posted_asks[seat] += 1
                    elif isinstance(decoded, TakeAction):
                        per_seat_takes[seat] += 1

            for seat, agent in scripts.items():
                actions[seat] = agent.act(obs_raw[seat], env.legal_actions(seat))

            obs_raw, _, _, _ = env.step(actions)

            # Attribute new trades to the buyer / seller among learned seats.
            new_trades = env.state.trades[trades_seen:]
            trades_seen = len(env.state.trades)
            for t in new_trades:
                # taker_order_id == -1 → this was a TAKE. Buyer/seller
                # attribution is enough for a coarse maker/taker split.
                if t.buyer_idx in learned_idxs:
                    per_seat_completed_taker_trades[t.buyer_idx] += 1
                if t.seller_idx in learned_idxs:
                    per_seat_completed_maker_trades[t.seller_idx] += 1

        res = env.get_result()
        survivors_hist.append(sum(1 for p in res["players"] if p["status"] == "alive"))
        time_limit_hist.append(1 if res["end_reason"] == "time_limit" else 0)
        trades_hist.append(res["n_trades"])

        winners_set = set(res["winner_idxs"])
        horizon = env.config.game_duration_seconds
        for i in range(4):
            if is_learned_mask[i]:
                p = res["players"][i]
                death_tick = p["died_at"] if p["died_at"] is not None else res["elapsed_seconds"]
                alive = p["status"] == "alive"
                per_seat["survived"].append(1 if alive else 0)
                per_seat["death_tick"].append(death_tick)
                per_seat["final_cash"].append(p["final_cash"])
                # v2b: terminal cash + required-food inventory diagnostics.
                # Required foods = the two the seat consumes (= all foods it
                # does NOT produce). For dead players we read their inventory
                # at the point of death (env preserves it after kill_player
                # zeroes reservations but not inventory).
                inv = p["final_inventory"]
                produced = tuple(p["produces"])
                required = tuple(f for f in env.config.food_types if f not in produced)
                # Guard: should always be exactly 2 required foods.
                req_invs = [int(inv.get(f, 0)) for f in required]
                inv_total = sum(int(inv.get(f, 0)) for f in env.config.food_types)
                inv_produced_total = sum(int(inv.get(f, 0)) for f in produced)
                inv_required_total = sum(req_invs)
                inv_required_mean = (
                    inv_required_total / len(req_invs) if req_invs else 0.0
                )
                inv_required_min = min(req_invs) if req_invs else 0
                if i in winners_set:
                    outcome = "winner"
                elif alive:
                    outcome = "alive_nonwinner"
                else:
                    outcome = "dead"
                per_seat["outcome"].append(outcome)
                per_seat["terminal_cash"].append(p["final_cash"])
                per_seat["inv_required_mean"].append(inv_required_mean)
                per_seat["inv_required_min"].append(inv_required_min)
                per_seat["inv_required_total"].append(inv_required_total)
                per_seat["inv_total"].append(inv_total)
                per_seat["inv_produced_total"].append(inv_produced_total)
                per_seat["inv_required_1"].append(req_invs[0] if len(req_invs) > 0 else 0)
                per_seat["inv_required_2"].append(req_invs[1] if len(req_invs) > 1 else 0)
                per_seat["completed_trades_as_buyer"].append(per_seat_completed_taker_trades[i])
                per_seat["completed_trades_as_seller"].append(per_seat_completed_maker_trades[i])
                per_seat["posted_bids"].append(per_seat_posted_bids[i])
                per_seat["posted_asks"].append(per_seat_posted_asks[i])
                per_seat["takes"].append(per_seat_takes[i])
                total_actions = per_seat_action_count[i]
                per_seat["frac_wait"].append(
                    per_seat_type_counts[i]["WaitAction"] / max(1, total_actions)
                )
                per_seat["frac_cancel"].append(
                    per_seat_type_counts[i]["CancelAction"] / max(1, total_actions)
                )
                per_seat["frac_take"].append(
                    per_seat_type_counts[i]["TakeAction"] / max(1, total_actions)
                )
                per_seat["frac_post"].append(
                    per_seat_type_counts[i]["PostAction"] / max(1, total_actions)
                )

    total_actions = sum(type_freq.values())

    # ---- v2b: terminal cash + required-food inventory aggregates ----
    def _stats(xs):
        if not xs:
            return {"n": 0, "mean": float("nan"), "median": float("nan"),
                    "std": float("nan")}
        arr = np.asarray(xs, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=0)),
        }

    outcomes = per_seat["outcome"]
    idx_by_outcome = {
        "all": list(range(len(outcomes))),
        "dead": [i for i, o in enumerate(outcomes) if o == "dead"],
        "alive_nonwinner": [
            i for i, o in enumerate(outcomes) if o == "alive_nonwinner"
        ],
        "winner": [i for i, o in enumerate(outcomes) if o == "winner"],
    }

    def _sel(field, idxs):
        return [per_seat[field][i] for i in idxs]

    terminal_metrics: dict = {"terminal_cash_by_outcome": {}}
    for outcome_label, idxs in idx_by_outcome.items():
        terminal_metrics["terminal_cash_by_outcome"][outcome_label] = _stats(
            _sel("terminal_cash", idxs)
        )

    inv_fields = (
        "inv_required_mean",
        "inv_required_min",
        "inv_required_total",
        "inv_required_1",
        "inv_required_2",
        "inv_total",
        "inv_produced_total",
    )
    terminal_metrics["required_food_by_outcome"] = {}
    for outcome_label, idxs in idx_by_outcome.items():
        terminal_metrics["required_food_by_outcome"][outcome_label] = {
            f: _stats(_sel(f, idxs)) for f in inv_fields
        }

    # Fraction of SURVIVING seats with min-required inventory in
    # {0-1, 2-3, 4-5, >5} buckets — the "hidden imbalance" diagnostic.
    surv_idxs = [i for i, o in enumerate(outcomes) if o in ("alive_nonwinner", "winner")]
    surv_mins = _sel("inv_required_min", surv_idxs)
    def _bucket_frac(vals, lo, hi):
        if not vals:
            return float("nan")
        return float(sum(1 for v in vals if lo <= v <= hi) / len(vals))
    def _bucket_frac_gt(vals, thresh):
        if not vals:
            return float("nan")
        return float(sum(1 for v in vals if v > thresh) / len(vals))
    terminal_metrics["survivor_min_required_buckets"] = {
        "n_survivors": len(surv_mins),
        "frac_0_to_1": _bucket_frac(surv_mins, 0, 1),
        "frac_2_to_3": _bucket_frac(surv_mins, 2, 3),
        "frac_4_to_5": _bucket_frac(surv_mins, 4, 5),
        "frac_gt_5": _bucket_frac_gt(surv_mins, 5),
    }

    # Outcome counts — sanity check on breakdowns above.
    terminal_metrics["outcome_counts"] = {
        "dead": len(idx_by_outcome["dead"]),
        "alive_nonwinner": len(idx_by_outcome["alive_nonwinner"]),
        "winner": len(idx_by_outcome["winner"]),
    }

    return {
        "n_games": n_games,
        "mode": "deterministic" if deterministic else "stochastic",
        "environment_class": environment_class.__name__,
        "game_duration_seconds": game_duration_seconds,
        "terminal_metrics": terminal_metrics,
        "n_learned_seats": sum(is_learned_mask),
        "lineup_mean_survivors": float(np.mean(survivors_hist)) if survivors_hist else float("nan"),
        "lineup_reach_time_limit_frac": float(np.mean(time_limit_hist)) if time_limit_hist else float("nan"),
        "lineup_mean_trades": float(np.mean(trades_hist)) if trades_hist else float("nan"),
        "learned_seat_survival_rate": (
            float(np.mean(per_seat["survived"])) if per_seat["survived"] else float("nan")
        ),
        "learned_seat_mean_death_tick": (
            float(np.mean(per_seat["death_tick"])) if per_seat["death_tick"] else float("nan")
        ),
        "learned_seat_mean_final_cash": (
            float(np.mean(per_seat["final_cash"])) if per_seat["final_cash"] else float("nan")
        ),
        "learned_seat_completed_trades_as_buyer_mean": (
            float(np.mean(per_seat["completed_trades_as_buyer"])) if per_seat["completed_trades_as_buyer"] else 0.0
        ),
        "learned_seat_completed_trades_as_seller_mean": (
            float(np.mean(per_seat["completed_trades_as_seller"])) if per_seat["completed_trades_as_seller"] else 0.0
        ),
        "learned_seat_posted_bids_mean": (
            float(np.mean(per_seat["posted_bids"])) if per_seat["posted_bids"] else 0.0
        ),
        "learned_seat_posted_asks_mean": (
            float(np.mean(per_seat["posted_asks"])) if per_seat["posted_asks"] else 0.0
        ),
        "learned_seat_takes_mean": (
            float(np.mean(per_seat["takes"])) if per_seat["takes"] else 0.0
        ),
        "action_type_freq": {
            "wait": type_freq.get("WaitAction", 0) / max(1, total_actions),
            "cancel": type_freq.get("CancelAction", 0) / max(1, total_actions),
            "take": type_freq.get("TakeAction", 0) / max(1, total_actions),
            "post": type_freq.get("PostAction", 0) / max(1, total_actions),
        },
        "learned_seat_type_freq_mean": {
            "wait": float(np.mean(per_seat["frac_wait"])) if per_seat["frac_wait"] else 0.0,
            "cancel": float(np.mean(per_seat["frac_cancel"])) if per_seat["frac_cancel"] else 0.0,
            "take": float(np.mean(per_seat["frac_take"])) if per_seat["frac_take"] else 0.0,
            "post": float(np.mean(per_seat["frac_post"])) if per_seat["frac_post"] else 0.0,
        },
        "_head_diag_sample": sample_obs_for_head_diag,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


LINEUPS = {
    "learned_selfplay": [True, True, True, True],
    "1learned_3scripted": [True, False, False, False],
    "4scripted_baseline": [False, False, False, False],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--algo_label", type=str, required=True)
    p.add_argument("--n_games", type=int, default=500)
    p.add_argument("--seed_base", type=int, default=200_000)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main(args) -> int:
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False,
             log_to_driver=False)
    register_nash_hunger_env()
    algo = Algorithm.from_checkpoint(str(args.checkpoint.resolve()))
    module = algo.get_module(SHARED_POLICY_ID)
    report: dict = {"checkpoint": str(args.checkpoint),
                    "algo_label": args.algo_label,
                    "lineups": {}}
    try:
        for lineup_name, mask in LINEUPS.items():
            report["lineups"][lineup_name] = {}
            for mode in ("deterministic", "stochastic"):
                # Skip stochastic 4scripted (no learned seats to be stochastic about)
                if not any(mask) and mode == "stochastic":
                    continue
                print(f"[{args.algo_label}] {lineup_name} {mode} — playing {args.n_games} games")
                r = play_games(
                    module, mask, args.n_games,
                    seed_base=args.seed_base + (0 if mode == "deterministic" else 500_000),
                    deterministic=(mode == "deterministic"),
                )
                # If we have learned seats, compute head diagnostics from
                # the collected observation sample.
                sample_obs = r.pop("_head_diag_sample")
                if sample_obs:
                    r["head_diagnostics"] = sample_head_diagnostics(module, sample_obs)
                report["lineups"][lineup_name][mode] = r
                print(f"    survival={r.get('learned_seat_survival_rate', float('nan')):.3f}  "
                      f"trades={r['lineup_mean_trades']:.1f}  "
                      f"reach_horizon={r['lineup_reach_time_limit_frac']:.3f}  "
                      f"type_freq={ {k: f'{v:.2f}' for k, v in r['action_type_freq'].items()} }")
    finally:
        algo.stop()
        ray.shutdown()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n[deep_eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
