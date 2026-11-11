"""Play a single game with Scripted-v1 agents and log everything.

One call to `run_game(...)` produces a `GameLog` dict that contains:
    - metadata: seed, cadence-per-seat, seat producers, ...
    - per-tick log: at every 10-second consumption tick, pre/post
      inventory + buffers + cash for every living player
    - per-player behavioral counters
    - per-food market statistics
    - starvation classification per dead player
    - final outcome

The dict is designed to be `json.dumps`-able and small enough to save
one-per-game as a JSONL line.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.cadenced_scripted import CadencedScripted
from env.actions import (
    ACTION_WAIT,
    CancelAction,
    PostAction,
    TakeAction,
    WaitAction,
    decode,
)
from env.config import DEFAULT_CONFIG, FOOD_TYPES
from env.env import FoodTradingEnv

CONSUMPTION_INTERVAL_SECONDS = DEFAULT_CONFIG.consumption_interval_seconds
CONSUME_PER_TICK = DEFAULT_CONFIG.consumption_per_required_food_per_second


@dataclass
class _PlayerTelemetry:
    """Per-player counters accumulated during a game."""

    decisions_taken: int = 0
    decisions_skipped: int = 0

    posts_bid: int = 0
    posts_ask: int = 0
    posts_replaced: int = 0
    validation_failures: int = 0
    cancels: int = 0
    takes: int = 0

    taker_trades: int = 0  # trades where this player initiated (post or take)
    maker_trades: int = 0

    units_bought: int = 0
    units_sold: int = 0
    cash_spent: int = 0
    cash_received: int = 0

    time_first_order: Optional[int] = None
    time_first_trade: Optional[int] = None

    # Set for starvation classification if died.
    starvation_cause: Optional[str] = None


@dataclass
class _FoodTelemetry:
    n_trades: int = 0
    volume: int = 0
    prices: List[int] = field(default_factory=list)  # execution prices
    time_first_trade: Optional[int] = None

    posted_bids: int = 0
    posted_asks: int = 0

    # Sampled at each tick (post-action, before consumption).
    best_bid_series: List[Optional[int]] = field(default_factory=list)
    best_ask_series: List[Optional[int]] = field(default_factory=list)
    depth_bid_series: List[int] = field(default_factory=list)
    depth_ask_series: List[int] = field(default_factory=list)

    ticks_no_bid: int = 0
    ticks_no_ask: int = 0

    # Set at game end.
    unused_surplus: int = 0


def _classify_starvation(
    death_tick: int,
    per_player_pre_tick_snapshots: List[Dict[str, Any]],
    per_player_last_action: List[Optional[dict]],
    per_player_orders_pre_death: List[List[dict]],
    player_idx: int,
    obs_at_death_tick: Dict[str, Any],
) -> str:
    """Best-effort classification of why a player starved.

    We look at the observation snapshot from the tick BEFORE the fatal
    consumption tick and try to distinguish the six causes the spec asks
    for. Categories:

        insufficient_cash    — no ask affordable at 1 unit (or no cash at all)
        no_ask_available     — had cash but no ask visible on that food
        affordable_ask_no_buy — ask was there + affordable, but no order placed
        order_pending_no_fill — order was submitted but did not fill in time
        cash_tied_in_reservations — cash was blocked by open orders
        other                — none of the above cleanly matches
    """
    snapshot = per_player_pre_tick_snapshots[player_idx]
    obs = obs_at_death_tick
    required = obs["required_foods"]
    inv = obs["available_inventory"]
    # Find the food that killed us: any required food with < consume-per-tick units.
    missing = [
        f for f in required if inv.get(f, 0) < CONSUME_PER_TICK
    ]
    if not missing:
        return "other"
    food = missing[0]

    cash_free = snapshot.get("available_cash", 0)
    cash_total = snapshot.get("cash", 0)
    reserved = cash_total - cash_free

    asks = obs["order_books"][food]["asks"]
    if not asks:
        # Distinguish: had-cash-but-no-ask vs no-cash-anyway.
        if cash_total < 1:
            return "insufficient_cash"
        return "no_ask_available"
    cheapest_ask = min(a["price"] for a in asks)
    if cash_free < cheapest_ask:
        if reserved >= cheapest_ask:
            return "cash_tied_in_reservations"
        return "insufficient_cash"

    # Cash was free AND an affordable ask exists — did we submit anything?
    orders = per_player_orders_pre_death[player_idx]
    pending = [o for o in orders if o["side"] == "bid" and o["food"] == food]
    if pending:
        return "order_pending_no_fill"
    return "affordable_ask_no_buy"


def _snapshot_player(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Compact per-tick per-player snapshot used for tick-log entries."""
    inv = obs["available_inventory"]
    required = obs["required_foods"]
    buffers = {f: inv[f] - CONSUME_PER_TICK for f in required}
    min_buffer = min(buffers.values()) if buffers else 0
    return {
        "cash": obs["cash"],
        "reserved_cash": obs["reserved_cash"],
        "available_cash": obs["available_cash"],
        "inventory": dict(obs["inventory"]),
        "reserved_inventory": dict(obs["reserved_inventory"]),
        "available_inventory": dict(inv),
        "buffers": buffers,
        "min_buffer": int(min_buffer),
        "status": obs["status"],
    }


def _record_action_kind(
    tele: _PlayerTelemetry,
    action_int: int,
    result: Optional[dict],
    tick: int,
) -> None:
    """Update per-player counters based on the action decode + apply result."""
    if action_int == ACTION_WAIT:
        return
    decoded = decode(action_int)
    if isinstance(decoded, WaitAction):
        return
    if result is None:
        return
    applied = result.get("applied", False)
    kind = result.get("kind")

    if isinstance(decoded, CancelAction):
        if applied:
            tele.cancels += 1
        else:
            tele.validation_failures += 1
        return

    if isinstance(decoded, TakeAction):
        if applied:
            tele.takes += 1
            filled = result.get("quantity_filled", 0) or 0
            if filled > 0 and tele.time_first_order is None:
                tele.time_first_order = tick
        else:
            tele.validation_failures += 1
        return

    if isinstance(decoded, PostAction):
        if not applied:
            tele.validation_failures += 1
            return
        # Post-side counter — bids vs asks
        if decoded.side == "bid":
            tele.posts_bid += 1
        else:
            tele.posts_ask += 1
        if tele.time_first_order is None:
            tele.time_first_order = tick
        # Result includes n_trades if the post hit the book on submission.
        if (result.get("n_trades") or 0) > 0:
            # These are "taker trades" from this player's post — they'll
            # also be logged from the trade-side loop below (trade.taker_idx
            # will equal this player). We rely on the trade loop to update
            # taker/maker + volume so we don't double-count here.
            pass
        # NOTE: The env's per-food-side replacement (posting on an existing
        # (food, side) auto-cancels the prior order) is not directly
        # observable from result. We approximate `posts_replaced` from
        # cancel-of-own counts inferred elsewhere — see game loop below.


def run_game(
    *,
    game_seed: int,
    cadences: Sequence[str],
    seat_permutation_seed: int,
    inner_bot_seed_stride: int = 977,
) -> Dict[str, Any]:
    """Play one game and return the full log.

    Args:
        game_seed: drives producers, initial inventory, and the env's
            market RNG (order-of-application on simultaneous ticks).
        cadences: the 4 cadence labels (order matches the *configuration*,
            not the seat order — we permute seats using
            `seat_permutation_seed`).
        seat_permutation_seed: RNG seed for the seat permutation. Two
            games sharing `game_seed` but different `seat_permutation_seed`
            get identical environmental setups with different seat
            assignments — useful for paired-seed A/B analysis.
        inner_bot_seed_stride: coprime prime used to derive per-seat
            bot RNG seeds from `game_seed`.
    """
    if len(cadences) != 4:
        raise ValueError("run_game expects exactly 4 cadences")

    # ---- seat permutation -------------------------------------------
    perm_rng = np.random.default_rng(seat_permutation_seed)
    seat_perm: List[int] = list(perm_rng.permutation(4))
    seat_cadences: List[str] = [cadences[seat_perm[s]] for s in range(4)]

    # ---- build env + agents -----------------------------------------
    env = FoodTradingEnv(
        seed=game_seed,
        player_names=[f"P{i}[{seat_cadences[i]}]" for i in range(4)],
    )
    obs = env.reset()
    agents = [
        CadencedScripted(
            cadence=seat_cadences[i],
            seed=game_seed + i * inner_bot_seed_stride,
        )
        for i in range(4)
    ]
    for a in agents:
        a.reset()

    # ---- log initial state ------------------------------------------
    initial_players = [
        {
            "idx": p.idx,
            "cadence": seat_cadences[p.idx],
            "produces": list(p.produces),
            "cash": p.cash,
            "inventory": dict(p.inventory),
        }
        for p in env.state.players
    ]

    # ---- telemetry accumulators -------------------------------------
    ptele = [_PlayerTelemetry() for _ in range(4)]
    ftele: Dict[str, _FoodTelemetry] = {f: _FoodTelemetry() for f in FOOD_TYPES}

    tick_logs: List[Dict[str, Any]] = []
    all_trades_seen: int = 0  # to slice new-trades per step
    pre_tick_snapshots: List[Dict[str, Any]] = [
        _snapshot_player(obs[i]) for i in range(4)
    ]
    per_player_orders_snapshots: List[List[dict]] = [
        list(obs[i]["own_orders"]) for i in range(4)
    ]
    death_tick_obs_snapshot: Dict[int, Dict[str, Any]] = {}

    # Wall-clock cadence sanity: track decisions per game-second per seat.
    decisions_per_second: List[List[int]] = [[] for _ in range(4)]

    # ---- main loop ---------------------------------------------------
    while not env.is_done():
        current_tick = env.state.elapsed_seconds
        actions_dict: Dict[int, int] = {}
        pre_action_counts = [
            (a.n_decisions_taken, a.n_decisions_skipped) for a in agents
        ]
        for i in range(4):
            # Snapshot own_orders BEFORE acting — used later to detect
            # replacements (own_order count on same food+side drops by 1
            # despite our count increasing by 1 = replacement).
            actions_dict[i] = agents[i].act(obs[i], env.legal_actions(i))
            taken, skipped = agents[i].n_decisions_taken, agents[i].n_decisions_skipped
            decisions_per_second[i].append(
                (taken - pre_action_counts[i][0]) + (skipped - pre_action_counts[i][1])
            )

        # Pre-tick snapshot happens on the RESULT AFTER actions apply but
        # BEFORE the consumption tick fires. That's what the spec asks
        # for — "immediately before the tick".
        obs_next, rewards, done, info = env.step(actions_dict)

        # ---- update behavioural counters from action applied ---------
        for i in range(4):
            _record_action_kind(
                ptele[i], actions_dict[i], info.actions_applied.get(i), current_tick
            )

        # Detect posts_replaced approximation: we count how many of this
        # player's open (side, food) buckets had an order before AND still
        # have one after but with a different order id. That's a
        # replacement. Cheaper approximation: compare set of order ids.
        for i in range(4):
            pre_ids = {o["id"] for o in per_player_orders_snapshots[i]}
            post_ids = {o["id"] for o in obs_next[i]["own_orders"]}
            # Orders that vanished AND a new one appeared on same side/food
            # in the same step → replacement. Count = new_ids that
            # displaced an existing (side, food).
            pre_by_key = defaultdict(list)
            for o in per_player_orders_snapshots[i]:
                pre_by_key[(o["side"], o["food"])].append(o["id"])
            post_by_key = defaultdict(list)
            for o in obs_next[i]["own_orders"]:
                post_by_key[(o["side"], o["food"])].append(o["id"])
            for k, post_orders in post_by_key.items():
                pre_orders = pre_by_key.get(k, [])
                if pre_orders and post_orders and set(pre_orders) != set(post_orders):
                    # At least one id changed → replacement occurred.
                    ptele[i].posts_replaced += 1
            per_player_orders_snapshots[i] = list(obs_next[i]["own_orders"])

        # ---- absorb new trades ---------------------------------------
        new_trades = env.state.trades[all_trades_seen:]
        for t in new_trades:
            ft = ftele[t.food_type]
            ft.n_trades += 1
            ft.volume += t.quantity
            ft.prices.append(t.price_per_unit)
            if ft.time_first_trade is None:
                ft.time_first_trade = t.elapsed_second

            # Taker id: for takes we stored -1 as sentinel. For post-that-crossed,
            # the taker is the buyer or seller whose order id == taker_order_id.
            taker_idx = t.buyer_idx if t.buyer_idx != -1 else t.seller_idx
            # More reliable: use maker/taker relationship. Maker had a
            # resting order id in taker_order_id? No — taker_order_id
            # names the incoming aggressor. Find its player idx by
            # matching maker_order_id back to a player.
            maker_player = None
            for p in env.state.players:
                # We don't have direct order lookup here; skip precise
                # maker attribution — approximate as: whichever side of
                # the trade didn't submit *this* step. For clarity we set
                # taker = buyer if a bid post happened this step for
                # buyer, else seller, else unknown.
                _ = p
            # Coarse: attribute both counters conservatively. Taker &
            # maker sums won't add to 2×n_trades unless we can be more
            # precise. We DO always know units_bought / units_sold / cash
            # from the trade endpoints:
            buyer_i, seller_i = t.buyer_idx, t.seller_idx
            if 0 <= buyer_i < 4:
                ptele[buyer_i].units_bought += t.quantity
                ptele[buyer_i].cash_spent += t.total_price
                if ptele[buyer_i].time_first_trade is None:
                    ptele[buyer_i].time_first_trade = t.elapsed_second
            if 0 <= seller_i < 4:
                ptele[seller_i].units_sold += t.quantity
                ptele[seller_i].cash_received += t.total_price
                if ptele[seller_i].time_first_trade is None:
                    ptele[seller_i].time_first_trade = t.elapsed_second

            # Attribute maker/taker: if taker_order_id == -1 → TAKE
            # action from taker side. We look up which side of the trade
            # doesn't have an order in the book right now to distinguish.
            # Simpler + still useful: the aggressor is whoever's action
            # this tick was a POST or TAKE that matched this trade.
            # (Precise attribution is skipped for cost; the aggregate
            # units_bought/sold captures the primary quantity signal.)

        all_trades_seen = len(env.state.trades)

        # ---- track posted_bids / posted_asks per food ---------------
        for i in range(4):
            a = actions_dict[i]
            if a == ACTION_WAIT:
                continue
            decoded = decode(a)
            if isinstance(decoded, PostAction) and info.actions_applied.get(i, {}).get("applied"):
                if decoded.side == "bid":
                    ftele[decoded.food_type].posted_bids += 1
                else:
                    ftele[decoded.food_type].posted_asks += 1

        # ---- market-depth samples after this step -------------------
        for f in FOOD_TYPES:
            book = env.state.order_books[f]
            best_bid = max((o.price_per_unit for o in book.bids), default=None)
            best_ask = min((o.price_per_unit for o in book.asks), default=None)
            depth_bid = sum(o.remaining_quantity for o in book.bids)
            depth_ask = sum(o.remaining_quantity for o in book.asks)
            ftele[f].best_bid_series.append(best_bid)
            ftele[f].best_ask_series.append(best_ask)
            ftele[f].depth_bid_series.append(depth_bid)
            ftele[f].depth_ask_series.append(depth_ask)
            if best_bid is None:
                ftele[f].ticks_no_bid += 1
            if best_ask is None:
                ftele[f].ticks_no_ask += 1

        # ---- consumption-tick log -----------------------------------
        # A consumption tick fires when the NEW elapsed_seconds is a
        # multiple of consumption_interval_seconds. `info.tick` is the
        # elapsed_seconds AFTER the step, which is what we want.
        if info.tick > 0 and info.tick % CONSUMPTION_INTERVAL_SECONDS == 0:
            entry: Dict[str, Any] = {
                "tick": info.tick,
                "pre_tick_snapshots": pre_tick_snapshots,
                "post_tick_snapshots": [_snapshot_player(obs_next[i]) for i in range(4)],
                "deaths_this_tick": list(info.deaths_this_tick),
            }
            tick_logs.append(entry)

        # ---- starvation classification: capture obs snapshot at death
        for died_idx in info.deaths_this_tick:
            death_tick_obs_snapshot[died_idx] = obs[died_idx]  # obs BEFORE the fatal step

        # ---- shift buffers for next iteration ----------------------
        obs = obs_next
        pre_tick_snapshots = [_snapshot_player(obs[i]) for i in range(4)]

    # ---- game finished — classify starvations ---------------------
    result = env.get_result()
    # Compact snapshot of the exact state the classifier saw at the fatal
    # tick, so an independent auditor can re-classify from the log alone.
    death_contexts: dict[int, dict] = {}
    for i in range(4):
        p_res = result["players"][i]
        if p_res["status"] == "dead":
            death_obs = death_tick_obs_snapshot.get(i)
            if death_obs is not None:
                ptele[i].starvation_cause = _classify_starvation(
                    death_tick=p_res["died_at"] or 0,
                    per_player_pre_tick_snapshots=[
                        _snapshot_player(death_obs) if j == i
                        else pre_tick_snapshots[j]
                        for j in range(4)
                    ],
                    per_player_last_action=[None] * 4,
                    per_player_orders_pre_death=[
                        list(death_obs.get("own_orders", [])) if j == i else []
                        for j in range(4)
                    ],
                    player_idx=i,
                    obs_at_death_tick=death_obs,
                )
                death_contexts[i] = {
                    "tick": p_res["died_at"],
                    "cash": death_obs["cash"],
                    "reserved_cash": death_obs["reserved_cash"],
                    "available_cash": death_obs["available_cash"],
                    "inventory": dict(death_obs["inventory"]),
                    "reserved_inventory": dict(death_obs["reserved_inventory"]),
                    "available_inventory": dict(death_obs["available_inventory"]),
                    "required_foods": list(death_obs["required_foods"]),
                    "asks_per_food": {
                        f: list(death_obs["order_books"][f]["asks"])
                        for f in death_obs["required_foods"]
                    },
                    "bids_per_food": {
                        f: list(death_obs["order_books"][f]["bids"])
                        for f in death_obs["required_foods"]
                    },
                    "own_orders": list(death_obs.get("own_orders", [])),
                }
            else:
                ptele[i].starvation_cause = "other"

    # ---- unused surplus per food ----------------------------------
    for i in range(4):
        p_res = result["players"][i]
        for f in p_res["produces"]:
            ftele[f].unused_surplus += p_res["final_inventory"].get(f, 0)

    # ---- pull decision counters from the cadence wrappers ---------
    for i, a in enumerate(agents):
        ptele[i].decisions_taken = a.n_decisions_taken
        ptele[i].decisions_skipped = a.n_decisions_skipped

    # ---- gather every attempted post-price the bots computed -------
    # The ScriptedV1 core inside each CadencedScripted logs each price
    # BEFORE any clamp / cap / affordability / legality filter fires, so
    # we can characterise the true unbounded distribution the heuristic
    # wants to emit — and count cap violations if any occurred.
    price_attempts: List[dict] = []
    for i, a in enumerate(agents):
        inner = getattr(a, "_inner", None)
        if inner is None:
            continue
        # Copy so a jsonl encoder doesn't chase a reference back into the
        # (soon-to-be-collected) agent instance.
        price_attempts.extend(list(getattr(inner, "attempted_prices", [])))

    # ---- executed trade prices (side/food attached so we can filter) --
    executed_trade_prices: List[dict] = [
        {
            "tick": t.elapsed_second,
            "food": t.food_type,
            "price": t.price_per_unit,
            "qty": t.quantity,
        }
        for t in env.state.trades
    ]

    # ---- assemble the log ----------------------------------------
    game_log: Dict[str, Any] = {
        "game_seed": game_seed,
        "seat_permutation_seed": seat_permutation_seed,
        "cadences_configuration_order": list(cadences),
        "seat_permutation": seat_perm,
        "seat_cadences": seat_cadences,
        "initial_players": initial_players,
        "final": {
            "end_reason": result["end_reason"],
            "elapsed_seconds": result["elapsed_seconds"],
            "reached_time_limit": result["end_reason"] == "time_limit",
            "n_survivors": sum(
                1 for p in result["players"] if p["status"] == "alive"
            ),
            "no_survivors": all(
                p["status"] != "alive" for p in result["players"]
            ),
            "winners": list(result["winner_idxs"]),
            "n_trades_total": result["n_trades"],
            "players": [
                {
                    "idx": p["idx"],
                    "cadence": seat_cadences[p["idx"]],
                    "produces": p["produces"],
                    "status": p["status"],
                    "died_at": p["died_at"],
                    "survived": 1 if p["status"] == "alive" else 0,
                    "final_cash": p["final_cash"],
                    "final_inventory": p["final_inventory"],
                    "total_inventory": sum(p["final_inventory"].values()),
                    "total_bought": p["total_bought"],
                    "total_sold": p["total_sold"],
                }
                for p in result["players"]
            ],
            "first_death_tick": min(
                (p["died_at"] for p in result["players"] if p["died_at"] is not None),
                default=None,
            ),
            "death_times": [p["died_at"] for p in result["players"]],
        },
        "players_behavior": [
            {
                "idx": i,
                "cadence": seat_cadences[i],
                "decisions_taken": ptele[i].decisions_taken,
                "decisions_skipped": ptele[i].decisions_skipped,
                "posts_bid": ptele[i].posts_bid,
                "posts_ask": ptele[i].posts_ask,
                "posts_total": ptele[i].posts_bid + ptele[i].posts_ask,
                "posts_replaced": ptele[i].posts_replaced,
                "validation_failures": ptele[i].validation_failures,
                "cancels": ptele[i].cancels,
                "takes": ptele[i].takes,
                "taker_trades": ptele[i].taker_trades,
                "maker_trades": ptele[i].maker_trades,
                "units_bought": ptele[i].units_bought,
                "units_sold": ptele[i].units_sold,
                "cash_spent": ptele[i].cash_spent,
                "cash_received": ptele[i].cash_received,
                "time_first_order": ptele[i].time_first_order,
                "time_first_trade": ptele[i].time_first_trade,
                "starvation_cause": ptele[i].starvation_cause,
                # State at the fatal-tick — enough for an auditor to
                # re-run the classifier from scratch.
                "death_context": death_contexts.get(i),
                "actions_per_second_series": decisions_per_second[i],
            }
            for i in range(4)
        ],
        "market_by_food": {
            f: {
                "n_trades": ftele[f].n_trades,
                "volume": ftele[f].volume,
                "mean_price": float(np.mean(ftele[f].prices)) if ftele[f].prices else None,
                "median_price": float(np.median(ftele[f].prices)) if ftele[f].prices else None,
                "time_first_trade": ftele[f].time_first_trade,
                "posted_bids": ftele[f].posted_bids,
                "posted_asks": ftele[f].posted_asks,
                "execution_rate": (
                    ftele[f].n_trades / max(1, ftele[f].posted_bids + ftele[f].posted_asks)
                ),
                "ticks_no_bid": ftele[f].ticks_no_bid,
                "ticks_no_ask": ftele[f].ticks_no_ask,
                "unused_surplus": ftele[f].unused_surplus,
                # Best-bid/ask + depth series are stored compactly — the
                # summarizer uses them for spread + depth aggregates.
                "best_bid_series": ftele[f].best_bid_series,
                "best_ask_series": ftele[f].best_ask_series,
                "depth_bid_series": ftele[f].depth_bid_series,
                "depth_ask_series": ftele[f].depth_ask_series,
            }
            for f in FOOD_TYPES
        },
        "tick_logs": tick_logs,
        # Price telemetry — see the notes above `price_attempts` for the
        # meaning of "attempted vs submitted vs executed" here.
        "price_attempts": price_attempts,
        "executed_trade_prices": executed_trade_prices,
        # Convenience: quick per-game cap-violation total so the
        # summarizer can flag configs where the price cap bites.
        "n_cap_violations": sum(1 for a in price_attempts if a["cap_violation"]),
    }
    return game_log
