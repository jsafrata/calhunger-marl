"""Stable vector encoding for per-seat Nash Hunger observations.

The 171 features are intentionally ordered and versioned by position:

    0:19    own private state
    19:35   public market summary
    35:50   opponents
    50:74   own open orders
    74:170  public trades from previous tick
    170:171 elapsed time

Optional Shared+ID (Condition B, v3): callers may request a 4-dim seat
one-hot appended at indices 171:175.  The public ``OBS_DIM`` remains the
171-D v2 contract; ``observation_dim(include_agent_id=True)`` returns 175.
Keeping the switch per environment/module (instead of process-global) lets
the evaluator load v2 and Shared+ID checkpoints in the same process.

Optional survival features (v4): ``include_survival_features=True`` appends
5 dims after the base 171 — per-food starvation cover in consumption cycles
(clipped and scaled) and the phase of the current consumption cycle. These
encode explicitly what an MLP would otherwise have to reconstruct as a
format-dependent sawtooth from the single elapsed-time scalar: how close
each required food is to killing the player, and how close the next
consumption tick is. Both are cycle-denominated, so they transfer across
game durations unchanged.

Feature order when both switches are on: base 171, then survival features,
then the agent-id one-hot — so the fixed ``AGENT_ID_SLICE`` constant is
only valid for the survival-features-off layouts.

Keep this module independent of any particular RL framework so Gymnasium,
RLlib, and future adapters all train against the same representation.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..config import GameConfig

# Fixed normalization scales retained from the original Gymnasium adapter.
_CASH_SCALE = 200.0
_INV_SCALE = 50.0
_QTY_SCALE = 20.0
_PRICE_SCALE = 10.0

OWN_PRIVATE_SLICE = slice(0, 19)
PUBLIC_MARKET_SLICE = slice(19, 35)
OPPONENTS_SLICE = slice(35, 50)
OWN_ORDERS_SLICE = slice(50, 74)
PUBLIC_TRADES_SLICE = slice(74, 170)
TIME_SLICE = slice(170, 171)


AGENT_ID_DIM: int = 4  # matches config.player_count; enforced at encode time
OBS_DIM: int = 171
AGENT_ID_SLICE = slice(171, 171 + AGENT_ID_DIM)

# v4 survival features: 4 per-food starvation covers + 1 cycle phase.
SURVIVAL_FEATURES_DIM: int = 5
SURVIVAL_FEATURES_SLICE = slice(171, 171 + SURVIVAL_FEATURES_DIM)
# Cover is measured in consumption cycles and clipped here before scaling to
# [0, 1]. Ten cycles of cover is far beyond any survival-relevant margin
# (the deepest scripted stockpiler targets five), so the clip loses nothing
# while keeping the feature bounded across formats.
COVER_CYCLES_CAP: float = 10.0


def observation_dim(
    *,
    include_agent_id: bool = False,
    include_survival_features: bool = False,
) -> int:
    return (
        OBS_DIM
        + (SURVIVAL_FEATURES_DIM if include_survival_features else 0)
        + (AGENT_ID_DIM if include_agent_id else 0)
    )


def _produces_food(produces: Any, food: str) -> bool:
    """Support both the current list representation and legacy single strings."""
    if isinstance(produces, str):
        return produces == food
    if isinstance(produces, Iterable):
        return food in produces
    return False


def _participant_order(obs: dict[str, Any]) -> list[int]:
    own_idx = obs["player_idx"]
    opponents = [p["idx"] for p in obs.get("players", ()) if p["idx"] != own_idx]
    return [own_idx, *opponents[:3]]


def _append_previous_tick_trades(
    out: list[float],
    *,
    obs: dict[str, Any],
    foods: tuple[str, ...],
) -> None:
    participants = _participant_order(obs)
    participant_pos = {player_idx: idx for idx, player_idx in enumerate(participants)}
    trades = obs.get("previous_tick_trades", ())

    for food in foods:
        qty_matrix = [[0.0 for _ in range(4)] for _ in range(4)]
        price_qty_matrix = [[0.0 for _ in range(4)] for _ in range(4)]

        for trade in trades:
            if trade.get("food") != food:
                continue
            buyer_pos = participant_pos.get(trade.get("buyer_idx"))
            seller_pos = participant_pos.get(trade.get("seller_idx"))
            if buyer_pos is None or seller_pos is None or buyer_pos == seller_pos:
                continue
            qty = float(trade.get("qty", 0))
            price = float(trade.get("price", 0))
            qty_matrix[buyer_pos][seller_pos] += qty
            price_qty_matrix[buyer_pos][seller_pos] += price * qty

        for buyer_pos in range(4):
            for seller_pos in range(4):
                if buyer_pos != seller_pos:
                    out.append(qty_matrix[buyer_pos][seller_pos] / _QTY_SCALE)

        for buyer_pos in range(4):
            for seller_pos in range(4):
                if buyer_pos == seller_pos:
                    continue
                qty = qty_matrix[buyer_pos][seller_pos]
                price = price_qty_matrix[buyer_pos][seller_pos] / qty if qty else 0.0
                out.append(price / _PRICE_SCALE)


def _append_survival_features(
    out: list[float],
    *,
    obs: dict[str, Any],
    foods: tuple[str, ...],
) -> None:
    """Per-food starvation cover (in cycles) + consumption-cycle phase.

    Cover for a required food is `starvation_seconds / interval` — the number
    of consumption ticks the current available inventory survives. Produced
    foods are never consumed by their producer, so they encode the cap
    (never starving). The phase feature is the fraction of the current cycle
    still to run before the next consumption tick, in (0, 1] — at 1.0 the
    tick just fired; near 0.0 it is imminent. Both are denominated in
    cycles, not wall-clock, so the same values mean the same thing in every
    game-duration format.
    """
    interval = float(obs["consumption_interval_seconds"])
    starvation = obs.get("starvation_seconds", {})
    for food in foods:
        if food in starvation:
            cover_cycles = float(starvation[food]) / interval
        else:
            cover_cycles = COVER_CYCLES_CAP
        out.append(min(cover_cycles, COVER_CYCLES_CAP) / COVER_CYCLES_CAP)
    steps_to_tick = interval - (float(obs["tick"]) % interval)
    out.append(steps_to_tick / interval)


def encode_observation(
    obs: dict[str, Any],
    config: GameConfig,
    *,
    include_agent_id: bool = False,
    include_survival_features: bool = False,
) -> np.ndarray:
    """Encode one legal per-seat observation as a normalized float32 vector."""
    foods = config.food_types
    duration = float(config.game_duration_seconds)
    out: list[float] = []

    # 0:19 — own private state.
    out.append(obs["cash"] / _CASH_SCALE)
    out.append(obs["reserved_cash"] / _CASH_SCALE)
    out.append(obs["available_cash"] / _CASH_SCALE)
    for food in foods:
        out.append(obs["inventory"][food] / _INV_SCALE)
    for food in foods:
        out.append(obs["reserved_inventory"][food] / _INV_SCALE)
    for food in foods:
        out.append(obs["available_inventory"][food] / _INV_SCALE)
    for food in foods:
        out.append(1.0 if _produces_food(obs.get("produces"), food) else 0.0)

    # 19:35 — best public bid/ask levels for each food.
    books = obs.get("order_books", {})
    for food in foods:
        book = books.get(food, {})
        bids = book.get("bids", ())
        asks = book.get("asks", ())
        best_bid = bids[0].get("price", 0) if bids else 0
        best_ask = asks[0].get("price", 0) if asks else 0
        bid_qty = bids[0].get("qty", 0) if bids else 0
        ask_qty = asks[0].get("qty", 0) if asks else 0
        out.extend(
            [
                best_bid / _PRICE_SCALE,
                best_ask / _PRICE_SCALE,
                bid_qty / _QTY_SCALE,
                ask_qty / _QTY_SCALE,
            ]
        )

    # 35:50 — opponents in the stable order supplied by the game observation.
    own_idx = obs["player_idx"]
    opponents = [p for p in obs.get("players", ()) if p["idx"] != own_idx]
    for opponent_idx in range(3):
        if opponent_idx >= len(opponents):
            out.extend([0.0] * (1 + len(foods)))
            continue
        opponent = opponents[opponent_idx]
        out.append(1.0 if opponent["status"] == "alive" else 0.0)
        for food in foods:
            out.append(
                1.0
                if _produces_food(opponent.get("produces"), food)
                else 0.0
            )

    # 50:74 — own open order state by food, then bid/ask.
    buckets = {(food, side): [] for food in foods for side in ("bid", "ask")}
    for order in obs.get("own_orders", ()):
        key = (order["food"], order["side"])
        if key in buckets:
            buckets[key].append(order)
    for food in foods:
        for side in ("bid", "ask"):
            orders = buckets[(food, side)]
            total_qty = sum(order["remaining_qty"] for order in orders)
            avg_price = (
                sum(
                    order["price"] * order["remaining_qty"]
                    for order in orders
                )
                / total_qty
                if total_qty
                else 0
            )
            out.extend(
                [
                    1.0 if orders else 0.0,
                    total_qty / _QTY_SCALE,
                    avg_price / _PRICE_SCALE,
                ]
            )

    # 74:170 — previous-tick public trade matrices.
    _append_previous_tick_trades(out, obs=obs, foods=foods)

    # 170:171 — elapsed game fraction.
    out.append(obs["tick"] / duration)

    # 171:176 — optional v4 survival features (cover + cycle phase).
    if include_survival_features:
        _append_survival_features(out, obs=obs, foods=foods)

    # Optional seat one-hot for Condition B (Shared+ID). At 171:175 in the
    # v3 layout; shifted after the survival block when that switch is on.
    if include_agent_id:
        own_idx = int(obs["player_idx"])
        if not (0 <= own_idx < AGENT_ID_DIM):
            raise ValueError(
                f"agent-id one-hot requires player_idx in [0,{AGENT_ID_DIM}), "
                f"got {own_idx}"
            )
        one_hot = [0.0] * AGENT_ID_DIM
        one_hot[own_idx] = 1.0
        out.extend(one_hot)

    encoded = np.ascontiguousarray(out, dtype=np.float32)
    expected_dim = observation_dim(
        include_agent_id=include_agent_id,
        include_survival_features=include_survival_features,
    )
    if encoded.shape != (expected_dim,):
        raise ValueError(
            f"observation encoded to {encoded.shape}, expected ({expected_dim},)"
        )
    return encoded
