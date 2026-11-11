"""Direct, rotated GameState encoding for the centralized critic.

Optional Shared+ID (Condition B, v3): callers may append a 4-dim seat
one-hot.  ``GLOBAL_OBS_DIM`` remains the 305-D v2 contract and
``global_observation_dim(include_agent_id=True)`` returns 309.  The
rotation is unchanged: slot 0 is still the target player.

Optional trade history (v4): ``include_trade_history=True`` appends the
previous tick's executed-trade matrices (per food, buyer x seller quantity
and average price, rotated so slot 0 is the target player) — the same 96
features the *actor's* local encoding has always carried. Without them the
centralized critic sees strictly less market information than the actor it
is a baseline for: resting orders are in the state, but *executed* trades
are not reconstructible by a memoryless network. Feature order when both
switches are on: base 305, then trade history, then the agent-id one-hot.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..config import GameConfig
from ..types import GameState, Order

SEAT_COUNT = 4
PLAYER_FEATURES = 20
ORDER_FEATURES = 7
ORDER_SLOTS_PER_PLAYER = 4 * 2
PLAYER_STATE_DIM = SEAT_COUNT * PLAYER_FEATURES
ORDER_STATE_DIM = SEAT_COUNT * ORDER_SLOTS_PER_PLAYER * ORDER_FEATURES
GLOBAL_OBS_DIM = PLAYER_STATE_DIM + ORDER_STATE_DIM + 1
AGENT_ID_DIM: int = 4

# v4 trade history: 4 foods x (12 off-diagonal qty + 12 off-diagonal price).
TRADE_HISTORY_DIM: int = 4 * 2 * (SEAT_COUNT * SEAT_COUNT - SEAT_COUNT)


def global_observation_dim(
    *,
    include_agent_id: bool = False,
    include_trade_history: bool = False,
) -> int:
    return (
        GLOBAL_OBS_DIM
        + (TRADE_HISTORY_DIM if include_trade_history else 0)
        + (AGENT_ID_DIM if include_agent_id else 0)
    )

_CASH_SCALE = 200.0
_INV_SCALE = 50.0
_QTY_SCALE = 20.0
_PRICE_SCALE = 10.0


def _produces_food(produces: Any, food: str) -> bool:
    if isinstance(produces, str):
        return produces == food
    if isinstance(produces, Iterable):
        return food in produces
    return False


def _rotated_player_indices(target_player_idx: int) -> list[int]:
    return [
        (target_player_idx + offset) % SEAT_COUNT
        for offset in range(SEAT_COUNT)
    ]


def _find_order(
    state: GameState,
    player_idx: int,
    food: str,
    side: str,
) -> Order | None:
    book = state.order_books[food]
    orders = book.bids if side == "bid" else book.asks
    return next(
        (order for order in orders if order.player_idx == player_idx),
        None,
    )


def _append_trade_history(
    out: list[float],
    *,
    state: GameState,
    foods: tuple[str, ...],
    rotated_indices: list[int],
) -> None:
    """Previous-tick executed-trade matrices, rotated like everything else.

    Mirrors `observation_encoder._append_previous_tick_trades`: per food, a
    buyer x seller quantity matrix and an average-price matrix, both with
    the diagonal dropped (self-trades cannot happen). Seat order is the
    critic's rotation — slot 0 is the target player — so the same physical
    trade lands in different matrix cells for different target seats,
    exactly as it does in the local encoding.
    """
    slot_of = {
        player_idx: slot for slot, player_idx in enumerate(rotated_indices)
    }
    previous_tick = state.elapsed_seconds - 1
    trades = [
        t for t in state.trades if t.elapsed_second == previous_tick
    ] if previous_tick >= 0 else []

    for food in foods:
        qty_matrix = [[0.0] * SEAT_COUNT for _ in range(SEAT_COUNT)]
        price_qty_matrix = [[0.0] * SEAT_COUNT for _ in range(SEAT_COUNT)]
        for trade in trades:
            if trade.food_type != food:
                continue
            buyer_slot = slot_of[trade.buyer_idx]
            seller_slot = slot_of[trade.seller_idx]
            if buyer_slot == seller_slot:
                continue
            qty = float(trade.quantity)
            qty_matrix[buyer_slot][seller_slot] += qty
            price_qty_matrix[buyer_slot][seller_slot] += (
                float(trade.price_per_unit) * qty
            )
        for buyer_slot in range(SEAT_COUNT):
            for seller_slot in range(SEAT_COUNT):
                if buyer_slot != seller_slot:
                    out.append(qty_matrix[buyer_slot][seller_slot] / _QTY_SCALE)
        for buyer_slot in range(SEAT_COUNT):
            for seller_slot in range(SEAT_COUNT):
                if buyer_slot == seller_slot:
                    continue
                qty = qty_matrix[buyer_slot][seller_slot]
                price = (
                    price_qty_matrix[buyer_slot][seller_slot] / qty
                    if qty else 0.0
                )
                out.append(price / _PRICE_SCALE)


def encode_global_observation(
    state: GameState,
    target_player_idx: int,
    config: GameConfig,
    *,
    include_agent_id: bool = False,
    include_trade_history: bool = False,
) -> np.ndarray:
    """Encode the exact current state with the target player rotated to slot 0."""
    if config.player_count != SEAT_COUNT:
        raise ValueError(
            f"global encoder requires {SEAT_COUNT} players, "
            f"got {config.player_count}"
        )
    if len(state.players) != SEAT_COUNT:
        raise ValueError(
            f"global encoder requires {SEAT_COUNT} state players, "
            f"got {len(state.players)}"
        )
    if target_player_idx < 0 or target_player_idx >= SEAT_COUNT:
        raise ValueError(f"invalid target player index: {target_player_idx}")

    foods = config.food_types
    duration = float(config.game_duration_seconds)
    rotated_indices = _rotated_player_indices(target_player_idx)
    out: list[float] = []

    # Four players × 20 features. Slot 0 is always the target player.
    for player_idx in rotated_indices:
        player = state.players[player_idx]
        out.append(1.0 if player.status == "alive" else 0.0)
        for food in foods:
            out.append(
                1.0 if _produces_food(player.produces, food) else 0.0
            )
        out.extend(
            [
                player.cash / _CASH_SCALE,
                player.reserved_cash / _CASH_SCALE,
                player.available_cash() / _CASH_SCALE,
            ]
        )
        for food in foods:
            out.append(player.inventory[food] / _INV_SCALE)
        for food in foods:
            out.append(player.reserved_inventory[food] / _INV_SCALE)
        for food in foods:
            out.append(player.available(food) / _INV_SCALE)

    # Four rotated owners × four foods × bid/ask × seven order features.
    for player_idx in rotated_indices:
        for food in foods:
            for side in ("bid", "ask"):
                order = _find_order(state, player_idx, food, side)
                if order is None:
                    out.extend([0.0] * ORDER_FEATURES)
                    continue
                out.extend(
                    [
                        1.0,
                        order.price_per_unit / _PRICE_SCALE,
                        order.original_quantity / _QTY_SCALE,
                        order.remaining_quantity / _QTY_SCALE,
                        1.0 if order.status == "partially_filled" else 0.0,
                        order.created_at_tick / duration,
                        order.sequence
                        / float(config.max_open_orders_per_player * SEAT_COUNT),
                    ]
                )

    out.append(state.elapsed_seconds / duration)

    if include_trade_history:
        _append_trade_history(
            out, state=state, foods=foods, rotated_indices=rotated_indices
        )

    if include_agent_id:
        one_hot = [0.0] * AGENT_ID_DIM
        one_hot[target_player_idx] = 1.0
        out.extend(one_hot)

    encoded = np.ascontiguousarray(out, dtype=np.float32)
    expected_dim = global_observation_dim(
        include_agent_id=include_agent_id,
        include_trade_history=include_trade_history,
    )
    if encoded.shape != (expected_dim,):
        raise ValueError(
            f"global observation encoded to {encoded.shape}, "
            f"expected ({expected_dim},)"
        )
    return encoded
