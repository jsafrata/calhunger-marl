"""Discrete action space for the simulator — full positive-integer action space.

Every action carries at most the tuple `(action_type, food, side, price, qty)`.
The action-type dimension has four values: `wait`, `post`, `cancel`, `take`.
For each type, only some of the other fields are meaningful:

    wait   : (nothing else — a single index)
    cancel : (side, food)                       — no price, no qty
    take   : (side, food, qty)                  — best-available price
    post   : (side, food, qty, price)           — all fields matter

There is no `cancel_all` and no explicit `replace`: `post` on the same
`(side, food)` automatically cancels the player's prior open order on that
book, mirroring the real game.

Layout (flat Discrete):

    [0]                 = WAIT
    [1 ..  8]           = CANCEL(side, food)              — 8 actions
    [9 .. 168]          = TAKE  (side, food, qty)         — 2·4·MAX_QTY
    [169 .. NUM-1]      = POST  (side, food, qty, price)  — 2·4·MAX_QTY·MAX_PRICE

`MAX_PRICE` and `MAX_QTY` are module constants. They cap the space so we can
represent it as a finite Discrete — raise them if you want the agent to be
able to bid/ask higher prices or bigger quantities. Bear in mind that the
POST band grows linearly in both, so NUM_ACTIONS scales as O(MAX_PRICE·MAX_QTY).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from .config import GameConfig
from .market import (
    OrderError,
    cancel_order,
    cancel_oldest_for_player_side_food,
    clear_books,
    post_order,
    post_order_without_matching,
    take_from_book,
    validate_order_replacement,
)
from .types import GameState

# ---------------------------------------------------------------------------
# Space parameters — positive integers only. Bump if you need bigger orders.
#
# The v2 defaults are MAX_PRICE=50 and MAX_QTY=5. Override for a single
# process via env vars HG_MAX_PRICE / HG_MAX_QTY, or programmatically via
# `set_action_space(...)` (tests only — downstream modules that already
# imported the names by value will not see the update).
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        val = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if val < 1:
        raise ValueError(f"{name} must be >= 1, got {val}")
    return val


MAX_PRICE: int = _env_int("HG_MAX_PRICE", 50)  # inclusive: price ∈ {1, ..., MAX_PRICE}
MAX_QTY: int = _env_int("HG_MAX_QTY", 5)       # inclusive: qty   ∈ {1, ..., MAX_QTY}

SIDES = ("bid", "ask")
FOODS = ("A", "B", "C", "D")
NUM_SIDES = len(SIDES)
NUM_FOODS = len(FOODS)

# ---------------------------------------------------------------------------
# Factored action-type ids. Used by the factored / autoregressive policy.
# Order is important — the policy conditions food on type, side on (type,
# food), etc. — so keep these stable.
# ---------------------------------------------------------------------------

TYPE_WAIT: int = 0
TYPE_CANCEL: int = 1
TYPE_TAKE: int = 2
TYPE_POST: int = 3
NUM_ACTION_TYPES: int = 4
ACTION_TYPES = ("wait", "cancel", "take", "post")

# Index layout — recomputed if MAX_PRICE / MAX_QTY change at import time.
ACTION_WAIT: int = 0
CANCEL_BASE: int = 1                           # [1 .. 8]
NUM_CANCEL_ACTIONS = NUM_SIDES * NUM_FOODS     # 8

TAKE_BASE: int = CANCEL_BASE + NUM_CANCEL_ACTIONS  # 9
NUM_TAKE_ACTIONS = NUM_SIDES * NUM_FOODS * MAX_QTY

POST_BASE: int = TAKE_BASE + NUM_TAKE_ACTIONS
NUM_POST_ACTIONS = NUM_SIDES * NUM_FOODS * MAX_QTY * MAX_PRICE

NUM_ACTIONS: int = POST_BASE + NUM_POST_ACTIONS


def set_action_space(
    *, max_price: Optional[int] = None, max_qty: Optional[int] = None
) -> dict:
    """Update MAX_PRICE / MAX_QTY and every derived constant in place.

    Intended for in-process test harnesses that want to exercise more than
    one bound configuration without a subprocess. Production code should
    instead set HG_MAX_PRICE / HG_MAX_QTY *before* any RL import — modules
    that already did `from env.actions import MAX_PRICE, ...` hold their
    own references and will not observe subsequent mutations.
    """
    global MAX_PRICE, MAX_QTY
    global NUM_TAKE_ACTIONS, POST_BASE, NUM_POST_ACTIONS, NUM_ACTIONS
    global QTY_BUCKETS, PRICE_BUCKETS, NUM_QTY, NUM_PRICE
    if max_price is not None:
        if max_price < 1:
            raise ValueError(f"max_price must be >= 1, got {max_price}")
        MAX_PRICE = int(max_price)
    if max_qty is not None:
        if max_qty < 1:
            raise ValueError(f"max_qty must be >= 1, got {max_qty}")
        MAX_QTY = int(max_qty)
    NUM_TAKE_ACTIONS = NUM_SIDES * NUM_FOODS * MAX_QTY
    POST_BASE = TAKE_BASE + NUM_TAKE_ACTIONS
    NUM_POST_ACTIONS = NUM_SIDES * NUM_FOODS * MAX_QTY * MAX_PRICE
    NUM_ACTIONS = POST_BASE + NUM_POST_ACTIONS
    QTY_BUCKETS = tuple(range(1, MAX_QTY + 1))
    PRICE_BUCKETS = tuple(range(1, MAX_PRICE + 1))
    NUM_QTY = MAX_QTY
    NUM_PRICE = MAX_PRICE
    return {
        "MAX_PRICE": MAX_PRICE,
        "MAX_QTY": MAX_QTY,
        "NUM_ACTIONS": NUM_ACTIONS,
    }

# ---------------------------------------------------------------------------
# Back-compat aliases. The bucketed action space this replaced exposed a few
# module-level names that agents / tests / parity code still import. Rather
# than touch a dozen unrelated call sites, we alias them onto the new
# integer space:
#   QTY_BUCKETS   = (1, 2, ..., MAX_QTY)   — every positive integer is its
#                                             own "bucket" now
#   PRICE_BUCKETS = (1, 2, ..., MAX_PRICE)  — likewise for prices
#   encode_trade  = compat wrapper around encode_post, treating its qty/price
#                   args as 0-indexed bucket indices into the tuples above
#   ACTION_NOOP   = ACTION_WAIT             — semantically identical
#
# `ACTION_CANCEL_ALL` and `CancelAllAction` are *not* aliased — the new
# space does not have a cancel-all action (post auto-replaces + individual
# cancels cover every use case). Callers that referenced them are patched
# explicitly.
# ---------------------------------------------------------------------------

ACTION_NOOP: int = ACTION_WAIT
ACTION_CANCEL_BASE: int = CANCEL_BASE

QTY_BUCKETS = tuple(range(1, MAX_QTY + 1))
PRICE_BUCKETS = tuple(range(1, MAX_PRICE + 1))
NUM_QTY = MAX_QTY
NUM_PRICE = MAX_PRICE


def encode_trade(side: str, food: str, qty_bucket: int, price_bucket: int) -> int:
    """Compat wrapper: legacy callers passed 0-indexed bucket ids into fixed
    tuples. In the new space every integer is its own bucket, so bucket idx k
    corresponds directly to `k + 1` (the 1-based positive integer)."""
    return encode_post(
        side, food, qty=qty_bucket + 1, price=price_bucket + 1
    )


# ---------------------------------------------------------------------------
# Typed decoded actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaitAction:
    pass


@dataclass(frozen=True)
class CancelAction:
    side: str
    food_type: str


@dataclass(frozen=True)
class TakeAction:
    """Immediate-or-cancel market-take against the resting book.

    `side` follows the same convention as post/cancel — it is the *book side
    the agent posts to conceptually*, NOT the book side it consumes from:

      side == "bid"  →  the agent is BUYING.  It walks the ASK book,
                        hitting the lowest asks first (best price for a
                        buyer). The agent PAYS cash and receives food.

      side == "ask"  →  the agent is SELLING. It walks the BID book,
                        hitting the highest bids first (best price for a
                        seller). The agent DELIVERS food and receives cash.

    Use `.direction` if you want the agent-centric verb ("buy" / "sell")
    instead of the book-side descriptor.
    """

    side: str
    food_type: str
    quantity: int

    @property
    def direction(self) -> str:
        """Plain-English agent verb: "buy" for side="bid", "sell" for side="ask"."""
        return "buy" if self.side == "bid" else "sell"


@dataclass(frozen=True)
class PostAction:
    side: str
    food_type: str
    quantity: int
    price_per_unit: int


# Preserve the old name so isinstance checks in agents/tests keep pointing at
# something. PostAction supersedes TradeAction — they carry the same fields.
TradeAction = PostAction
NoopAction = WaitAction


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def _validate_side_food(side: str, food: str) -> None:
    if side not in SIDES:
        raise ValueError(f"unknown side: {side!r}")
    if food not in FOODS:
        raise ValueError(f"unknown food: {food!r}")


def encode_wait() -> int:
    return ACTION_WAIT


def encode_cancel(side: str, food: str) -> int:
    _validate_side_food(side, food)
    return CANCEL_BASE + SIDES.index(side) * NUM_FOODS + FOODS.index(food)


def encode_take(side: str, food: str, qty: int) -> int:
    _validate_side_food(side, food)
    if not (1 <= qty <= MAX_QTY):
        raise ValueError(f"take qty must be in 1..{MAX_QTY}, got {qty}")
    return (
        TAKE_BASE
        + SIDES.index(side) * (NUM_FOODS * MAX_QTY)
        + FOODS.index(food) * MAX_QTY
        + (qty - 1)
    )


def encode_post(side: str, food: str, qty: int, price: int) -> int:
    _validate_side_food(side, food)
    if not (1 <= qty <= MAX_QTY):
        raise ValueError(f"post qty must be in 1..{MAX_QTY}, got {qty}")
    if not (1 <= price <= MAX_PRICE):
        raise ValueError(f"post price must be in 1..{MAX_PRICE}, got {price}")
    return (
        POST_BASE
        + SIDES.index(side) * (NUM_FOODS * MAX_QTY * MAX_PRICE)
        + FOODS.index(food) * (MAX_QTY * MAX_PRICE)
        + (qty - 1) * MAX_PRICE
        + (price - 1)
    )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def decode(action: int):
    if action < 0 or action >= NUM_ACTIONS:
        raise ValueError(f"action {action} out of range [0, {NUM_ACTIONS})")
    if action == ACTION_WAIT:
        return WaitAction()
    if action < TAKE_BASE:
        rel = action - CANCEL_BASE
        side = SIDES[rel // NUM_FOODS]
        food = FOODS[rel % NUM_FOODS]
        return CancelAction(side=side, food_type=food)
    if action < POST_BASE:
        rel = action - TAKE_BASE
        side_idx, rest = divmod(rel, NUM_FOODS * MAX_QTY)
        food_idx, qty_m1 = divmod(rest, MAX_QTY)
        return TakeAction(
            side=SIDES[side_idx],
            food_type=FOODS[food_idx],
            quantity=qty_m1 + 1,
        )
    rel = action - POST_BASE
    side_idx, rest = divmod(rel, NUM_FOODS * MAX_QTY * MAX_PRICE)
    food_idx, rest = divmod(rest, MAX_QTY * MAX_PRICE)
    qty_m1, price_m1 = divmod(rest, MAX_PRICE)
    return PostAction(
        side=SIDES[side_idx],
        food_type=FOODS[food_idx],
        quantity=qty_m1 + 1,
        price_per_unit=price_m1 + 1,
    )


# ---------------------------------------------------------------------------
# Legal-action mask
# ---------------------------------------------------------------------------


def _opposite_book_side(state: GameState, food: str, side: str):
    """The pool the taker/poster would trade against.

    A bid (buyer) trades against the asks; an ask (seller) trades against the
    bids. Returned pool is *not* sorted here — the market module sorts as
    needed when it matches.
    """
    book = state.order_books[food]
    return book.asks if side == "bid" else book.bids


def _best_takeable_price(state: GameState, food: str, side: str, player_idx: int):
    """Best price for a taker on `side` (side is what the taker *is*).

    Returns None if there's nothing to take (empty book or the only resting
    orders are the player's own — self-trades aren't allowed).
    """
    pool = _opposite_book_side(state, food, side)
    if side == "bid":
        best = None
        for o in pool:
            if o.player_idx == player_idx:
                continue
            if best is None or o.price_per_unit < best.price_per_unit:
                best = o
    else:
        best = None
        for o in pool:
            if o.player_idx == player_idx:
                continue
            if best is None or o.price_per_unit > best.price_per_unit:
                best = o
    return best.price_per_unit if best is not None else None


def legal_action_mask(state: GameState, player_idx: int) -> np.ndarray:
    """Boolean mask of shape (NUM_ACTIONS,). True bits are legal for this player.

    WAIT is always legal (even for dead players — so the mask has ≥1 True bit,
    which agents / masked softmaxes rely on).
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[ACTION_WAIT] = True

    p = state.players[player_idx]
    if p.status != "alive" or state.phase != "active":
        return mask

    _config: GameConfig = state.config  # type: ignore[assignment]
    avail_cash = p.available_cash()

    for side_idx, side in enumerate(SIDES):
        for food_idx, food in enumerate(FOODS):
            avail_food = p.available(food)

            # ---- CANCEL(side, food) ----
            book = state.order_books[food]
            pool = book.bids if side == "bid" else book.asks
            if any(o.player_idx == player_idx for o in pool):
                mask[CANCEL_BASE + side_idx * NUM_FOODS + food_idx] = True

            # ---- TAKE(side, food, qty) ----
            best_price = _best_takeable_price(state, food, side, player_idx)
            if best_price is not None:
                # Take needs at least 1 unit of capacity on the constraining side.
                # Buying: qty * best_ask ≤ cash. Selling: qty ≤ available_food.
                if side == "bid":
                    max_affordable = avail_cash // max(best_price, 1)
                    max_qty = min(MAX_QTY, max_affordable)
                else:
                    max_qty = min(MAX_QTY, avail_food)
                if max_qty >= 1:
                    take_base = (
                        TAKE_BASE
                        + side_idx * (NUM_FOODS * MAX_QTY)
                        + food_idx * MAX_QTY
                    )
                    for q in range(1, max_qty + 1):
                        mask[take_base + (q - 1)] = True

            # ---- POST(side, food, qty, price) ----
            post_base_sf = (
                POST_BASE
                + side_idx * (NUM_FOODS * MAX_QTY * MAX_PRICE)
                + food_idx * (MAX_QTY * MAX_PRICE)
            )
            if side == "bid":
                # qty * price ≤ avail_cash → for each price, qty ≤ avail_cash/price
                for price in range(1, MAX_PRICE + 1):
                    max_qty = min(MAX_QTY, avail_cash // price)
                    if max_qty < 1:
                        continue
                    for q in range(1, max_qty + 1):
                        mask[post_base_sf + (q - 1) * MAX_PRICE + (price - 1)] = True
            else:
                if avail_food >= 1:
                    max_qty = min(MAX_QTY, avail_food)
                    for q in range(1, max_qty + 1):
                        row = post_base_sf + (q - 1) * MAX_PRICE
                        mask[row : row + MAX_PRICE] = True

    return mask


# ---------------------------------------------------------------------------
# Factored ↔ flat converters
# ---------------------------------------------------------------------------
#
# The factored policy sees each action as a 5-tuple of 0-indexed ids:
#
#     (type_id, food_id, side_id, price_id, qty_id)
#
# Only the ids that are meaningful for the sampled action-type matter — the
# rest are placeholders (always 0) and don't contribute to log-probability.
#
#     WAIT   →  (0, _, _, _, _)                     placeholders in all others
#     CANCEL →  (1, food_id, side_id, _, _)         price/qty are placeholders
#     TAKE   →  (2, food_id, side_id, _, qty_id)    price is placeholder
#     POST   →  (3, food_id, side_id, price_id, qty_id)    all five used
#
# price_id and qty_id are 0-based, so `price_id + 1` is the actual price and
# `qty_id + 1` is the actual quantity (both positive integers).


def flat_to_factored(action: int) -> tuple[int, int, int, int, int]:
    """Decode a flat action index into `(type, food, side, price, qty)` ids.

    Irrelevant fields are set to 0. This means two different flat actions
    (e.g. WAIT and POST(bid, A, qty=1, price=1)) can share the same tuple
    when the ids overlap — that's fine: consumers rely on the action-type
    id to know which fields to honor.
    """
    if action < 0 or action >= NUM_ACTIONS:
        raise ValueError(f"action {action} out of range [0, {NUM_ACTIONS})")
    if action == ACTION_WAIT:
        return (TYPE_WAIT, 0, 0, 0, 0)
    if action < TAKE_BASE:  # CANCEL
        rel = action - CANCEL_BASE
        side_id = rel // NUM_FOODS
        food_id = rel % NUM_FOODS
        return (TYPE_CANCEL, food_id, side_id, 0, 0)
    if action < POST_BASE:  # TAKE
        rel = action - TAKE_BASE
        side_id, rest = divmod(rel, NUM_FOODS * MAX_QTY)
        food_id, qty_m1 = divmod(rest, MAX_QTY)
        return (TYPE_TAKE, food_id, side_id, 0, qty_m1)
    # POST
    rel = action - POST_BASE
    side_id, rest = divmod(rel, NUM_FOODS * MAX_QTY * MAX_PRICE)
    food_id, rest = divmod(rest, MAX_QTY * MAX_PRICE)
    qty_m1, price_m1 = divmod(rest, MAX_PRICE)
    return (TYPE_POST, food_id, side_id, price_m1, qty_m1)


def factored_to_flat(
    type_id: int, food_id: int, side_id: int, price_id: int, qty_id: int
) -> int:
    """Encode a factored tuple back into a flat action index.

    `food_id`, `side_id`, `price_id`, and `qty_id` are ignored when the
    action type doesn't depend on them — pass 0 (or anything else, we drop
    the value) in those positions.
    """
    if type_id == TYPE_WAIT:
        return ACTION_WAIT
    if type_id == TYPE_CANCEL:
        return CANCEL_BASE + side_id * NUM_FOODS + food_id
    if type_id == TYPE_TAKE:
        return (
            TAKE_BASE
            + side_id * (NUM_FOODS * MAX_QTY)
            + food_id * MAX_QTY
            + qty_id
        )
    if type_id == TYPE_POST:
        return (
            POST_BASE
            + side_id * (NUM_FOODS * MAX_QTY * MAX_PRICE)
            + food_id * (MAX_QTY * MAX_PRICE)
            + qty_id * MAX_PRICE
            + price_id
        )
    raise ValueError(f"unknown type_id: {type_id}")


# ---------------------------------------------------------------------------
# Per-head legal masks for the factored policy
# ---------------------------------------------------------------------------
#
# The flat legal mask has one entry per action id (NUM_ACTIONS bits). For an
# autoregressive policy we need masks per head, and each head's mask is
# *conditioned* on the sampled ids of previous heads:
#
#   type_mask[t]                              — is any action of type t legal?
#   food_mask[t, f]                           — is (t, f, *, *, *)  legal for some (side, price, qty)?
#   side_mask[t, f, s]                        — is (t, f, s, *, *)  legal?
#   price_mask[t, f, s, p]                    — is (t, f, s, p, *)  legal? (only meaningful for POST)
#   qty_mask[t, f, s, p, q]                   — is (t, f, s, p, q)  legal?
#
# For heads that a given action type doesn't use (e.g. `price` for TAKE), the
# corresponding mask along that dimension is all-True — we don't want to
# accidentally mask out the placeholder id 0.


@dataclass(frozen=True)
class FactoredMasks:
    """Per-head boolean masks derived from a flat legal-action mask.

    Every field is a numpy bool array. Shapes:
        type_mask     : (NUM_ACTION_TYPES,)
        food_mask     : (NUM_ACTION_TYPES, NUM_FOODS)
        side_mask     : (NUM_ACTION_TYPES, NUM_FOODS, NUM_SIDES)
        price_mask    : (NUM_ACTION_TYPES, NUM_FOODS, NUM_SIDES, MAX_PRICE)
        qty_mask      : (NUM_ACTION_TYPES, NUM_FOODS, NUM_SIDES, MAX_PRICE, MAX_QTY)

    A head's mask entries default to True in dimensions the action type does
    not depend on, so sampling a placeholder id (0) is always "legal" for
    those heads.
    """

    type_mask: np.ndarray
    food_mask: np.ndarray
    side_mask: np.ndarray
    price_mask: np.ndarray
    qty_mask: np.ndarray


def factored_masks_from_flat(flat_mask: np.ndarray) -> FactoredMasks:
    """Derive per-head autoregressive masks from a flat legal-action mask.

    The returned masks encode the *conditional* legality of each head given
    every possible combination of previously-sampled ids. This is small
    (total ≈ NUM_ACTIONS bits) so we materialise it directly rather than
    computing lazily.
    """
    if flat_mask.shape != (NUM_ACTIONS,):
        raise ValueError(
            f"flat_mask must have shape ({NUM_ACTIONS},), got {flat_mask.shape}"
        )

    type_mask = np.zeros((NUM_ACTION_TYPES,), dtype=bool)
    food_mask = np.zeros(
        (NUM_ACTION_TYPES, NUM_FOODS), dtype=bool
    )
    side_mask = np.zeros(
        (NUM_ACTION_TYPES, NUM_FOODS, NUM_SIDES), dtype=bool
    )
    price_mask = np.zeros(
        (NUM_ACTION_TYPES, NUM_FOODS, NUM_SIDES, MAX_PRICE), dtype=bool
    )
    qty_mask = np.zeros(
        (NUM_ACTION_TYPES, NUM_FOODS, NUM_SIDES, MAX_PRICE, MAX_QTY),
        dtype=bool,
    )

    # ---- WAIT ----
    if flat_mask[ACTION_WAIT]:
        type_mask[TYPE_WAIT] = True
        # WAIT doesn't use food/side/price/qty — mark everything legal so
        # the placeholder id 0 always samples cleanly.
        food_mask[TYPE_WAIT, :] = True
        side_mask[TYPE_WAIT, :, :] = True
        price_mask[TYPE_WAIT, :, :, :] = True
        qty_mask[TYPE_WAIT, :, :, :, :] = True

    # ---- CANCEL ----
    for side_id, side in enumerate(SIDES):
        for food_id, food in enumerate(FOODS):
            idx = CANCEL_BASE + side_id * NUM_FOODS + food_id
            if flat_mask[idx]:
                type_mask[TYPE_CANCEL] = True
                food_mask[TYPE_CANCEL, food_id] = True
                side_mask[TYPE_CANCEL, food_id, side_id] = True
    # CANCEL doesn't use price/qty — placeholder-id 0 must always be legal.
    price_mask[TYPE_CANCEL, :, :, :] = True
    qty_mask[TYPE_CANCEL, :, :, :, :] = True

    # ---- TAKE ----
    take_span = NUM_SIDES * NUM_FOODS * MAX_QTY
    take_slice = flat_mask[TAKE_BASE : TAKE_BASE + take_span]
    take_grid = take_slice.reshape(NUM_SIDES, NUM_FOODS, MAX_QTY)
    for side_id in range(NUM_SIDES):
        for food_id in range(NUM_FOODS):
            legal_qtys = take_grid[side_id, food_id]
            if legal_qtys.any():
                type_mask[TYPE_TAKE] = True
                food_mask[TYPE_TAKE, food_id] = True
                side_mask[TYPE_TAKE, food_id, side_id] = True
                # Copy the same qty-mask under every price slot — TAKE doesn't
                # use price, so the placeholder id 0 will be sampled from any
                # price slot, and the qty head must be legal there.
                qty_mask[TYPE_TAKE, food_id, side_id, :, :] = legal_qtys[None, :]
    # TAKE doesn't use price — placeholder-id 0 must always be legal.
    price_mask[TYPE_TAKE, :, :, :] = True

    # ---- POST ----
    post_span = NUM_SIDES * NUM_FOODS * MAX_QTY * MAX_PRICE
    post_slice = flat_mask[POST_BASE : POST_BASE + post_span]
    post_grid = post_slice.reshape(NUM_SIDES, NUM_FOODS, MAX_QTY, MAX_PRICE)
    for side_id in range(NUM_SIDES):
        for food_id in range(NUM_FOODS):
            per_food = post_grid[side_id, food_id]  # (MAX_QTY, MAX_PRICE)
            if per_food.any():
                type_mask[TYPE_POST] = True
                food_mask[TYPE_POST, food_id] = True
                side_mask[TYPE_POST, food_id, side_id] = True
                # For each (price), any legal qty makes the price legal.
                # per_food.T is (MAX_PRICE, MAX_QTY) — .any(axis=-1) collapses qty.
                price_legal = per_food.any(axis=0)  # (MAX_PRICE,) — any qty legal?
                price_mask[TYPE_POST, food_id, side_id, :] = price_legal
                # qty is (price → qty legal set) — per_food.T is (MAX_PRICE, MAX_QTY)
                qty_mask[TYPE_POST, food_id, side_id, :, :] = per_food.T

    return FactoredMasks(
        type_mask=type_mask,
        food_mask=food_mask,
        side_mask=side_mask,
        price_mask=price_mask,
        qty_mask=qty_mask,
    )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_action(state: GameState, player_idx: int, action: int) -> dict:
    """Decode + apply. Never raises: illegal actions become a silent no-op with
    `{"applied": False, "code": ...}` so RL agents can't crash the env.
    """
    decoded = decode(action)
    p = state.players[player_idx]
    if p.status != "alive":
        return {"applied": False, "code": "player_dead"}
    if state.phase != "active":
        return {"applied": False, "code": "game_not_active"}

    if isinstance(decoded, WaitAction):
        return {"applied": True, "kind": "wait"}

    if isinstance(decoded, CancelAction):
        cancelled = cancel_oldest_for_player_side_food(
            state,
            player_idx=player_idx,
            side=decoded.side,
            food_type=decoded.food_type,
        )
        if cancelled is None:
            return {"applied": False, "code": "no_order_to_cancel"}
        return {"applied": True, "kind": "cancel", "order_id": cancelled.id}

    if isinstance(decoded, TakeAction):
        trades = take_from_book(
            state,
            player_idx=player_idx,
            side=decoded.side,
            food_type=decoded.food_type,
            quantity=decoded.quantity,
        )
        return {
            "applied": True,
            "kind": "take",
            "n_trades": len(trades),
            "quantity_filled": sum(t.quantity for t in trades),
        }

    # PostAction
    try:
        order, trades = post_order(
            state,
            player_idx=player_idx,
            side=decoded.side,
            food_type=decoded.food_type,
            quantity=decoded.quantity,
            price_per_unit=decoded.price_per_unit,
        )
    except OrderError as e:
        return {"applied": False, "code": e.code}
    return {
        "applied": True,
        "kind": "post",
        "order_id": order.id,
        "n_trades": len(trades),
    }


def _has_cancel_target(
    state: GameState, *, player_idx: int, side: str, food_type: str
) -> bool:
    book = state.order_books[food_type]
    pool = book.bids if side == "bid" else book.asks
    return any(o.player_idx == player_idx for o in pool)


def apply_actions_simultaneous(
    state: GameState,
    actions: Mapping[int, int],
    *,
    player_order: Sequence[int],
    food_order: Sequence[str],
) -> dict[int, dict]:
    """Apply one simultaneous market tick and then clear crossed books.

    - Waits, cancels, and takes execute immediately in `player_order` (they
      don't leave lasting orders that need arbitrage).
    - Posts go through a two-phase pipeline: validate all posts against the
      pre-tick book, then queue them into the book WITHOUT matching, then run
      one book-clearing pass so no seat gets to trade before later seats have
      submitted their quote.
    """
    decoded_by_idx = {
        idx: decode(actions.get(idx, ACTION_WAIT))
        for idx in range(len(state.players))
    }
    applied: dict[int, dict] = {}

    for idx, decoded in decoded_by_idx.items():
        p = state.players[idx]
        if p.status != "alive":
            applied[idx] = {"applied": False, "code": "player_dead"}
            continue
        if state.phase != "active":
            applied[idx] = {"applied": False, "code": "game_not_active"}
            continue
        if isinstance(decoded, WaitAction):
            applied[idx] = {"applied": True, "kind": "wait"}
        elif isinstance(decoded, CancelAction):
            if _has_cancel_target(
                state,
                player_idx=idx,
                side=decoded.side,
                food_type=decoded.food_type,
            ):
                applied[idx] = {"applied": True, "kind": "cancel", "order_id": None}
            else:
                applied[idx] = {"applied": False, "code": "no_order_to_cancel"}
        elif isinstance(decoded, TakeAction):
            # Take is evaluated eagerly in the second pass — no pre-validation
            # here, we just mark it eligible.
            applied[idx] = {"applied": True, "kind": "take", "n_trades": 0}
        else:  # PostAction
            try:
                validate_order_replacement(
                    state,
                    player_idx=idx,
                    side=decoded.side,
                    food_type=decoded.food_type,
                    quantity=decoded.quantity,
                    price_per_unit=decoded.price_per_unit,
                )
            except OrderError as e:
                applied[idx] = {"applied": False, "code": e.code}
            else:
                applied[idx] = {
                    "applied": True,
                    "kind": "post",
                    "order_id": None,
                    "n_trades": 0,
                }

    posted_order_ids: dict[int, int] = {}
    for idx in player_order:
        if not applied[idx]["applied"]:
            continue
        decoded = decoded_by_idx[idx]
        if isinstance(decoded, WaitAction):
            continue
        if isinstance(decoded, CancelAction):
            cancelled = cancel_oldest_for_player_side_food(
                state,
                player_idx=idx,
                side=decoded.side,
                food_type=decoded.food_type,
            )
            if cancelled is None:
                applied[idx] = {"applied": False, "code": "no_order_to_cancel"}
            else:
                applied[idx]["order_id"] = cancelled.id
        elif isinstance(decoded, TakeAction):
            trades = take_from_book(
                state,
                player_idx=idx,
                side=decoded.side,
                food_type=decoded.food_type,
                quantity=decoded.quantity,
            )
            applied[idx]["n_trades"] = len(trades)
            applied[idx]["quantity_filled"] = sum(t.quantity for t in trades)
        else:  # PostAction
            order = post_order_without_matching(
                state,
                player_idx=idx,
                side=decoded.side,
                food_type=decoded.food_type,
                quantity=decoded.quantity,
                price_per_unit=decoded.price_per_unit,
                sequence=state.next_sequence(),
            )
            posted_order_ids[idx] = order.id
            applied[idx]["order_id"] = order.id

    trades = clear_books(state, food_order)
    for idx, order_id in posted_order_ids.items():
        applied[idx]["n_trades"] = sum(
            1
            for trade in trades
            if trade.maker_order_id == order_id or trade.taker_order_id == order_id
        )

    return applied
