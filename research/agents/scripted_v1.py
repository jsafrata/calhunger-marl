"""ScriptedV1 — the live-game bot's decision logic, ported to Python.

This is a **frozen heuristic policy**. It mirrors `backend/src/game/bot.ts::
runBotActions` line-for-line: same survival tiers, same reference-price
lookup, same 1-unit orders, same 5-unit sell buffer. We do not tune or
"improve" it — this file exists so we can characterise the scripted bot
against itself and against future learned policies.

The one intentional difference from the TS bot is the **decision cadence**.
In the live game the bot runs on its own `setInterval`, so a "Fast" bot
can fire multiple times per game-second. Our env advances one game-second
per `step()` call, so the cadence wrapper (see `CadencedScripted` below)
gates the underlying decision logic per env step. Practically this means:

    Fast   : bot decides on ~every env step  (real game: ~1.33/sec)
    Medium : bot decides on ~half of steps    (real game: ~0.5/sec)
    Slow   : bot decides on ~one step in ~4-5 (real game: ~0.22/sec)

The ordinal relationship (Fast > Medium > Slow) is preserved; Fast is
mildly under-simulated (1 action/sec vs the game's 1.33). That's a known
caveat for the harness — noted in the eval report.

Multi-action-per-step behavior of the TS bot's `runBotActions`
------------------------------------------------------------
On each fire the TS bot iterates over ALL required foods and posts a bid
for each that hits the tier ladder, then iterates over ALL produced foods
and posts an ask for each with ≥5 surplus. That's up to 4 orders per fire
(2 required foods + 2 produced foods). Because our env accepts one action
per player per step, `act()` here **cannot** post all 4 orders in one
step. We rotate through the (side, food) slots across successive fires
using a small round-robin counter so that, over enough steps, the same
composite behaviour is achieved. Each single `act()` call still returns
exactly one action — the one the TS bot would have emitted first (or
next-in-rotation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.actions import (
    ACTION_WAIT,
    MAX_PRICE,
    MAX_QTY,
    encode_post,
)

from .base import Agent

# Tuning constants — taken 1:1 from bot.ts. Any change here changes the frozen
# heuristic; only touch them alongside a version bump (ScriptedV2, ...).
_SELL_MIN_SURPLUS = 5
_SELL_QTY = 1
_BUY_QTY = 1


def _best_ask(obs: Dict[str, Any], food: str) -> Optional[int]:
    asks = obs["order_books"][food]["asks"]
    return asks[0]["price"] if asks else None


def _best_bid(obs: Dict[str, Any], food: str) -> Optional[int]:
    bids = obs["order_books"][food]["bids"]
    return bids[0]["price"] if bids else None


def _last_trade(obs: Dict[str, Any], food: str) -> Optional[int]:
    return obs["order_books"][food]["last_trade_price"]


def _tiered_bid_price(
    inventory: int, ref_ask: Optional[int], last_trade_price: Optional[int]
) -> Optional[int]:
    """Return the bid price the bot would post for a required food at the
    given available inventory, or None if the bot should skip this food.

    Faithful port of bot.ts:
        ≤ 2   → panic:       max(bestAsk ∥ lastTrade ∥ 15, 12)
        ≤ 6   → concerned:   clamp(ref + 1, 6, 14) where ref = bestAsk ∥ lastTrade ∥ 8
        ≤ 20  → comfortable: clamp(ref, 3, 8)      where ref = bestAsk ∥ lastTrade ∥ 5
        > 20  → well-stocked: clamp(lastTrade - 1, 2, 5), and 50% coin-flip skip

    The 50% coin-flip in the well-stocked tier is delegated to the caller
    (we return the price and let the caller flip using their RNG).
    """
    if inventory <= 2:
        ref = ref_ask if ref_ask is not None else (
            last_trade_price if last_trade_price is not None else 15
        )
        return max(ref, 12)
    if inventory <= 6:
        ref = ref_ask if ref_ask is not None else (
            last_trade_price if last_trade_price is not None else 8
        )
        return max(6, min(ref + 1, 14))
    if inventory <= 20:
        ref = ref_ask if ref_ask is not None else (
            last_trade_price if last_trade_price is not None else 5
        )
        return max(3, min(ref, 8))
    # well-stocked
    last = last_trade_price if last_trade_price is not None else 4
    return max(2, min(last - 1, 5))


def _sell_price(
    ref_bid: Optional[int], last_trade_price: Optional[int], jitter: int
) -> int:
    """Ask-side price = max(3, ref + 1 + jitter{0,1})."""
    ref = ref_bid if ref_bid is not None else (
        last_trade_price if last_trade_price is not None else 5
    )
    return max(3, ref + 1 + jitter)


@dataclass
class _RotationState:
    """Round-robin index over (side, food) slots so multiple orders the TS
    bot would emit per fire get spread across successive env steps."""

    idx: int = 0

    def next(self, n: int) -> int:
        v = self.idx % max(1, n)
        self.idx += 1
        return v


class ScriptedV1Agent(Agent):
    """Frozen scripted policy — mirror of the live-game bot.

    The agent maintains a small RNG (for the well-stocked coin-flip and the
    sell-price jitter, both identical to the TS bot's `Math.random()` calls)
    plus a round-robin counter used only to pick which of the multiple
    orders the TS bot would emit on one fire we emit first on this call.
    """

    name = "scripted_v1"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._rot = _RotationState()
        # Public telemetry the harness collects at game end. One entry per
        # attempted post — the price the bot computed via its tier logic,
        # BEFORE any clamp / cap / affordability / legality check. Lets us
        # see the true unbounded distribution the heuristic wants to emit.
        self.attempted_prices: List[dict] = []

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._rot = _RotationState()
        self.attempted_prices = []

    # -----------------------------------------------------------------
    # Candidate generation — one entry per (side, food) the TS bot would
    # emit an order for on this fire. Order matches bot.ts: required-food
    # bids first, produced-food asks second.
    # -----------------------------------------------------------------

    def _candidate_actions(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> List[int]:
        candidates: List[int] = []
        avail_cash = obs["available_cash"]

        tick = int(obs.get("tick", 0))
        player_idx = int(obs.get("player_idx", -1))

        # ---- BIDS for required foods ----
        for food in obs["required_foods"]:
            inventory = obs["available_inventory"][food]
            price = _tiered_bid_price(
                inventory=inventory,
                ref_ask=_best_ask(obs, food),
                last_trade_price=_last_trade(obs, food),
            )
            if price is None:
                continue
            # Well-stocked coin-flip skip. Log NOTHING here — the bot
            # actively chose not to post; there was no price attempt.
            if inventory > 20 and self._rng.random() > 0.5:
                continue

            attempt = {
                "tick": tick,
                "player_idx": player_idx,
                "food": food,
                "side": "bid",
                "attempted_price": int(price),
                "cap_violation": price > MAX_PRICE,
                "submitted": False,
                "reason_not_submitted": None,
            }
            self.attempted_prices.append(attempt)

            if attempt["cap_violation"]:
                # Do NOT try to encode — encode_post would raise. Record
                # and move on. The bot's *intent* is preserved in the log.
                attempt["reason_not_submitted"] = "cap_violation"
                continue
            # Affordability check — must have ≥ price cash for 1 unit.
            if avail_cash < price or _BUY_QTY * price > avail_cash:
                attempt["reason_not_submitted"] = "insufficient_cash"
                continue
            action = encode_post("bid", food, qty=_BUY_QTY, price=price)
            if not legal_mask[action]:
                attempt["reason_not_submitted"] = "illegal_under_mask"
                continue
            attempt["submitted"] = True
            candidates.append(action)

        # ---- ASKS for own produced foods ----
        for own in obs["produces"]:
            own_avail = obs["available_inventory"][own]
            if own_avail < _SELL_MIN_SURPLUS:
                continue
            # TS bot: floor(Math.random() * 2) → 0 or 1
            jitter = int(self._rng.integers(0, 2))
            price = _sell_price(
                ref_bid=_best_bid(obs, own),
                last_trade_price=_last_trade(obs, own),
                jitter=jitter,
            )
            attempt = {
                "tick": tick,
                "player_idx": player_idx,
                "food": own,
                "side": "ask",
                "attempted_price": int(price),
                "cap_violation": price > MAX_PRICE,
                "submitted": False,
                "reason_not_submitted": None,
            }
            self.attempted_prices.append(attempt)

            if attempt["cap_violation"]:
                attempt["reason_not_submitted"] = "cap_violation"
                continue
            action = encode_post("ask", own, qty=_SELL_QTY, price=price)
            if not legal_mask[action]:
                attempt["reason_not_submitted"] = "illegal_under_mask"
                continue
            attempt["submitted"] = True
            candidates.append(action)

        return candidates

    # -----------------------------------------------------------------
    # act — return ONE action per env step. See module docstring for
    # rationale (env step is 1s; TS bot can emit up to 4 orders per fire).
    # -----------------------------------------------------------------

    def act(self, obs: Dict[str, Any], legal_mask: np.ndarray) -> int:
        if obs["status"] != "alive":
            return ACTION_WAIT

        candidates = self._candidate_actions(obs, legal_mask)
        if not candidates:
            return ACTION_WAIT

        # Round-robin across successive fires so all the bot's intended
        # orders eventually get emitted, rather than always the first one.
        pick = self._rot.next(len(candidates))
        return candidates[pick]
