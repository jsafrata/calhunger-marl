"""ScriptedV2 — inventory-skewed market maker.

Economic archetype: a commodity dealer that quotes continuously, earns the
bid-ask spread, and skews its prices to control inventory risk. Structurally
this is the Avellaneda-Stoikov shape — a reservation price displaced from fair
value by inventory deviation, with quotes posted symmetrically around it.

How it differs from ScriptedV1
------------------------------
V1 is reactive and post-only: it never emits `take`, never emits `cancel`, and
every order is exactly 1 unit at a price read off a 4-tier inventory ladder.
V2 quotes *two-sided*, sizes 1-5 units from near-term flow, and crosses the
spread with `take` whenever the resting book is better than its own reservation
price. It will buy and sell any food, including the ones it produces.

Two calibration choices worth stating explicitly
------------------------------------------------
1. **Targets are set on the remaining horizon, not the next two ticks.** Each
   player needs `2 x 18 = 36` units of each required food across a full game
   but opens with only 20-40 units of *each* food. A fixed short-horizon target
   (say 4 units) would read that opening position as ~10x overstocked and skew
   the bot into liquidating the very food it needs to survive. `horizon_need`
   starts at 36 and declines, so required food correctly reads as scarce for
   most of the game and the bot is a natural buyer.
2. **Price skews on the horizon deviation; size comes from near-term flow.**
   Sizing off the horizon deviation would make the bot order the maximum every
   single step from t=0. Skew answers "how badly do I want this over the rest of
   the game", quantity answers "how much do I need in the next two cycles" —
   conflating them produces a bot that is simultaneously desperate and broke.

Breadth is deliberately narrow: two-sided on the two required foods, ask-only
on the two produced foods. Six resting orders. With one action per env step and
only ~5-10 decisions per 10-step consumption cycle, quoting all four books
two-sided cannot even be *established* at medium cadence, let alone refreshed.

Unlike V1 these constants are calibration, not a port of anything — they are
expected to be tuned against self-play survival rather than held frozen.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.actions import ACTION_WAIT, encode_post, encode_take

from .base import Agent
from .scripted_common import (
    CONSUME_PER_CYCLE,
    FairValueTracker,
    RoundRobin,
    best_ask,
    best_bid,
    clamp_price,
    clamp_qty,
    cycle_progress,
    deficit_for_cycles,
    foods_by_urgency,
    horizon_need,
    largest_legal_qty,
    new_attempt,
    own_order,
    safe_ask_qty,
    starving_foods,
    survival_cash_estimate,
    unwind_ask_action,
)

# ---- calibration knobs ----------------------------------------------------
TARGET_BUFFER = 2           # units held above the bare horizon need
PRODUCED_WORKING_STOCK = 2  # produced food is for selling; keep only a little
SKEW_GAIN = 0.35            # max +/- fraction of fair value the skew can move
HALF_SPREAD = 1.0           # quote distance either side of the reservation price
COVER_CYCLES = 2            # near-term flow horizon used for sizing
LARGE_DEVIATION = 4         # deviation at which we size up to the full deficit
REQUOTE_PRICE_TOL = 2       # price drift that justifies spending an action
REQUOTE_QTY_TOL = 2         # quantity drift that justifies spending an action
URGENT_COVER_CYCLES = 2     # below this much cover we start paying up
URGENT_PROGRESS = 0.5       # ...but only past this point in the cycle
URGENT_TAKE_MULT = 1.6      # how far above fair value urgency will chase
PANIC_TAKE_MULT = 2.5       # price multiple used when starving and must post
MAX_CONSECUTIVE_CROSSES = 2  # crosses before a stale quote gets the next turn


class ScriptedV2Agent(Agent):
    """Two-sided liquidity provider with inventory-skewed reservation prices."""

    name = "scripted_v2"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._fv = FairValueTracker()
        self._rot = RoundRobin()
        self._consecutive_crosses = 0
        # Same telemetry contract as ScriptedV1 — `eval/game_runner.py` reads
        # `_inner.attempted_prices` off the cadence wrapper.
        self.attempted_prices: List[dict] = []

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._fv.reset()
        self._rot = RoundRobin()
        self._consecutive_crosses = 0
        self.attempted_prices = []

    # -----------------------------------------------------------------
    # Valuation
    # -----------------------------------------------------------------

    def _target(self, obs: Dict[str, Any], food: str) -> int:
        """Desired holding of `food`, on the remaining-game horizon."""
        if food in obs["required_foods"]:
            return horizon_need(obs, food) + TARGET_BUFFER
        return PRODUCED_WORKING_STOCK

    def _reservation_price(self, obs: Dict[str, Any], food: str) -> float:
        """Fair value displaced by normalised inventory deviation.

        Below target the reservation rises (keener buyer, more reluctant
        seller); above target it falls. Normalising by the target keeps the
        skew scale-free as the horizon shrinks — an absolute deviation would
        saturate at t=0, when every required food is ~13 units short of a
        36-unit horizon need.
        """
        fv = self._fv.value(obs, food)
        target = self._target(obs, food)
        inv = int(obs["available_inventory"][food])
        shortfall = (target - inv) / max(target, 1)
        shortfall = max(-1.0, min(1.0, shortfall))
        return fv * (1.0 + SKEW_GAIN * shortfall)

    def _quotes(self, obs: Dict[str, Any], food: str) -> Tuple[int, int]:
        """(bid price, ask price) straddling the reservation price."""
        r = self._reservation_price(obs, food)
        return clamp_price(r - HALF_SPREAD), clamp_price(r + HALF_SPREAD)

    # -----------------------------------------------------------------
    # Sizing
    # -----------------------------------------------------------------

    def _bid_qty(self, obs: Dict[str, Any], food: str) -> int:
        deficit = deficit_for_cycles(obs, food, COVER_CYCLES)
        if deficit >= LARGE_DEVIATION:
            return clamp_qty(deficit)
        if deficit >= 1:
            return 2
        return 1  # token quote: we are stocked, but a maker still shows a bid

    def _ask_qty(self, obs: Dict[str, Any], food: str) -> int:
        """Sell size, hard-capped by what cannot starve us.

        A resting ask reserves its inventory immediately, and the death check
        reads *available* inventory — so `safe_ask_qty` is what keeps a
        two-sided quote on a required food from being a slow suicide.
        """
        safe = safe_ask_qty(obs, food)
        if safe < 1:
            return 0
        surplus = int(obs["available_inventory"][food]) - self._target(obs, food)
        if surplus >= LARGE_DEVIATION:
            desired = clamp_qty(surplus)
        elif surplus >= 1:
            desired = 2
        else:
            desired = 1
        return min(safe, desired)

    # -----------------------------------------------------------------
    # Urgency
    # -----------------------------------------------------------------

    def _is_urgent(self, obs: Dict[str, Any], food: str) -> bool:
        """True once cover is thin and the cycle is far enough along.

        Deliberately fires *before* the bot is actually below the consumption
        amount: with one action per step it needs a head start to fix both
        required foods before the deadline.
        """
        if food not in obs["required_foods"]:
            return False
        cover = int(obs["available_inventory"][food])
        return (
            cover < URGENT_COVER_CYCLES * CONSUME_PER_CYCLE
            and cycle_progress(obs) >= URGENT_PROGRESS
        )

    def _buy_value(self, obs: Dict[str, Any], food: str) -> float:
        """Highest price we will pay right now by crossing the spread."""
        bid_price, _ = self._quotes(obs, food)
        if self._is_urgent(obs, food):
            return self._fv.value(obs, food) * URGENT_TAKE_MULT
        return float(bid_price)

    def _cash_guard_ok(self, obs: Dict[str, Any], food: str, cost: int) -> bool:
        """Don't spend the cash the *other* required food needs to survive.

        Skipped when the purchase is itself urgent — survival on this food
        outranks reserving for the next one.
        """
        if self._is_urgent(obs, food):
            return True
        others = [f for f in obs["required_foods"] if f != food]
        if not others:
            return True
        reserve = sum(
            deficit_for_cycles(obs, f, 1) * max(1, int(round(self._fv.value(obs, f))))
            for f in others
        )
        return int(obs["available_cash"]) - cost >= reserve

    # -----------------------------------------------------------------
    # Action layers, highest priority first
    # -----------------------------------------------------------------

    def _emergency_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """We die at the next consumption tick unless this food arrives now."""
        for food in starving_foods(obs):
            need = CONSUME_PER_CYCLE - int(obs["available_inventory"][food])
            action = largest_legal_qty(
                lambda q, f=food: encode_take("bid", f, q), legal_mask, max(1, need)
            )
            if action is not None:
                return action
            # Nothing resting to hit — leave a bid rich enough that anyone
            # holding the food will lift it before the deadline.
            fv = self._fv.value(obs, food)
            ask = best_ask(obs, food)
            price = clamp_price(max(fv * PANIC_TAKE_MULT, (ask + 2) if ask else fv * 2))
            attempt = new_attempt(
                obs, food=food, side="bid", price=price, qty=max(1, need)
            )
            self.attempted_prices.append(attempt)
            action = largest_legal_qty(
                lambda q, f=food, p=price: encode_post("bid", f, q, p),
                legal_mask,
                max(1, need),
            )
            if action is not None:
                attempt["submitted"] = True
                return action
            attempt["reason_not_submitted"] = "illegal_under_mask"
        return None

    def _cross_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Take resting liquidity that is better than our own reservation price."""
        candidates: List[Tuple[float, int]] = []

        for food in obs["required_foods"]:
            ask = best_ask(obs, food)
            if ask is None:
                continue
            value = self._buy_value(obs, food)
            if ask > value:
                continue
            qty = self._bid_qty(obs, food)
            if not self._cash_guard_ok(obs, food, ask * qty):
                continue
            action = largest_legal_qty(
                lambda q, f=food: encode_take("bid", f, q), legal_mask, qty
            )
            if action is not None:
                candidates.append((value - ask, action))

        for food in list(obs["produces"]) + list(obs["required_foods"]):
            bid = best_bid(obs, food)
            if bid is None:
                continue
            _, ask_price = self._quotes(obs, food)
            if bid < ask_price:
                continue
            qty = self._ask_qty(obs, food)
            if qty < 1:
                continue
            action = largest_legal_qty(
                lambda q, f=food: encode_take("ask", f, q), legal_mask, qty
            )
            if action is not None:
                candidates.append((bid - ask_price, action))

        if not candidates:
            return None
        # Best edge first; the action index breaks ties deterministically.
        candidates.sort(key=lambda c: (-c[0], c[1]))
        return candidates[0][1]

    def _quote_slots(self, obs: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """(rank, side, food) for the six slots we maintain, best rank first."""
        slots: List[Tuple[int, str, str]] = []
        for rank, food in enumerate(foods_by_urgency(obs)):
            slots.append((rank, "bid", food))
        for food in obs["produces"]:
            slots.append((10, "ask", food))
        for food in obs["required_foods"]:
            slots.append((20, "ask", food))
        return slots

    def _requote_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Refresh the single most stale quote.

        The staleness test is a filter, not a budget: with ~5-10 actions per
        cycle across six slots, something has to decide which one gets the
        step. Slots are ranked, and the round-robin only breaks ties *within*
        the most-stale group so a permanently-stale top slot cannot monopolise
        the bot.
        """
        # (staleness, rank, action, attempt-record)
        stale: List[Tuple[int, int, int, dict]] = []

        for rank, side, food in self._quote_slots(obs):
            bid_price, ask_price = self._quotes(obs, food)
            price = bid_price if side == "bid" else ask_price
            qty = self._bid_qty(obs, food) if side == "bid" else self._ask_qty(obs, food)
            if qty < 1:
                continue
            if side == "bid" and not self._cash_guard_ok(obs, food, price * qty):
                continue

            resting = own_order(obs, side, food)
            if resting is None:
                staleness = 0
            elif abs(int(resting["price"]) - price) >= REQUOTE_PRICE_TOL:
                staleness = 1
            elif abs(int(resting["remaining_qty"]) - qty) >= REQUOTE_QTY_TOL:
                staleness = 2
            else:
                continue

            attempt = new_attempt(obs, food=food, side=side, price=price, qty=qty)
            action = largest_legal_qty(
                lambda q, s=side, f=food, p=price: encode_post(s, f, q, p),
                legal_mask,
                qty,
            )
            if action is None:
                attempt["reason_not_submitted"] = "illegal_under_mask"
                self.attempted_prices.append(attempt)
                continue
            stale.append((staleness, rank, action, attempt))

        if not stale:
            return None
        stale.sort(key=lambda s: (s[0], s[1], s[2]))
        best_staleness = stale[0][0]
        top = [s for s in stale if s[0] == best_staleness]
        _, _, action, attempt = top[self._rot.next(len(top))]
        # Only the quote we actually emit counts as an attempt; the others were
        # merely considered and will be reconsidered next fire.
        attempt["submitted"] = True
        self.attempted_prices.append(attempt)
        return action

    # -----------------------------------------------------------------

    def act(self, obs: Dict[str, Any], legal_mask: np.ndarray) -> int:
        if obs["status"] != "alive":
            return ACTION_WAIT

        self._fv.observe(obs)

        # In self-play there is nearly always *some* edge on the book, so a
        # strict cross-before-quote order turns the market maker into a pure
        # taker: it never gets back to refreshing its own quotes. Yielding the
        # turn after a run of crosses keeps both sides of the archetype alive.
        maker_turn = self._consecutive_crosses >= MAX_CONSECUTIVE_CROSSES
        trading_layers = (
            (self._requote_action, self._cross_action)
            if maker_turn
            else (self._cross_action, self._requote_action)
        )

        # Releasing a stale ask outranks even the starvation layer: the food it
        # is holding hostage is food we already own, so freeing it is strictly
        # cheaper than buying a replacement.
        for layer in (unwind_ask_action, self._emergency_action) + trading_layers:
            action = layer(obs, legal_mask)
            if action is not None:
                if layer is self._cross_action:
                    self._consecutive_crosses += 1
                else:
                    self._consecutive_crosses = 0
                return action
        return ACTION_WAIT
