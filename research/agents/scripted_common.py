"""Shared machinery for the calibrated scripted baselines (ScriptedV2 / V3).

`ScriptedV1` deliberately shares nothing with this module: it is a line-for-line
port of the live game's `bot.ts` and must stay frozen. V2 and V3 have no
external referent, so the mechanics they *agree* on — what fair value means,
how far away the next consumption tick is, how much stock is safe to offer —
live here. That keeps the two bots from drifting apart on plumbing so their
only differences are strategic.

Three env facts drive almost everything in this module; all three are easy to
get wrong from the spec alone:

1. **Death reads *available* inventory.** `tick.consume_and_kill` kills on
   `inventory - reserved_inventory < consume_amount`. Posting an ask reserves
   the food immediately (`market._reserve_for_order`), so a *resting, unfilled*
   ask on a required food starves you exactly like a completed sale would.
   Every sell decision therefore goes through `safe_ask_qty`.
2. **`take` reserves nothing.** `market.take_from_book` is immediate-or-cancel
   and never locks cash or inventory. Crossing the spread is strictly safer for
   survival than posting, which is why both bots escalate to `take` under
   urgency rather than to an aggressive `post`.
3. **Production and consumption both fire only every `interval` seconds**, and
   production lands immediately *before* consumption on the same tick
   (`tick.step_tick`). So the cycle boundary is the only moment inventory moves
   for non-market reasons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from env.actions import MAX_PRICE, MAX_QTY, encode_cancel
from env.config import DEFAULT_CONFIG

# Units of each required food consumed per consumption cycle. Read from the
# shared config the same way `GreedyAgent` does, since the observation exposes
# only the derived `starvation_seconds` and not the rate itself.
CONSUME_PER_CYCLE: int = DEFAULT_CONFIG.consumption_per_required_food_per_second

# Cycles of cover a resting ask on a required food must leave behind. See
# `resting_ask_floor` for why one cycle is not enough.
ASK_COVER_CYCLES: int = 2

# Price used before any trade has printed and the book is empty. Matches the
# order of magnitude of `bot.ts`'s own no-history fallbacks.
FALLBACK_VALUE: float = 5.0


# ---------------------------------------------------------------------------
# Cycle / deadline arithmetic
# ---------------------------------------------------------------------------


def steps_to_next_consumption(obs: Dict[str, Any]) -> int:
    """Env steps remaining until the next consumption tick, inclusive of now.

    `step_tick` increments the clock and *then* consumes when
    `elapsed_seconds % interval == 0`, and the agent acts before the clock
    advances. So at `tick == t` the agent has exactly `interval - (t % interval)`
    more action opportunities before that consumption resolves.
    """
    interval = int(obs["consumption_interval_seconds"])
    return interval - (int(obs["tick"]) % interval)


def cycles_left(obs: Dict[str, Any]) -> int:
    """Consumption ticks still to come, including the one at the final second.

    The game ends at `game_duration_seconds`, but `step_tick` runs that tick's
    consumption *before* the end-of-game check — so the last consumption does
    happen and can still kill you.
    """
    interval = int(obs["consumption_interval_seconds"])
    return max(0, math.ceil(int(obs["remaining_seconds"]) / interval))


def cycle_progress(obs: Dict[str, Any]) -> float:
    """Fraction of the current consumption cycle already elapsed, in [0, 1)."""
    interval = int(obs["consumption_interval_seconds"])
    return (interval - steps_to_next_consumption(obs)) / interval


# ---------------------------------------------------------------------------
# Survival accounting
# ---------------------------------------------------------------------------


def survival_floor(obs: Dict[str, Any], food: str) -> int:
    """Units of `food` that must remain *available* to survive the next tick."""
    return CONSUME_PER_CYCLE if food in obs["required_foods"] else 0


def resting_ask_floor(obs: Dict[str, Any], food: str) -> int:
    """Floor to apply when committing inventory to a *resting* ask.

    Strictly larger than `survival_floor`, and the difference is a trap worth
    spelling out: `consume_and_kill` decrements **total** inventory but leaves
    reservations untouched. An ask sized against the immediate floor is safe
    when posted and lethal one tick later —

        hold 4, reserve 2  -> available 2, survives, inventory drops to 2
        hold 2, reserve 2  -> available 0, dies

    — because the unfilled order carried its reservation across the boundary.
    Requiring cover for two cycles means a resting ask can survive a
    consumption tick without eating into the next one's food. Orders that go
    unfilled for longer than that are unwound by `unwind_ask_action`.
    """
    return ASK_COVER_CYCLES * CONSUME_PER_CYCLE if food in obs["required_foods"] else 0


def safe_ask_qty(obs: Dict[str, Any], food: str) -> int:
    """Largest ask quantity that cannot starve us.

    `available_inventory` already nets out inventory reserved by our resting
    asks, so this is measured against what is still free. We deliberately do
    *not* credit the refund a same-slot replacement would produce
    (`market._validate_post`'s `reusable`): `legal_action_mask` doesn't credit
    it either, so counting it would just produce actions the mask rejects.
    """
    free = int(obs["available_inventory"][food]) - resting_ask_floor(obs, food)
    return max(0, min(MAX_QTY, free))


def unsafe_ask_food(obs: Dict[str, Any]) -> Optional[str]:
    """A required food whose resting ask has decayed into a starvation risk.

    Sizing an ask safely at post time is not enough: consumption keeps eating
    the total while the reservation stays put, so a quote that goes unfilled
    across cycles eventually corners its own owner. This spots that case so the
    bot can release the order.
    """
    for food in obs["required_foods"]:
        reserved = int(obs["reserved_inventory"][food])
        if reserved <= 0:
            continue
        if int(obs["inventory"][food]) - reserved < resting_ask_floor(obs, food):
            return food
    return None


def unwind_ask_action(obs: Dict[str, Any], legal_mask) -> Optional[int]:
    """Cancel a resting ask that has become unsafe, if there is one.

    Cancelling rather than re-posting smaller: it refunds the whole reservation
    in a single action, and at slow cadence the bot cannot count on getting a
    second turn before the deadline.
    """
    food = unsafe_ask_food(obs)
    if food is None:
        return None
    action = encode_cancel("ask", food)
    return action if legal_mask[action] else None


def deficit_for_cycles(obs: Dict[str, Any], food: str, n_cycles: int) -> int:
    """Units of `food` still needed to cover the next `n_cycles` consumptions.

    `n_cycles` is clamped to the cycles the game actually has left. Without the
    clamp a bot planning three cycles ahead in a game with one cycle remaining
    buys food it can never eat — harmless-looking, but every unit is paid for in
    cash *and* locks `price * qty` in `reserved_cash` while the bid rests, which
    is exactly the budget the last consumption tick needs. The 30-second variant
    makes this acute (the whole game is three cycles), but it also bites the
    final cycles of the 180-second game.
    """
    if food not in obs["required_foods"]:
        return 0
    need = min(int(n_cycles), cycles_left(obs)) * CONSUME_PER_CYCLE
    return max(0, need - int(obs["available_inventory"][food]))


def horizon_need(obs: Dict[str, Any], food: str) -> int:
    """Units of `food` needed to survive every remaining consumption tick.

    This is the quantity that makes required food structurally scarce: players
    open with 20-40 units of each food but need `2 x 18 = 36` of each required
    food across a full game.
    """
    if food not in obs["required_foods"]:
        return 0
    return cycles_left(obs) * CONSUME_PER_CYCLE


def is_starving(obs: Dict[str, Any], food: str) -> bool:
    """True if the next consumption tick kills us on this food as things stand."""
    return food in obs["required_foods"] and (
        int(obs["available_inventory"][food]) < CONSUME_PER_CYCLE
    )


def starving_foods(obs: Dict[str, Any]) -> List[str]:
    """Required foods that would kill us at the next tick, most urgent first."""
    short = [f for f in obs["required_foods"] if is_starving(obs, f)]
    return sorted(short, key=lambda f: int(obs["available_inventory"][f]))


def foods_by_urgency(obs: Dict[str, Any]) -> List[str]:
    """Required foods ordered by projected post-consumption inventory, lowest first."""
    return sorted(
        obs["required_foods"],
        key=lambda f: int(obs["available_inventory"][f]) - CONSUME_PER_CYCLE,
    )


def survival_cash_estimate(obs: Dict[str, Any], unit_price: float) -> int:
    """Cash needed to buy the next cycle's shortfall across both required foods."""
    total = 0
    for food in obs["required_foods"]:
        total += deficit_for_cycles(obs, food, 1) * max(1, int(round(unit_price)))
    return total


# ---------------------------------------------------------------------------
# Book reads (own orders excluded — you cannot trade with yourself)
# ---------------------------------------------------------------------------
#
# `market._match_incoming` and `market.take_from_book` both skip resting orders
# belonging to the acting player, so a "best ask" that is our own order is not
# takeable. V1 reads the aggregated levels and so can be fooled by its own
# quote; V2/V3 read the per-order view and filter.


def best_ask(obs: Dict[str, Any], food: str) -> Optional[int]:
    """Lowest ask price posted by someone else, or None."""
    me = obs["player_idx"]
    prices = [
        o["price"]
        for o in obs["order_books"][food]["ask_orders"]
        if o["player_idx"] != me
    ]
    return min(prices) if prices else None


def best_bid(obs: Dict[str, Any], food: str) -> Optional[int]:
    """Highest bid price posted by someone else, or None."""
    me = obs["player_idx"]
    prices = [
        o["price"]
        for o in obs["order_books"][food]["bid_orders"]
        if o["player_idx"] != me
    ]
    return max(prices) if prices else None


def book_mid(obs: Dict[str, Any], food: str) -> Optional[float]:
    bid = best_bid(obs, food)
    ask = best_ask(obs, food)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return None


def own_order(obs: Dict[str, Any], side: str, food: str) -> Optional[Dict[str, Any]]:
    """Our resting order on this (side, food), if any.

    There is at most one: `market.post_order` replaces any prior order on the
    same (player, food, side).
    """
    for o in obs["own_orders"]:
        if o["side"] == side and o["food"] == food:
            return o
    return None


# ---------------------------------------------------------------------------
# Fair value
# ---------------------------------------------------------------------------


class FairValueTracker:
    """Slow-moving per-food value estimate.

    An EMA over executed trades is the anchor; the book midpoint only *seeds*
    a food that has never traded. Deliberately not blended with live quotes on
    every call — both specs require an estimate that "changes gradually rather
    than following every individual quote", and a mid-blend would inherit the
    full volatility of a two-order book.
    """

    def __init__(self, alpha: float = 0.3, fallback: float = FALLBACK_VALUE) -> None:
        self.alpha = alpha
        self.fallback = fallback
        self._ema: Dict[str, float] = {}
        self._last_tick_seen: int = -1

    def reset(self) -> None:
        self._ema = {}
        self._last_tick_seen = -1

    def observe(self, obs: Dict[str, Any]) -> None:
        """Fold every trade we haven't already folded into the EMA.

        `recent_trades` is the last 20 trades game-wide, so we de-duplicate by
        tick: trades at or before `_last_tick_seen` have already been counted.
        """
        newest = self._last_tick_seen
        for trade in obs["recent_trades"]:
            tick = int(trade["tick"])
            if tick <= self._last_tick_seen:
                continue
            food = trade["food"]
            price = float(trade["price"])
            if food in self._ema:
                self._ema[food] = self.alpha * price + (1 - self.alpha) * self._ema[food]
            else:
                self._ema[food] = price
            newest = max(newest, tick)
        self._last_tick_seen = newest

    def value(self, obs: Dict[str, Any], food: str) -> float:
        if food in self._ema:
            return self._ema[food]
        mid = book_mid(obs, food)
        if mid is not None:
            return mid
        return self.fallback

    def trade_anchor(self, food: str) -> Optional[float]:
        """EMA over executed trades only — None until `food` has printed.

        `value` seeds an untraded food from the book midpoint, which is right
        for a bot that quotes (it must price *relative to the book*), and
        wrong for a bot that judges the book (in a one-sided book the "mid"
        IS the quote under judgement, so every edge test against it becomes
        a tautology). Take-driven bots use this and fall back on their own
        prior instead.
        """
        return self._ema.get(food)


# ---------------------------------------------------------------------------
# Action-space guards
# ---------------------------------------------------------------------------


def clamp_price(price: float) -> int:
    """Bound a computed price into the encodable range.

    `encode_post` raises outside `1..MAX_PRICE`, so this must be applied before
    encoding. V1 instead logs a `cap_violation` and skips the order; V2/V3 clamp
    and still trade, because for them an unencodable price is a calibration
    artifact rather than a faithful reproduction of the TS bot's intent.
    """
    return int(max(1, min(MAX_PRICE, round(price))))


def clamp_qty(qty: float) -> int:
    return int(max(1, min(MAX_QTY, round(qty))))


def largest_legal_qty(
    encoder: Callable[[int], int], legal_mask, desired: int
) -> Optional[int]:
    """Largest quantity <= `desired` whose encoded action is legal.

    Both bots size orders from their own accounting (deficits, safe surplus),
    but the mask applies its own affordability rules — e.g. a bid needs
    `qty * price <= available_cash`. Walking down finds the biggest order that
    actually clears rather than silently dropping the intent.
    """
    for qty in range(int(desired), 0, -1):
        action = encoder(qty)
        if legal_mask[action]:
            return action
    return None


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
#
# `eval/game_runner.py` reaches through `CadencedScripted._inner` and reads
# `attempted_prices`, so V2/V3 emit the same record shape V1 does. `submitted`
# and `reason_not_submitted` are filled in by the caller after the legality
# check.


def new_attempt(
    obs: Dict[str, Any], *, food: str, side: str, price: int, qty: int
) -> Dict[str, Any]:
    return {
        "tick": int(obs.get("tick", 0)),
        "player_idx": int(obs.get("player_idx", -1)),
        "food": food,
        "side": side,
        "attempted_price": int(price),
        "attempted_qty": int(qty),
        "cap_violation": False,
        "submitted": False,
        "reason_not_submitted": None,
    }


class DecisionRateEstimator:
    """How many more times will this bot get to act before a deadline?

    `CadencedScripted` gates `act()`, so the inner bot is called on only a
    fraction of env steps — every step at "fast", roughly one in four at
    "slow". Any schedule reasoned in *env steps* therefore overstates the
    opportunities remaining, by up to 5x, and a deadline-driven bot will
    discover it has run out of turns while still believing it has time.

    Rather than plumb the cadence into the inner bot (which would couple the
    strategy to its wrapper), we measure the realised gap between calls.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._last_tick: Optional[int] = None
        self._gap: float = 1.0
        self._samples: int = 0

    def reset(self) -> None:
        self._last_tick = None
        self._gap = 1.0
        self._samples = 0

    def observe(self, obs: Dict[str, Any]) -> None:
        tick = int(obs["tick"])
        if self._last_tick is not None and tick > self._last_tick:
            gap = float(tick - self._last_tick)
            self._gap = gap if self._samples == 0 else (
                self.alpha * gap + (1 - self.alpha) * self._gap
            )
            self._samples += 1
        self._last_tick = tick

    @property
    def gap(self) -> float:
        """Mean env steps between consecutive decisions (>= 1)."""
        return max(1.0, self._gap)

    def decisions_within(self, steps: int) -> int:
        """Decisions we expect to get within the next `steps` env steps."""
        return int(steps / self.gap)


@dataclass
class RoundRobin:
    """Tie-break counter, so equally-ranked slots don't starve each other.

    Same role as `ScriptedV1`'s `_RotationState`: with one action per env step,
    a strict priority order would let the top-ranked slot monopolise the bot.
    """

    idx: int = 0

    def next(self, n: int) -> int:
        if n <= 0:
            return 0
        v = self.idx % n
        self.idx += 1
        return v
