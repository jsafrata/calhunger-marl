"""ScriptedV5 — opportunistic sniper / edge-only scalper.

Economic archetype: a patient bargain hunter — the trader who keeps no shop,
posts no standing quotes, and spends most of the session doing nothing at
all. It marks every food to a slow fair-value anchor and acts only when the
resting book is *mispriced* against that anchor: it lifts asks sitting below
value and hits bids standing above it. Money comes exclusively from the edge
captured on each snipe; there is no spread income and no carry.

Survival is handled by a deadline-triggered burst, not by standing inventory:
the bot runs lean (a couple of cycles of cover) and, when the next
consumption tick approaches with cover still short, it first lifts any
acceptably-priced ask and otherwise posts a deliberately rich bid — bait that
any stocked opponent will sell into — sized to exactly the shortfall.

How it differs from ScriptedV1 / V2 / V3 / V4
---------------------------------------------
Every existing bot is defined by the orders it *rests*: V1 reposts a reactive
1-unit ladder, V2 maintains six two-sided quotes, V3 works one patient order
per food down a concession schedule, V4 sits on standing cost-plus quotes
around a stockpile. V5 is the inverse: its default action is WAIT, its trades
are almost all `take`, and the only orders it ever posts are the survival
bait bid and an occasional cash-raising ask. Where V4 holds five cycles of
stock and V3 buys just in time on a *price* schedule, V5 runs lean and buys
just in time on an *edge* rule — it never concedes on price until the
deadline forces the escalation burst.

Format adaptability
-------------------
All scheduling is derived from the observation: cover targets are counted in
consumption cycles, the escalation trigger is counted in *realised decisions*
before the next tick (so cadence changes are absorbed automatically), and the
keep-level for required food clamps to `horizon_need`. Nothing reads the game
duration except through `remaining_seconds`.

Like V2/V3/V4 these constants are calibration, not a port — they are expected
to be tuned against self-play survival rather than held frozen.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from env.actions import ACTION_WAIT, encode_cancel, encode_post, encode_take

from .base import Agent
from .scripted_common import (
    CONSUME_PER_CYCLE,
    DecisionRateEstimator,
    FairValueTracker,
    best_ask,
    best_bid,
    clamp_price,
    clamp_qty,
    deficit_for_cycles,
    foods_by_urgency,
    horizon_need,
    largest_legal_qty,
    new_attempt,
    own_order,
    safe_ask_qty,
    starving_foods,
    steps_to_next_consumption,
    unwind_ask_action,
)

# ---- calibration knobs ----------------------------------------------------
COVER_CYCLES_TARGET = 2    # lean working cover, in consumption cycles
MAX_COVER_CYCLES = 4       # ceiling once cover is stretched for infrequency
COMFORTABLE_DECISIONS = 4  # decisions/cycle at which the lean target suffices
KEEP_CYCLES = 4            # required food retained before any is sold
WORKING_STOCK = 1          # produced units retained rather than sold
BUY_EDGE = 0.06            # lift asks priced <= FV * (1 - BUY_EDGE)
SELL_EDGE = 0.10           # hit bids priced >= FV * (1 + SELL_EDGE)
RESALE_MAX_QTY = 3         # bargain units bought purely to resell, per snipe
RESALE_CASH_CUSHION = 2.0  # war chests a resale buy must leave untouched
ESCALATE_SLACK = 2         # spare decisions kept when judging the deadline
ESCALATION_ACCEPT = 1.35   # ask multiple the deadline burst will still lift
ESCALATION_BID_MULT = 1.25 # bait-bid premium over fair value
CHEAP_ASK_MULT = 0.95      # discount used when raising cash for survival
RAISE_CASH_COVER_CYCLES = 2  # cycles of purchases the cash floor must fund
REPOST_PRICE_TOL = 1       # anchor drift a resting bait absorbs un-reposted
PANIC_MULT = 2.5           # price multiple when starving with an empty book


class ScriptedV5Agent(Agent):
    """Waits for mispricing; snipes it; posts only survival bait."""

    name = "scripted_v5"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._fv = FairValueTracker()
        self._rate = DecisionRateEstimator()
        # Same telemetry contract as ScriptedV1 — `eval/game_runner.py` reads
        # `_inner.attempted_prices` off the cadence wrapper.
        self.attempted_prices: List[dict] = []

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._fv.reset()
        self._rate.reset()
        self.attempted_prices = []

    # -----------------------------------------------------------------
    # Valuation
    # -----------------------------------------------------------------

    def _value(self, obs: Dict[str, Any], food: str) -> float:
        """Mark to *prints*, never to quotes.

        A sniper's whole premise is that resting quotes are suspect, so its
        anchor must not be seeded from them: in a one-sided book the mid is
        the very quote being judged, and every edge test against it becomes
        a tautology (a bid can never clear `bid >= mid * 1.1` when it *is*
        the mid). Until a food has traded we keep the fixed prior instead.
        """
        anchor = self._fv.trade_anchor(food)
        return anchor if anchor is not None else self._fv.fallback

    # -----------------------------------------------------------------
    # Cover accounting
    # -----------------------------------------------------------------

    def _target_cycles(self, obs: Dict[str, Any]) -> int:
        """Lean cover target, stretched when decisions are scarce.

        A sniper that only gets two turns per cycle cannot afford to run at
        two cycles of cover — its escalation window may hold a single turn —
        so the target rises as the realised decision rate falls. Same
        inventory-policy logic as V3's coverage stretch, applied to a much
        leaner base.
        """
        interval = int(obs["consumption_interval_seconds"])
        per_cycle = self._rate.decisions_within(interval)
        stretch = max(0, COMFORTABLE_DECISIONS - per_cycle)
        return min(MAX_COVER_CYCLES, COVER_CYCLES_TARGET + stretch)

    def _keep_level(self, obs: Dict[str, Any], food: str) -> int:
        """Units of `food` never offered for sale."""
        if food in obs["required_foods"]:
            return min(horizon_need(obs, food), KEEP_CYCLES * CONSUME_PER_CYCLE)
        return WORKING_STOCK

    def _sellable(self, obs: Dict[str, Any], food: str) -> int:
        return int(obs["available_inventory"][food]) - self._keep_level(obs, food)

    def _escalating(self, obs: Dict[str, Any]) -> bool:
        """True once patience would risk missing the consumption deadline.

        Counted in realised decisions, not env steps: the cadence wrapper may
        only let us act every few steps, and a sniper by construction has no
        resting orders working for it in the meantime.
        """
        outstanding = sum(
            1
            for food in obs["required_foods"]
            if deficit_for_cycles(obs, food, self._target_cycles(obs)) > 0
        )
        if outstanding == 0:
            return False
        left = self._rate.decisions_within(steps_to_next_consumption(obs))
        return left <= outstanding + ESCALATE_SLACK

    def _war_chest(self, obs: Dict[str, Any]) -> int:
        """Cash that must survive any discretionary purchase.

        One full cycle's consumption cost per required food — not merely the
        *current* deficit, which is 0 whenever the larder happens to be
        stocked and so guards nothing. Consumption never stops, and a sniper
        whose sales channel is illiquid (nobody bidding rich) cannot count on
        topping the purse back up before the next cycle bills it.
        """
        return sum(
            CONSUME_PER_CYCLE * max(1, int(round(self._value(obs, f))))
            for f in obs["required_foods"]
        )

    def _cash_ok(self, obs: Dict[str, Any], food: str, cost: int) -> bool:
        """Never let a snipe spend the cash the coming consumption ticks need."""
        reserve = sum(
            max(deficit_for_cycles(obs, f, 1), CONSUME_PER_CYCLE)
            * max(1, int(round(self._value(obs, f))))
            for f in obs["required_foods"]
            if f != food
        )
        return int(obs["available_cash"]) - cost >= reserve

    # -----------------------------------------------------------------
    # Action layers, highest priority first
    # -----------------------------------------------------------------

    def _emergency_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """We die at the next consumption tick unless this food arrives now.

        Crucially this layer *falls through* once its panic bid is resting:
        on an all-sniper table several players can be starving on different
        foods at once, and each of them is also the only available seller of
        what the others need. A bot that spends every remaining turn
        re-posting its own bid deadlocks the whole table; a bot that posts
        the bid once and then moves on gets to the sale layer below, sells
        into the *other* players' panic bids, and the market clears.
        """
        for food in starving_foods(obs):
            need = CONSUME_PER_CYCLE - int(obs["available_inventory"][food])
            action = largest_legal_qty(
                lambda q, f=food: encode_take("bid", f, q), legal_mask, max(1, need)
            )
            if action is not None:
                return action
            fv = self._value(obs, food)
            ask = best_ask(obs, food)
            price = clamp_price(max(fv * PANIC_MULT, (ask + 2) if ask else fv * 2))
            resting = own_order(obs, "bid", food)
            qty = max(1, need)
            if resting is not None:
                if int(resting["price"]) >= price:
                    continue  # bid already working at panic level — don't re-post
                # Replacing the resting bid must never shrink it.
                qty = clamp_qty(max(qty, int(resting["remaining_qty"])))
            attempt = new_attempt(
                obs, food=food, side="bid", price=price, qty=qty
            )
            self.attempted_prices.append(attempt)
            action = largest_legal_qty(
                lambda q, f=food, p=price: encode_post("bid", f, q, p),
                legal_mask,
                qty,
            )
            if action is not None:
                attempt["submitted"] = True
                return action
            attempt["reason_not_submitted"] = "illegal_under_mask"
            # The panic bid was unaffordable — usually because our own stale,
            # too-cheap bait on this very food has the cash locked in
            # `reserved_cash` while attracting no seller. Freeing it now is
            # what makes next turn's panic bid fundable.
            if resting is not None:
                cancel = encode_cancel("bid", food)
                if legal_mask[cancel]:
                    return cancel
        return None

    def _retire_bait_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Cancel a bait bid whose job is done.

        The bait rests at a deliberately rich price, so the moment cover is
        restored it is both a cash lock (`price * qty` in `reserved_cash`)
        and a standing invitation to sell us food we no longer need.
        """
        for order in obs["own_orders"]:
            if order["side"] != "bid":
                continue
            food = order["food"]
            if deficit_for_cycles(obs, food, self._target_cycles(obs)) > 0:
                continue
            action = encode_cancel("bid", food)
            if legal_mask[action]:
                return action
        return None

    def _escalation_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """The deadline burst: lift what is acceptable, else post the bait."""
        if not self._escalating(obs):
            return None
        for food in foods_by_urgency(obs):
            deficit = deficit_for_cycles(obs, food, self._target_cycles(obs))
            if deficit < 1:
                continue
            fv = self._value(obs, food)
            ask = best_ask(obs, food)
            if ask is not None and ask <= fv * ESCALATION_ACCEPT:
                action = largest_legal_qty(
                    lambda q, f=food: encode_take("bid", f, q),
                    legal_mask,
                    clamp_qty(deficit),
                )
                if action is not None:
                    return action
            price = clamp_price(fv * ESCALATION_BID_MULT)
            qty = clamp_qty(deficit)
            resting = own_order(obs, "bid", food)
            if resting is not None:
                if (
                    int(resting["price"]) >= price - REPOST_PRICE_TOL
                    and int(resting["remaining_qty"]) >= qty
                ):
                    continue  # bait already set; don't burn the turn re-posting
                # `post` replaces the resting order, so never replace a panic
                # bid with a cheaper or smaller one — that is a downgrade.
                price = max(price, int(resting["price"]))
                qty = clamp_qty(max(qty, int(resting["remaining_qty"])))
            attempt = new_attempt(obs, food=food, side="bid", price=price, qty=qty)
            self.attempted_prices.append(attempt)
            action = largest_legal_qty(
                lambda q, f=food, p=price: encode_post("bid", f, q, p),
                legal_mask,
                qty,
            )
            if action is None:
                attempt["reason_not_submitted"] = "illegal_under_mask"
                continue
            attempt["submitted"] = True
            return action
        return None

    def _bargain_buy_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Lift any ask resting below fair value — the buying half of the edge.

        Required foods are restocked up to the keep-level whenever they are
        cheap (a bargain brought forward is a bargain); other foods are bought
        purely for resale, in small size, and only with the survival budget
        untouched.
        """
        for food in list(obs["required_foods"]) + list(obs["produces"]):
            fv = self._value(obs, food)
            ask = best_ask(obs, food)
            if ask is None or ask > fv * (1 - BUY_EDGE):
                continue
            if food in obs["required_foods"]:
                desired = self._keep_level(obs, food) - int(
                    obs["available_inventory"][food]
                )
            else:
                desired = RESALE_MAX_QTY
            if desired < 1:
                continue
            qty = clamp_qty(desired)
            if not self._cash_ok(obs, food, ask * qty):
                continue
            if food not in obs["required_foods"]:
                # Resale is pure speculation — it needs a fat purse, not just
                # a clear one. (This also stops the bot "bargain-buying" its
                # own produce back while its sale asks sit unfilled.)
                cushion = RESALE_CASH_CUSHION * self._war_chest(obs)
                if int(obs["available_cash"]) - ask * qty < cushion:
                    continue
            action = largest_legal_qty(
                lambda q, f=food: encode_take("bid", f, q), legal_mask, qty
            )
            if action is not None:
                return action
        return None

    def _rich_bid_sale_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Hit any bid resting above fair value — the selling half of the edge.

        In self-play these bids are mostly other snipers' bait, which is the
        entire circulation mechanism of an all-sniper table: the hungry post
        rich, the stocked sell into it, and the market clears at a premium.
        """
        for food in list(obs["produces"]) + list(obs["required_foods"]):
            sellable = self._sellable(obs, food)
            if sellable < 1:
                continue
            bid = best_bid(obs, food)
            if bid is None or bid < self._value(obs, food) * (1 + SELL_EDGE):
                continue
            action = largest_legal_qty(
                lambda q, f=food: encode_take("ask", f, q),
                legal_mask,
                clamp_qty(sellable),
            )
            if action is not None:
                return action
        return None

    def _raise_cash_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Post a discounted ask when the survival budget is short of cash.

        The one non-bait order this bot ever rests. Priced *below* fair value
        so it is the first thing another bargain hunter lifts — speed of fill
        matters more than the last tick of margin when the shortfall is
        funding the next consumption tick.

        The trigger looks two cycles ahead, not one: against opponents who
        never bid rich (so the snipe-sale layer never fires), production
        piles up unsold while escalation buys bleed cash, and a bot that only
        starts raising cash inside the final cycle posts its first ask with
        no time left for anyone to lift it.
        """
        needed = sum(
            deficit_for_cycles(obs, f, RAISE_CASH_COVER_CYCLES)
            * max(1, int(round(self._value(obs, f))))
            for f in obs["required_foods"]
        )
        if int(obs["available_cash"]) >= needed:
            return None
        for food in obs["produces"]:
            qty = min(clamp_qty(self._sellable(obs, food)), safe_ask_qty(obs, food))
            if qty < 1:
                continue
            price = clamp_price(self._value(obs, food) * CHEAP_ASK_MULT)
            resting = own_order(obs, "ask", food)
            if resting is not None and int(resting["price"]) <= price:
                continue
            attempt = new_attempt(obs, food=food, side="ask", price=price, qty=qty)
            self.attempted_prices.append(attempt)
            action = largest_legal_qty(
                lambda q, f=food, p=price: encode_post("ask", f, q, p),
                legal_mask,
                qty,
            )
            if action is None:
                attempt["reason_not_submitted"] = "illegal_under_mask"
                continue
            attempt["submitted"] = True
            return action
        return None

    # -----------------------------------------------------------------

    def act(self, obs: Dict[str, Any], legal_mask: np.ndarray) -> int:
        if obs["status"] != "alive":
            return ACTION_WAIT

        self._fv.observe(obs)
        self._rate.observe(obs)

        for layer in (
            unwind_ask_action,
            self._emergency_action,
            self._retire_bait_action,
            self._escalation_action,
            self._bargain_buy_action,
            self._rich_bid_sale_action,
            self._raise_cash_action,
        ):
            action = layer(obs, legal_mask)
            if action is not None:
                return action
        return ACTION_WAIT
