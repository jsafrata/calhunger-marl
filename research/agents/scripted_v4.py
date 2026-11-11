"""ScriptedV4 — precautionary stockpiler / storage arbitrageur.

Economic archetype: a grain elevator — a storage merchant in a seasonal
commodity market. It accumulates stock early while goods are plentiful,
carries a deep safety stock through the middle of the game, and liquidates
the pile into late-game scarcity when everyone else is running lean. Money is
made on *carry* — the spread between early acquisition prices and late
scarcity prices — plus a steady cost-plus markup on own production.

How it differs from ScriptedV1 / V2 / V3
----------------------------------------
V1 is a reactive 1-unit post-only ladder. V2 earns the bid-ask spread and
holds inventory *near* a target with symmetric two-sided quotes; it is
agnostic about when in the game it holds stock. V3 is the opposite pole of
this bot: just-in-time, holding as little as possible, conceding on a
within-cycle schedule. V4 is defined by its *intertemporal* position:

- it front-loads buying into a safety stock several cycles deep — far above
  V3's JIT band — and pays up whenever cover drops below its low-water mark;
- it never quotes two-sided on the same book (it is a buyer of required foods
  and a seller of surplus, full stop), so it earns no spread;
- its selling premium is a function of *game* progress (scarcity), not of
  cycle progress (V3's deadline) or of the book (V1/V2).

Format adaptability
-------------------
The stock target self-clamps to the remaining horizon (`horizon_need`), so
the same code is a net hoarder in the opening, a holder in the middle, and a
net liquidator in the endgame — in a 30-second game just as in a 180-second
one. Nothing here reads the game duration except through the observation.

Like V2/V3 these constants are calibration, not a port — they are expected to
be tuned against self-play survival rather than held frozen.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.actions import ACTION_WAIT, encode_cancel, encode_post, encode_take

from .base import Agent
from .scripted_common import (
    CONSUME_PER_CYCLE,
    FairValueTracker,
    RoundRobin,
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
    survival_cash_estimate,
    unwind_ask_action,
)

# ---- calibration knobs ----------------------------------------------------
STOCK_CYCLES = 5           # safety-stock depth, in consumption cycles
STOCK_BUFFER = 2           # units on top of the cycle-derived pile
LOW_WATER_CYCLES = 2       # cover below this = pay up to restock immediately
# Produced food is merchandise, and the producer never eats it: production
# lands every cycle regardless, so holding any of it back starves the very
# customers whose bids fund the stockpile. Sell it all.
PRODUCED_WORKING_STOCK = 0
URGENT_ACCEPT = 1.5        # take asks up to FV*this when below low water
BUILD_ACCEPT = 1.2         # take asks up to FV*this while building the pile
URGENT_BID_MULT = 1.1      # resting bid price when below low water
BUILD_BID_MULT = 0.9       # patient standing bid while merely building
BASE_MARKUP = 0.10         # ask premium over fair value at t=0
LATE_MARKUP = 0.35         # extra premium as scarcity peaks at game end
RICH_BID_EDGE = 1.15       # sell surplus into any bid >= FV*this
LIQUIDATE_PROGRESS = 0.75  # game progress past which any at-value bid is hit
REQUOTE_PRICE_TOL = 2      # price drift that justifies spending an action
REQUOTE_QTY_TOL = 2        # quantity drift that justifies spending an action
PANIC_MULT = 2.5           # price multiple used when starving and must post


class ScriptedV4Agent(Agent):
    """Deep safety stock early, cost-plus liquidation into late scarcity."""

    name = "scripted_v4"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._fv = FairValueTracker()
        self._rot = RoundRobin()
        # Same telemetry contract as ScriptedV1 — `eval/game_runner.py` reads
        # `_inner.attempted_prices` off the cadence wrapper.
        self.attempted_prices: List[dict] = []

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._fv.reset()
        self._rot = RoundRobin()
        self.attempted_prices = []

    # -----------------------------------------------------------------
    # Stock accounting
    # -----------------------------------------------------------------

    def _target_stock(self, obs: Dict[str, Any], food: str) -> int:
        """The pile this bot wants to sit on.

        Clamped by `horizon_need` so the elevator never stores food the game
        has no time to consume — which is also what turns it into a seller at
        the end: as the horizon shrinks below the pile, the difference becomes
        surplus and goes on offer.
        """
        if food in obs["required_foods"]:
            cap = STOCK_CYCLES * CONSUME_PER_CYCLE + STOCK_BUFFER
            return min(horizon_need(obs, food), cap)
        return PRODUCED_WORKING_STOCK

    def _cover_cycles(self, obs: Dict[str, Any], food: str) -> int:
        return int(obs["available_inventory"][food]) // CONSUME_PER_CYCLE

    def _sale_floor(self, obs: Dict[str, Any], food: str) -> int:
        """Holdings below this are never offered — the other half of carry.

        Deliberately *not* the buy target: a required food is only surplus
        above the full remaining-horizon need. Flooring sales at the buy
        target instead would make the bot dump a large endowment in the
        opening (when prices are at their cheapest) and re-buy the same units
        later at scarcity prices — the exact opposite of the storage trade.
        The floor tracks `horizon_need` down over time, which is what turns
        the held pile into sellable surplus on schedule at the end.
        """
        if food in obs["required_foods"]:
            return horizon_need(obs, food) + STOCK_BUFFER
        return PRODUCED_WORKING_STOCK

    def _surplus(self, obs: Dict[str, Any], food: str) -> int:
        return int(obs["available_inventory"][food]) - self._sale_floor(obs, food)

    def _game_progress(self, obs: Dict[str, Any]) -> float:
        total = int(obs["tick"]) + int(obs["remaining_seconds"])
        return int(obs["tick"]) / total if total > 0 else 1.0

    # -----------------------------------------------------------------
    # Pricing
    # -----------------------------------------------------------------

    def _ask_price(self, obs: Dict[str, Any], food: str) -> int:
        """Cost-plus markup that rises with scarcity, i.e. with game progress.

        This is the carry trade: the same unit is offered at ~FV*1.1 in the
        opening and ~FV*1.45 at the end, when JIT players are short and the
        horizon leaves them no room to wait the quote out.

        Capped at the best competing ask (as long as that ask is itself at or
        above fair value): a merchant who always quotes formula-above-FV into
        a market that then trades at his price ratchets the fair value EMA —
        and therefore his own next quote — upward forever. Matching the
        standing level breaks the spiral without ever undercutting below
        value.
        """
        fv = self._fv.value(obs, food)
        price = fv * (1 + BASE_MARKUP + LATE_MARKUP * self._game_progress(obs))
        other = best_ask(obs, food)
        if other is not None and fv <= other < price:
            price = float(other)
        return clamp_price(price)

    def _bid_price(self, obs: Dict[str, Any], food: str) -> int:
        fv = self._fv.value(obs, food)
        mult = (
            URGENT_BID_MULT
            if self._cover_cycles(obs, food) < LOW_WATER_CYCLES
            else BUILD_BID_MULT
        )
        return clamp_price(fv * mult)

    def _cash_guard_ok(self, obs: Dict[str, Any], food: str, cost: int) -> bool:
        """Don't spend the cash the *other* required food needs to survive.

        Skipped when this food is itself below low water — restocking it now
        outranks reserving for the next one.
        """
        if self._cover_cycles(obs, food) < LOW_WATER_CYCLES:
            return True
        reserve = sum(
            deficit_for_cycles(obs, f, 1) * max(1, int(round(self._fv.value(obs, f))))
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

        A stockpiler that reaches this layer has already failed at its job;
        it exists as the same backstop V2/V3 carry. Once the panic bid is
        resting the layer falls through rather than re-posting every turn —
        on an all-V4 table several players can starve at once, and each is
        the only seller of what the others need, so a bot locked into its
        own bid would deadlock the table.
        """
        for food in starving_foods(obs):
            need = CONSUME_PER_CYCLE - int(obs["available_inventory"][food])
            action = largest_legal_qty(
                lambda q, f=food: encode_take("bid", f, q), legal_mask, max(1, need)
            )
            if action is not None:
                return action
            fv = self._fv.value(obs, food)
            ask = best_ask(obs, food)
            price = clamp_price(max(fv * PANIC_MULT, (ask + 2) if ask else fv * 2))
            resting = own_order(obs, "bid", food)
            qty = max(1, need)
            if resting is not None:
                if int(resting["price"]) >= price:
                    continue  # bid already working at panic level — don't re-post
                # `post` replaces the resting order — never shrink it.
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
            # too-cheap bid on this very food has the cash locked in
            # `reserved_cash` while attracting no seller. Freeing it now is
            # what makes next turn's panic bid fundable.
            if resting is not None:
                cancel = encode_cancel("bid", food)
                if legal_mask[cancel]:
                    return cancel
        return None

    def _restock_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Cover has broken the low-water mark: pay up and refill *now*.

        Preference order is take-then-post: `take` reserves nothing and lands
        immediately, so it is strictly safer than resting a bid this close to
        the floor.
        """
        for food in foods_by_urgency(obs):
            if self._cover_cycles(obs, food) >= LOW_WATER_CYCLES:
                continue
            desired = self._target_stock(obs, food) - int(
                obs["available_inventory"][food]
            )
            if desired < 1:
                continue
            ask = best_ask(obs, food)
            fv = self._fv.value(obs, food)
            if ask is not None and ask <= fv * URGENT_ACCEPT:
                action = largest_legal_qty(
                    lambda q, f=food: encode_take("bid", f, q),
                    legal_mask,
                    clamp_qty(desired),
                )
                if action is not None:
                    return action
            # Nothing takeable at an acceptable price — work a rich bid
            # instead, but only if the one already resting has drifted.
            price = self._bid_price(obs, food)
            action = self._maintain_post(
                obs, legal_mask, side="bid", food=food,
                price=price, qty=clamp_qty(desired), no_downgrade=True,
            )
            if action is not None:
                return action
        return None

    def _rich_bid_sale_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Someone is bidding above our scarcity ask — hit them immediately.

        This is where the carry profit is realised: late-game panic bids from
        under-stocked players land well above fair value, and the pile exists
        precisely to be sold into them. In the endgame the acceptance bar
        drops to fair value itself — the pile is worthless once the game
        ends, so any at-or-above-value bid is better than carrying the food
        over the final whistle.
        """
        edge = RICH_BID_EDGE if self._game_progress(obs) < LIQUIDATE_PROGRESS else 1.0
        for food in list(obs["produces"]) + list(obs["required_foods"]):
            surplus = self._surplus(obs, food)
            if surplus < 1:
                continue
            bid = best_bid(obs, food)
            if bid is None or bid < self._fv.value(obs, food) * edge:
                continue
            action = largest_legal_qty(
                lambda q, f=food: encode_take("ask", f, q),
                legal_mask,
                clamp_qty(surplus),
            )
            if action is not None:
                return action
        return None

    def _build_take_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Above low water but below target: lift asks while they are cheap.

        This is the front-loading half of the archetype — early in the game
        `horizon_need` dwarfs the endowment, so this layer fires cycle after
        cycle and builds the pile while quotes still sit near fair value.
        """
        for food in foods_by_urgency(obs):
            deficit = self._target_stock(obs, food) - int(
                obs["available_inventory"][food]
            )
            if deficit < 1 or self._cover_cycles(obs, food) < LOW_WATER_CYCLES:
                continue
            ask = best_ask(obs, food)
            if ask is None or ask > self._fv.value(obs, food) * BUILD_ACCEPT:
                continue
            qty = clamp_qty(deficit)
            if not self._cash_guard_ok(obs, food, ask * qty):
                continue
            action = largest_legal_qty(
                lambda q, f=food: encode_take("bid", f, q), legal_mask, qty
            )
            if action is not None:
                return action
        return None

    def _maintain_post(
        self,
        obs: Dict[str, Any],
        legal_mask: np.ndarray,
        *,
        side: str,
        food: str,
        price: int,
        qty: int,
        no_downgrade: bool = False,
    ) -> Optional[int]:
        """Post (side, food, qty, price) unless the resting order is close enough.

        The tolerance is what keeps the bot from burning every turn re-posting
        the same quote — an action spent requoting is an action not spent
        restocking the other food.

        `no_downgrade` protects an emergency bid: `post` replaces the resting
        order on the same (side, food), so a routine restock quote following a
        panic bid would silently swap a price-30 lifeline for a price-11 one.
        While the flag is set a richer resting bid is kept (topping up only
        its quantity), and normal downward repricing resumes once the caller
        stops passing it — i.e. once cover is back above the low-water mark.
        """
        resting = own_order(obs, side, food)
        min_qty = 1
        if no_downgrade and resting is not None and int(resting["price"]) >= price:
            if int(resting["remaining_qty"]) >= qty:
                return None
            # Top up the richer resting bid — but a replacement that is
            # *smaller* than what already rests (e.g. because the rest of the
            # cash is reserved by that very order) is still a downgrade.
            price = int(resting["price"])
            min_qty = int(resting["remaining_qty"]) + 1
        if (
            resting is not None
            and abs(int(resting["price"]) - price) < REQUOTE_PRICE_TOL
            and abs(int(resting["remaining_qty"]) - qty) < REQUOTE_QTY_TOL
        ):
            return None
        attempt = new_attempt(obs, food=food, side=side, price=price, qty=qty)
        self.attempted_prices.append(attempt)
        action = None
        for q in range(int(qty), min_qty - 1, -1):
            candidate = encode_post(side, food, q, price)
            if legal_mask[candidate]:
                action = candidate
                break
        if action is None:
            attempt["reason_not_submitted"] = "illegal_under_mask"
            return None
        attempt["submitted"] = True
        return action

    def _quote_slots(
        self, obs: Dict[str, Any]
    ) -> List[Tuple[int, str, str, int, int]]:
        """(rank, side, food, price, qty) for every quote worth resting."""
        slots: List[Tuple[int, str, str, int, int]] = []

        for rank, food in enumerate(foods_by_urgency(obs)):
            deficit = self._target_stock(obs, food) - int(
                obs["available_inventory"][food]
            )
            if deficit < 1:
                continue
            price = self._bid_price(obs, food)
            qty = clamp_qty(deficit)
            if not self._cash_guard_ok(obs, food, price * qty):
                continue
            slots.append((rank, "bid", food, price, qty))

        # Produced merchandise first, then late-game liquidation of the pile.
        #
        # Sizing counts the resting ask's own reservation back in: `post`
        # replaces the prior order on the same slot, refunding it, so sizing
        # off bare `available_inventory` makes the desired quantity *depend
        # on the current quote* — q4 reads as surplus 2, q2 reads as surplus
        # 4 — and the bot oscillates between the two, one wasted action per
        # turn, forever.
        for rank_base, foods in ((10, obs["produces"]), (20, obs["required_foods"])):
            for i, food in enumerate(foods):
                resting = own_order(obs, "ask", food)
                own_reserved = int(resting["remaining_qty"]) if resting else 0
                offerable = int(obs["available_inventory"][food]) + own_reserved
                surplus = offerable - self._sale_floor(obs, food)
                qty = min(
                    clamp_qty(surplus),
                    safe_ask_qty(obs, food) + own_reserved,
                ) if surplus >= 1 else 0
                if qty < 1:
                    continue
                slots.append(
                    (rank_base + i, "ask", food, self._ask_price(obs, food), qty)
                )
        return slots

    def _maintain_quotes_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Refresh the most stale of the standing quotes, one per turn."""
        stale: List[Tuple[int, int, str, str, int, int]] = []
        for rank, side, food, price, qty in self._quote_slots(obs):
            resting = own_order(obs, side, food)
            if resting is None:
                staleness = 0
            elif abs(int(resting["price"]) - price) >= REQUOTE_PRICE_TOL:
                staleness = 1
            elif abs(int(resting["remaining_qty"]) - qty) >= REQUOTE_QTY_TOL:
                staleness = 2
            else:
                continue
            stale.append((staleness, rank, side, food, price, qty))

        if not stale:
            return None
        stale.sort(key=lambda s: (s[0], s[1]))
        best_staleness = stale[0][0]
        top = [s for s in stale if s[0] == best_staleness]
        _, _, side, food, price, qty = top[self._rot.next(len(top))]
        return self._maintain_post(
            obs, legal_mask, side=side, food=food, price=price, qty=qty,
            no_downgrade=(
                side == "bid"
                and self._cover_cycles(obs, food) < LOW_WATER_CYCLES
            ),
        )

    def _free_cash_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Cancel a non-urgent resting bid when survival purchases are underfunded."""
        needed = survival_cash_estimate(
            obs, self._fv.value(obs, obs["required_foods"][0])
        )
        if int(obs["available_cash"]) >= needed:
            return None
        urgent = set(starving_foods(obs))
        for order in obs["own_orders"]:
            if order["side"] != "bid" or order["food"] in urgent:
                continue
            action = encode_cancel("bid", order["food"])
            if legal_mask[action]:
                return action
        return None

    # -----------------------------------------------------------------

    def act(self, obs: Dict[str, Any], legal_mask: np.ndarray) -> int:
        if obs["status"] != "alive":
            return ACTION_WAIT

        self._fv.observe(obs)

        for layer in (
            unwind_ask_action,
            self._emergency_action,
            self._restock_action,
            self._rich_bid_sale_action,
            self._build_take_action,
            self._maintain_quotes_action,
            self._free_cash_action,
        ):
            action = layer(obs, legal_mask)
            if action is not None:
                return action
        return ACTION_WAIT
