"""ScriptedV3 — deadline-aware procurement trader.

Economic archetype: a firm running just-in-time procurement against a known
delivery deadline, plus scheduled liquidation of its own production. It plans
around the *next consumption tick* rather than reacting to the book, and
concedes on price as that deadline approaches — the shape of a scheduled
execution algorithm crossed with a bargaining concession curve.

How it differs from ScriptedV1 and ScriptedV2
---------------------------------------------
V1 reprices off the book on every fire and always trades 1 unit. V3 computes a
plan once per consumption cycle and then executes it, in batches, on a fixed
concession schedule — its price path is a function of *time remaining*, not of
what the book just did. V2 quotes both sides continuously to earn the spread;
V3 never tries to earn a spread at all. It is one-sided per food: it buys what
it must consume and sells what it produces.

The two bots converge under high urgency — both end up crossing the spread —
so their observable separation lives in the early and middle of each cycle,
where V2 is quoting two-sided and V3 is sitting on a single patient order.

A note on early-game passivity
------------------------------
Planning against a 2-cycle coverage horizon means V3 buys nothing for the first
~11 cycles: players open holding far more than two cycles of each required
food. That is not a bug — it is what just-in-time procurement *is*, and it is
the behaviour a learned opponent should have to anticipate. The bot stays
solvent in the meantime by selling produced surplus every cycle, and the
`_escalate` rule guarantees it starts crossing while it still has enough action
opportunities left to fix both required foods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    cycle_progress,
    deficit_for_cycles,
    foods_by_urgency,
    largest_legal_qty,
    new_attempt,
    own_order,
    safe_ask_qty,
    starving_foods,
    steps_to_next_consumption,
    survival_cash_estimate,
    unwind_ask_action,
)

# ---- calibration knobs ----------------------------------------------------
# Baseline JIT coverage, in consumption ticks. The original design called for
# 1-2 cycles; a 30-seed sweep across all three cadences put the survival optimum
# at 3 (mean survivors 3.83 vs 3.48 at 2). Two cycles of cover is too lean for a
# game that is *structurally* short of required food — each player needs 36 units
# of each required food over 18 cycles but opens holding 20-40 units in total.
COVERAGE_CYCLES = 3
MAX_COVERAGE_CYCLES = 5    # ceiling once coverage is stretched for infrequency
COMFORTABLE_DECISIONS = 6  # decisions/cycle at which baseline coverage suffices
WORKING_STOCK = 2          # produced units retained rather than offered
EARLY_UNTIL = 0.4          # cycle-progress boundaries for the concession stages
MID_UNTIL = 0.7
LATE_UNTIL = 0.9
PATIENT_EDGE = 0.20        # early-stage distance from fair value
MID_EDGE = 0.05            # mid-stage distance
LATE_EDGE = 0.10           # late-stage *adverse* distance (pay up / undercut)
PANIC_MULT = 2.5           # price multiple when starving with an empty book


@dataclass
class CyclePlan:
    """What this consumption cycle is supposed to achieve."""

    cycle: int
    buy: Dict[str, int] = field(default_factory=dict)
    sell: Dict[str, int] = field(default_factory=dict)
    # Actions already spent this cycle, per side. Used to guarantee production
    # still reaches the market on cycles where procurement wants every turn.
    buy_actions: int = 0
    sell_actions: int = 0


def concession_stage(progress: float) -> str:
    """Where we are on the concession curve: early -> mid -> late -> final."""
    if progress < EARLY_UNTIL:
        return "early"
    if progress < MID_UNTIL:
        return "mid"
    if progress < LATE_UNTIL:
        return "late"
    return "final"


class ScriptedV3Agent(Agent):
    """Plans per consumption cycle; concedes on price as the deadline nears."""

    name = "scripted_v3"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._fv = FairValueTracker()
        self._rate = DecisionRateEstimator()
        self._plan: Optional[CyclePlan] = None
        self.attempted_prices: List[dict] = []

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._fv.reset()
        self._rate.reset()
        self._plan = None
        self.attempted_prices = []

    # -----------------------------------------------------------------
    # Cycle planning
    # -----------------------------------------------------------------

    def _cycle_index(self, obs: Dict[str, Any]) -> int:
        interval = int(obs["consumption_interval_seconds"])
        return int(obs["tick"]) // interval

    def _coverage_cycles(self, obs: Dict[str, Any]) -> int:
        """How many cycles ahead to procure, given how often we get to act.

        Textbook inventory policy: safety stock rises as order frequency falls.
        A desk that can place an order every step can afford to run lean; one
        that gets two turns per cycle and must service two required foods plus
        its own liquidation cannot, and has to buy further ahead to compensate.
        Without this, a slow-cadence bot plans a deficit it has no remaining
        turns to execute.
        """
        interval = int(obs["consumption_interval_seconds"])
        per_cycle = self._rate.decisions_within(interval)
        stretch = max(0, COMFORTABLE_DECISIONS - per_cycle)
        return min(MAX_COVERAGE_CYCLES, COVERAGE_CYCLES + stretch)

    def _ensure_plan(self, obs: Dict[str, Any]) -> CyclePlan:
        """Recompute the plan at each cycle boundary.

        Production and consumption both land on the boundary tick, so this is
        the only moment inventory moves for non-market reasons — everything the
        plan needs to know is settled exactly here.
        """
        cycle = self._cycle_index(obs)
        if self._plan is not None and self._plan.cycle == cycle:
            return self._plan

        plan = CyclePlan(cycle=cycle)
        coverage = self._coverage_cycles(obs)
        for food in obs["required_foods"]:
            deficit = deficit_for_cycles(obs, food, coverage)
            if deficit > 0:
                plan.buy[food] = deficit
        for food in obs["produces"]:
            surplus = int(obs["available_inventory"][food]) - WORKING_STOCK
            if surplus > 0:
                plan.sell[food] = surplus
        self._plan = plan
        return plan

    def _escalate(self, obs: Dict[str, Any], plan: CyclePlan) -> bool:
        """True once we can no longer afford to be patient.

        Counted in *decisions*, not env steps: the cadence wrapper may only let
        us act every 2nd or 5th step, so a bot that waits until
        `steps_remaining == outstanding` has already missed its window. This is
        the rule that makes "start emergency execution while enough action
        opportunities remain to fix both required-food deficits" concrete.
        """
        outstanding = sum(
            1
            for food in obs["required_foods"]
            if deficit_for_cycles(obs, food, 1) > 0 or food in plan.buy
        )
        if outstanding == 0:
            return False
        left = self._rate.decisions_within(steps_to_next_consumption(obs))
        return left <= outstanding + 1

    # -----------------------------------------------------------------
    # Concession pricing
    # -----------------------------------------------------------------

    def _bid_price(self, obs: Dict[str, Any], food: str, stage: str) -> int:
        fv = self._fv.value(obs, food)
        if stage == "early":
            return clamp_price(fv * (1 - PATIENT_EDGE))
        if stage == "mid":
            return clamp_price(fv * (1 - MID_EDGE))
        ask = best_ask(obs, food)
        if stage == "late":
            # Pay up, and at least meet the book if it is cheap enough to touch.
            target = fv * (1 + LATE_EDGE)
            return clamp_price(max(target, ask) if ask is not None else target)
        return clamp_price(max(fv * PANIC_MULT, (ask + 2) if ask is not None else fv * 2))

    def _ask_price(self, obs: Dict[str, Any], food: str, stage: str) -> int:
        fv = self._fv.value(obs, food)
        if stage == "early":
            return clamp_price(fv * (1 + PATIENT_EDGE))
        if stage == "mid":
            return clamp_price(fv * (1 + MID_EDGE))
        bid = best_bid(obs, food)
        target = fv * (1 - LATE_EDGE)
        if stage == "late":
            return clamp_price(min(target, bid) if bid is not None else target)
        # final: release the goods
        return clamp_price(bid if bid is not None else fv * (1 - 2 * LATE_EDGE))

    # -----------------------------------------------------------------
    # Execution primitives
    # -----------------------------------------------------------------

    def _buy(
        self,
        obs: Dict[str, Any],
        legal_mask: np.ndarray,
        food: str,
        qty: int,
        stage: str,
    ) -> Optional[int]:
        """One procurement action for `food`, appropriate to the stage."""
        qty = clamp_qty(qty)

        if stage in ("late", "final"):
            ask = best_ask(obs, food)
            price = self._bid_price(obs, food, stage)
            if ask is not None and ask <= price:
                action = largest_legal_qty(
                    lambda q, f=food: encode_take("bid", f, q), legal_mask, qty
                )
                if action is not None:
                    return action

        price = self._bid_price(obs, food, stage)
        resting = own_order(obs, "bid", food)
        if resting is not None and int(resting["price"]) == price and int(
            resting["remaining_qty"]
        ) == qty:
            return None  # already working the right order; don't burn the step

        attempt = new_attempt(obs, food=food, side="bid", price=price, qty=qty)
        self.attempted_prices.append(attempt)
        action = largest_legal_qty(
            lambda q, f=food, p=price: encode_post("bid", f, q, p), legal_mask, qty
        )
        if action is None:
            attempt["reason_not_submitted"] = "illegal_under_mask"
            return None
        attempt["submitted"] = True
        return action

    def _sell(
        self,
        obs: Dict[str, Any],
        legal_mask: np.ndarray,
        food: str,
        qty: int,
        stage: str,
    ) -> Optional[int]:
        """One liquidation action for `food`, capped by the survival floor."""
        qty = min(clamp_qty(qty), safe_ask_qty(obs, food))
        if qty < 1:
            return None

        if stage in ("late", "final"):
            bid = best_bid(obs, food)
            price = self._ask_price(obs, food, stage)
            if bid is not None and bid >= price:
                action = largest_legal_qty(
                    lambda q, f=food: encode_take("ask", f, q), legal_mask, qty
                )
                if action is not None:
                    return action

        price = self._ask_price(obs, food, stage)
        resting = own_order(obs, "ask", food)
        if resting is not None and int(resting["price"]) == price and int(
            resting["remaining_qty"]
        ) == qty:
            return None

        attempt = new_attempt(obs, food=food, side="ask", price=price, qty=qty)
        self.attempted_prices.append(attempt)
        action = largest_legal_qty(
            lambda q, f=food, p=price: encode_post("ask", f, q, p), legal_mask, qty
        )
        if action is None:
            attempt["reason_not_submitted"] = "illegal_under_mask"
            return None
        attempt["submitted"] = True
        return action

    def _free_cash_action(
        self, obs: Dict[str, Any], legal_mask: np.ndarray
    ) -> Optional[int]:
        """Cancel a non-urgent resting bid when survival purchases are underfunded.

        A resting bid locks `price * qty` of cash in `reserved_cash` the moment
        it is posted, so an over-eager speculative order can price the bot out
        of the food it actually needs. Cancelling refunds the reservation.
        """
        needed = survival_cash_estimate(obs, self._fv.value(obs, obs["required_foods"][0]))
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
        self._rate.observe(obs)

        # A resting ask whose reservation has outlived its safety margin is
        # holding our own survival food hostage — free it before anything else.
        action = unwind_ask_action(obs, legal_mask)
        if action is not None:
            return action

        plan = self._ensure_plan(obs)
        stage = concession_stage(cycle_progress(obs))
        if self._escalate(obs, plan):
            stage = "final"

        # 1. Starving right now — nothing else matters.
        for food in starving_foods(obs):
            need = CONSUME_PER_CYCLE - int(obs["available_inventory"][food])
            action = self._buy(obs, legal_mask, food, max(1, need), "final")
            if action is not None:
                plan.buy_actions += 1
                return action

        # 2/3. Procurement and liquidation, earliest-deadline-first within each.
        #
        # Selling normally yields to buying, but not unconditionally: once a
        # cycle has spent an action on procurement and none on liquidation, the
        # sell goes first. Without this the bot stops offering its production
        # exactly when every other player is short — which starves the market it
        # is itself trying to buy from, and at medium cadence that feedback loop
        # kills the whole table.
        owe_sell = plan.buy_actions >= 1 and plan.sell_actions == 0
        order = ("sell", "buy") if owe_sell else ("buy", "sell")

        for phase in order:
            if phase == "buy":
                for food in foods_by_urgency(obs):
                    want = max(plan.buy.get(food, 0), deficit_for_cycles(obs, food, 1))
                    if want < 1:
                        continue
                    action = self._buy(obs, legal_mask, food, want, stage)
                    if action is not None:
                        plan.buy_actions += 1
                        return action
            else:
                for food in obs["produces"]:
                    want = plan.sell.get(food, 0)
                    if want < 1:
                        continue
                    action = self._sell(obs, legal_mask, food, want, stage)
                    if action is not None:
                        plan.sell_actions += 1
                        return action

        # 4. Unblock survival cash if a speculative bid is sitting on it.
        action = self._free_cash_action(obs, legal_mask)
        if action is not None:
            return action

        return ACTION_WAIT
