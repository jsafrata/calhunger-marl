"""Per-tick game progression: production, consumption, death, end-of-game.

Mirrors backend/src/game/{production, consumption, death, winner, tick}.ts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import GameConfig
from .market import cancel_all_for_player
from .types import GameEndReason, GameState, Player


@dataclass
class TickResult:
    deaths: List[dict]  # [{"player_idx": int, "missing_foods": [str, ...]}]
    ended: bool
    end_reason: Optional[GameEndReason]


def produce_food(state: GameState) -> None:
    config: GameConfig = state.config  # type: ignore[assignment]
    for p in state.players:
        if p.status != "alive":
            continue
        # Each player produces TWO foods now — add the per-cycle production
        # rate to EACH of their producer foods.
        for f in p.produces:
            p.inventory[f] += config.production_per_second


def consume_and_kill(state: GameState) -> List[dict]:
    """Consume consumption_per_required_food_per_second of each required food
    per alive player. A player dies when any required food has less than the
    full consumption amount available.

    Uses *available* (not total) inventory: food locked in open asks cannot
    save you.
    """
    config: GameConfig = state.config  # type: ignore[assignment]
    consume_amount = config.consumption_per_required_food_per_second
    deaths: List[dict] = []

    for p in state.players:
        if p.status != "alive":
            continue
        required = config.required_foods(p.produces)
        missing = [
            f
            for f in required
            if (p.inventory[f] - p.reserved_inventory[f]) < consume_amount
        ]
        if missing:
            deaths.append({"player_idx": p.idx, "missing_foods": list(missing)})
            continue
        for f in required:
            p.inventory[f] -= consume_amount

    return deaths


def kill_player(state: GameState, player: Player, _missing: List[str]) -> None:
    """Mark dead, cancel all open orders, zero out reservations.
    Mirrors backend/src/game/death.ts: orders are removed (not refunded into avail
    cash/inventory). Reservations on the dead player are simply zeroed.
    """
    player.status = "dead"
    player.died_at_second = state.elapsed_seconds
    cancel_all_for_player(state, player_idx=player.idx, release_reservations=False)
    player.reserved_cash = 0
    player.reserved_inventory = {"A": 0, "B": 0, "C": 0, "D": 0}


def step_tick(state: GameState) -> TickResult:
    """One simulation tick. Order:

      1. advance clock
      2. every consumption_interval_seconds: production + consumption + death
      3. end-of-game checks (time, single survivor, no survivors)

    Production and consumption are synced (mirrors backend/src/game/tick.ts):
    both fire once per consumption interval. This preserves the per-cycle
    balance (+2 own / -1 each of 3 required, net -1 per cycle) and stops food
    from accumulating between consumption ticks.

    Trade processing is NOT in here — orders are processed when posted (during
    the env.step() preamble). This matches the live game where orders execute
    immediately on submission rather than waiting for a tick.
    """
    result = TickResult(deaths=[], ended=False, end_reason=None)
    if state.phase != "active":
        result.ended = True
        return result

    config: GameConfig = state.config  # type: ignore[assignment]
    state.elapsed_seconds += 1

    if state.elapsed_seconds % config.consumption_interval_seconds == 0:
        produce_food(state)

        death_infos = consume_and_kill(state)
        for d in death_infos:
            kill_player(state, state.players[d["player_idx"]], d["missing_foods"])
            result.deaths.append(d)

    alive = state.alive_count()

    # v2: only end at the fixed horizon or at zero survivors. A sole
    # surviving player who later dies before 180s ends dead (and cannot
    # win); this makes "survival" mean "alive at t=180" rather than
    # "alive when the game happened to end early". See the archive
    # README for the pre-v2 semantics.
    if state.elapsed_seconds >= config.game_duration_seconds:
        end_game(state, "time_limit")
        result.ended = True
        result.end_reason = "time_limit"
    elif alive == 0:
        end_game(state, "no_survivors")
        result.ended = True
        result.end_reason = "no_survivors"

    return result


# ---------------------------------------------------------------------------
# End-of-game / winner
# ---------------------------------------------------------------------------


def total_inventory(player: Player) -> int:
    return sum(player.inventory.values())


def determine_winners(state: GameState) -> List[int]:
    survivors = [p for p in state.players if p.status == "alive"]
    if not survivors:
        return []
    if len(survivors) == 1:
        return [survivors[0].idx]
    max_cash = max(p.cash for p in survivors)
    leaders = [p for p in survivors if p.cash == max_cash]
    if len(leaders) == 1:
        return [leaders[0].idx]
    max_food = max(total_inventory(p) for p in leaders)
    return [p.idx for p in leaders if total_inventory(p) == max_food]


def end_game(state: GameState, reason: GameEndReason) -> None:
    if state.phase == "ended":
        return
    state.phase = "ended"
    state.end_reason = reason
    state.winner_idxs = determine_winners(state)
