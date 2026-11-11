"""Game configuration constants. Mirrors backend/src/config.ts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


FOOD_TYPES: Tuple[str, ...] = ("A", "B", "C", "D")

FOOD_DISPLAY_NAMES = {
    "A": "Grain",
    "B": "Veggie",
    "C": "Meat",
    "D": "Milk",
}


@dataclass(frozen=True)
class GameConfig:
    player_count: int = 4
    food_types: Tuple[str, ...] = FOOD_TYPES
    initial_cash: int = 100
    initial_units_per_food_type: int = 100
    initial_units_per_player: int = 100
    min_initial_units_per_food_per_player: int = 20
    game_duration_seconds: int = 180
    # Each player produces +2 of each of their two produced foods per cycle,
    # and consumes −2 of each of their two required foods per cycle. Net per
    # player: produces 4 units / cycle, consumes 4 → break-even per cycle.
    production_per_second: int = 2
    consumption_per_required_food_per_second: int = 2
    consumption_interval_seconds: int = 10
    min_price: int = 0
    min_order_quantity: int = 1
    max_open_orders_per_player: int = 50
    max_new_orders_per_player_per_second: int = 10
    max_cancels_per_player_per_second: int = 20
    simultaneous_order_clearing: bool = True

    def required_foods(self, produces) -> Tuple[str, ...]:
        """Foods the player must consume each cycle = food_types \\ produces.
        Accepts either a single food (str) for back-compat or an iterable of
        foods (the new multi-producer regime: each player produces 2 foods)."""
        if produces is None:
            produced = set()
        elif isinstance(produces, str):
            produced = {produces}
        else:
            produced = set(produces)
        return tuple(f for f in self.food_types if f not in produced)


DEFAULT_CONFIG = GameConfig()
