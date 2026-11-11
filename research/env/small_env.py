"""Short-horizon CalHunger environments with scaled initial food budgets.

Purpose
-------
Reduced-difficulty variants used to isolate the horizon and initial-food
axes of task complexity.  The 30-second environment is the baseline.  The
60-, 90-, 120-, and 150-second environments multiply its duration, total
initial food, required-food sum, and required-food bounds by 2, 3, 4, and 5
respectively.  Everything else is inherited unchanged from
``FoodTradingEnv``.

Changes vs. the default (v2) env
--------------------------------
For a scale factor ``r = duration / 30``:

- Total initial food per player is ``16 * r``.
- The two required foods sum to ``8 * r`` (exactly half the total).
- Each required food is in ``[3 * r, 5 * r]``.
- The two produced foods split the remaining ``8 * r`` units.  As in the
  original 30-second environment, they have no individual min/max bound.
- All valid integer splits are sampled uniformly and food identities are
  randomized, so no food is privileged by the dealer.
- The pool-based dealer in ``env/setup.py`` is bypassed entirely.
  ``initial_units_per_food_type`` is meaningless here since the dealer
  no longer draws from a fixed deck.

Everything else is inherited unchanged, including:
- 4 players, 4 foods, per-player 2-producer / 2-required split,
- +2 / -2 production/consumption per cycle, 10 s consumption interval,
- ``initial_cash`` = 100, same action space, same market clearing, and
  the same v2 terminal-only reward + full-horizon termination when the
  RL adapter is configured with ``reward_kind="terminal_only"``.

Public API
----------
- ``SMALL_CONFIG`` / ``SmallFoodTradingEnv`` retain the existing 30-second
  API.
- ``CONFIG_60`` / ``FoodTradingEnv60`` (and the corresponding 90, 120, and
  150 names) expose the new scaled variants.
- ``SCALED_ENVIRONMENTS`` maps RL-facing variant names (``small``,
  ``small_60``, ..., ``small_150``) to their config and environment class.

The default ``FoodTradingEnv`` and ``DEFAULT_CONFIG`` are NOT modified;
importing this module has no side effects on the base env.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import GameConfig
from .env import FoodTradingEnv
from .rng import make_rng, shuffled
from .setup import assign_producers
from .types import GameState, OrderBook, Player, empty_inventory


SMALL_GAME_DURATION_SECONDS: int = 30
SMALL_INITIAL_UNITS_PER_PLAYER: int = 16
SMALL_REQUIRED_SUM: int = 8
SMALL_REQUIRED_MIN: int = 3
SMALL_REQUIRED_MAX: int = 5


@dataclass(frozen=True)
class ScaledFoodSpec:
    """Initial-food constraints for one short-horizon environment."""

    game_duration_seconds: int
    initial_units_per_player: int
    required_sum: int
    required_min: int
    required_max: int

    def __post_init__(self) -> None:
        if self.game_duration_seconds <= 0:
            raise ValueError("game duration must be positive")
        if self.initial_units_per_player <= 0:
            raise ValueError("initial food must be positive")
        if 2 * self.required_sum != self.initial_units_per_player:
            raise ValueError(
                "required-food sum must equal half of total initial food"
            )
        if not (0 <= self.required_min <= self.required_max):
            raise ValueError("required-food bounds must be ordered and non-negative")
        if not (
            2 * self.required_min <= self.required_sum <= 2 * self.required_max
        ):
            raise ValueError("required-food bounds admit no two-food split")

    @property
    def produced_sum(self) -> int:
        return self.initial_units_per_player - self.required_sum


def _make_scaled_spec(game_duration_seconds: int) -> ScaledFoodSpec:
    """Scale every 30-second baseline food constraint by an integer ratio."""
    ratio, remainder = divmod(game_duration_seconds, SMALL_GAME_DURATION_SECONDS)
    if ratio < 1 or remainder:
        raise ValueError(
            "scaled game duration must be a positive multiple of "
            f"{SMALL_GAME_DURATION_SECONDS}, got {game_duration_seconds}"
        )
    return ScaledFoodSpec(
        game_duration_seconds=game_duration_seconds,
        initial_units_per_player=SMALL_INITIAL_UNITS_PER_PLAYER * ratio,
        required_sum=SMALL_REQUIRED_SUM * ratio,
        required_min=SMALL_REQUIRED_MIN * ratio,
        required_max=SMALL_REQUIRED_MAX * ratio,
    )


FOOD_SPEC_30 = _make_scaled_spec(30)
FOOD_SPEC_60 = _make_scaled_spec(60)
FOOD_SPEC_90 = _make_scaled_spec(90)
FOOD_SPEC_120 = _make_scaled_spec(120)
FOOD_SPEC_150 = _make_scaled_spec(150)


def _config_for_spec(spec: ScaledFoodSpec) -> GameConfig:
    # ``initial_units_per_food_type`` is retained only to keep the base config
    # meaningful.  The scaled dealer constructs inventories per player and
    # never draws from the base environment's shared food pool.
    return GameConfig(
        game_duration_seconds=spec.game_duration_seconds,
        initial_units_per_player=spec.initial_units_per_player,
        initial_units_per_food_type=spec.initial_units_per_player,
        min_initial_units_per_food_per_player=0,
    )


SMALL_CONFIG = _config_for_spec(FOOD_SPEC_30)
CONFIG_60 = _config_for_spec(FOOD_SPEC_60)
CONFIG_90 = _config_for_spec(FOOD_SPEC_90)
CONFIG_120 = _config_for_spec(FOOD_SPEC_120)
CONFIG_150 = _config_for_spec(FOOD_SPEC_150)


def _sample_required_split(
    rng: np.random.Generator,
    spec: ScaledFoodSpec = FOOD_SPEC_30,
) -> tuple[int, int]:
    """Sample uniformly from every valid integer required-food split."""
    lo, hi = spec.required_min, spec.required_max
    total = spec.required_sum
    valid = [(a, total - a) for a in range(lo, hi + 1) if lo <= total - a <= hi]
    if not valid:
        raise ValueError(
            f"no valid required-food split for sum={total}, range=[{lo},{hi}]"
        )
    return valid[int(rng.integers(0, len(valid)))]


def _sample_produced_split(
    rng: np.random.Generator,
    spec: ScaledFoodSpec = FOOD_SPEC_30,
) -> tuple[int, int]:
    """Sample uniformly over produced-food partitions of the remaining half.

    The spec constrains only the required foods; the produced foods are
    free apart from summing to ``spec.produced_sum``. Uniform over
    all integer partitions keeps the sampler symmetric and avoids any
    hidden preference between produced and required behavior.
    """
    remainder = spec.produced_sum
    a = int(rng.integers(0, remainder + 1))
    return a, remainder - a


def deal_scaled_initial_food(
    rng: np.random.Generator,
    config: GameConfig,
    players: List[Player],
    spec: ScaledFoodSpec,
) -> None:
    """Assign initial inventories satisfying one scaled environment spec."""
    if config.game_duration_seconds != spec.game_duration_seconds:
        raise ValueError(
            "config/spec duration mismatch: "
            f"{config.game_duration_seconds} != {spec.game_duration_seconds}"
        )
    if config.initial_units_per_player != spec.initial_units_per_player:
        raise ValueError(
            "config/spec initial-food mismatch: "
            f"{config.initial_units_per_player} != {spec.initial_units_per_player}"
        )

    for player in players:
        player.inventory = empty_inventory()
        produced = tuple(player.produces)
        required = config.required_foods(produced)
        if len(produced) != 2 or len(required) != 2:
            raise ValueError(
                f"scaled dealer expects 2 produced + 2 required foods per "
                f"player, got produced={produced} required={required}"
            )

        required_split = _sample_required_split(rng, spec)
        for food, units in zip(shuffled(rng, list(required)), required_split):
            player.inventory[food] = int(units)

        produced_split = _sample_produced_split(rng, spec)
        for food, units in zip(shuffled(rng, list(produced)), produced_split):
            player.inventory[food] = int(units)

        total = sum(player.inventory.values())
        if total != spec.initial_units_per_player:
            raise ValueError(
                f"scaled dealer produced total={total} for player {player.idx}, "
                f"expected {spec.initial_units_per_player}"
            )


def deal_small_initial_food(
    rng: np.random.Generator,
    config: GameConfig,
    players: List[Player],
) -> None:
    """Assign per-player initial inventories that satisfy the small-env spec.

    For each player:
      - the two required foods sum to `SMALL_REQUIRED_SUM` with each in
        [SMALL_REQUIRED_MIN, SMALL_REQUIRED_MAX];
      - the two produced foods sum to the remainder (16 - 8 = 8), split
        uniformly over integer partitions;
      - foods the player neither produces nor requires (should be empty
        under a 4-food / 2-produce / 2-consume setup) get 0.
    """
    deal_scaled_initial_food(rng, config, players, FOOD_SPEC_30)


def scaled_initial_state(
    config: GameConfig,
    rng: np.random.Generator,
    player_names: List[str],
    spec: ScaledFoodSpec,
) -> GameState:
    """Build an initial state using a scaled per-player dealer."""
    if len(player_names) != config.player_count:
        raise ValueError(
            f"expected {config.player_count} player names, got {len(player_names)}"
        )

    players: List[Player] = [
        Player(idx=i, name=name) for i, name in enumerate(player_names)
    ]
    state = GameState(
        config=config,
        players=players,
        order_books={f: OrderBook(food_type=f) for f in config.food_types},
        elapsed_seconds=0,
        phase="active",
    )

    assign_producers(rng, config, players)
    deal_scaled_initial_food(rng, config, players, spec)
    for player in players:
        player.cash = config.initial_cash
    return state


def small_initial_state(
    config: GameConfig,
    rng: np.random.Generator,
    player_names: List[str],
) -> GameState:
    """Build an initial state using the small dealer.

    Mirrors ``env.setup.initial_state`` but calls ``deal_small_initial_food``
    instead of the pool-based dealer. Producer assignment reuses the base
    ``assign_producers`` so the two-producer-per-food derangement contract
    is unchanged.
    """
    return scaled_initial_state(config, rng, player_names, FOOD_SPEC_30)


class ScaledFoodTradingEnv(FoodTradingEnv):
    """Base class for duration-scaled variants of ``FoodTradingEnv``."""

    FOOD_SPEC: ScaledFoodSpec = FOOD_SPEC_30
    DEFAULT_SCALED_CONFIG: GameConfig = SMALL_CONFIG

    def __init__(
        self,
        seed: int = 0,
        config: Optional[GameConfig] = None,
        player_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            seed=seed,
            config=config or self.DEFAULT_SCALED_CONFIG,
            player_names=player_names,
        )

    def _build_state(self, seed: int) -> GameState:
        self._market_rng = make_rng(seed)
        return scaled_initial_state(
            self.config,
            make_rng(seed),
            self.player_names,
            self.FOOD_SPEC,
        )


class SmallFoodTradingEnv(ScaledFoodTradingEnv):
    """The original 30-second, 16-unit short-horizon environment.

    Reuses every piece of the base environment except state construction.
    The RL adapters (``NashHungerMultiAgentEnv``, ``NashHungerGym``) do
    not need to be aware of this subclass — they operate on the
    ``FoodTradingEnv`` interface, which this class satisfies.
    """

    FOOD_SPEC = FOOD_SPEC_30
    DEFAULT_SCALED_CONFIG = SMALL_CONFIG


class FoodTradingEnv60(ScaledFoodTradingEnv):
    """60-second variant with 32 initial units per player."""

    FOOD_SPEC = FOOD_SPEC_60
    DEFAULT_SCALED_CONFIG = CONFIG_60


class FoodTradingEnv90(ScaledFoodTradingEnv):
    """90-second variant with 48 initial units per player."""

    FOOD_SPEC = FOOD_SPEC_90
    DEFAULT_SCALED_CONFIG = CONFIG_90


class FoodTradingEnv120(ScaledFoodTradingEnv):
    """120-second variant with 64 initial units per player."""

    FOOD_SPEC = FOOD_SPEC_120
    DEFAULT_SCALED_CONFIG = CONFIG_120


class FoodTradingEnv150(ScaledFoodTradingEnv):
    """150-second variant with 80 initial units per player."""

    FOOD_SPEC = FOOD_SPEC_150
    DEFAULT_SCALED_CONFIG = CONFIG_150


SCALED_ENVIRONMENTS = {
    "small": (SMALL_CONFIG, SmallFoodTradingEnv),
    "small_60": (CONFIG_60, FoodTradingEnv60),
    "small_90": (CONFIG_90, FoodTradingEnv90),
    "small_120": (CONFIG_120, FoodTradingEnv120),
    "small_150": (CONFIG_150, FoodTradingEnv150),
}
