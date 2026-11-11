"""Game setup: deck creation, dealing, producer assignment."""
from __future__ import annotations

from typing import Dict, List
import numpy as np

from .config import GameConfig
from .types import GameState, OrderBook, Player, empty_inventory
from .rng import shuffle_in_place, shuffled


def create_initial_deck(config: GameConfig) -> List[str]:
    deck: List[str] = []
    for food in config.food_types:
        for _ in range(config.initial_units_per_food_type):
            deck.append(food)
    return deck


def deal_initial_food(
    config: GameConfig,
    players: List[Player],
    deck: List[str],
    rng: np.random.Generator,
) -> None:
    """Two-phase dealer (mirrors backend/src/game/setup.ts dealInitialFood):

    Phase 1: guarantee every player gets `min_initial_units_per_food_per_player`
             of each food type.
    Phase 2: top up each player to `initial_units_per_player` from the remaining
             pool (round-robin from a re-shuffled remainder).
    """
    min_per = config.min_initial_units_per_food_per_player
    per_player = config.initial_units_per_player
    per_food = config.initial_units_per_food_type

    if min_per * config.player_count > per_food:
        raise ValueError(
            f"min_initial_units_per_food_per_player ({min_per}) * players "
            f"({config.player_count}) exceeds initial_units_per_food_type ({per_food})"
        )
    if min_per * len(config.food_types) > per_player:
        raise ValueError(
            f"min per food * food types ({min_per * len(config.food_types)}) "
            f"exceeds initial_units_per_player ({per_player})"
        )

    for player in players:
        player.inventory = empty_inventory()

    # Bucket the deck by food type, preserving shuffled order within each bucket.
    buckets: Dict[str, List[str]] = {f: [] for f in config.food_types}
    for f in deck:
        buckets[f].append(f)

    # Phase 1: pull `min_per` of each food per player out of the buckets.
    for food in config.food_types:
        for player in players:
            for _ in range(min_per):
                if not buckets[food]:
                    raise ValueError(f"bucket {food} exhausted during guarantee")
                buckets[food].pop()
                player.inventory[food] += 1

    # Phase 2: reassemble remaining cards and reshuffle to interleave foods.
    remaining: List[str] = []
    for f in config.food_types:
        remaining.extend(buckets[f])
    shuffle_in_place(rng, remaining)

    idx = 0
    for player in players:
        while sum(player.inventory.values()) < per_player:
            if idx >= len(remaining):
                raise ValueError("remaining pool exhausted during top-up")
            card = remaining[idx]
            idx += 1
            player.inventory[card] += 1


def assign_producers(
    rng: np.random.Generator, config: GameConfig, players: List[Player]
) -> None:
    if len(players) != config.player_count:
        raise ValueError(
            f"need exactly {config.player_count} players, got {len(players)}"
        )
    # Each player produces TWO foods; each food has exactly two producers.
    # Pick two permutations of the food list, retry until they form a
    # derangement (no shared positions) so no player gets the same food twice.
    # Mirrors backend/src/game/setup.ts assignProducerFoodTypes.
    perm1 = shuffled(rng, list(config.food_types))
    perm2 = shuffled(rng, list(config.food_types))
    while any(a == b for a, b in zip(perm1, perm2)):
        perm2 = shuffled(rng, list(config.food_types))
    for i, p in enumerate(players):
        p.produces = [perm1[i], perm2[i]]


def initial_state(
    config: GameConfig,
    rng: np.random.Generator,
    player_names: List[str],
) -> GameState:
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

    deck = create_initial_deck(config)
    shuffle_in_place(rng, deck)
    deal_initial_food(config, players, deck, rng)

    for p in players:
        p.cash = config.initial_cash

    return state
