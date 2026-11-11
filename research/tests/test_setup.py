from collections import Counter

from env.config import DEFAULT_CONFIG, FOOD_TYPES
from env.rng import make_rng
from env.setup import (
    assign_producers,
    create_initial_deck,
    deal_initial_food,
    initial_state,
)
from env.types import Player


def test_deck_has_400_units_100_each():
    deck = create_initial_deck(DEFAULT_CONFIG)
    assert len(deck) == 400
    counts = Counter(deck)
    for f in FOOD_TYPES:
        assert counts[f] == 100


def test_each_player_gets_100_units():
    rng = make_rng(0)
    state = initial_state(
        DEFAULT_CONFIG, rng, ["A0", "A1", "A2", "A3"]
    )
    for p in state.players:
        assert sum(p.inventory.values()) == 100


def test_global_food_distribution_is_100_each():
    rng = make_rng(7)
    state = initial_state(
        DEFAULT_CONFIG, rng, ["A0", "A1", "A2", "A3"]
    )
    totals = {f: 0 for f in FOOD_TYPES}
    for p in state.players:
        for f in FOOD_TYPES:
            totals[f] += p.inventory[f]
    for f in FOOD_TYPES:
        assert totals[f] == 100


def test_producer_assignment_covers_every_food_twice():
    """Each player produces exactly 2 distinct foods; each food is produced
    by exactly 2 players. Mirrors backend/src/game/setup.ts's two-permutation
    derangement strategy."""
    rng = make_rng(123)
    state = initial_state(DEFAULT_CONFIG, rng, ["A", "B", "C", "D"])
    counts = {f: 0 for f in FOOD_TYPES}
    for p in state.players:
        assert len(p.produces) == 2
        assert p.produces[0] != p.produces[1]
        for f in p.produces:
            counts[f] += 1
    for f in FOOD_TYPES:
        assert counts[f] == 2, (f, counts)


def test_starting_cash():
    rng = make_rng(0)
    state = initial_state(DEFAULT_CONFIG, rng, ["x"] * 4)
    for p in state.players:
        assert p.cash == 100
        assert p.reserved_cash == 0


def test_assign_producers_requires_full_table():
    import pytest

    rng = make_rng(0)
    players = [Player(idx=i, name=f"p{i}") for i in range(3)]
    with pytest.raises(ValueError):
        assign_producers(rng, DEFAULT_CONFIG, players)


def test_deal_guarantees_min_per_food_per_player():
    """Every player must receive at least MIN_INITIAL_UNITS_PER_FOOD_PER_PLAYER
    of every food type. Mirrors the TS two-phase algorithm."""
    rng = make_rng(42)
    state = initial_state(DEFAULT_CONFIG, rng, ["P0", "P1", "P2", "P3"])
    min_per = DEFAULT_CONFIG.min_initial_units_per_food_per_player
    for p in state.players:
        for f in FOOD_TYPES:
            assert p.inventory[f] >= min_per, (p.name, f, p.inventory[f])


def test_deal_initial_food_signature():
    """deal_initial_food consumes exactly initial_units_per_player per player
    and respects the min-per guarantee."""
    rng = make_rng(42)
    deck = create_initial_deck(DEFAULT_CONFIG)
    players = [Player(idx=i, name=f"p{i}") for i in range(4)]
    deal_initial_food(DEFAULT_CONFIG, players, deck, rng)
    min_per = DEFAULT_CONFIG.min_initial_units_per_food_per_player
    for p in players:
        assert sum(p.inventory.values()) == DEFAULT_CONFIG.initial_units_per_player
        for f in FOOD_TYPES:
            assert p.inventory[f] >= min_per
