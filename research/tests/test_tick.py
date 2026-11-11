from env.config import DEFAULT_CONFIG
from env.rng import make_rng
from env.setup import initial_state
from env.tick import consume_and_kill, kill_player, produce_food, step_tick


def fresh():
    rng = make_rng(0)
    s = initial_state(DEFAULT_CONFIG, rng, ["P0", "P1", "P2", "P3"])
    return s


def test_production_adds_per_food_amount_for_alive_only():
    """produce_food() adds production_per_second to EACH of the player's
    produced foods (each player produces TWO foods)."""
    s = fresh()
    prod = DEFAULT_CONFIG.production_per_second
    before = [{f: p.inventory[f] for f in p.produces} for p in s.players]
    produce_food(s)
    for p, b in zip(s.players, before):
        for f in p.produces:
            assert p.inventory[f] == b[f] + prod

    # Now kill one and verify it doesn't produce
    s.players[0].status = "dead"
    snapshot = dict(s.players[0].inventory)
    produce_food(s)
    assert dict(s.players[0].inventory) == snapshot


def test_step_tick_only_produces_and_consumes_on_cycle_boundary():
    """Production AND consumption only fire when elapsed_seconds is a multiple
    of consumption_interval_seconds. Each player produces TWO foods (+2 each)
    and consumes ONE of each of TWO required foods (-1 each) per cycle."""
    s = fresh()
    interval = DEFAULT_CONFIG.consumption_interval_seconds
    baseline_inv = [dict(p.inventory) for p in s.players]
    for _ in range(interval - 1):
        step_tick(s)
    for p, b in zip(s.players, baseline_inv):
        assert p.inventory == b
    step_tick(s)
    assert s.elapsed_seconds == interval
    for p, b in zip(s.players, baseline_inv):
        produced = set(p.produces)
        for f in DEFAULT_CONFIG.food_types:
            if f in produced:
                assert p.inventory[f] == b[f] + DEFAULT_CONFIG.production_per_second
            else:
                assert p.inventory[f] == b[f] - DEFAULT_CONFIG.consumption_per_required_food_per_second


def test_consumption_removes_per_food_amount_from_required_only():
    """consume_and_kill() debits consumption_per_required_food_per_second
    from each of the two required foods; produced foods are untouched."""
    s = fresh()
    consume = DEFAULT_CONFIG.consumption_per_required_food_per_second
    start = 5
    for p in s.players:
        p.inventory = {f: start for f in ("A", "B", "C", "D")}
        p.reserved_inventory = {"A": 0, "B": 0, "C": 0, "D": 0}
    deaths = consume_and_kill(s)
    assert deaths == []
    for p in s.players:
        produced = set(p.produces)
        for f in ("A", "B", "C", "D"):
            if f in produced:
                assert p.inventory[f] == start  # producer doesn't consume own foods
            else:
                assert p.inventory[f] == start - consume


def test_player_dies_when_missing_required_food():
    s = fresh()
    p0 = s.players[0]
    # Zero out all of the player's required foods so they must die.
    required = [f for f in ("A", "B", "C", "D") if f not in p0.produces]
    p0.inventory = {f: 100 for f in ("A", "B", "C", "D")}
    for f in required:
        p0.inventory[f] = 0
    p0.reserved_inventory = {"A": 0, "B": 0, "C": 0, "D": 0}
    deaths = consume_and_kill(s)
    found = [d for d in deaths if d["player_idx"] == 0]
    assert len(found) == 1
    assert set(found[0]["missing_foods"]) == set(required)


def test_reserved_inventory_excluded_from_consumption_check():
    """Even if total inventory has enough, if reserved makes available < 1, dies."""
    s = fresh()
    p0 = s.players[0]
    needed_food = next(f for f in ("A", "B", "C", "D") if f not in p0.produces)
    p0.inventory = {f: 100 for f in ("A", "B", "C", "D")}
    p0.reserved_inventory = {f: 0 for f in ("A", "B", "C", "D")}
    p0.reserved_inventory[needed_food] = 100
    deaths = consume_and_kill(s)
    found = [d for d in deaths if d["player_idx"] == 0]
    assert len(found) == 1


def test_player_dies_with_less_than_full_required_food_amount():
    """A player must have the full per-cycle amount of every required food."""
    s = fresh()
    p0 = s.players[0]
    consume = DEFAULT_CONFIG.consumption_per_required_food_per_second
    required = [f for f in ("A", "B", "C", "D") if f not in p0.produces]
    p0.inventory = {f: 100 for f in ("A", "B", "C", "D")}
    # Set one required food strictly below consume_amount (but > 0).
    p0.inventory[required[0]] = max(1, consume - 1)
    p0.reserved_inventory = {"A": 0, "B": 0, "C": 0, "D": 0}

    deaths = consume_and_kill(s)
    assert any(d["player_idx"] == 0 for d in deaths)
    # Death is decided before consumption, so neither required food is debited.
    assert p0.inventory[required[0]] == max(1, consume - 1)
    assert p0.inventory[required[1]] == 100


def test_kill_player_zeros_reservations_and_cancels_orders():
    from env.market import post_order

    s = fresh()
    # Stack the deck for P0
    s.players[0].cash = 1000
    s.players[0].inventory = {"A": 50, "B": 50, "C": 50, "D": 50}
    post_order(s, player_idx=0, side="bid", food_type="A", quantity=2, price_per_unit=3)
    post_order(s, player_idx=0, side="ask", food_type="C", quantity=4, price_per_unit=99)
    assert s.players[0].reserved_cash > 0
    assert s.players[0].reserved_inventory["C"] > 0

    kill_player(s, s.players[0], ["B"])
    assert s.players[0].status == "dead"
    assert s.players[0].reserved_cash == 0
    assert all(v == 0 for v in s.players[0].reserved_inventory.values())
    # Books should have no orders from P0
    for f in ("A", "B", "C", "D"):
        for o in s.order_books[f].bids + s.order_books[f].asks:
            assert o.player_idx != 0


def test_tick_clock_advances():
    s = fresh()
    assert s.elapsed_seconds == 0
    step_tick(s)
    assert s.elapsed_seconds == 1


def test_tick_ends_at_max_seconds():
    from env.config import GameConfig

    config = GameConfig(game_duration_seconds=2)
    rng = make_rng(0)
    s = initial_state(config, rng, ["P0", "P1", "P2", "P3"])
    # Pad inventories so nobody dies of starvation in 2 ticks
    for p in s.players:
        p.inventory = {"A": 100, "B": 100, "C": 100, "D": 100}
    r = step_tick(s)
    assert not r.ended
    r = step_tick(s)
    assert r.ended
    assert r.end_reason == "time_limit"
    assert s.phase == "ended"


def test_tick_does_not_end_early_when_only_one_survivor_remains():
    """v2: a single-survivor state must NOT end the game. The episode
    continues until t=180 or until the sole survivor also dies. The
    archived v1 behavior (early single_survivor end) is intentionally
    removed — see the archive README."""
    s = fresh()
    # Force 3 players to die on the next consumption tick.
    for idx in [0, 1, 2]:
        p = s.players[idx]
        for f in ("A", "B", "C", "D"):
            if f != p.produces:
                p.inventory[f] = 0
    # Player 3 stays alive.
    s.players[3].inventory = {"A": 100, "B": 100, "C": 100, "D": 100}
    interval = DEFAULT_CONFIG.consumption_interval_seconds
    # Advance through the first consumption cycle: 3 seats die, 1 survives.
    # Under v1 the game would end here; under v2 it must NOT end.
    for _ in range(interval):
        r = step_tick(s)
        if r.ended:
            break
    assert s.alive_count() == 1
    assert s.players[3].status == "alive"
    assert not r.ended, (
        "v2: sole-survivor states must not end the game early"
    )
    # A further step should also not end the game as long as player 3 stays alive.
    r2 = step_tick(s)
    assert not r2.ended
    # If we skip ahead to the horizon, THEN the game ends (time_limit).
    while s.elapsed_seconds < DEFAULT_CONFIG.game_duration_seconds:
        step_tick(s)
    assert s.phase == "ended"
    assert s.end_reason == "time_limit"
