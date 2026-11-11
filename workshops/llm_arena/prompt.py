"""One shared prompt template for every provider and every seat (v2).

Design (debugged on the 60 s variant, frozen before any scoring run):

- Every game constant (horizon, feeding cycle, production/consumption
  amounts, price/quantity bounds) is derived from the actual
  ``GameConfig`` of the environment being played, so the same template
  is correct for the default 180 s game and every scaled variant.
- The state section surfaces *derived survival facts* the raw JSON blob
  of the v1 prompt made the model compute itself: spendable cash,
  available (post-reservation) stock, feedings of cover per required
  food, and time to the next feeding.  This mirrors the survival
  features the RL observation already contains — parity, not help.
- No strategy advice: rules, state, and the output contract only.
  Strategy discovery is part of what is being measured.
- The information diet is unchanged from the v1 prompt and matches the
  game's partial observability: own private state plus the public book;
  nothing about other players' cash or inventory.

``minimal_game_state`` (the v1 machine-readable state) is retained for
the mock provider and tests; it is no longer embedded in the prompt.
"""
from __future__ import annotations

from typing import Any

from ._paths import bootstrap

bootstrap()

from env.actions import MAX_PRICE, MAX_QTY  # noqa: E402
from env.config import GameConfig  # noqa: E402


PROTOCOL_VERSION = "llm-arena-v3"

FOOD_NAMES = {"A": "Grain", "B": "Veggie", "C": "Meat", "D": "Milk"}


def build_rules(config: GameConfig, observation: dict[str, Any]) -> str:
    """Rules header; static per (config, seat role, alive count)."""
    T = config.game_duration_seconds
    interval = config.consumption_interval_seconds
    consume = config.consumption_per_required_food_per_second
    produce = config.production_per_second
    produces = list(observation["produces"])
    required = list(observation["required_foods"])
    t = observation["tick"]
    alive = sum(1 for p in observation["players"] if p["status"] == "alive")

    return f"""CALHUNGER — a trading survival game. Second {t} of {T}. You are one of 4 players ({alive} alive). You act once per second.

SURVIVAL RULES
- Foods: A=Grain, B=Veggie, C=Meat, D=Milk. You produce {produces[0]} and {produces[1]}, and must EAT {required[0]} and {required[1]}. No player produces what they eat, so required food can only be obtained by buying it.
- Every {interval}th second is a FEEDING: you first gain +{produce} of each food you produce, then you must eat {consume} of EACH required food. If your AVAILABLE stock of either required food is below {consume} at that moment, you die instantly.
- AVAILABLE stock = inventory minus units locked in your own open asks. Food you are offering for sale cannot be eaten. Cash locked in your open bids cannot fund new orders. Never leave an ask resting on a food you need to eat.

MARKET RULES
- One order book per food. bid = buy, ask = sell. Whole-number prices 1..{MAX_PRICE}, quantities 1..{MAX_QTY}.
- post: place a resting limit order (a bid locks price*qty cash; an ask locks the food). It replaces your earlier order on the same side+food. If it crosses a resting order it trades at the resting order's price.
- take: trade instantly against the best resting orders at their prices (side "bid" buys from the cheapest asks, side "ask" sells to the highest bids). Any unfilled remainder just disappears; nothing gets locked.
- cancel: removes your oldest order on that side+food and unlocks it. wait: do nothing. You cannot trade with yourself.

WINNING
The game ends after second {T}. Dead players lose. Among survivors, the most CASH wins (tie broken by total food); a sole survivor wins regardless of cash. Dying rich is still losing."""


OUTPUT_BLOCK = """OUTPUT
Reply with ONE JSON object only, exactly these five keys, null for unused fields:
{"action":"wait","side":null,"food":null,"quantity":null,"price":null}
{"action":"cancel","side":"bid","food":"A","quantity":null,"price":null}
{"action":"take","side":"bid","food":"A","quantity":2,"price":null}
{"action":"post","side":"ask","food":"C","quantity":3,"price":6}
Remember: side "bid" means BUY, side "ask" means SELL — for both post and take."""


def build_state_block(observation: dict[str, Any], config: GameConfig) -> str:
    t = observation["tick"]
    T = config.game_duration_seconds
    interval = config.consumption_interval_seconds
    consume = config.consumption_per_required_food_per_second
    produce = config.production_per_second

    next_feed = ((t // interval) + 1) * interval
    feeds_left = T // interval - t // interval
    inv = observation["inventory"]
    avail = observation["available_inventory"]

    lines = [
        f"YOUR STATE (second {t}; next feeding at second {next_feed}, "
        f"{next_feed - t}s away; {feeds_left} feedings left)",
        f"- Cash: {observation['cash']} total, "
        f"{observation['available_cash']} spendable"
        + (
            f" ({observation['reserved_cash']} locked in your bids)"
            if observation["reserved_cash"]
            else ""
        ),
    ]
    for f in observation["required_foods"]:
        cover = avail[f] // consume
        if avail[f] < consume:
            flag = (
                f"  ⚠ BELOW {consume} — YOU DIE at second {next_feed} "
                "unless you buy NOW"
            )
        elif cover == 1:
            flag = "  (only covers the next feeding)"
        else:
            flag = ""
        held = (
            f"{inv[f]} held, {avail[f]} available"
            if inv[f] != avail[f]
            else f"{avail[f]} available"
        )
        plural = "s" if cover != 1 else ""
        lines.append(
            f"- EAT {f} ({FOOD_NAMES[f]}): {held} → survives {cover} "
            f"feeding{plural}{flag}"
        )
    for f in observation["produces"]:
        held = (
            f"{inv[f]} held, {avail[f]} available"
            if inv[f] != avail[f]
            else f"{avail[f]} available"
        )
        lines.append(
            f"- SELL {f} ({FOOD_NAMES[f]}): {held} to sell "
            f"(+{produce} at every feeding)"
        )
    if observation["own_orders"]:
        orders = ", ".join(
            f"{o['side']} {o['remaining_qty']}x{o['food']}@{o['price']}"
            for o in observation["own_orders"]
        )
        lines.append(f"- Your open orders: {orders}")
    else:
        lines.append("- Your open orders: none")
    return "\n".join(lines)


def build_market_block(observation: dict[str, Any], config: GameConfig) -> str:
    me = observation["player_idx"]
    required = set(observation["required_foods"])
    lines = [
        "MARKET (other players' resting orders, best price first; qty@price)"
    ]
    for f in config.food_types:
        book = observation["order_books"][f]

        def fmt(order_key: str, reverse: bool) -> str:
            others = [o for o in book[order_key] if o["player_idx"] != me]
            others.sort(key=lambda o: o["price"], reverse=reverse)
            return ", ".join(f"{o['qty']}@{o['price']}" for o in others) or "none"

        role = "you EAT this" if f in required else "you SELL this"
        last = book["last_trade_price"]
        last_txt = f" | last trade @{last}" if last is not None else ""
        lines.append(
            f"- {f} ({FOOD_NAMES[f]}, {role}): asks {fmt('ask_orders', False)}"
            f" | bids {fmt('bid_orders', True)}{last_txt}"
        )
    return "\n".join(lines)


def _other_orders(observation: dict[str, Any], food: str, order_key: str):
    me = observation["player_idx"]
    return [
        o
        for o in observation["order_books"][food][order_key]
        if o["player_idx"] != me
    ]


def build_legal_block(observation: dict[str, Any], config: GameConfig) -> str:
    """Render, in words, exactly what the env's legal-action mask allows.

    Derived from the same observation the rest of the prompt uses; the
    consistency with ``env.legal_actions`` is asserted by a test.  With
    this block in the prompt, a rejected action means the model ignored
    an explicit list — a coherence failure, not a bookkeeping one.
    """
    cash = observation["available_cash"]
    avail = observation["available_inventory"]

    buys: list[str] = []
    cant_afford: list[str] = []
    sells: list[str] = []
    for f in config.food_types:
        asks = _other_orders(observation, f, "ask_orders")
        if asks:
            best_ask = min(o["price"] for o in asks)
            max_buy = min(MAX_QTY, cash // best_ask)
            if max_buy >= 1:
                buys.append(f"{f} up to {max_buy} (best ask {best_ask})")
            else:
                cant_afford.append(f"{f} (best ask {best_ask})")
        bids = _other_orders(observation, f, "bid_orders")
        if bids and avail[f] >= 1:
            best_bid = max(o["price"] for o in bids)
            max_sell = min(MAX_QTY, avail[f])
            sells.append(f"{f} up to {max_sell} (best bid {best_bid})")

    postable_asks = [
        f"{f} up to {min(MAX_QTY, avail[f])}"
        for f in config.food_types
        if avail[f] >= 1
    ]
    cancels = sorted(
        {(o["side"], o["food"]) for o in observation["own_orders"]}
    )

    buy_txt = ", ".join(buys) if buys else "nothing"
    if cant_afford:
        buy_txt += "; cannot afford " + ", ".join(cant_afford)
    lines = [
        "LEGAL ACTIONS RIGHT NOW (anything else is rejected and wastes "
        "your turn as a wait)",
        f'- take side "bid" (buy): {buy_txt}',
        '- take side "ask" (sell): '
        + (", ".join(sells) if sells else "nothing"),
        '- post side "bid" (buy order): '
        + (
            f"any food, quantity*price up to {cash}"
            if cash >= 1
            else "nothing (no spendable cash)"
        ),
        '- post side "ask" (sell order): '
        + (", ".join(postable_asks) if postable_asks else "nothing"),
        "- cancel: "
        + (
            ", ".join(f"your {side} on {food}" for side, food in cancels)
            if cancels
            else "nothing to cancel"
        ),
        "- wait: always legal",
    ]
    return "\n".join(lines)


def build_prompt(observation: dict[str, Any], config: GameConfig) -> str:
    return (
        f"{build_rules(config, observation)}\n\n"
        f"{build_state_block(observation, config)}\n\n"
        f"{build_market_block(observation, config)}\n\n"
        f"{build_legal_block(observation, config)}\n\n"
        f"{OUTPUT_BLOCK}\n"
    )


def prompt_template_text(config: GameConfig) -> str:
    """Stable text hashed into the job identity: the rules and output
    contract for a canonical seat (state/market blocks vary per tick)."""
    dummy_obs = {
        "tick": 0,
        "produces": ["A", "B"],
        "required_foods": ["C", "D"],
        "players": [{"status": "alive"}] * 4,
    }
    return build_rules(config, dummy_obs) + "\n" + OUTPUT_BLOCK


# ---------------------------------------------------------------------------
# v1 machine-readable state — retained for the mock provider and tests
# ---------------------------------------------------------------------------

STATE_KEYS = {
    "produces",
    "consumes",
    "food_units",
    "cash",
    "second",
    "quotes_from_other_players",
    "own_open_quotes",
}


def minimal_game_state(observation: dict[str, Any]) -> dict[str, Any]:
    """The exact state categories the prompt exposes, as a dict."""
    player_idx = observation["player_idx"]
    other_quotes: list[dict[str, Any]] = []
    for food, book in observation["order_books"].items():
        for side, order_key in (("bid", "bid_orders"), ("ask", "ask_orders")):
            quotes = [
                {
                    "side": side,
                    "food": food,
                    "price": order["price"],
                    "quantity": order["qty"],
                }
                for order in book[order_key]
                if order["player_idx"] != player_idx
            ]
            quotes.sort(
                key=lambda quote: (
                    -quote["price"] if side == "bid" else quote["price"]
                )
            )
            other_quotes.extend(quotes)

    own_quotes = [
        {
            "side": order["side"],
            "food": order["food"],
            "price": order["price"],
            "quantity": order["remaining_qty"],
        }
        for order in observation["own_orders"]
    ]

    state = {
        "produces": list(observation["produces"]),
        "consumes": list(observation["required_foods"]),
        "food_units": dict(observation["inventory"]),
        "cash": observation["cash"],
        "second": observation["tick"],
        "quotes_from_other_players": other_quotes,
        "own_open_quotes": own_quotes,
    }
    assert set(state) == STATE_KEYS
    return state
