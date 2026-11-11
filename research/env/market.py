"""Order book + matching engine. Mirrors backend/src/market/*."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from .config import GameConfig
from .types import GameState, Order, OrderBook, Trade


class OrderError(Exception):
    """Raised when an order cannot be posted (insufficient resources, bad params, etc.)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def _bid_sort_key(o: Order) -> Tuple[int, int]:
    # Highest price first → negate. Then earliest sequence → ascending.
    return (-o.price_per_unit, o.sequence)


def _ask_sort_key(o: Order) -> Tuple[int, int]:
    # Lowest price first → ascending. Then earliest sequence → ascending.
    return (o.price_per_unit, o.sequence)


def sort_bids(bids: List[Order]) -> None:
    bids.sort(key=_bid_sort_key)


def sort_asks(asks: List[Order]) -> None:
    asks.sort(key=_ask_sort_key)


# ---------------------------------------------------------------------------
# Reservations
# ---------------------------------------------------------------------------


def _reserve_for_order(state: GameState, order: Order) -> None:
    p = state.players[order.player_idx]
    if order.side == "bid":
        p.reserved_cash += order.price_per_unit * order.remaining_quantity
    else:
        p.reserved_inventory[order.food_type] += order.remaining_quantity


def _release_remaining(state: GameState, order: Order) -> None:
    if order.remaining_quantity <= 0:
        return
    p = state.players[order.player_idx]
    if order.side == "bid":
        refund = order.price_per_unit * order.remaining_quantity
        p.reserved_cash = max(0, p.reserved_cash - refund)
    else:
        cur = p.reserved_inventory[order.food_type]
        p.reserved_inventory[order.food_type] = max(0, cur - order.remaining_quantity)


def _apply_trade_reservations(
    state: GameState,
    buyer_order: Order,
    seller_order: Order,
    quantity: int,
) -> None:
    """Release the buyer's reserved cash at the buyer's *limit price* (which is what was
    locked at order time), and release the seller's reserved inventory by `quantity`.
    Note the trade itself executes at the maker's price; the difference between
    buyer's limit and maker price is returned to available cash automatically because
    we only deduct `total_price` from cash but release `limit*qty` from reserved.
    """
    buyer = state.players[buyer_order.player_idx]
    seller = state.players[seller_order.player_idx]

    reserved_at_limit = buyer_order.price_per_unit * quantity
    buyer.reserved_cash = max(0, buyer.reserved_cash - reserved_at_limit)

    cur = seller.reserved_inventory[seller_order.food_type]
    seller.reserved_inventory[seller_order.food_type] = max(0, cur - quantity)


# ---------------------------------------------------------------------------
# Book ops
# ---------------------------------------------------------------------------


def _add_to_book(book: OrderBook, order: Order) -> None:
    if order.side == "bid":
        book.bids.append(order)
        sort_bids(book.bids)
    else:
        book.asks.append(order)
        sort_asks(book.asks)


def _remove_from_book(book: OrderBook, order_id: int) -> Optional[Order]:
    for i, o in enumerate(book.bids):
        if o.id == order_id:
            return book.bids.pop(i)
    for i, o in enumerate(book.asks):
        if o.id == order_id:
            return book.asks.pop(i)
    return None


def _find_in_book(book: OrderBook, order_id: int) -> Optional[Order]:
    for o in book.bids:
        if o.id == order_id:
            return o
    for o in book.asks:
        if o.id == order_id:
            return o
    return None


def _find_prior_player_order(
    state: GameState, *, player_idx: int, side: str, food_type: str
) -> Optional[Order]:
    book = state.order_books[food_type]
    pool = book.bids if side == "bid" else book.asks
    return next((o for o in pool if o.player_idx == player_idx), None)


def find_order_anywhere(state: GameState, order_id: int) -> Optional[Order]:
    for f in state.config.food_types:  # type: ignore[attr-defined]
        o = _find_in_book(state.order_books[f], order_id)
        if o:
            return o
    return None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _execute_trade(
    state: GameState,
    *,
    buyer_order: Order,
    seller_order: Order,
    maker_order: Order,
    taker_order: Order,
    quantity: int,
    food_type: str,
) -> Trade:
    buyer = state.players[buyer_order.player_idx]
    seller = state.players[seller_order.player_idx]

    trade_price = maker_order.price_per_unit
    total_price = trade_price * quantity

    buyer.cash -= total_price
    seller.cash += total_price

    seller.inventory[food_type] -= quantity
    buyer.inventory[food_type] += quantity

    buyer_order.remaining_quantity -= quantity
    seller_order.remaining_quantity -= quantity

    _apply_trade_reservations(
        state, buyer_order=buyer_order, seller_order=seller_order, quantity=quantity
    )

    buyer.total_bought += quantity
    buyer.cash_spent_on_trades += total_price
    seller.total_sold += quantity
    seller.cash_from_trades += total_price

    buyer_order.status = "filled" if buyer_order.remaining_quantity == 0 else "partially_filled"
    seller_order.status = "filled" if seller_order.remaining_quantity == 0 else "partially_filled"

    trade = Trade(
        id=state.next_trade_id(),
        food_type=food_type,
        buyer_idx=buyer.idx,
        seller_idx=seller.idx,
        price_per_unit=trade_price,
        quantity=quantity,
        total_price=total_price,
        maker_order_id=maker_order.id,
        taker_order_id=taker_order.id,
        elapsed_second=state.elapsed_seconds,
    )
    state.trades.append(trade)
    return trade


def _match_incoming(state: GameState, incoming: Order) -> List[Trade]:
    book = state.order_books[incoming.food_type]
    trades: List[Trade] = []

    if incoming.side == "bid":
        while incoming.remaining_quantity > 0:
            sort_asks(book.asks)
            ask = next(
                (o for o in book.asks if o.player_idx != incoming.player_idx),
                None,
            )
            if ask is None or incoming.price_per_unit < ask.price_per_unit:
                break
            qty = min(incoming.remaining_quantity, ask.remaining_quantity)
            trades.append(
                _execute_trade(
                    state,
                    buyer_order=incoming,
                    seller_order=ask,
                    maker_order=ask,
                    taker_order=incoming,
                    quantity=qty,
                    food_type=incoming.food_type,
                )
            )
            if ask.remaining_quantity == 0:
                _remove_from_book(book, ask.id)
    else:  # ask
        while incoming.remaining_quantity > 0:
            sort_bids(book.bids)
            bid = next(
                (o for o in book.bids if o.player_idx != incoming.player_idx),
                None,
            )
            if bid is None or bid.price_per_unit < incoming.price_per_unit:
                break
            qty = min(incoming.remaining_quantity, bid.remaining_quantity)
            trades.append(
                _execute_trade(
                    state,
                    buyer_order=bid,
                    seller_order=incoming,
                    maker_order=bid,
                    taker_order=incoming,
                    quantity=qty,
                    food_type=incoming.food_type,
                )
            )
            if bid.remaining_quantity == 0:
                _remove_from_book(book, bid.id)

    if incoming.remaining_quantity > 0:
        _add_to_book(book, incoming)
    else:
        incoming.status = "filled"

    return trades


def _matching_counterparty(
    own: Order,
    candidates: List[Order],
    *,
    own_is_bid: bool,
) -> Optional[Order]:
    for candidate in candidates:
        if candidate.player_idx == own.player_idx:
            continue
        if own_is_bid:
            if own.price_per_unit >= candidate.price_per_unit:
                return candidate
        elif candidate.price_per_unit >= own.price_per_unit:
            return candidate
    return None


def _best_crossed_pair(book: OrderBook) -> Optional[Tuple[Order, Order]]:
    sort_bids(book.bids)
    sort_asks(book.asks)

    best_bid = next(
        (
            bid
            for bid in book.bids
            if bid.remaining_quantity > 0
            and _matching_counterparty(bid, book.asks, own_is_bid=True) is not None
        ),
        None,
    )
    best_ask = next(
        (
            ask
            for ask in book.asks
            if ask.remaining_quantity > 0
            and _matching_counterparty(ask, book.bids, own_is_bid=False) is not None
        ),
        None,
    )

    if best_bid is None or best_ask is None:
        return None

    if best_bid.sequence <= best_ask.sequence:
        ask = _matching_counterparty(best_bid, book.asks, own_is_bid=True)
        if ask is None:
            return None
        return best_bid, ask

    bid = _matching_counterparty(best_ask, book.bids, own_is_bid=False)
    if bid is None:
        return None
    return bid, best_ask


def clear_books(state: GameState, foods: Iterable[str]) -> List[Trade]:
    """Clear already-posted crossed books without introducing new orders.

    Used by simultaneous order clearing: all players' actions are first applied
    to the books, then each food book is matched as a market-clearing phase.
    """
    trades: List[Trade] = []
    for food_type in foods:
        book = state.order_books[food_type]
        while True:
            pair = _best_crossed_pair(book)
            if pair is None:
                break
            bid, ask = pair
            qty = min(bid.remaining_quantity, ask.remaining_quantity)
            maker, taker = (bid, ask) if bid.sequence <= ask.sequence else (ask, bid)
            trades.append(
                _execute_trade(
                    state,
                    buyer_order=bid,
                    seller_order=ask,
                    maker_order=maker,
                    taker_order=taker,
                    quantity=qty,
                    food_type=food_type,
                )
            )
            if bid.remaining_quantity == 0:
                _remove_from_book(book, bid.id)
            if ask.remaining_quantity == 0:
                _remove_from_book(book, ask.id)
    return trades


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _count_open_orders(state: GameState, player_idx: int) -> int:
    n = 0
    for f in state.config.food_types:  # type: ignore[attr-defined]
        book = state.order_books[f]
        n += sum(1 for o in book.bids if o.player_idx == player_idx)
        n += sum(1 for o in book.asks if o.player_idx == player_idx)
    return n


def _validate_post(
    state: GameState,
    *,
    player_idx: int,
    side: str,
    food_type: str,
    quantity: int,
    price_per_unit: int,
    replacing_prior: Optional[Order] = None,
) -> None:
    config: GameConfig = state.config  # type: ignore[assignment]

    if state.phase != "active":
        raise OrderError("game_not_active", "Game is not active.")
    p = state.players[player_idx]
    if p.status != "alive":
        raise OrderError("player_dead", "Dead players cannot post orders.")
    if food_type not in config.food_types:
        raise OrderError("bad_food_type", f"unknown food: {food_type}")
    if side not in ("bid", "ask"):
        raise OrderError("bad_side", "side must be bid|ask")
    if not isinstance(quantity, int) or quantity < config.min_order_quantity:
        raise OrderError("bad_qty", f"quantity must be int ≥ {config.min_order_quantity}")
    if not isinstance(price_per_unit, int) or price_per_unit < config.min_price:
        raise OrderError("bad_price", f"price must be int ≥ {config.min_price}")
    if (
        replacing_prior is None
        and _count_open_orders(state, player_idx) >= config.max_open_orders_per_player
    ):
        raise OrderError(
            "too_many_open_orders",
            f"max {config.max_open_orders_per_player} open orders.",
        )
    if side == "bid":
        cost = price_per_unit * quantity
        reusable = (
            replacing_prior.price_per_unit * replacing_prior.remaining_quantity
            if replacing_prior is not None and replacing_prior.side == "bid"
            else 0
        )
        available = p.available_cash() + reusable
        if cost > available:
            raise OrderError(
                "insufficient_cash",
                f"need {cost} cash, have {available}.",
            )
    else:
        reusable = (
            replacing_prior.remaining_quantity
            if replacing_prior is not None and replacing_prior.side == "ask"
            else 0
        )
        avail = p.available(food_type) + reusable
        if quantity > avail:
            raise OrderError(
                "insufficient_inventory",
                f"need {quantity} {food_type}, have {avail}.",
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def post_order(
    state: GameState,
    *,
    player_idx: int,
    side: str,
    food_type: str,
    quantity: int,
    price_per_unit: int,
) -> Tuple[Order, List[Trade]]:
    _validate_post(
        state,
        player_idx=player_idx,
        side=side,
        food_type=food_type,
        quantity=quantity,
        price_per_unit=price_per_unit,
    )

    # Figgie-style: at most one resting order per (player, food, side). Posting
    # a new one replaces any prior open order on the same food+side. Mirrors
    # backend/src/market/orderActions.ts postOrder.
    book = state.order_books[food_type]
    pool = book.bids if side == "bid" else book.asks
    prior = next((o for o in pool if o.player_idx == player_idx), None)
    if prior is not None:
        _release_remaining(state, prior)
        prior.status = "cancelled"
        if prior.side == "bid":
            book.bids = [o for o in book.bids if o.id != prior.id]
        else:
            book.asks = [o for o in book.asks if o.id != prior.id]

    order = Order(
        id=state.next_order_id(),
        player_idx=player_idx,
        food_type=food_type,
        side=side,  # type: ignore[arg-type]
        price_per_unit=price_per_unit,
        original_quantity=quantity,
        remaining_quantity=quantity,
        status="open",
        created_at_tick=state.elapsed_seconds,
        sequence=state.next_sequence(),
    )

    _reserve_for_order(state, order)
    trades = _match_incoming(state, order)
    return order, trades


def post_order_without_matching(
    state: GameState,
    *,
    player_idx: int,
    side: str,
    food_type: str,
    quantity: int,
    price_per_unit: int,
    sequence: Optional[int] = None,
) -> Order:
    prior = _find_prior_player_order(
        state, player_idx=player_idx, side=side, food_type=food_type
    )
    _validate_post(
        state,
        player_idx=player_idx,
        side=side,
        food_type=food_type,
        quantity=quantity,
        price_per_unit=price_per_unit,
        replacing_prior=prior,
    )

    if prior is not None:
        _release_remaining(state, prior)
        prior.status = "cancelled"
        _remove_from_book(state.order_books[food_type], prior.id)

    order = Order(
        id=state.next_order_id(),
        player_idx=player_idx,
        food_type=food_type,
        side=side,  # type: ignore[arg-type]
        price_per_unit=price_per_unit,
        original_quantity=quantity,
        remaining_quantity=quantity,
        status="open",
        created_at_tick=state.elapsed_seconds,
        sequence=sequence if sequence is not None else state.next_sequence(),
    )

    _reserve_for_order(state, order)
    _add_to_book(state.order_books[food_type], order)
    return order


def validate_order_replacement(
    state: GameState,
    *,
    player_idx: int,
    side: str,
    food_type: str,
    quantity: int,
    price_per_unit: int,
) -> None:
    prior = _find_prior_player_order(
        state, player_idx=player_idx, side=side, food_type=food_type
    )
    _validate_post(
        state,
        player_idx=player_idx,
        side=side,
        food_type=food_type,
        quantity=quantity,
        price_per_unit=price_per_unit,
        replacing_prior=prior,
    )


def cancel_order(
    state: GameState, *, player_idx: int, order_id: int
) -> Order:
    if state.phase != "active":
        raise OrderError("game_not_active", "Game is not active.")
    order = find_order_anywhere(state, order_id)
    if order is None:
        raise OrderError("order_not_found", "no such order")
    if order.player_idx != player_idx:
        raise OrderError("not_owner", "not your order")
    if order.status in ("filled", "cancelled"):
        raise OrderError("order_finalized", "order is already finalized")
    _release_remaining(state, order)
    order.status = "cancelled"
    _remove_from_book(state.order_books[order.food_type], order.id)
    return order


def cancel_all_for_player(
    state: GameState, *, player_idx: int, release_reservations: bool = True
) -> List[Order]:
    cancelled: List[Order] = []
    for f in state.config.food_types:  # type: ignore[attr-defined]
        book = state.order_books[f]
        own = [o for o in book.bids if o.player_idx == player_idx] + [
            o for o in book.asks if o.player_idx == player_idx
        ]
        for o in own:
            if release_reservations:
                _release_remaining(state, o)
            o.status = "cancelled"
            _remove_from_book(book, o.id)
            cancelled.append(o)
    return cancelled


def cancel_oldest_for_player_side_food(
    state: GameState, *, player_idx: int, side: str, food_type: str
) -> Optional[Order]:
    """Convenience for the discrete action space: cancel this player's oldest open
    order on the given side+food. Returns None if no such order exists.
    """
    book = state.order_books[food_type]
    pool = book.bids if side == "bid" else book.asks
    own = [o for o in pool if o.player_idx == player_idx]
    if not own:
        return None
    own.sort(key=lambda o: o.sequence)  # earliest first
    target = own[0]
    return cancel_order(state, player_idx=player_idx, order_id=target.id)


# ---------------------------------------------------------------------------
# Take (IOC / market-take)
# ---------------------------------------------------------------------------
#
# Take walks the opposite side of the book best-price first and executes
# trades directly against resting makers. Unlike post_order it never leaves
# an order on the book — anything not filled at the current book depth is
# simply dropped (there is no reservation to release because a take never
# reserves cash or inventory of its own).


def _execute_take_trade(
    state: GameState,
    *,
    taker_idx: int,
    maker_order: Order,
    take_side: str,
    quantity: int,
    food_type: str,
) -> Trade:
    """Same effect as `_execute_trade` but the taker has no order (and no
    reservation), so we skip the taker-side reservation release and only
    release the maker's reservation. Cash/inventory are moved directly.
    """
    taker = state.players[taker_idx]
    maker = state.players[maker_order.player_idx]
    price = maker_order.price_per_unit
    total = price * quantity

    if take_side == "bid":  # taker buys, maker sells
        taker.cash -= total
        taker.inventory[food_type] += quantity
        taker.total_bought += quantity
        taker.cash_spent_on_trades += total

        maker.cash += total
        maker.inventory[food_type] -= quantity
        maker.total_sold += quantity
        maker.cash_from_trades += total
        # Maker had reserved `quantity` units of food for this order.
        cur = maker.reserved_inventory[food_type]
        maker.reserved_inventory[food_type] = max(0, cur - quantity)

        buyer_idx = taker_idx
        seller_idx = maker.idx
    else:  # take_side == "ask": taker sells, maker buys
        taker.cash += total
        taker.inventory[food_type] -= quantity
        taker.total_sold += quantity
        taker.cash_from_trades += total

        maker.cash -= total
        maker.inventory[food_type] += quantity
        maker.total_bought += quantity
        maker.cash_spent_on_trades += total
        # Maker had reserved `maker.price_per_unit * quantity` cash for this order.
        released = maker_order.price_per_unit * quantity
        maker.reserved_cash = max(0, maker.reserved_cash - released)

        buyer_idx = maker.idx
        seller_idx = taker_idx

    maker_order.remaining_quantity -= quantity
    maker_order.status = (
        "filled" if maker_order.remaining_quantity == 0 else "partially_filled"
    )

    trade = Trade(
        id=state.next_trade_id(),
        food_type=food_type,
        buyer_idx=buyer_idx,
        seller_idx=seller_idx,
        price_per_unit=price,
        quantity=quantity,
        total_price=total,
        maker_order_id=maker_order.id,
        # Sentinel: a take never has a taker order id. Downstream code that
        # filters trades by `taker_order_id == some_order_id` won't match.
        taker_order_id=-1,
        elapsed_second=state.elapsed_seconds,
    )
    state.trades.append(trade)
    return trade


def take_from_book(
    state: GameState,
    *,
    player_idx: int,
    side: str,
    food_type: str,
    quantity: int,
) -> List[Trade]:
    """Immediate-or-cancel take: match up to `quantity` units against the
    opposite side of the book (best price first), skipping the player's own
    orders. Stops early when the book empties or when the taker runs out of
    resources (cash for a bid, food for an ask).

    Returns the trades executed. No order is created, no reservation is held,
    and no unfilled remainder rests on the book.
    """
    if state.phase != "active":
        raise OrderError("game_not_active", "Game is not active.")
    if side not in ("bid", "ask"):
        raise OrderError("bad_side", f"unknown side: {side!r}")
    if quantity < 1:
        raise OrderError("bad_qty", "quantity must be ≥ 1")

    taker = state.players[player_idx]
    book = state.order_books[food_type]
    trades: List[Trade] = []
    remaining = quantity

    if side == "bid":
        while remaining > 0:
            sort_asks(book.asks)
            ask = next(
                (o for o in book.asks if o.player_idx != player_idx), None
            )
            if ask is None:
                break
            price = ask.price_per_unit
            affordable = (
                taker.available_cash() // price if price > 0 else remaining
            )
            take_qty = min(remaining, ask.remaining_quantity, affordable)
            if take_qty < 1:
                break
            trades.append(
                _execute_take_trade(
                    state,
                    taker_idx=player_idx,
                    maker_order=ask,
                    take_side="bid",
                    quantity=take_qty,
                    food_type=food_type,
                )
            )
            remaining -= take_qty
            if ask.remaining_quantity == 0:
                _remove_from_book(book, ask.id)
    else:  # ask (sell)
        while remaining > 0:
            sort_bids(book.bids)
            bid = next(
                (o for o in book.bids if o.player_idx != player_idx), None
            )
            if bid is None:
                break
            take_qty = min(
                remaining, bid.remaining_quantity, taker.available(food_type)
            )
            if take_qty < 1:
                break
            trades.append(
                _execute_take_trade(
                    state,
                    taker_idx=player_idx,
                    maker_order=bid,
                    take_side="ask",
                    quantity=take_qty,
                    food_type=food_type,
                )
            )
            remaining -= take_qty
            if bid.remaining_quantity == 0:
                _remove_from_book(book, bid.id)

    return trades
