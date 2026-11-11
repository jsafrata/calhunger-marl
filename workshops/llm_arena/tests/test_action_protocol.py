"""The wire protocol accepts exactly the four action forms and nothing else."""
import json

import pytest

from llm_arena.action_protocol import (
    ACTION_SCHEMA,
    DecisionError,
    parse_decision,
)
from env.actions import (
    ACTION_WAIT,
    MAX_PRICE,
    MAX_QTY,
    encode_cancel,
    encode_post,
    encode_take,
)


def wire(action, side=None, food=None, quantity=None, price=None) -> str:
    return json.dumps(
        {
            "action": action,
            "side": side,
            "food": food,
            "quantity": quantity,
            "price": price,
        }
    )


def test_wait_maps_to_wait_index():
    parsed = parse_decision(wire("wait"))
    assert parsed.action_index == ACTION_WAIT
    assert parsed.value == {"action": "wait"}


def test_cancel_take_post_map_to_env_encoders():
    assert (
        parse_decision(wire("cancel", side="bid", food="A")).action_index
        == encode_cancel("bid", "A")
    )
    assert (
        parse_decision(wire("take", side="ask", food="C", quantity=3)).action_index
        == encode_take("ask", "C", 3)
    )
    assert (
        parse_decision(
            wire("post", side="ask", food="D", quantity=MAX_QTY, price=MAX_PRICE)
        ).action_index
        == encode_post("ask", "D", MAX_QTY, MAX_PRICE)
    )


def test_bounds_come_from_the_env_action_space():
    assert ACTION_SCHEMA["properties"]["quantity"]["maximum"] == MAX_QTY
    assert ACTION_SCHEMA["properties"]["price"]["maximum"] == MAX_PRICE
    with pytest.raises(DecisionError):
        parse_decision(wire("take", side="bid", food="A", quantity=MAX_QTY + 1))
    with pytest.raises(DecisionError):
        parse_decision(
            wire("post", side="bid", food="A", quantity=1, price=MAX_PRICE + 1)
        )


@pytest.mark.parametrize(
    "raw",
    [
        "the answer is " + wire("wait"),  # wrapper text
        "```json\n" + wire("wait") + "\n```",  # markdown fence
        "[" + wire("wait") + "]",  # not an object
        json.dumps({"action": "wait", "side": None, "food": None}),  # missing keys
        json.dumps(
            {
                "action": "wait",
                "side": None,
                "food": None,
                "quantity": None,
                "price": None,
                "note": "hi",
            }
        ),  # extra key
        wire("take", side="bid", food="A"),  # take without quantity
        wire("wait", side="bid"),  # wait with a meaningful field
        wire("post", side="up", food="A", quantity=1, price=1),  # bad side
        wire("take", side="bid", food="E", quantity=1),  # bad food
        wire("take", side="bid", food="A", quantity=True),  # bool quantity
    ],
)
def test_rejects_malformed_responses(raw):
    with pytest.raises(DecisionError):
        parse_decision(raw)
