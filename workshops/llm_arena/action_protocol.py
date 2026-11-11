"""Strict JSON wire protocol between an LLM seat and the unchanged env.

Adapted from the preliminary arena (jsafrata/calhunger-llm,
``arena/action_protocol.py``).  Two deliberate changes:

- The quantity/price bounds are imported from ``env.actions`` instead of
  being hardcoded, so the protocol always matches the research action
  space (v2 default: qty 1..5, price 1..50, 2049 flat actions).
- The schemas used for provider structured output live here next to the
  parser so there is exactly one definition of the wire format.

The parser is unchanged in spirit: exactly one bare JSON object, all
five keys mandatory, nulls required for unused fields, no wrapper text.
A parsed action is converted to the env's flat action index; whether the
env then *accepts* it (its original legality rules) is measured, not
enforced — illegal actions are silently no-op'd by the env and recorded
via ``StepInfo.actions_applied``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ._paths import bootstrap

bootstrap()

from env.actions import (  # noqa: E402
    ACTION_WAIT,
    MAX_PRICE,
    MAX_QTY,
    encode_cancel,
    encode_post,
    encode_take,
)


ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["wait", "cancel", "take", "post"],
        },
        "side": {
            "type": ["string", "null"],
            "enum": ["bid", "ask", None],
        },
        "food": {
            "type": ["string", "null"],
            "enum": ["A", "B", "C", "D", None],
        },
        "quantity": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": MAX_QTY,
        },
        "price": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": MAX_PRICE,
        },
    },
    "required": ["action", "side", "food", "quantity", "price"],
    "additionalProperties": False,
}


# Anthropic's structured-output validator accepts nullable unions, but
# enum members must live inside the matching string branch rather than
# beside a multi-type declaration.  Same wire format as ACTION_SCHEMA.
ANTHROPIC_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["wait", "cancel", "take", "post"],
        },
        "side": {
            "anyOf": [
                {"type": "string", "enum": ["bid", "ask"]},
                {"type": "null"},
            ]
        },
        "food": {
            "anyOf": [
                {"type": "string", "enum": ["A", "B", "C", "D"]},
                {"type": "null"},
            ]
        },
        "quantity": {
            "anyOf": [
                {
                    "type": "integer",
                    "description": f"An integer from 1 through {MAX_QTY}.",
                },
                {"type": "null"},
            ]
        },
        "price": {
            "anyOf": [
                {
                    "type": "integer",
                    "description": f"An integer from 1 through {MAX_PRICE}.",
                },
                {"type": "null"},
            ]
        },
    },
    "required": ["action", "side", "food", "quantity", "price"],
    "additionalProperties": False,
}


class DecisionError(ValueError):
    """The model did not return one valid action object."""


@dataclass(frozen=True)
class ParsedDecision:
    value: dict[str, Any]
    action_index: int


def extract_json_object(raw: str) -> Mapping[str, Any]:
    """Parse exactly one bare JSON object with no wrapper text."""
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DecisionError(
            f"response must be one bare JSON object: {exc.msg}"
        ) from None
    if not isinstance(value, dict):
        raise DecisionError("top-level response must be a JSON object")
    return value


def parse_decision(raw: str) -> ParsedDecision:
    value = dict(extract_json_object(raw))
    allowed = {"action", "side", "food", "quantity", "price"}
    extras = set(value) - allowed
    if extras:
        raise DecisionError(f"unexpected fields: {sorted(extras)}")
    missing_wire_fields = allowed - set(value)
    if missing_wire_fields:
        raise DecisionError(
            "response must include all five fields; missing: "
            f"{sorted(missing_wire_fields)}"
        )

    # Null fields are required on the wire, then removed from the
    # normalized action submitted to the unchanged environment.
    value = {key: item for key, item in value.items() if item is not None}

    kind = value.get("action")
    if kind not in {"wait", "cancel", "take", "post"}:
        raise DecisionError("action must be wait, cancel, take, or post")

    required_by_kind = {
        "wait": set(),
        "cancel": {"side", "food"},
        "take": {"side", "food", "quantity"},
        "post": {"side", "food", "quantity", "price"},
    }
    meaningful = {"action"} | required_by_kind[kind]
    unexpected = set(value) - meaningful
    if unexpected:
        raise DecisionError(f"fields not used by {kind}: {sorted(unexpected)}")
    missing = required_by_kind[kind] - set(value)
    if missing:
        raise DecisionError(f"missing fields for {kind}: {sorted(missing)}")

    if kind == "wait":
        return ParsedDecision(value={"action": "wait"}, action_index=ACTION_WAIT)

    side = value.get("side")
    food = value.get("food")
    if side not in {"bid", "ask"}:
        raise DecisionError("side must be bid (buy) or ask (sell)")
    if food not in {"A", "B", "C", "D"}:
        raise DecisionError("food must be A, B, C, or D")

    if kind == "cancel":
        normalized = {"action": kind, "side": side, "food": food}
        return ParsedDecision(normalized, encode_cancel(side, food))

    quantity = value.get("quantity")
    if (
        isinstance(quantity, bool)
        or not isinstance(quantity, int)
        or not 1 <= quantity <= MAX_QTY
    ):
        raise DecisionError(
            f"quantity must be an integer from 1 through {MAX_QTY}"
        )

    if kind == "take":
        normalized = {
            "action": kind,
            "side": side,
            "food": food,
            "quantity": quantity,
        }
        return ParsedDecision(normalized, encode_take(side, food, quantity))

    price = value.get("price")
    if (
        isinstance(price, bool)
        or not isinstance(price, int)
        or not 1 <= price <= MAX_PRICE
    ):
        raise DecisionError(
            f"price must be an integer from 1 through {MAX_PRICE}"
        )
    normalized = {
        "action": kind,
        "side": side,
        "food": food,
        "quantity": quantity,
        "price": price,
    }
    return ParsedDecision(normalized, encode_post(side, food, quantity, price))
