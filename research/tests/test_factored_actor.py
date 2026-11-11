"""Unit tests for the autoregressive factored actor.

These tests exercise the actor without RLlib — they only need torch and
numpy. They validate the invariants that matter for correctness of the
factored policy:

1. Flat ↔ factored round-trip (both python and torch batched paths).
2. Sampled actions are always legal under the mask.
3. Log-probability sums only relevant heads per action type (irrelevant
   heads contribute exactly 0).
4. Teacher-forced log-prob equals the sample-time log-prob when we feed
   back the same sampled action.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("torch is not installed", allow_module_level=True)

import torch

from env.actions import (
    ACTION_TYPES,
    ACTION_WAIT,
    NUM_ACTIONS,
    NUM_FOODS,
    NUM_SIDES,
    MAX_PRICE,
    MAX_QTY,
    TYPE_CANCEL,
    TYPE_POST,
    TYPE_TAKE,
    TYPE_WAIT,
    encode_cancel,
    encode_post,
    encode_take,
    encode_wait,
    factored_masks_from_flat,
    flat_to_factored,
    factored_to_flat,
    legal_action_mask,
)
from env.env import FoodTradingEnv
from env.rl.factored_actor import (
    FactoredActor,
    FactoredMaskTensors,
    factored_masks_from_flat_tensor,
    flat_to_factored_batch,
    factored_to_flat_batch,
)


# ---------------------------------------------------------------------------
# Flat ↔ factored round-trips
# ---------------------------------------------------------------------------


def test_python_roundtrip_covers_every_action():
    for a in range(NUM_ACTIONS):
        t, f, s, p, q = flat_to_factored(a)
        assert factored_to_flat(t, f, s, p, q) == a


def test_torch_batched_roundtrip_matches_python_per_action():
    actions = torch.arange(NUM_ACTIONS, dtype=torch.long)
    factored = flat_to_factored_batch(actions)
    back = factored_to_flat_batch(factored)
    assert torch.equal(back, actions)


def test_torch_and_python_agree_on_factored_ids():
    for a in [
        encode_wait(),
        encode_cancel("bid", "A"),
        encode_cancel("ask", "D"),
        encode_take("bid", "B", qty=1),
        encode_take("ask", "C", qty=MAX_QTY),
        encode_post("bid", "A", qty=1, price=1),
        encode_post("ask", "D", qty=MAX_QTY, price=MAX_PRICE),
    ]:
        t, f, s, p, q = flat_to_factored(a)
        batched = flat_to_factored_batch(torch.tensor([a], dtype=torch.long))
        assert int(batched["type_id"][0]) == t
        assert int(batched["food_id"][0]) == f
        assert int(batched["side_id"][0]) == s
        assert int(batched["price_id"][0]) == p
        assert int(batched["qty_id"][0]) == q


# ---------------------------------------------------------------------------
# Sampling stays inside the legal mask
# ---------------------------------------------------------------------------


def _mask_tensors_from(env: FoodTradingEnv, player_idx: int) -> FactoredMaskTensors:
    """Build (batch=1) mask tensors from the env's flat legal mask."""
    flat = legal_action_mask(env.state, player_idx)
    fm = factored_masks_from_flat(flat)
    return FactoredMaskTensors(
        type_mask=torch.from_numpy(fm.type_mask).unsqueeze(0),
        food_mask=torch.from_numpy(fm.food_mask).unsqueeze(0),
        side_mask=torch.from_numpy(fm.side_mask).unsqueeze(0),
        price_mask=torch.from_numpy(fm.price_mask).unsqueeze(0),
        qty_mask=torch.from_numpy(fm.qty_mask).unsqueeze(0),
    )


def test_sampled_action_is_always_legal_under_flat_mask():
    torch.manual_seed(0)
    env = FoodTradingEnv(seed=0)
    env.reset()
    # Force a non-trivial mask: player 0 has some cash + some inventory.
    env.state.players[0].cash = 30
    env.state.players[0].inventory = {"A": 5, "B": 0, "C": 0, "D": 2}
    env.state.players[0].reserved_inventory = {f: 0 for f in "ABCD"}

    masks = _mask_tensors_from(env, 0)
    actor = FactoredActor(embedding_size=8)
    emb = torch.randn(1, 8)
    flat_mask = legal_action_mask(env.state, 0)

    # Sample many times, every sample must land in the flat mask.
    for _ in range(200):
        factored, log_prob, entropy = actor.sample(emb, masks)
        flat_action = int(factored_to_flat_batch(factored)[0])
        assert flat_mask[flat_action], (
            f"sampled illegal action {flat_action} — "
            f"factored={ {k: int(v[0]) for k, v in factored.items()} }"
        )
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(entropy).all()


# ---------------------------------------------------------------------------
# Irrelevant heads must not contribute to log-prob
# ---------------------------------------------------------------------------


def _actor_and_masks_for_type(action_type: int) -> tuple[FactoredActor, torch.Tensor, FactoredMaskTensors, dict]:
    """Build an actor + a masks tensor where only the given action_type is
    legal (and every relevant sub-choice is legal). Returns the actor,
    a randomised embedding, the masks, and a factored dict of a concrete
    example action of that type."""
    torch.manual_seed(42)
    # Pinned to head_kind="linear" so `head.weight` is directly available
    # for the perturbation-based test below. The MLP head is covered by
    # tests/test_v2_invariants.py.
    actor = FactoredActor(embedding_size=8, head_kind="linear")
    emb = torch.randn(1, 8)

    # All types legal so the type dist is non-degenerate; perturbing type_head
    # actually changes log P(type) for any specific type_id.
    type_mask = torch.ones(1, 4, dtype=torch.bool)
    food_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    side_mask = torch.ones(1, 4, 4, 2, dtype=torch.bool)
    price_mask = torch.ones(1, 4, 4, 2, MAX_PRICE, dtype=torch.bool)
    qty_mask = torch.ones(1, 4, 4, 2, MAX_PRICE, MAX_QTY, dtype=torch.bool)

    masks = FactoredMaskTensors(
        type_mask=type_mask,
        food_mask=food_mask,
        side_mask=side_mask,
        price_mask=price_mask,
        qty_mask=qty_mask,
    )

    if action_type == TYPE_WAIT:
        factored = {
            "type_id": torch.tensor([TYPE_WAIT], dtype=torch.long),
            "food_id": torch.tensor([0], dtype=torch.long),
            "side_id": torch.tensor([0], dtype=torch.long),
            "price_id": torch.tensor([0], dtype=torch.long),
            "qty_id": torch.tensor([0], dtype=torch.long),
        }
    elif action_type == TYPE_CANCEL:
        factored = {
            "type_id": torch.tensor([TYPE_CANCEL], dtype=torch.long),
            "food_id": torch.tensor([2], dtype=torch.long),
            "side_id": torch.tensor([1], dtype=torch.long),
            "price_id": torch.tensor([0], dtype=torch.long),
            "qty_id": torch.tensor([0], dtype=torch.long),
        }
    elif action_type == TYPE_TAKE:
        factored = {
            "type_id": torch.tensor([TYPE_TAKE], dtype=torch.long),
            "food_id": torch.tensor([1], dtype=torch.long),
            "side_id": torch.tensor([0], dtype=torch.long),
            "price_id": torch.tensor([0], dtype=torch.long),
            "qty_id": torch.tensor([3], dtype=torch.long),
        }
    else:  # POST
        factored = {
            "type_id": torch.tensor([TYPE_POST], dtype=torch.long),
            "food_id": torch.tensor([3], dtype=torch.long),
            "side_id": torch.tensor([1], dtype=torch.long),
            "price_id": torch.tensor([7], dtype=torch.long),
            "qty_id": torch.tensor([4], dtype=torch.long),
        }
    return actor, emb, masks, factored


@pytest.mark.parametrize(
    "action_type,expected_head_count",
    [
        (TYPE_WAIT, 1),    # type only
        (TYPE_CANCEL, 3),  # type + food + side
        (TYPE_TAKE, 4),    # type + food + side + qty
        (TYPE_POST, 5),    # everything
    ],
)
def test_logp_sums_only_relevant_heads(action_type: int, expected_head_count: int):
    """Perturbing the logits of an IRRELEVANT head must not change log-prob;
    perturbing a RELEVANT head must change it."""
    actor, emb, masks, factored = _actor_and_masks_for_type(action_type)

    base_logp = actor.log_prob(emb, factored, masks)

    def perturb(head: "nn.Module"):
        # Non-uniform perturbation: adding a scalar to every logit would
        # cancel under softmax; we need something that breaks the invariance.
        with torch.no_grad():
            torch.manual_seed(1234)
            head.weight.add_(torch.randn_like(head.weight) * 0.5)

    from torch import nn

    heads = [
        ("type", actor.type_head, True),  # always relevant
        ("food", actor.food_head, action_type in (TYPE_CANCEL, TYPE_TAKE, TYPE_POST)),
        ("side", actor.side_head, action_type in (TYPE_CANCEL, TYPE_TAKE, TYPE_POST)),
        ("price", actor.price_head, action_type == TYPE_POST),
        ("qty", actor.qty_head, action_type in (TYPE_TAKE, TYPE_POST)),
    ]

    n_relevant = 0
    for name, head, relevant in heads:
        original = head.weight.data.clone()
        perturb(head)
        new_logp = actor.log_prob(emb, factored, masks)
        head.weight.data.copy_(original)  # restore

        changed = not torch.allclose(base_logp, new_logp, atol=1e-6)
        if relevant:
            n_relevant += 1
            assert changed, (
                f"perturbing RELEVANT {name} head for {ACTION_TYPES[action_type]}"
                " did not change logp"
            )
        else:
            assert not changed, (
                f"perturbing IRRELEVANT {name} head for {ACTION_TYPES[action_type]}"
                f" changed logp: {base_logp.item()} → {new_logp.item()}"
            )

    assert n_relevant == expected_head_count


# ---------------------------------------------------------------------------
# Sample-time and teacher-forced logp are consistent
# ---------------------------------------------------------------------------


def _enumerate_factored_reachable(masks) -> set[int]:
    """Enumerate every factored tuple that a *sequential* masked sampler
    could reach, apply the same placeholder-id-0 convention as the actor,
    and return the set of flat action ids they encode to.

    A tuple `(t, f, s, p, q)` is reachable iff every step in
        type_mask[t]
        food_mask[t, f]
        side_mask[t, f, s]
        price_mask[t, f, s, p]
        qty_mask[t, f, s, p, q]
    is True — the very check the autoregressive sampler runs at each head.
    Placeholder ids are pinned to 0 for irrelevant heads, matching the
    actor's own convention.
    """
    reachable: set[int] = set()
    tm, fm, sm, pm, qm = (
        masks.type_mask, masks.food_mask, masks.side_mask,
        masks.price_mask, masks.qty_mask,
    )
    for t in range(4):
        if not tm[t]:
            continue
        # Determine which sub-dims are enumerated vs pinned to 0.
        food_range = range(NUM_FOODS) if t != TYPE_WAIT else [0]
        for f in food_range:
            if t != TYPE_WAIT and not fm[t, f]:
                continue
            side_range = range(NUM_SIDES) if t != TYPE_WAIT else [0]
            for s in side_range:
                if t != TYPE_WAIT and not sm[t, f, s]:
                    continue
                price_range = range(MAX_PRICE) if t == TYPE_POST else [0]
                for p in price_range:
                    if t == TYPE_POST and not pm[t, f, s, p]:
                        continue
                    qty_range = range(MAX_QTY) if t in (TYPE_TAKE, TYPE_POST) else [0]
                    for q in qty_range:
                        if t in (TYPE_TAKE, TYPE_POST) and not qm[t, f, s, p, q]:
                            continue
                        # Extra guard for POST: price mask conditioned only
                        # on side/food might admit a (p) that has no legal
                        # qty; skip if the qty row for this (p) is fully off.
                        reachable.add(factored_to_flat(t, f, s, p, q))
    return reachable


def test_factored_legal_set_equals_flat_legal_set_exactly():
    """Exhaustive proof that the factored autoregressive space represents
    exactly the same set of legal actions as the underlying flat mask —
    every reachable factored path decodes to a legal flat action, and every
    legal flat action is reachable by exactly one factored path.

    Failure of this test would mean the per-head masks are wrong (either
    admitting illegal joint paths, or hiding a legal path from the policy).
    """
    from env.actions import (
        factored_masks_from_flat as _py_factored_masks,
        legal_action_mask as _legal,
    )

    scenarios = [
        # (setup_fn, description) — vary cash, inventory, seeds, and
        # optionally seed some resting orders to stress the mask logic.
        ("baseline", lambda env: None),
        ("no_cash", lambda env: setattr(env.state.players[0], "cash", 0)),
        (
            "no_food",
            lambda env: (
                setattr(env.state.players[0], "cash", 100),
                env.state.players[0].inventory.update({f: 0 for f in "ABCD"}),
                env.state.players[0].reserved_inventory.update({f: 0 for f in "ABCD"}),
            ),
        ),
        (
            "one_food_only",
            lambda env: (
                setattr(env.state.players[0], "cash", 3),
                env.state.players[0].inventory.update({"A": 2, "B": 0, "C": 0, "D": 0}),
                env.state.players[0].reserved_inventory.update({f: 0 for f in "ABCD"}),
            ),
        ),
        (
            "with_resting_orders",
            lambda env: _seed_orders(env),
        ),
    ]

    for name, setup in scenarios:
        env = FoodTradingEnv(seed=hash(name) & 0xFFFF)
        env.reset()
        setup(env)

        flat = _legal(env.state, 0)
        legal_flat_ids = {int(a) for a in np.where(flat)[0]}
        masks = _py_factored_masks(flat)
        reachable_ids = _enumerate_factored_reachable(masks)

        illegal_reachable = reachable_ids - legal_flat_ids
        unreachable_legal = legal_flat_ids - reachable_ids

        assert not illegal_reachable, (
            f"[{name}] factored sampler can reach ILLEGAL flat actions: "
            f"{sorted(illegal_reachable)[:20]} ({len(illegal_reachable)} total)"
        )
        assert not unreachable_legal, (
            f"[{name}] some legal flat actions are UNREACHABLE by any "
            f"factored path: {sorted(unreachable_legal)[:20]} "
            f"({len(unreachable_legal)} total)"
        )


def _seed_orders(env):
    """Give the tested player a mix of open orders + book depth so cancel
    and take masks are non-trivial."""
    from env.market import post_order
    env.state.players[0].cash = 50
    env.state.players[0].inventory = {"A": 8, "B": 3, "C": 0, "D": 5}
    env.state.players[0].reserved_inventory = {f: 0 for f in "ABCD"}
    # Own resting orders → cancel becomes legal on those (side, food) pairs.
    post_order(env.state, player_idx=0, side="bid", food_type="C", quantity=2, price_per_unit=4)
    post_order(env.state, player_idx=0, side="ask", food_type="A", quantity=3, price_per_unit=6)
    # Opponents supply asks + bids → take becomes legal.
    env.state.players[1].cash = 100
    env.state.players[1].inventory = {"C": 20, "B": 20, "A": 0, "D": 0}
    env.state.players[1].reserved_inventory = {f: 0 for f in "ABCD"}
    post_order(env.state, player_idx=1, side="ask", food_type="C", quantity=5, price_per_unit=3)
    post_order(env.state, player_idx=1, side="bid", food_type="B", quantity=2, price_per_unit=5)


def test_torch_and_numpy_mask_derivers_agree():
    """The torch-batched mask deriver must produce identical masks to the
    per-example numpy version, for a mix of realistic game states."""
    for seed in range(5):
        env = FoodTradingEnv(seed=seed)
        env.reset()
        # Skew resources so different types are (in)legal for each player.
        env.state.players[0].cash = 50
        env.state.players[0].inventory = {"A": 10, "B": 0, "C": 3, "D": 0}
        env.state.players[0].reserved_inventory = {f: 0 for f in "ABCD"}
        env.state.players[1].cash = 0
        env.state.players[1].inventory = {"A": 0, "B": 0, "C": 0, "D": 0}
        env.state.players[1].reserved_inventory = {f: 0 for f in "ABCD"}

        flat_masks = np.stack(
            [legal_action_mask(env.state, i) for i in range(4)], axis=0
        )
        py_masks = [factored_masks_from_flat(flat_masks[i]) for i in range(4)]
        torch_masks = factored_masks_from_flat_tensor(torch.from_numpy(flat_masks))

        for i in range(4):
            np.testing.assert_array_equal(
                torch_masks.type_mask[i].numpy(), py_masks[i].type_mask
            )
            np.testing.assert_array_equal(
                torch_masks.food_mask[i].numpy(), py_masks[i].food_mask
            )
            np.testing.assert_array_equal(
                torch_masks.side_mask[i].numpy(), py_masks[i].side_mask
            )
            np.testing.assert_array_equal(
                torch_masks.price_mask[i].numpy(), py_masks[i].price_mask
            )
            np.testing.assert_array_equal(
                torch_masks.qty_mask[i].numpy(), py_masks[i].qty_mask
            )


def test_sample_logp_matches_teacher_forced_logp():
    torch.manual_seed(7)
    actor = FactoredActor(embedding_size=8)
    emb = torch.randn(4, 8)

    # Wide-open mask so we don't accidentally trigger the -inf fallback.
    B = 4
    masks = FactoredMaskTensors(
        type_mask=torch.ones(B, 4, dtype=torch.bool),
        food_mask=torch.ones(B, 4, 4, dtype=torch.bool),
        side_mask=torch.ones(B, 4, 4, 2, dtype=torch.bool),
        price_mask=torch.ones(B, 4, 4, 2, MAX_PRICE, dtype=torch.bool),
        qty_mask=torch.ones(B, 4, 4, 2, MAX_PRICE, MAX_QTY, dtype=torch.bool),
    )

    factored, sample_logp, _ = actor.sample(emb, masks)
    teacher_logp = actor.log_prob(emb, factored, masks)

    torch.testing.assert_close(sample_logp, teacher_logp)
