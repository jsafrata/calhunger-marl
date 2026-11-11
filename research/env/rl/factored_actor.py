"""Autoregressive, masked, factored policy head for the food-trading env.

The flat Discrete(4969) action space factors naturally as

    action = (type, food, side, price, quantity)

with `type ∈ {wait, cancel, take, post}` and only the fields relevant to
each type contributing to the action's log-probability. This module
provides the neural machinery (heads + sampling + log-prob) for that
factorization; the wider RLModule (see rllib_module.py) wraps it with a
shared torso and a critic.

Factorization
-------------
The five heads are conditionally dependent in the order

    type → food → side → price → quantity.

Each head sees the shared torso embedding plus embeddings of the
previously-sampled ids (autoregressive conditioning).

Per-type relevance
------------------
Not every head is used for every action type. During sampling we set
the placeholder ids of unused heads to 0 and add 0 to log-prob for them.
During training we compute log-prob per head and mask the irrelevant
head contributions to 0 before summing.

    wait   : type only
    cancel : type + food + side
    take   : type + food + side + quantity
    post   : type + food + side + price + quantity

Masking
-------
Each head's logits are masked with `-inf` where the corresponding
`FactoredMasks` field says the option is illegal. `_masked_categorical`
guards against the "all-illegal" edge case by falling back to a uniform
distribution over the mask itself. In practice `type_mask` always
contains at least WAIT so the fallback is a defensive check, not the
common path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from ..actions import (
    MAX_PRICE,
    MAX_QTY,
    NUM_ACTION_TYPES,
    NUM_FOODS,
    NUM_SIDES,
    TYPE_CANCEL,
    TYPE_POST,
    TYPE_TAKE,
    TYPE_WAIT,
)

# ---------------------------------------------------------------------------
# Per-type relevance masks. Shape (NUM_ACTION_TYPES,). True where a given
# head contributes to log-probability for that action type.
# ---------------------------------------------------------------------------

_FOOD_RELEVANT = torch.tensor(
    [False, True, True, True], dtype=torch.bool
)  # wait, cancel, take, post
_SIDE_RELEVANT = torch.tensor(
    [False, True, True, True], dtype=torch.bool
)
_PRICE_RELEVANT = torch.tensor(
    [False, False, False, True], dtype=torch.bool
)  # only POST uses price
_QTY_RELEVANT = torch.tensor(
    [False, False, True, True], dtype=torch.bool
)  # take and post use qty


# ---------------------------------------------------------------------------
# Factored-mask tensor container (torch mirror of actions.FactoredMasks).
# ---------------------------------------------------------------------------


@dataclass
class FactoredMaskTensors:
    """Torch tensors of the same shapes as ``actions.FactoredMasks``, on the
    device that the actor will use them from. All fields are bool tensors."""

    type_mask: torch.Tensor
    food_mask: torch.Tensor
    side_mask: torch.Tensor
    price_mask: torch.Tensor
    qty_mask: torch.Tensor


# ---------------------------------------------------------------------------
# The FactoredActor module
# ---------------------------------------------------------------------------


HEAD_HIDDEN_SIZE: int = 128
DEFAULT_HEAD_KIND: str = "mlp"


def _build_head(input_size: int, output_size: int, kind: str) -> nn.Module:
    """Build one autoregressive head.

    ``kind='linear'`` returns a single ``nn.Linear`` (v1 default, kept for
    ablation). ``kind='mlp'`` inserts one hidden layer:
    ``Linear(input, 128) → LayerNorm → SiLU → Linear(128, output)`` — the
    v2 default. Both variants take the same conditioning input and emit
    logits of shape ``(B, output_size)``; the caller applies the mask
    identically for either shape.
    """
    if kind == "linear":
        return nn.Linear(input_size, output_size)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(input_size, HEAD_HIDDEN_SIZE),
            nn.LayerNorm(HEAD_HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(HEAD_HIDDEN_SIZE, output_size),
        )
    raise ValueError(f"unknown head_kind {kind!r}; expected 'linear' or 'mlp'")


class FactoredActor(nn.Module):
    """Autoregressive masked head for the food-trading action space.

    Args:
        embedding_size: dim of the shared torso embedding fed in for each
            example.
        cond_dim: size of the embedding used to convey each sampled id to
            downstream heads. Kept small; 16 is a defensible default.
        head_kind: ``"mlp"`` (v2 default) inserts a 128-unit hidden layer
            with LayerNorm + SiLU on every head; ``"linear"`` uses a
            single linear projection (v1). Autoregressive conditioning,
            masks, and the zero-log-prob contribution from irrelevant
            heads are the same for both.
    """

    def __init__(
        self,
        embedding_size: int,
        *,
        cond_dim: int = 16,
        head_kind: str = DEFAULT_HEAD_KIND,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.cond_dim = cond_dim
        self.head_kind = head_kind

        # Heads. Each head produces logits for its own dimension.
        self.type_head = _build_head(embedding_size, NUM_ACTION_TYPES, head_kind)
        self.food_head = _build_head(embedding_size + cond_dim, NUM_FOODS, head_kind)
        self.side_head = _build_head(
            embedding_size + 2 * cond_dim, NUM_SIDES, head_kind
        )
        self.price_head = _build_head(
            embedding_size + 3 * cond_dim, MAX_PRICE, head_kind
        )
        self.qty_head = _build_head(
            embedding_size + 4 * cond_dim, MAX_QTY, head_kind
        )

        # Conditioning embeddings — one per sampled dimension. All fed
        # together into downstream heads.
        self.type_embed = nn.Embedding(NUM_ACTION_TYPES, cond_dim)
        self.food_embed = nn.Embedding(NUM_FOODS, cond_dim)
        self.side_embed = nn.Embedding(NUM_SIDES, cond_dim)
        self.price_embed = nn.Embedding(MAX_PRICE, cond_dim)

        # Diagnostic stash — refreshed (detached) on every sample() /
        # teacher-forced pass. The joint-entropy scalar RLlib logs hides
        # which head still carries the entropy (a 50-way price head can
        # dominate it); `eval/head_entropy_diagnostic.py` reads this to
        # answer that question per checkpoint. Not part of any gradient.
        self.last_head_stats: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Sampling (used at exploration & inference time)
    # ------------------------------------------------------------------

    def sample(
        self,
        emb: torch.Tensor,
        masks: FactoredMaskTensors,
        *,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Sample an action autoregressively.

        Args:
            emb:   torso embedding, shape (B, embedding_size)
            masks: per-head bool masks (each shape (B, ...))
            deterministic: if True, take argmax at each head instead of
                sampling. Useful for greedy-inference rollouts.

        Returns:
            factored: dict with keys type_id/food_id/side_id/price_id/qty_id,
                each shape (B,) long.
            log_prob: shape (B,) — log-probability of the sampled action
                summing only the RELEVANT heads for its sampled type.
            entropy: shape (B,) — sum of per-head entropies, only over
                relevant heads given the sampled type.
        """
        batch_size = emb.shape[0]
        device = emb.device

        # ---- type ----
        type_logits = self.type_head(emb)
        type_logits = _apply_mask(type_logits, masks.type_mask)
        type_dist = _safe_categorical(type_logits)
        type_id = (
            type_dist.probs.argmax(dim=-1) if deterministic else type_dist.sample()
        )
        type_logp = type_dist.log_prob(type_id)
        type_entropy = type_dist.entropy()

        # Precompute per-row relevance masks — used both to zero out
        # irrelevant sampled ids *before* they're fed into downstream
        # heads' embeddings (so sample() and _teacher_forced() see the
        # same conditioning inputs) AND to zero out irrelevant heads'
        # log-prob / entropy contributions at the end.
        rel_food = _FOOD_RELEVANT.to(device)[type_id]
        rel_side = _SIDE_RELEVANT.to(device)[type_id]
        rel_price = _PRICE_RELEVANT.to(device)[type_id]
        rel_qty = _QTY_RELEVANT.to(device)[type_id]

        # ---- food ----
        type_e = self.type_embed(type_id)
        food_logits = self.food_head(torch.cat([emb, type_e], dim=-1))
        food_mask = _gather_along_batch(masks.food_mask, type_id)  # (B, NUM_FOODS)
        food_logits = _apply_mask(food_logits, food_mask)
        food_dist = _safe_categorical(food_logits)
        food_id = (
            food_dist.probs.argmax(dim=-1) if deterministic else food_dist.sample()
        )
        food_logp = food_dist.log_prob(food_id)
        food_entropy = food_dist.entropy()
        # Zero the id BEFORE using it downstream — matches what
        # _teacher_forced() sees when it receives the factored dict.
        food_id = torch.where(rel_food, food_id, torch.zeros_like(food_id))

        # ---- side ----
        food_e = self.food_embed(food_id)
        side_logits = self.side_head(torch.cat([emb, type_e, food_e], dim=-1))
        side_mask = _gather2(masks.side_mask, type_id, food_id)  # (B, NUM_SIDES)
        side_logits = _apply_mask(side_logits, side_mask)
        side_dist = _safe_categorical(side_logits)
        side_id = (
            side_dist.probs.argmax(dim=-1) if deterministic else side_dist.sample()
        )
        side_logp = side_dist.log_prob(side_id)
        side_entropy = side_dist.entropy()
        side_id = torch.where(rel_side, side_id, torch.zeros_like(side_id))

        # ---- price ----
        side_e = self.side_embed(side_id)
        price_logits = self.price_head(
            torch.cat([emb, type_e, food_e, side_e], dim=-1)
        )
        price_mask = _gather3(
            masks.price_mask, type_id, food_id, side_id
        )  # (B, MAX_PRICE)
        price_logits = _apply_mask(price_logits, price_mask)
        price_dist = _safe_categorical(price_logits)
        price_id = (
            price_dist.probs.argmax(dim=-1)
            if deterministic
            else price_dist.sample()
        )
        price_logp = price_dist.log_prob(price_id)
        price_entropy = price_dist.entropy()
        price_id = torch.where(rel_price, price_id, torch.zeros_like(price_id))

        # ---- qty ----
        price_e = self.price_embed(price_id)
        qty_logits = self.qty_head(
            torch.cat([emb, type_e, food_e, side_e, price_e], dim=-1)
        )
        qty_mask = _gather4(
            masks.qty_mask, type_id, food_id, side_id, price_id
        )  # (B, MAX_QTY)
        qty_logits = _apply_mask(qty_logits, qty_mask)
        qty_dist = _safe_categorical(qty_logits)
        qty_id = (
            qty_dist.probs.argmax(dim=-1) if deterministic else qty_dist.sample()
        )
        qty_logp = qty_dist.log_prob(qty_id)
        qty_entropy = qty_dist.entropy()
        qty_id = torch.where(rel_qty, qty_id, torch.zeros_like(qty_id))

        # Sum log-probs only across relevant heads for each example's type.
        log_prob = (
            type_logp
            + torch.where(rel_food, food_logp, torch.zeros_like(food_logp))
            + torch.where(rel_side, side_logp, torch.zeros_like(side_logp))
            + torch.where(rel_price, price_logp, torch.zeros_like(price_logp))
            + torch.where(rel_qty, qty_logp, torch.zeros_like(qty_logp))
        )

        entropy = (
            type_entropy
            + torch.where(rel_food, food_entropy, torch.zeros_like(food_entropy))
            + torch.where(rel_side, side_entropy, torch.zeros_like(side_entropy))
            + torch.where(
                rel_price, price_entropy, torch.zeros_like(price_entropy)
            )
            + torch.where(rel_qty, qty_entropy, torch.zeros_like(qty_entropy))
        )

        self._stash_head_stats(
            type_id=type_id,
            type_entropy=type_entropy,
            food_entropy=food_entropy,
            side_entropy=side_entropy,
            price_entropy=price_entropy,
            qty_entropy=qty_entropy,
            rel_food=rel_food,
            rel_side=rel_side,
            rel_price=rel_price,
            rel_qty=rel_qty,
        )

        return (
            {
                "type_id": type_id,
                "food_id": food_id,
                "side_id": side_id,
                "price_id": price_id,
                "qty_id": qty_id,
            },
            log_prob,
            entropy,
        )

    def _stash_head_stats(self, **tensors: torch.Tensor) -> None:
        self.last_head_stats = {
            name: tensor.detach() for name, tensor in tensors.items()
        }

    # ------------------------------------------------------------------
    # Teacher-forced log-prob (used at training time)
    # ------------------------------------------------------------------

    def log_prob(
        self,
        emb: torch.Tensor,
        factored: Dict[str, torch.Tensor],
        masks: FactoredMaskTensors,
    ) -> torch.Tensor:
        """Log-probability of the given factored action under the current
        policy, teacher-forcing the previously-sampled ids into each
        subsequent head. Only relevant heads contribute for each type."""
        _, log_prob, _ = self._teacher_forced(emb, factored, masks)
        return log_prob

    def entropy(
        self,
        emb: torch.Tensor,
        factored: Dict[str, torch.Tensor],
        masks: FactoredMaskTensors,
    ) -> torch.Tensor:
        """**Single-sample estimator** of the autoregressive joint entropy.

        For each row, returns
            H(type) + Σ_i H(child_i | teacher-forced prefix from this row)
        where each child's entropy is evaluated at the specific prefix
        that this row's `factored` dict encodes.

        Averaged over a batch of samples this is an unbiased estimator of
        the true joint entropy — the outer expectation over prefixes is
        replaced by the empirical average across rows. Per row it is
        *stochastic*: two different sampled prefixes with the same
        observation give different per-row entropies.

        If you need the exact joint entropy (e.g. for an entropy
        regularizer that must be non-noisy), compute it by expanding the
        expectation:
            H(a) = H(type)
                 + Σ_t P(type=t) · [ H(food|t) + Σ_f P(food=f|t) · [ ... ] ]
        which is tractable at our cardinalities but not what this method
        does. PPO's entropy bonus tolerates the noisier single-sample form.
        """
        _, _, entropy = self._teacher_forced(emb, factored, masks)
        return entropy

    def log_prob_and_entropy(
        self,
        emb: torch.Tensor,
        factored: Dict[str, torch.Tensor],
        masks: FactoredMaskTensors,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, log_prob, entropy = self._teacher_forced(emb, factored, masks)
        return log_prob, entropy

    def _teacher_forced(
        self,
        emb: torch.Tensor,
        factored: Dict[str, torch.Tensor],
        masks: FactoredMaskTensors,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        device = emb.device
        type_id = factored["type_id"].to(device).long()
        food_id = factored["food_id"].to(device).long()
        side_id = factored["side_id"].to(device).long()
        price_id = factored["price_id"].to(device).long()
        qty_id = factored["qty_id"].to(device).long()

        # ---- type ----
        type_logits = self.type_head(emb)
        type_logits = _apply_mask(type_logits, masks.type_mask)
        type_dist = _safe_categorical(type_logits)
        type_logp = type_dist.log_prob(type_id)
        type_entropy = type_dist.entropy()

        # ---- food ----
        type_e = self.type_embed(type_id)
        food_logits = self.food_head(torch.cat([emb, type_e], dim=-1))
        food_mask = _gather_along_batch(masks.food_mask, type_id)
        food_logits = _apply_mask(food_logits, food_mask)
        food_dist = _safe_categorical(food_logits)
        food_logp = food_dist.log_prob(food_id)
        food_entropy = food_dist.entropy()

        # ---- side ----
        food_e = self.food_embed(food_id)
        side_logits = self.side_head(torch.cat([emb, type_e, food_e], dim=-1))
        side_mask = _gather2(masks.side_mask, type_id, food_id)
        side_logits = _apply_mask(side_logits, side_mask)
        side_dist = _safe_categorical(side_logits)
        side_logp = side_dist.log_prob(side_id)
        side_entropy = side_dist.entropy()

        # ---- price ----
        side_e = self.side_embed(side_id)
        price_logits = self.price_head(
            torch.cat([emb, type_e, food_e, side_e], dim=-1)
        )
        price_mask = _gather3(masks.price_mask, type_id, food_id, side_id)
        price_logits = _apply_mask(price_logits, price_mask)
        price_dist = _safe_categorical(price_logits)
        price_logp = price_dist.log_prob(price_id)
        price_entropy = price_dist.entropy()

        # ---- qty ----
        price_e = self.price_embed(price_id)
        qty_logits = self.qty_head(
            torch.cat([emb, type_e, food_e, side_e, price_e], dim=-1)
        )
        qty_mask = _gather4(masks.qty_mask, type_id, food_id, side_id, price_id)
        qty_logits = _apply_mask(qty_logits, qty_mask)
        qty_dist = _safe_categorical(qty_logits)
        qty_logp = qty_dist.log_prob(qty_id)
        qty_entropy = qty_dist.entropy()

        rel_food = _FOOD_RELEVANT.to(device)[type_id]
        rel_side = _SIDE_RELEVANT.to(device)[type_id]
        rel_price = _PRICE_RELEVANT.to(device)[type_id]
        rel_qty = _QTY_RELEVANT.to(device)[type_id]
        zero = torch.zeros_like(type_logp)

        log_prob = (
            type_logp
            + torch.where(rel_food, food_logp, zero)
            + torch.where(rel_side, side_logp, zero)
            + torch.where(rel_price, price_logp, zero)
            + torch.where(rel_qty, qty_logp, zero)
        )
        entropy = (
            type_entropy
            + torch.where(rel_food, food_entropy, zero)
            + torch.where(rel_side, side_entropy, zero)
            + torch.where(rel_price, price_entropy, zero)
            + torch.where(rel_qty, qty_entropy, zero)
        )
        self._stash_head_stats(
            type_id=type_id,
            type_entropy=type_entropy,
            food_entropy=food_entropy,
            side_entropy=side_entropy,
            price_entropy=price_entropy,
            qty_entropy=qty_entropy,
            rel_food=rel_food,
            rel_side=rel_side,
            rel_price=rel_price,
            rel_qty=rel_qty,
        )
        heads = {
            "type_id": type_id,
            "food_id": food_id,
            "side_id": side_id,
            "price_id": price_id,
            "qty_id": qty_id,
        }
        return heads, log_prob, entropy


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Add -inf to illegal logits so the softmax puts 0 mass on them."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return logits.masked_fill(~mask, float("-inf"))


def _safe_categorical(logits: torch.Tensor) -> Categorical:
    """Build a Categorical, defending against rows where every entry is
    -inf (which would make softmax NaN). If a whole row is masked out we
    fall back to a uniform distribution over the row — a defensive path;
    in practice the WAIT bit in `type_mask` guarantees at least one legal
    option upstream, but nested heads for exotic corners can still hit
    an empty mask (e.g., PRICE conditioned on an all-illegal food)."""
    is_finite = torch.isfinite(logits)
    all_masked = ~is_finite.any(dim=-1, keepdim=True)
    fallback = torch.zeros_like(logits)
    safe_logits = torch.where(all_masked, fallback, logits)
    return Categorical(logits=safe_logits)


def _gather_along_batch(
    mask: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """Given a mask of shape (B, D0, ...) and index of shape (B,), returns
    mask[b, index[b], ...] — the row of D0 selected per batch element."""
    b = index.shape[0]
    return mask[torch.arange(b, device=mask.device), index]


def _gather2(mask: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor) -> torch.Tensor:
    b = i0.shape[0]
    return mask[torch.arange(b, device=mask.device), i0, i1]


def _gather3(
    mask: torch.Tensor,
    i0: torch.Tensor,
    i1: torch.Tensor,
    i2: torch.Tensor,
) -> torch.Tensor:
    b = i0.shape[0]
    return mask[torch.arange(b, device=mask.device), i0, i1, i2]


def _gather4(
    mask: torch.Tensor,
    i0: torch.Tensor,
    i1: torch.Tensor,
    i2: torch.Tensor,
    i3: torch.Tensor,
) -> torch.Tensor:
    b = i0.shape[0]
    return mask[torch.arange(b, device=mask.device), i0, i1, i2, i3]


# ---------------------------------------------------------------------------
# Batched conversion between flat action ids and factored dicts.
# ---------------------------------------------------------------------------

from ..actions import (  # noqa: E402  — after class defs to avoid partial init
    ACTION_WAIT as _ACTION_WAIT,
    CANCEL_BASE as _CANCEL_BASE,
    POST_BASE as _POST_BASE,
    TAKE_BASE as _TAKE_BASE,
)


def flat_to_factored_batch(actions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Torch-side batched version of ``actions.flat_to_factored``.

    Args:
        actions: long tensor of shape (B,) with flat action ids in [0,
            NUM_ACTIONS).

    Returns:
        dict with the five head-id tensors, each shape (B,) long. Uses
        placeholder 0 for irrelevant ids (matches the Python helper).
    """
    a = actions.long()
    b = a.shape[0]
    device = a.device
    zero = torch.zeros_like(a)

    # WAIT rows.
    is_wait = a == _ACTION_WAIT
    # CANCEL rows.
    is_cancel = (a >= _CANCEL_BASE) & (a < _TAKE_BASE)
    # TAKE rows.
    is_take = (a >= _TAKE_BASE) & (a < _POST_BASE)
    # POST rows (the rest).
    is_post = a >= _POST_BASE

    type_id = torch.zeros(b, dtype=torch.long, device=device)
    type_id = torch.where(is_cancel, torch.full_like(a, TYPE_CANCEL), type_id)
    type_id = torch.where(is_take, torch.full_like(a, TYPE_TAKE), type_id)
    type_id = torch.where(is_post, torch.full_like(a, TYPE_POST), type_id)

    # CANCEL: rel = a - CANCEL_BASE; side = rel // NUM_FOODS; food = rel % NUM_FOODS
    cancel_rel = a - _CANCEL_BASE
    cancel_side = torch.where(is_cancel, cancel_rel // NUM_FOODS, zero)
    cancel_food = torch.where(is_cancel, cancel_rel % NUM_FOODS, zero)

    # TAKE: rel = a - TAKE_BASE; side = rel // (NUM_FOODS*MAX_QTY); ...
    take_rel = a - _TAKE_BASE
    take_side = torch.where(is_take, take_rel // (NUM_FOODS * MAX_QTY), zero)
    take_rest = torch.where(is_take, take_rel % (NUM_FOODS * MAX_QTY), zero)
    take_food = torch.where(is_take, take_rest // MAX_QTY, zero)
    take_qty = torch.where(is_take, take_rest % MAX_QTY, zero)

    # POST: rel = a - POST_BASE; side = rel // (NUM_FOODS*MAX_QTY*MAX_PRICE); ...
    post_rel = a - _POST_BASE
    post_side = torch.where(
        is_post, post_rel // (NUM_FOODS * MAX_QTY * MAX_PRICE), zero
    )
    post_rest = torch.where(
        is_post, post_rel % (NUM_FOODS * MAX_QTY * MAX_PRICE), zero
    )
    post_food = torch.where(is_post, post_rest // (MAX_QTY * MAX_PRICE), zero)
    post_rest2 = torch.where(is_post, post_rest % (MAX_QTY * MAX_PRICE), zero)
    post_qty = torch.where(is_post, post_rest2 // MAX_PRICE, zero)
    post_price = torch.where(is_post, post_rest2 % MAX_PRICE, zero)

    food_id = cancel_food + take_food + post_food
    side_id = cancel_side + take_side + post_side
    price_id = post_price
    qty_id = take_qty + post_qty
    # Rows in `is_wait` naturally get zeros for all placeholders.
    del is_wait

    return {
        "type_id": type_id,
        "food_id": food_id,
        "side_id": side_id,
        "price_id": price_id,
        "qty_id": qty_id,
    }


def factored_masks_from_flat_tensor(flat: torch.Tensor) -> FactoredMaskTensors:
    """Torch-side batched derivation of the per-head masks from a flat mask.

    Args:
        flat: bool/float tensor of shape (B, NUM_ACTIONS) — RLlib delivers
            action_mask as float from a Box space; we cast to bool here.

    Returns:
        FactoredMaskTensors with all fields shaped (B, ...). Semantics
        match ``actions.factored_masks_from_flat`` — heads that a given
        action type does not use are all-True along their dimension so the
        placeholder id 0 always samples cleanly.
    """
    if flat.dim() != 2:
        raise ValueError(
            f"expected (B, NUM_ACTIONS) mask, got shape {tuple(flat.shape)}"
        )
    if not flat.is_floating_point() and flat.dtype != torch.bool:
        flat = flat.bool()
    if flat.is_floating_point():
        flat = flat > 0.5

    B = flat.shape[0]
    device = flat.device

    # Slice into the four action-type bands. Layout indices come from the
    # constants in actions.py; keep them in sync if that layout changes.
    wait_bit = flat[:, _ACTION_WAIT]                                            # (B,)
    cancel_slice = flat[:, _CANCEL_BASE:_TAKE_BASE]                             # (B, 2*4)
    take_slice = flat[:, _TAKE_BASE:_POST_BASE]                                 # (B, 2*4*Q)
    post_slice = flat[:, _POST_BASE:]                                           # (B, 2*4*Q*P)

    cancel_mask = cancel_slice.view(B, NUM_SIDES, NUM_FOODS)                    # (B, s, f)
    take_mask = take_slice.view(B, NUM_SIDES, NUM_FOODS, MAX_QTY)               # (B, s, f, q)
    post_mask = post_slice.view(
        B, NUM_SIDES, NUM_FOODS, MAX_QTY, MAX_PRICE
    )                                                                           # (B, s, f, q, p)

    # ---- type ----
    type_mask = torch.stack(
        [
            wait_bit,
            cancel_mask.any(dim=(-2, -1)),                    # any side, food legal
            take_mask.any(dim=(-3, -2, -1)),                  # any side, food, qty
            post_mask.any(dim=(-4, -3, -2, -1)),              # any side, food, qty, price
        ],
        dim=-1,
    )  # (B, 4)

    # ---- food ----
    # food_mask[b, t, f] = True iff any (s, ...) is legal for that (t, f).
    # For WAIT, food is a placeholder → all True.
    food_wait = torch.ones(B, NUM_FOODS, dtype=torch.bool, device=device)
    food_cancel = cancel_mask.any(dim=1)                                        # (B, f)
    food_take = take_mask.any(dim=(1, 3))                                       # (B, f)
    food_post = post_mask.any(dim=(1, 3, 4))                                    # (B, f)
    food_mask = torch.stack([food_wait, food_cancel, food_take, food_post], dim=1)  # (B, 4, f)

    # ---- side ----
    side_wait = torch.ones(B, NUM_FOODS, NUM_SIDES, dtype=torch.bool, device=device)
    # cancel_mask is (B, s, f) → we want (B, f, s)
    side_cancel = cancel_mask.transpose(1, 2)                                   # (B, f, s)
    # take_mask is (B, s, f, q) → any along q → (B, s, f) → transpose → (B, f, s)
    side_take = take_mask.any(dim=-1).transpose(1, 2)                           # (B, f, s)
    # post_mask is (B, s, f, q, p) → any along (q, p) → (B, s, f) → transpose
    side_post = post_mask.any(dim=(-2, -1)).transpose(1, 2)                     # (B, f, s)
    side_mask = torch.stack(
        [side_wait, side_cancel, side_take, side_post], dim=1
    )                                                                           # (B, 4, f, s)

    # ---- price ----
    # Only meaningful for POST; the other types get all-True placeholders.
    all_prices_true = torch.ones(
        B, NUM_FOODS, NUM_SIDES, MAX_PRICE, dtype=torch.bool, device=device
    )
    # post_mask (B, s, f, q, p) → any along q → (B, s, f, p) → transpose (s ↔ f)
    price_post = post_mask.any(dim=3).transpose(1, 2)                           # (B, f, s, p)
    price_mask = torch.stack(
        [all_prices_true, all_prices_true, all_prices_true, price_post], dim=1
    )                                                                           # (B, 4, f, s, p)

    # ---- qty ----
    # Placeholder-all-True for WAIT and CANCEL.
    all_q_true = torch.ones(
        B,
        NUM_FOODS,
        NUM_SIDES,
        MAX_PRICE,
        MAX_QTY,
        dtype=torch.bool,
        device=device,
    )
    # TAKE: qty depends on (side, food) only. take_mask is (B, s, f, q)
    # → transpose to (B, f, s, q) → broadcast across MAX_PRICE
    take_qty = take_mask.transpose(1, 2)                                        # (B, f, s, q)
    qty_take = take_qty.unsqueeze(3).expand(-1, -1, -1, MAX_PRICE, -1)          # (B, f, s, p, q)
    # POST: qty depends on (side, food, price). post_mask is (B, s, f, q, p)
    # → transpose to (B, f, s, q, p) → swap last two dims → (B, f, s, p, q)
    qty_post = post_mask.transpose(1, 2).transpose(-2, -1)                      # (B, f, s, p, q)
    qty_mask = torch.stack(
        [all_q_true, all_q_true, qty_take, qty_post], dim=1
    )                                                                           # (B, 4, f, s, p, q)

    return FactoredMaskTensors(
        type_mask=type_mask,
        food_mask=food_mask,
        side_mask=side_mask,
        price_mask=price_mask,
        qty_mask=qty_mask,
    )


def factored_to_flat_batch(factored: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Torch-side inverse of ``flat_to_factored_batch``."""
    type_id = factored["type_id"].long()
    food_id = factored["food_id"].long()
    side_id = factored["side_id"].long()
    price_id = factored["price_id"].long()
    qty_id = factored["qty_id"].long()

    is_wait = type_id == TYPE_WAIT
    is_cancel = type_id == TYPE_CANCEL
    is_take = type_id == TYPE_TAKE
    is_post = type_id == TYPE_POST

    cancel_flat = _CANCEL_BASE + side_id * NUM_FOODS + food_id
    take_flat = (
        _TAKE_BASE
        + side_id * (NUM_FOODS * MAX_QTY)
        + food_id * MAX_QTY
        + qty_id
    )
    post_flat = (
        _POST_BASE
        + side_id * (NUM_FOODS * MAX_QTY * MAX_PRICE)
        + food_id * (MAX_QTY * MAX_PRICE)
        + qty_id * MAX_PRICE
        + price_id
    )
    wait_flat = torch.full_like(type_id, _ACTION_WAIT)

    flat = wait_flat.clone()
    flat = torch.where(is_cancel, cancel_flat, flat)
    flat = torch.where(is_take, take_flat, flat)
    flat = torch.where(is_post, post_flat, flat)
    del is_wait
    return flat
