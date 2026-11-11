"""Behavioral-cloning representation diagnostic.

Tests whether the current MLP-torso + factored autoregressive actor can
represent frozen Scripted-v1 Fast. Cloned weights are ONLY used for this
diagnostic — never for the IPPO/MAPPO baselines.

Stages
------
    generate  — play Scripted-v1 Fast in randomized games, record every
                (obs, mask, action, factored) per seat per tick, sharded
                into train/val/test npz using disjoint game seeds.
    train     — instantiate the existing MLPTorso + FactoredActor (no
                critic), train with teacher-forced per-head cross-entropy,
                report per-head + flat + conditional-on-type accuracy.
    rollout   — 1000-game deterministic self-play + 1learned+3fast eval.
                Reports survival / reach-180 / trades / TAKE-freq /
                bid-vs-ask, plus a Scripted-v1 Fast reference row.
    all       — run the three stages sequentially.

Usage
-----
    python -m eval.bc_diagnostic --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval._warnings import silence_ray_deprecations  # noqa: E402
silence_ray_deprecations()

from agents.cadenced_scripted import CadencedScripted  # noqa: E402
from env.actions import (  # noqa: E402
    ACTION_WAIT,
    NUM_ACTIONS,
    NUM_ACTION_TYPES,
    NUM_FOODS,
    NUM_SIDES,
    MAX_PRICE,
    MAX_QTY,
    TYPE_WAIT,
    TYPE_CANCEL,
    TYPE_TAKE,
    TYPE_POST,
    decode,
    CancelAction,
    PostAction,
    TakeAction,
    WaitAction,
)
from env.config import DEFAULT_CONFIG  # noqa: E402
from env.env import FoodTradingEnv  # noqa: E402
from env.rl.factored_actor import (  # noqa: E402
    FactoredActor,
    _FOOD_RELEVANT,
    _SIDE_RELEVANT,
    _PRICE_RELEVANT,
    _QTY_RELEVANT,
    _apply_mask,
    _gather_along_batch,
    _gather2,
    _gather3,
    _gather4,
    factored_masks_from_flat_tensor,
    factored_to_flat_batch,
    flat_to_factored_batch,
)
from env.rl.observation_encoder import OBS_DIM, encode_observation  # noqa: E402
from env.rl.rllib_module import MLPTorso  # noqa: E402


DEFAULT_ROOT = Path("eval/results/bc_diagnostic")


# =============================================================================
# Stage 1 — dataset generation
# =============================================================================


def _pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.astype(np.uint8, copy=False), axis=-1)


def _unpack_mask(packed: np.ndarray, n_actions: int = NUM_ACTIONS) -> np.ndarray:
    m = np.unpackbits(packed, axis=-1)
    return m[..., :n_actions]


def _play_game_collect(game_seed: int, cadence: str = "fast") -> dict:
    """Play one game with 4 Scripted-v1 Fast agents; record per-seat, per-tick
    (obs, mask, flat_action, factored) rows for every decision."""
    env = FoodTradingEnv(seed=game_seed)
    obs = env.reset()
    cfg = env.config
    agents = [CadencedScripted(cadence, seed=game_seed * 977 + i) for i in range(4)]
    for a in agents:
        a.reset()

    rows = {
        "obs": [],           # (N, OBS_DIM) float32
        "mask_packed": [],   # (N, ceil(NUM_ACTIONS/8)) uint8
        "action": [],        # (N,) int32
        "type_id": [],
        "food_id": [],
        "side_id": [],
        "price_id": [],
        "qty_id": [],
    }
    while not env.is_done():
        actions_dict = {}
        for i in range(4):
            mask_np = env.legal_actions(i).astype(np.float32)
            action = agents[i].act(obs[i], mask_np)
            actions_dict[i] = action

            enc = encode_observation(obs[i], cfg)
            rows["obs"].append(enc.astype(np.float32))
            rows["mask_packed"].append(_pack_mask(mask_np > 0.5))
            rows["action"].append(np.int32(action))

            # Decode into factored.
            decoded = decode(action)
            if isinstance(decoded, WaitAction):
                t, f, s, p, q = TYPE_WAIT, 0, 0, 0, 0
            elif isinstance(decoded, CancelAction):
                t = TYPE_CANCEL
                f = cfg.food_types.index(decoded.food_type)
                s = 0 if decoded.side == "bid" else 1
                p, q = 0, 0
            elif isinstance(decoded, TakeAction):
                t = TYPE_TAKE
                f = cfg.food_types.index(decoded.food_type)
                s = 0 if decoded.side == "bid" else 1
                p = 0
                q = decoded.quantity - 1  # qty ids are 0..MAX_QTY-1 for qty 1..MAX_QTY
            elif isinstance(decoded, PostAction):
                t = TYPE_POST
                f = cfg.food_types.index(decoded.food_type)
                s = 0 if decoded.side == "bid" else 1
                p = decoded.price_per_unit - 1  # price ids 0..MAX_PRICE-1 for 1..MAX_PRICE
                q = decoded.quantity - 1
            else:
                raise RuntimeError(f"unknown decoded action: {decoded!r}")
            rows["type_id"].append(np.int32(t))
            rows["food_id"].append(np.int32(f))
            rows["side_id"].append(np.int32(s))
            rows["price_id"].append(np.int32(p))
            rows["qty_id"].append(np.int32(q))
        obs, _, _, _ = env.step(actions_dict)

    for k in rows:
        rows[k] = np.stack(rows[k], axis=0)
    return rows


def generate_dataset(
    root: Path,
    n_train: int,
    n_val: int,
    n_test: int,
    train_seed_base: int = 100_000,
    val_seed_base: int = 500_000,
    test_seed_base: int = 900_000,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    stats = {}
    splits = [
        ("train", n_train, train_seed_base),
        ("val",   n_val,   val_seed_base),
        ("test",  n_test,  test_seed_base),
    ]
    for split_name, n_games, seed_base in splits:
        print(f"[generate] {split_name}: {n_games} games starting at seed {seed_base}")
        t0 = time.time()
        obs_all, mask_all, act_all = [], [], []
        tp, fp, sp, pp, qp = [], [], [], [], []
        for g in range(n_games):
            game_seed = seed_base + g
            r = _play_game_collect(game_seed)
            obs_all.append(r["obs"])
            mask_all.append(r["mask_packed"])
            act_all.append(r["action"])
            tp.append(r["type_id"])
            fp.append(r["food_id"])
            sp.append(r["side_id"])
            pp.append(r["price_id"])
            qp.append(r["qty_id"])
            if (g + 1) % 200 == 0 or g == n_games - 1:
                dt = time.time() - t0
                total_rows = sum(a.shape[0] for a in act_all)
                print(f"    game {g+1}/{n_games}  rows={total_rows}  elapsed={dt:.0f}s")
        obs_all = np.concatenate(obs_all, axis=0)
        mask_all = np.concatenate(mask_all, axis=0)
        act_all = np.concatenate(act_all, axis=0)
        tp = np.concatenate(tp, axis=0)
        fp = np.concatenate(fp, axis=0)
        sp = np.concatenate(sp, axis=0)
        pp = np.concatenate(pp, axis=0)
        qp = np.concatenate(qp, axis=0)
        out = root / f"{split_name}.npz"
        np.savez_compressed(out,
            obs=obs_all, mask_packed=mask_all, action=act_all,
            type_id=tp, food_id=fp, side_id=sp, price_id=pp, qty_id=qp,
            seed_range=np.array([seed_base, seed_base + n_games], dtype=np.int64),
        )
        type_counts = np.bincount(tp, minlength=NUM_ACTION_TYPES).tolist()
        stats[split_name] = {
            "n_games": n_games,
            "n_rows": int(act_all.shape[0]),
            "type_counts": {
                "wait": type_counts[TYPE_WAIT],
                "cancel": type_counts[TYPE_CANCEL],
                "take": type_counts[TYPE_TAKE],
                "post": type_counts[TYPE_POST],
            },
            "type_fractions": {
                "wait": type_counts[TYPE_WAIT] / len(tp),
                "cancel": type_counts[TYPE_CANCEL] / len(tp),
                "take": type_counts[TYPE_TAKE] / len(tp),
                "post": type_counts[TYPE_POST] / len(tp),
            },
            "seed_range": [seed_base, seed_base + n_games],
            "file": str(out),
            "file_size_mb": out.stat().st_size / 1e6,
            "gen_seconds": time.time() - t0,
        }
        print(f"    wrote {out} ({stats[split_name]['file_size_mb']:.1f} MB)")
    Path(root / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# =============================================================================
# Stage 2 — BC training
# =============================================================================


class BCActor(torch.nn.Module):
    """MLP torso + FactoredActor, no critic. Reuses the exact classes used
    by the RL module so successful BC weights would slot in unchanged."""

    def __init__(self, *, head_kind: str = "mlp") -> None:
        super().__init__()
        self.torso = MLPTorso()
        self.head_kind = head_kind
        self.actor = FactoredActor(
            embedding_size=self.torso.output_size,
            head_kind=head_kind,
        )

    def head_logits_teacher_forced(
        self,
        obs: torch.Tensor,
        mask_flat: torch.Tensor,
        factored: dict,
    ) -> dict:
        """Return per-head **masked** logits with teacher forcing on the
        TRUE prefix. This is the training signal for BC."""
        emb, _ = self.torso(obs, None)
        masks = factored_masks_from_flat_tensor(mask_flat)
        actor = self.actor
        type_id = factored["type_id"]
        food_id = factored["food_id"]
        side_id = factored["side_id"]
        price_id = factored["price_id"]

        type_logits = _apply_mask(actor.type_head(emb), masks.type_mask)

        type_e = actor.type_embed(type_id)
        food_logits = _apply_mask(
            actor.food_head(torch.cat([emb, type_e], dim=-1)),
            _gather_along_batch(masks.food_mask, type_id),
        )

        food_e = actor.food_embed(food_id)
        side_logits = _apply_mask(
            actor.side_head(torch.cat([emb, type_e, food_e], dim=-1)),
            _gather2(masks.side_mask, type_id, food_id),
        )

        side_e = actor.side_embed(side_id)
        price_logits = _apply_mask(
            actor.price_head(torch.cat([emb, type_e, food_e, side_e], dim=-1)),
            _gather3(masks.price_mask, type_id, food_id, side_id),
        )

        price_e = actor.price_embed(price_id)
        qty_logits = _apply_mask(
            actor.qty_head(torch.cat([emb, type_e, food_e, side_e, price_e], dim=-1)),
            _gather4(masks.qty_mask, type_id, food_id, side_id, price_id),
        )
        return {
            "type": type_logits, "food": food_logits, "side": side_logits,
            "price": price_logits, "qty": qty_logits,
        }


def _safe_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over masked logits. Where every entry is -inf (a fully-
    illegal row that the actor would have to fall back on uniform for), the
    loss is set to 0 rather than NaN. This mirrors the actor's _safe_categorical
    fallback."""
    finite_mask = torch.isfinite(logits)
    row_all_neginf = ~finite_mask.any(dim=-1)
    fallback = torch.zeros_like(logits)
    safe = torch.where(row_all_neginf.unsqueeze(-1), fallback, logits)
    loss = F.cross_entropy(safe, target, reduction="none")
    loss = torch.where(row_all_neginf, torch.zeros_like(loss), loss)
    return loss


def _load_split(path: Path) -> dict:
    d = np.load(path)
    return {
        "obs": d["obs"],
        "mask_packed": d["mask_packed"],
        "action": d["action"],
        "type_id": d["type_id"],
        "food_id": d["food_id"],
        "side_id": d["side_id"],
        "price_id": d["price_id"],
        "qty_id": d["qty_id"],
    }


def _batches(split: dict, batch_size: int, shuffle: bool, seed: int = 0):
    n = split["action"].shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        b = idx[i:i+batch_size]
        # Unpack the mask on the fly to keep memory low.
        packed = split["mask_packed"][b]
        mask = _unpack_mask(packed)  # (B, NUM_ACTIONS) uint8
        yield {
            "obs":       torch.from_numpy(split["obs"][b]).float(),
            "mask":      torch.from_numpy(mask).float(),
            "action":    torch.from_numpy(split["action"][b]).long(),
            "type_id":   torch.from_numpy(split["type_id"][b]).long(),
            "food_id":   torch.from_numpy(split["food_id"][b]).long(),
            "side_id":   torch.from_numpy(split["side_id"][b]).long(),
            "price_id":  torch.from_numpy(split["price_id"][b]).long(),
            "qty_id":    torch.from_numpy(split["qty_id"][b]).long(),
        }


def _compute_metrics(model: BCActor, split: dict, batch_size: int, device: str) -> dict:
    """Per-head + flat accuracy + conditional-on-type metrics."""
    model.eval()
    tot = {k: 0 for k in ("n", "type_correct", "food_correct_rel", "food_rel",
                          "side_correct_rel", "side_rel",
                          "price_correct_rel", "price_rel", "price_l1_sum",
                          "qty_correct_rel", "qty_rel", "qty_l1_sum",
                          "flat_correct", "flat_legal")}
    conditional = {
        "type_correct_by_target": Counter(),
        "type_total_by_target": Counter(),
        # POST-specific
        "post_side_correct": 0, "post_side_total": 0,
        "post_food_correct": 0, "post_food_total": 0,
        "post_price_correct": 0, "post_price_total": 0, "post_price_l1_sum": 0,
        "post_qty_correct": 0, "post_qty_total": 0,
        # TAKE-specific
        "take_correct": 0, "take_total": 0,
        # posted-bid vs posted-ask breakdown
        "post_target_bid": 0, "post_target_ask": 0,
        "post_pred_bid_when_target_bid": 0, "post_pred_ask_when_target_bid": 0,
        "post_pred_bid_when_target_ask": 0, "post_pred_ask_when_target_ask": 0,
    }
    with torch.no_grad():
        for batch in _batches(split, batch_size, shuffle=False):
            obs = batch["obs"].to(device)
            mask = batch["mask"].to(device)
            action = batch["action"].to(device)
            factored = {k: batch[k].to(device) for k in
                        ("type_id","food_id","side_id","price_id","qty_id")}

            # Argmax rollout — use the actor's own sample(deterministic=True)
            emb, _ = model.torso(obs, None)
            masks_t = factored_masks_from_flat_tensor(mask)
            pred_factored, _, _ = model.actor.sample(emb, masks_t, deterministic=True)
            pred_flat = factored_to_flat_batch(pred_factored)

            # Flat accuracy
            flat_correct = (pred_flat == action)
            tot["flat_correct"] += int(flat_correct.sum().item())
            # Legality of predicted argmax (should be True by construction due to mask)
            legal_bits = mask.gather(1, pred_flat.unsqueeze(1)).squeeze(1)
            tot["flat_legal"] += int((legal_bits > 0.5).sum().item())

            # Per-head accuracy under TEACHER-FORCED prefix (this is the
            # supervised metric that matches the loss).
            logits = model.head_logits_teacher_forced(obs, mask, factored)
            type_pred = logits["type"].argmax(dim=-1)
            food_pred = logits["food"].argmax(dim=-1)
            side_pred = logits["side"].argmax(dim=-1)
            price_pred = logits["price"].argmax(dim=-1)
            qty_pred = logits["qty"].argmax(dim=-1)

            type_correct = (type_pred == factored["type_id"])
            tot["type_correct"] += int(type_correct.sum().item())

            rel_food = _FOOD_RELEVANT.to(device)[factored["type_id"]]
            rel_side = _SIDE_RELEVANT.to(device)[factored["type_id"]]
            rel_price = _PRICE_RELEVANT.to(device)[factored["type_id"]]
            rel_qty = _QTY_RELEVANT.to(device)[factored["type_id"]]

            def rel_acc(pred, target, rel):
                nrel = int(rel.sum().item())
                ncorrect = int(((pred == target) & rel).sum().item())
                return ncorrect, nrel

            fc, fn = rel_acc(food_pred, factored["food_id"], rel_food)
            sc, sn = rel_acc(side_pred, factored["side_id"], rel_side)
            pc, pn = rel_acc(price_pred, factored["price_id"], rel_price)
            qc, qn = rel_acc(qty_pred, factored["qty_id"], rel_qty)
            tot["food_correct_rel"] += fc; tot["food_rel"] += fn
            tot["side_correct_rel"] += sc; tot["side_rel"] += sn
            tot["price_correct_rel"] += pc; tot["price_rel"] += pn
            tot["qty_correct_rel"] += qc; tot["qty_rel"] += qn
            # L1 error on prices/qty (only over relevant rows)
            tot["price_l1_sum"] += float(((price_pred - factored["price_id"]).abs() * rel_price).sum().item())
            tot["qty_l1_sum"] += float(((qty_pred - factored["qty_id"]).abs() * rel_qty).sum().item())
            tot["n"] += obs.shape[0]

            # Conditional metrics
            for t_id in range(NUM_ACTION_TYPES):
                m = factored["type_id"] == t_id
                conditional["type_total_by_target"][t_id] += int(m.sum().item())
                conditional["type_correct_by_target"][t_id] += int((type_correct & m).sum().item())
            # POST subset (target type is POST): compute head accuracies using
            # the argmax on the teacher-forced logits
            is_post_tgt = factored["type_id"] == TYPE_POST
            npost = int(is_post_tgt.sum().item())
            if npost > 0:
                conditional["post_side_total"] += npost
                conditional["post_side_correct"] += int(((side_pred == factored["side_id"]) & is_post_tgt).sum().item())
                conditional["post_food_total"] += npost
                conditional["post_food_correct"] += int(((food_pred == factored["food_id"]) & is_post_tgt).sum().item())
                conditional["post_price_total"] += npost
                conditional["post_price_correct"] += int(((price_pred == factored["price_id"]) & is_post_tgt).sum().item())
                conditional["post_price_l1_sum"] += float(((price_pred - factored["price_id"]).abs() * is_post_tgt).sum().item())
                conditional["post_qty_total"] += npost
                conditional["post_qty_correct"] += int(((qty_pred == factored["qty_id"]) & is_post_tgt).sum().item())
                # side-of-post breakdown (0=bid, 1=ask)
                tgt_bid = is_post_tgt & (factored["side_id"] == 0)
                tgt_ask = is_post_tgt & (factored["side_id"] == 1)
                conditional["post_target_bid"] += int(tgt_bid.sum().item())
                conditional["post_target_ask"] += int(tgt_ask.sum().item())
                conditional["post_pred_bid_when_target_bid"] += int(((side_pred == 0) & tgt_bid).sum().item())
                conditional["post_pred_ask_when_target_bid"] += int(((side_pred == 1) & tgt_bid).sum().item())
                conditional["post_pred_bid_when_target_ask"] += int(((side_pred == 0) & tgt_ask).sum().item())
                conditional["post_pred_ask_when_target_ask"] += int(((side_pred == 1) & tgt_ask).sum().item())
            # TAKE: correct = argmax type == TAKE
            is_take_tgt = factored["type_id"] == TYPE_TAKE
            ntake = int(is_take_tgt.sum().item())
            if ntake > 0:
                conditional["take_total"] += ntake
                conditional["take_correct"] += int(((type_pred == TYPE_TAKE) & is_take_tgt).sum().item())

    n = max(1, tot["n"])
    def rate(a, b): return a / max(1, b)
    return {
        "n": n,
        "flat_accuracy": rate(tot["flat_correct"], n),
        "flat_legal_fraction": rate(tot["flat_legal"], n),
        "per_head_accuracy": {
            "type": rate(tot["type_correct"], n),
            "food_rel_only": rate(tot["food_correct_rel"], tot["food_rel"]),
            "side_rel_only": rate(tot["side_correct_rel"], tot["side_rel"]),
            "price_rel_only": rate(tot["price_correct_rel"], tot["price_rel"]),
            "qty_rel_only": rate(tot["qty_correct_rel"], tot["qty_rel"]),
        },
        "per_head_l1_mae_rel": {
            "price": rate(tot["price_l1_sum"], tot["price_rel"]),
            "qty":   rate(tot["qty_l1_sum"],   tot["qty_rel"]),
        },
        "type_accuracy_by_target": {
            name: rate(conditional["type_correct_by_target"][t_id],
                       conditional["type_total_by_target"][t_id])
            for name, t_id in (("wait",TYPE_WAIT),("cancel",TYPE_CANCEL),
                               ("take",TYPE_TAKE),("post",TYPE_POST))
        },
        "post_conditional": {
            "side_accuracy": rate(conditional["post_side_correct"], conditional["post_side_total"]),
            "food_accuracy": rate(conditional["post_food_correct"], conditional["post_food_total"]),
            "price_exact":   rate(conditional["post_price_correct"], conditional["post_price_total"]),
            "price_l1_mae":  rate(conditional["post_price_l1_sum"], conditional["post_price_total"]),
            "qty_accuracy":  rate(conditional["post_qty_correct"], conditional["post_qty_total"]),
        },
        "post_side_bid_vs_ask": {
            "n_target_bid": conditional["post_target_bid"],
            "n_target_ask": conditional["post_target_ask"],
            "acc_when_target_bid":
                rate(conditional["post_pred_bid_when_target_bid"], conditional["post_target_bid"]),
            "acc_when_target_ask":
                rate(conditional["post_pred_ask_when_target_ask"], conditional["post_target_ask"]),
        },
        "take_type_accuracy": rate(conditional["take_correct"], conditional["take_total"]),
    }


def train(
    root: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    *,
    head_kind: str = "mlp",
    patience: int = 5,
    dataset_root: Path | None = None,
) -> dict:
    ds_root = dataset_root or root
    train_split = _load_split(ds_root / "train.npz")
    val_split   = _load_split(ds_root / "val.npz")
    test_split  = _load_split(ds_root / "test.npz")
    print(f"[train] head_kind={head_kind}  patience={patience}  "
          f"loaded train={train_split['action'].shape[0]}  "
          f"val={val_split['action'].shape[0]}  test={test_split['action'].shape[0]}")

    model = BCActor(head_kind=head_kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    curves = []
    best_val_flat_acc = -1.0
    best_state = None
    epochs_since_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses_epoch = {"type": [], "food": [], "side": [], "price": [], "qty": [], "total": []}
        n_batches = 0
        for batch in _batches(train_split, batch_size, shuffle=True, seed=epoch):
            obs = batch["obs"].to(device)
            mask = batch["mask"].to(device)
            factored = {k: batch[k].to(device) for k in
                        ("type_id","food_id","side_id","price_id","qty_id")}

            logits = model.head_logits_teacher_forced(obs, mask, factored)
            type_loss  = _safe_ce(logits["type"],  factored["type_id"])
            food_loss  = _safe_ce(logits["food"],  factored["food_id"])
            side_loss  = _safe_ce(logits["side"],  factored["side_id"])
            price_loss = _safe_ce(logits["price"], factored["price_id"])
            qty_loss   = _safe_ce(logits["qty"],   factored["qty_id"])

            rel_food = _FOOD_RELEVANT.to(device)[factored["type_id"]]
            rel_side = _SIDE_RELEVANT.to(device)[factored["type_id"]]
            rel_price = _PRICE_RELEVANT.to(device)[factored["type_id"]]
            rel_qty = _QTY_RELEVANT.to(device)[factored["type_id"]]
            zero = torch.zeros_like(type_loss)
            food_loss  = torch.where(rel_food,  food_loss,  zero)
            side_loss  = torch.where(rel_side,  side_loss,  zero)
            price_loss = torch.where(rel_price, price_loss, zero)
            qty_loss   = torch.where(rel_qty,   qty_loss,   zero)

            total = (type_loss + food_loss + side_loss + price_loss + qty_loss).mean()
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses_epoch["type"].append(float(type_loss.mean().item()))
            losses_epoch["food"].append(float(food_loss.mean().item()))
            losses_epoch["side"].append(float(side_loss.mean().item()))
            losses_epoch["price"].append(float(price_loss.mean().item()))
            losses_epoch["qty"].append(float(qty_loss.mean().item()))
            losses_epoch["total"].append(float(total.item()))
            n_batches += 1
        train_time = time.time() - t0
        train_loss = {k: float(np.mean(v)) for k, v in losses_epoch.items()}

        # Evaluate on val
        val_metrics = _compute_metrics(model, val_split, batch_size, device)
        print(f"[train] epoch {epoch}/{epochs}  "
              f"loss_tot={train_loss['total']:.4f}  "
              f"loss_type={train_loss['type']:.4f}  "
              f"loss_post_price={train_loss['price']:.4f}  "
              f"val_flat_acc={val_metrics['flat_accuracy']:.4f}  "
              f"val_type_acc={val_metrics['per_head_accuracy']['type']:.4f}  "
              f"val_post_side_acc={val_metrics['post_conditional']['side_accuracy']:.4f}  "
              f"val_post_price_exact={val_metrics['post_conditional']['price_exact']:.4f}  "
              f"[{train_time:.0f}s]")
        curves.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
            "train_seconds": train_time,
        })
        improved = val_metrics["flat_accuracy"] > best_val_flat_acc + 1e-4
        if improved:
            best_val_flat_acc = val_metrics["flat_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"[train] early stop at epoch {epoch}: no val-flat-acc "
                      f"gain for {patience} epochs (best={best_val_flat_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), root / "bc_actor.pt")
    print(f"[train] wrote {root/'bc_actor.pt'}")

    test_metrics = _compute_metrics(model, test_split, batch_size, device)
    print(f"[train] TEST — flat_acc={test_metrics['flat_accuracy']:.4f}  "
          f"type_acc={test_metrics['per_head_accuracy']['type']:.4f}  "
          f"post_side_acc={test_metrics['post_conditional']['side_accuracy']:.4f}  "
          f"post_price_exact={test_metrics['post_conditional']['price_exact']:.4f}")

    report = {
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "head_kind": head_kind,
        "patience": patience,
        "n_train": int(train_split["action"].shape[0]),
        "n_val":   int(val_split["action"].shape[0]),
        "n_test":  int(test_split["action"].shape[0]),
        "curves": curves,
        "best_val_flat_accuracy": best_val_flat_acc,
        "test_metrics": test_metrics,
    }
    (root / "bc_train_report.json").write_text(json.dumps(report, indent=2))
    return report


# =============================================================================
# Stage 3 — rollout eval
# =============================================================================


class ClonedBCAgent:
    """Wraps a trained BCActor in the Agent contract used by the game loop."""
    name = "bc_cloned"

    def __init__(self, model: BCActor, device: str = "cpu") -> None:
        self.model = model
        self.model.eval()
        self.device = device
        self.cfg = DEFAULT_CONFIG

    def reset(self) -> None:
        pass

    def act(self, obs: dict, legal_mask: np.ndarray) -> int:
        enc = encode_observation(obs, self.cfg).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(enc).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(legal_mask.astype(np.float32)).unsqueeze(0).to(self.device)
            emb, _ = self.model.torso(obs_t, None)
            masks = factored_masks_from_flat_tensor(mask_t)
            factored, _, _ = self.model.actor.sample(emb, masks, deterministic=True)
            flat = factored_to_flat_batch(factored)
            return int(flat.item())


def _rollout_lineup(agents_by_seat: list, n_games: int, seed_base: int) -> dict:
    """Play n_games games with the given per-seat agents; return aggregate metrics."""
    per_seat_stats = [Counter() for _ in range(4)]
    per_seat_survival = [[] for _ in range(4)]
    per_seat_bids = [[] for _ in range(4)]
    per_seat_asks = [[] for _ in range(4)]
    per_seat_takes = [[] for _ in range(4)]
    per_seat_type_freq = [Counter() for _ in range(4)]
    per_seat_n_decisions = [[] for _ in range(4)]
    survivors_hist = []
    reach180_hist = []
    trades_hist = []

    for g in range(n_games):
        seed = seed_base + g
        env = FoodTradingEnv(seed=seed)
        obs = env.reset()
        for a in agents_by_seat:
            a.reset()
        seat_bids = [0]*4; seat_asks = [0]*4; seat_takes = [0]*4
        seat_type_counts = [Counter() for _ in range(4)]
        seat_ndec = [0]*4
        while not env.is_done():
            actions = {}
            for i in range(4):
                a = agents_by_seat[i].act(obs[i], env.legal_actions(i))
                actions[i] = a
                seat_ndec[i] += 1
                d = decode(a)
                seat_type_counts[i][type(d).__name__] += 1
                if isinstance(d, PostAction):
                    if d.side == "bid": seat_bids[i] += 1
                    else: seat_asks[i] += 1
                elif isinstance(d, TakeAction):
                    seat_takes[i] += 1
            obs, _, _, _ = env.step(actions)
        res = env.get_result()
        survivors_hist.append(sum(1 for p in res["players"] if p["status"]=="alive"))
        reach180_hist.append(1 if res["end_reason"]=="time_limit" else 0)
        trades_hist.append(res["n_trades"])
        for i in range(4):
            per_seat_survival[i].append(1 if res["players"][i]["status"]=="alive" else 0)
            per_seat_bids[i].append(seat_bids[i])
            per_seat_asks[i].append(seat_asks[i])
            per_seat_takes[i].append(seat_takes[i])
            per_seat_type_freq[i].update(seat_type_counts[i])
            per_seat_n_decisions[i].append(seat_ndec[i])
    def m(x): return float(np.mean(x)) if x else 0.0
    return {
        "n_games": n_games,
        "survivors_mean": m(survivors_hist),
        "reach180_frac": m(reach180_hist),
        "trades_mean": m(trades_hist),
        "per_seat": [
            {
                "survival_rate": m(per_seat_survival[i]),
                "bids_mean": m(per_seat_bids[i]),
                "asks_mean": m(per_seat_asks[i]),
                "takes_mean": m(per_seat_takes[i]),
                "decisions_mean": m(per_seat_n_decisions[i]),
                "type_freq": {
                    "wait":   per_seat_type_freq[i].get("WaitAction",0),
                    "cancel": per_seat_type_freq[i].get("CancelAction",0),
                    "take":   per_seat_type_freq[i].get("TakeAction",0),
                    "post":   per_seat_type_freq[i].get("PostAction",0),
                },
            }
            for i in range(4)
        ],
    }


def rollout(
    root: Path,
    n_games: int,
    device: str = "cpu",
    *,
    head_kind: str = "mlp",
) -> dict:
    model = BCActor(head_kind=head_kind)
    state = torch.load(root / "bc_actor.pt", map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    lineups = {}
    for name, mask in (("bc_selfplay", [True]*4), ("1bc_3fast", [True,False,False,False])):
        print(f"[rollout] {name}: {n_games} games")
        t0 = time.time()
        agents = []
        for i, is_bc in enumerate(mask):
            if is_bc:
                agents.append(ClonedBCAgent(model, device=device))
            else:
                a = CadencedScripted("fast", seed=1_000_000 + i)
                agents.append(a)
        r = _rollout_lineup(agents, n_games=n_games, seed_base=700_000)
        r["elapsed_seconds"] = time.time() - t0
        lineups[name] = r
        print(f"    survivors={r['survivors_mean']:.2f}  reach180={r['reach180_frac']:.3f}  trades={r['trades_mean']:.1f}  ({r['elapsed_seconds']:.0f}s)")

    # Scripted-v1 Fast reference
    print(f"[rollout] scripted_v1_fast_reference: {n_games} games")
    t0 = time.time()
    agents = [CadencedScripted("fast", seed=1_000_000 + i) for i in range(4)]
    r = _rollout_lineup(agents, n_games=n_games, seed_base=800_000)
    r["elapsed_seconds"] = time.time() - t0
    lineups["scripted_v1_fast_reference"] = r
    print(f"    survivors={r['survivors_mean']:.2f}  reach180={r['reach180_frac']:.3f}  trades={r['trades_mean']:.1f}  ({r['elapsed_seconds']:.0f}s)")

    report = {"n_games": n_games, "lineups": lineups}
    (root / "bc_rollout_report.json").write_text(json.dumps(report, indent=2))
    return report


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("generate","train","rollout","all"), default="all")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--dataset_root", type=Path, default=None,
                   help="If set, load train/val/test.npz from here instead of --root. "
                        "Useful when comparing linear vs mlp on the SAME dataset.")
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_val", type=int, default=300)
    p.add_argument("--n_test", type=int, default=300)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rollout_games", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--head_kind", choices=("mlp", "linear"), default="mlp")
    p.add_argument("--patience", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    ds_root = args.dataset_root or args.root
    if args.stage in ("generate","all"):
        generate_dataset(ds_root, args.n_train, args.n_val, args.n_test)
    if args.stage in ("train","all"):
        train(args.root, args.epochs, args.batch_size, args.lr, args.device,
              head_kind=args.head_kind, patience=args.patience,
              dataset_root=args.dataset_root)
    if args.stage in ("rollout","all"):
        rollout(args.root, args.rollout_games, args.device,
                head_kind=args.head_kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
