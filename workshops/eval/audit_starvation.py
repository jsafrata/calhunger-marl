"""Audit — re-classify each death independently from the saved
`death_context` and compare against the harness's stored label.

The classifier we're auditing lives in `game_runner._classify_starvation`.
This auditor reimplements the same six-way rule from the spec, using ONLY
the `death_context` snapshot that ships with each game log, then reports
per-category accuracy on a random sample.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

CONSUME_PER_TICK = 2  # mirrors env config
ROOT = Path(__file__).resolve().parent / "results" / "validation"
SAMPLE_PER_CATEGORY = 50


def independent_classify(ctx: dict) -> tuple[str, dict]:
    """Re-implement the classifier rule from the spec, using only the
    saved death context. Returns (predicted_cause, diagnostics)."""
    inv = ctx["available_inventory"]
    required = ctx["required_foods"]

    # The food that killed us = first required food with < CONSUME_PER_TICK units.
    missing = [f for f in required if inv.get(f, 0) < CONSUME_PER_TICK]
    if not missing:
        return "other", {"note": "no required food below survival threshold"}
    food = missing[0]

    cash_free = ctx["available_cash"]
    cash_total = ctx["cash"]
    reserved = cash_total - cash_free

    asks = ctx["asks_per_food"].get(food, [])
    if not asks:
        if cash_total < 1:
            return "insufficient_cash", {"food": food, "reason": "no cash and no ask"}
        return "no_ask_available", {"food": food, "cash_total": cash_total}
    cheapest_ask = min(a["price"] for a in asks)
    if cash_free < cheapest_ask:
        if reserved >= cheapest_ask:
            return "cash_tied_in_reservations", {
                "food": food, "cheapest_ask": cheapest_ask,
                "cash_free": cash_free, "reserved": reserved,
            }
        return "insufficient_cash", {
            "food": food, "cheapest_ask": cheapest_ask, "cash_free": cash_free,
        }

    own_pending = [
        o for o in ctx["own_orders"]
        if o["side"] == "bid" and o["food"] == food
    ]
    if own_pending:
        return "order_pending_no_fill", {
            "food": food, "pending_orders": len(own_pending),
        }
    return "affordable_ask_no_buy", {
        "food": food, "cheapest_ask": cheapest_ask, "cash_free": cash_free,
    }


def iter_deaths() -> Iterable[dict]:
    """Yield one dict per dead player across every game in the validation batch."""
    for f in sorted(ROOT.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                g = json.loads(line)
                if "error" in g:
                    continue
                for pb in g["players_behavior"]:
                    if pb.get("death_context") is None:
                        continue
                    yield {
                        "game_seed": g["game_seed"],
                        "config": f.stem,
                        "player_idx": pb["idx"],
                        "cadence": pb["cadence"],
                        "labeled_cause": pb["starvation_cause"],
                        "context": pb["death_context"],
                    }


def main() -> None:
    deaths = list(iter_deaths())
    total_by_cause: Counter = Counter(d["labeled_cause"] for d in deaths)
    print(f"[audit-starvation] {len(deaths)} total deaths in validation batch")
    print(f"[audit-starvation] labeled breakdown: {dict(total_by_cause)}\n")

    rng = random.Random(20260716)
    categories = ["affordable_ask_no_buy", "no_ask_available", "order_pending_no_fill"]

    total_agree = 0
    total_checked = 0
    mismatch_examples: list[dict] = []
    per_cat_stats = {}

    for cat in categories:
        pool = [d for d in deaths if d["labeled_cause"] == cat]
        if not pool:
            print(f"[{cat}] no deaths of this class in the validation batch")
            per_cat_stats[cat] = {"n_pool": 0, "n_sampled": 0, "n_agree": 0}
            continue
        rng.shuffle(pool)
        sample = pool[:SAMPLE_PER_CATEGORY]

        n_agree = 0
        n_disagree = 0
        confusion: Counter = Counter()
        for d in sample:
            predicted, diag = independent_classify(d["context"])
            if predicted == d["labeled_cause"]:
                n_agree += 1
            else:
                n_disagree += 1
                confusion[predicted] += 1
                if len(mismatch_examples) < 15:
                    mismatch_examples.append({
                        "config": d["config"], "seed": d["game_seed"],
                        "player": d["player_idx"], "cadence": d["cadence"],
                        "labeled": d["labeled_cause"], "predicted": predicted,
                        "diag": diag,
                    })
        n = len(sample)
        rate = n_agree / n
        per_cat_stats[cat] = {
            "n_pool": len(pool), "n_sampled": n, "n_agree": n_agree,
            "n_disagree": n_disagree, "confusion": dict(confusion),
        }
        total_agree += n_agree
        total_checked += n
        print(f"[{cat}] pool={len(pool)}  sampled={n}  agree={n_agree}  "
              f"disagree={n_disagree}  accuracy={rate:.3f}"
              + (f"  → predicted-instead: {dict(confusion)}" if confusion else ""))

    print(f"\n[audit-starvation] OVERALL: agree {total_agree}/{total_checked} = "
          f"{total_agree/max(1,total_checked):.3f}\n")

    if mismatch_examples:
        print("First few mismatches (label vs. independent recompute):")
        for m in mismatch_examples:
            print(f"  {m}")
    else:
        print("No mismatches — classifier is stable under independent re-computation.")


if __name__ == "__main__":
    main()
