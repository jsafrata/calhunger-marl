"""Aggregate one tag's per-game JSONL logs into summary tables.

Usage:
    python -m eval.summarize --tag smoke
    python -m eval.summarize --tag main

Reads eval/results/<tag>/<config>.jsonl. Writes:
    eval/results/<tag>/summary.csv               — one row per config,
                                                    all headline metrics
    eval/results/<tag>/summary_by_cadence.csv    — one row per (config,
                                                    cadence), player-side stats
    eval/results/<tag>/starvation_causes.csv     — breakdown of death causes
    eval/results/<tag>/market_by_food.csv        — per-config × per-food market stats
    eval/results/<tag>/primary_questions.md      — narrative answers to the
                                                    six questions in the spec
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.config import FOOD_TYPES  # noqa: E402

CADENCES = ("fast", "medium", "slow")


# -----------------------------------------------------------------------------
# Small stats helpers
# -----------------------------------------------------------------------------


def _mean_std(xs: Sequence[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


def _ci95_mean(xs: Sequence[float]) -> tuple[float, float]:
    """Normal-approx 95% CI around the mean. Cheap; fine for n≥30."""
    if not xs:
        return float("nan"), float("nan")
    mean = float(np.mean(xs))
    if len(xs) < 2:
        return mean, mean
    se = float(np.std(xs, ddof=1)) / math.sqrt(len(xs))
    return mean - 1.96 * se, mean + 1.96 * se


def _bootstrap_ci_median(
    xs: Sequence[float], n_boot: int = 500, seed: int = 0
) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) < 2:
        return float(xs[0]), float(xs[0])
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=float)
    boots = np.median(
        arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))], axis=1
    )
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _fmt_ci(lo: float, hi: float) -> str:
    if math.isnan(lo) or math.isnan(hi):
        return ""
    return f"[{lo:.3f}, {hi:.3f}]"


# -----------------------------------------------------------------------------
# JSONL loader — streams, we don't keep everything in memory.
# -----------------------------------------------------------------------------


def load_games(path: Path) -> Iterable[dict]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# -----------------------------------------------------------------------------
# Per-config aggregate — one pass over the JSONL
# -----------------------------------------------------------------------------


def summarize_config(path: Path) -> Dict:
    """Aggregate one config's games into a dict of headline metrics."""
    game_reached_time_limit: List[int] = []
    n_survivors: List[int] = []
    no_survivor: List[int] = []
    first_death_time: List[float] = []
    n_trades_total: List[int] = []

    # Player-side (aggregated across the 4 seats per game)
    posts_per_game: List[int] = []
    validation_fail_per_game: List[int] = []
    cancels_per_game: List[int] = []
    takes_per_game: List[int] = []

    # Per-cadence player buckets (for the by-cadence CSV)
    by_cadence: Dict[str, Dict[str, List[float]]] = {
        c: defaultdict(list) for c in CADENCES
    }

    # Starvation cause counts
    starvation_causes: Counter = Counter()

    # Market series → aggregate per-food
    per_food: Dict[str, Dict[str, List[float]]] = {
        f: defaultdict(list) for f in FOOD_TYPES
    }

    # Deaths despite affordable liquidity — spec-required "special" counter
    deaths_with_affordable_ask: int = 0
    total_deaths: int = 0

    # Price distributions across attempted / submitted / executed prices.
    # Kept as flat lists so the caller can compute max / percentiles / bin
    # counts in the writer below without having to re-stream the JSONL.
    attempted_prices_all: List[int] = []
    submitted_prices_all: List[int] = []
    executed_prices_all: List[int] = []
    total_cap_violations: int = 0

    n_games = 0
    n_errored = 0
    for g in load_games(path):
        if "error" in g:
            n_errored += 1
            continue
        n_games += 1
        final = g["final"]

        game_reached_time_limit.append(1 if final["reached_time_limit"] else 0)
        n_survivors.append(final["n_survivors"])
        no_survivor.append(1 if final["no_survivors"] else 0)
        n_trades_total.append(final["n_trades_total"])
        if final["first_death_tick"] is not None:
            first_death_time.append(float(final["first_death_tick"]))

        # Player-side aggregates
        posts_this_game = 0
        for pb in g["players_behavior"]:
            posts_this_game += pb["posts_total"]
            validation_fail_per_game.append(pb["validation_failures"])
            cancels_per_game.append(pb["cancels"])
            takes_per_game.append(pb["takes"])
            c = pb["cadence"]
            b = by_cadence[c]
            b["decisions_taken"].append(pb["decisions_taken"])
            b["decisions_skipped"].append(pb["decisions_skipped"])
            b["posts_total"].append(pb["posts_total"])
            b["posts_bid"].append(pb["posts_bid"])
            b["posts_ask"].append(pb["posts_ask"])
            b["posts_replaced"].append(pb["posts_replaced"])
            b["validation_failures"].append(pb["validation_failures"])
            b["cancels"].append(pb["cancels"])
            b["takes"].append(pb["takes"])
            b["units_bought"].append(pb["units_bought"])
            b["units_sold"].append(pb["units_sold"])
            b["cash_spent"].append(pb["cash_spent"])
            b["cash_received"].append(pb["cash_received"])
            if pb["time_first_order"] is not None:
                b["time_first_order"].append(pb["time_first_order"])
            if pb["time_first_trade"] is not None:
                b["time_first_trade"].append(pb["time_first_trade"])
            b["survived"].append(
                1 if final["players"][pb["idx"]]["survived"] else 0
            )
        posts_per_game.append(posts_this_game)

        # Starvation causes
        for pb in g["players_behavior"]:
            cause = pb.get("starvation_cause")
            if cause is not None:
                starvation_causes[cause] += 1
                total_deaths += 1
                if cause in ("affordable_ask_no_buy", "order_pending_no_fill"):
                    deaths_with_affordable_ask += 1

        # Price telemetry — attempted / submitted / executed distributions
        # and cap-violation totals.
        for pa in g.get("price_attempts", []):
            attempted_prices_all.append(int(pa["attempted_price"]))
            if pa.get("submitted"):
                submitted_prices_all.append(int(pa["attempted_price"]))
            if pa.get("cap_violation"):
                total_cap_violations += 1
        for tp in g.get("executed_trade_prices", []):
            executed_prices_all.append(int(tp["price"]))

        # Market per-food
        for f, m in g["market_by_food"].items():
            pf = per_food[f]
            pf["n_trades"].append(m["n_trades"])
            pf["volume"].append(m["volume"])
            if m["mean_price"] is not None:
                pf["mean_price"].append(m["mean_price"])
            if m["median_price"] is not None:
                pf["median_price"].append(m["median_price"])
            if m["time_first_trade"] is not None:
                pf["time_first_trade"].append(m["time_first_trade"])
            pf["posted_bids"].append(m["posted_bids"])
            pf["posted_asks"].append(m["posted_asks"])
            pf["execution_rate"].append(m["execution_rate"])
            pf["ticks_no_bid"].append(m["ticks_no_bid"])
            pf["ticks_no_ask"].append(m["ticks_no_ask"])
            pf["unused_surplus"].append(m["unused_surplus"])
            # Series → summary scalars
            bids = [x for x in m["best_bid_series"] if x is not None]
            asks = [x for x in m["best_ask_series"] if x is not None]
            spreads = [
                a - b for a, b in zip(m["best_ask_series"], m["best_bid_series"])
                if a is not None and b is not None
            ]
            if bids:
                pf["mean_best_bid"].append(float(np.mean(bids)))
            if asks:
                pf["mean_best_ask"].append(float(np.mean(asks)))
            if spreads:
                pf["mean_spread"].append(float(np.mean(spreads)))
            pf["mean_depth_bid"].append(float(np.mean(m["depth_bid_series"])))
            pf["mean_depth_ask"].append(float(np.mean(m["depth_ask_series"])))

    return {
        "n_games": n_games,
        "n_errored": n_errored,
        "game_reached_time_limit": game_reached_time_limit,
        "n_survivors": n_survivors,
        "no_survivor": no_survivor,
        "first_death_time": first_death_time,
        "n_trades_total": n_trades_total,
        "posts_per_game": posts_per_game,
        "validation_fail_per_game": validation_fail_per_game,
        "cancels_per_game": cancels_per_game,
        "takes_per_game": takes_per_game,
        "by_cadence": by_cadence,
        "starvation_causes": dict(starvation_causes),
        "per_food": per_food,
        "deaths_with_affordable_ask": deaths_with_affordable_ask,
        "total_deaths": total_deaths,
        "attempted_prices_all": attempted_prices_all,
        "submitted_prices_all": submitted_prices_all,
        "executed_prices_all": executed_prices_all,
        "total_cap_violations": total_cap_violations,
    }


# -----------------------------------------------------------------------------
# CSV writers
# -----------------------------------------------------------------------------


def write_summary_csv(all_summaries: Dict[str, Dict], out: Path) -> None:
    fields = [
        "config", "n_games",
        "reach_time_limit_frac",
        "no_survivor_frac",
        "n_survivors_mean", "n_survivors_std",
        "first_death_tick_median", "first_death_tick_ci95",
        "n_trades_total_mean", "n_trades_total_std",
        "posts_per_game_mean", "posts_per_game_std",
        "validation_fails_per_seat_mean",
        "deaths_with_affordable_ask_frac_of_deaths",
        "total_deaths",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cfg, s in all_summaries.items():
            n = s["n_games"]
            deaths = s["total_deaths"]
            m, sd = _mean_std(s["n_trades_total"])
            posts_m, posts_sd = _mean_std(s["posts_per_game"])
            vf_m, _ = _mean_std(s["validation_fail_per_game"])
            surv_m, surv_sd = _mean_std(s["n_survivors"])
            fd_median = (
                float(statistics.median(s["first_death_time"]))
                if s["first_death_time"] else float("nan")
            )
            fd_lo, fd_hi = _bootstrap_ci_median(s["first_death_time"])
            w.writerow({
                "config": cfg,
                "n_games": n,
                "reach_time_limit_frac": f"{sum(s['game_reached_time_limit'])/max(1,n):.3f}",
                "no_survivor_frac": f"{sum(s['no_survivor'])/max(1,n):.3f}",
                "n_survivors_mean": f"{surv_m:.3f}",
                "n_survivors_std": f"{surv_sd:.3f}",
                "first_death_tick_median": f"{fd_median:.1f}" if not math.isnan(fd_median) else "",
                "first_death_tick_ci95": _fmt_ci(fd_lo, fd_hi),
                "n_trades_total_mean": f"{m:.2f}",
                "n_trades_total_std": f"{sd:.2f}",
                "posts_per_game_mean": f"{posts_m:.2f}",
                "posts_per_game_std": f"{posts_sd:.2f}",
                "validation_fails_per_seat_mean": f"{vf_m:.2f}",
                "deaths_with_affordable_ask_frac_of_deaths": (
                    f"{s['deaths_with_affordable_ask']/max(1,deaths):.3f}"
                ),
                "total_deaths": deaths,
            })


def write_by_cadence_csv(all_summaries: Dict[str, Dict], out: Path) -> None:
    fields = [
        "config", "cadence", "n_seats",
        "survival_rate",
        "decisions_taken_mean", "decisions_taken_ci95",
        "posts_total_mean",
        "posts_bid_mean", "posts_ask_mean",
        "posts_replaced_mean",
        "validation_failures_mean",
        "takes_mean",
        "units_bought_mean", "units_sold_mean",
        "cash_spent_mean", "cash_received_mean",
        "time_first_order_median",
        "time_first_trade_median",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cfg, s in all_summaries.items():
            for c in CADENCES:
                b = s["by_cadence"][c]
                if not b or not b.get("survived"):
                    continue
                n_seats = len(b["survived"])
                surv_rate = sum(b["survived"]) / max(1, n_seats)
                dm_lo, dm_hi = _ci95_mean(b["decisions_taken"])
                w.writerow({
                    "config": cfg,
                    "cadence": c,
                    "n_seats": n_seats,
                    "survival_rate": f"{surv_rate:.3f}",
                    "decisions_taken_mean": f"{np.mean(b['decisions_taken']):.2f}",
                    "decisions_taken_ci95": _fmt_ci(dm_lo, dm_hi),
                    "posts_total_mean": f"{np.mean(b['posts_total']):.2f}",
                    "posts_bid_mean": f"{np.mean(b['posts_bid']):.2f}",
                    "posts_ask_mean": f"{np.mean(b['posts_ask']):.2f}",
                    "posts_replaced_mean": f"{np.mean(b['posts_replaced']):.2f}",
                    "validation_failures_mean": f"{np.mean(b['validation_failures']):.2f}",
                    "takes_mean": f"{np.mean(b['takes']):.2f}",
                    "units_bought_mean": f"{np.mean(b['units_bought']):.2f}",
                    "units_sold_mean": f"{np.mean(b['units_sold']):.2f}",
                    "cash_spent_mean": f"{np.mean(b['cash_spent']):.2f}",
                    "cash_received_mean": f"{np.mean(b['cash_received']):.2f}",
                    "time_first_order_median": (
                        f"{statistics.median(b['time_first_order']):.1f}"
                        if b.get("time_first_order") else ""
                    ),
                    "time_first_trade_median": (
                        f"{statistics.median(b['time_first_trade']):.1f}"
                        if b.get("time_first_trade") else ""
                    ),
                })


def write_starvation_csv(all_summaries: Dict[str, Dict], out: Path) -> None:
    causes = [
        "insufficient_cash", "no_ask_available", "affordable_ask_no_buy",
        "order_pending_no_fill", "cash_tied_in_reservations", "other",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "total_deaths"] + causes)
        for cfg, s in all_summaries.items():
            row = [cfg, s["total_deaths"]] + [
                s["starvation_causes"].get(c, 0) for c in causes
            ]
            w.writerow(row)


def write_price_distribution_csv(
    all_summaries: Dict[str, Dict], out: Path
) -> None:
    """Per-config: distributional summary of attempted / submitted /
    executed prices, plus the count of attempts above the 30 / 50 / 100
    thresholds the smoke-test spec calls out."""
    fields = [
        "config", "which",
        "n", "min", "max", "median", "p95", "p99",
        "count_above_30", "count_above_50", "count_above_100",
        "frac_above_30", "frac_above_50", "frac_above_100",
        "cap_violations_total",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cfg, s in all_summaries.items():
            for which, arr in (
                ("attempted", s["attempted_prices_all"]),
                ("submitted", s["submitted_prices_all"]),
                ("executed", s["executed_prices_all"]),
            ):
                n = len(arr)
                if n == 0:
                    row = {k: "" for k in fields}
                    row["config"] = cfg
                    row["which"] = which
                    row["n"] = 0
                    row["cap_violations_total"] = s["total_cap_violations"] if which == "attempted" else ""
                    w.writerow(row)
                    continue
                a = np.asarray(arr, dtype=float)
                c30 = int(np.sum(a > 30))
                c50 = int(np.sum(a > 50))
                c100 = int(np.sum(a > 100))
                row = {
                    "config": cfg, "which": which, "n": n,
                    "min": int(a.min()), "max": int(a.max()),
                    "median": f"{np.median(a):.1f}",
                    "p95": f"{np.percentile(a, 95):.1f}",
                    "p99": f"{np.percentile(a, 99):.1f}",
                    "count_above_30": c30,
                    "count_above_50": c50,
                    "count_above_100": c100,
                    # Scientific-notation for tiny fractions — otherwise a
                    # single attempt among millions rounds to "0.0000",
                    # hiding the very cap-violation the row is meant to flag.
                    "frac_above_30": f"{c30/n:.3e}" if c30 else "0",
                    "frac_above_50": f"{c50/n:.3e}" if c50 else "0",
                    "frac_above_100": f"{c100/n:.3e}" if c100 else "0",
                    "cap_violations_total": (
                        s["total_cap_violations"] if which == "attempted" else ""
                    ),
                }
                w.writerow(row)


def write_market_csv(all_summaries: Dict[str, Dict], out: Path) -> None:
    fields = [
        "config", "food",
        "n_trades_mean", "volume_mean",
        "mean_price_mean", "median_price_median",
        "time_first_trade_median",
        "posted_bids_mean", "posted_asks_mean",
        "execution_rate_mean",
        "mean_spread", "mean_depth_bid", "mean_depth_ask",
        "ticks_no_bid_mean", "ticks_no_ask_mean",
        "unused_surplus_mean",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cfg, s in all_summaries.items():
            for f in FOOD_TYPES:
                pf = s["per_food"][f]
                if not pf:
                    continue
                w.writerow({
                    "config": cfg, "food": f,
                    "n_trades_mean": f"{np.mean(pf['n_trades']):.2f}",
                    "volume_mean": f"{np.mean(pf['volume']):.2f}",
                    "mean_price_mean": (
                        f"{np.mean(pf['mean_price']):.3f}" if pf.get("mean_price") else ""
                    ),
                    "median_price_median": (
                        f"{np.median(pf['median_price']):.3f}"
                        if pf.get("median_price") else ""
                    ),
                    "time_first_trade_median": (
                        f"{np.median(pf['time_first_trade']):.1f}"
                        if pf.get("time_first_trade") else ""
                    ),
                    "posted_bids_mean": f"{np.mean(pf['posted_bids']):.2f}",
                    "posted_asks_mean": f"{np.mean(pf['posted_asks']):.2f}",
                    "execution_rate_mean": f"{np.mean(pf['execution_rate']):.3f}",
                    "mean_spread": (
                        f"{np.mean(pf['mean_spread']):.3f}" if pf.get("mean_spread") else ""
                    ),
                    "mean_depth_bid": f"{np.mean(pf['mean_depth_bid']):.2f}",
                    "mean_depth_ask": f"{np.mean(pf['mean_depth_ask']):.2f}",
                    "ticks_no_bid_mean": f"{np.mean(pf['ticks_no_bid']):.2f}",
                    "ticks_no_ask_mean": f"{np.mean(pf['ticks_no_ask']):.2f}",
                    "unused_surplus_mean": f"{np.mean(pf['unused_surplus']):.2f}",
                })


# -----------------------------------------------------------------------------
# Primary-questions narrative
# -----------------------------------------------------------------------------


def _survival_rate(s: Dict, cadence: str) -> float:
    b = s["by_cadence"].get(cadence, {})
    surv = b.get("survived", [])
    return (sum(surv) / len(surv)) if surv else float("nan")


def _fraction_time_limit(s: Dict) -> float:
    return sum(s["game_reached_time_limit"]) / max(1, s["n_games"])


def write_primary_questions_md(all_summaries: Dict[str, Dict], out: Path) -> None:
    lines: List[str] = []
    lines.append("# Scripted-v1 evaluation — primary questions\n")
    n_any = next(iter(all_summaries.values()))["n_games"]
    lines.append(
        f"Based on {n_any} games per configuration across "
        f"{len(all_summaries)} configurations "
        f"(total games: {sum(s['n_games'] for s in all_summaries.values())}).\n"
    )

    # Q1: functioning market?
    lines.append("## Q1: Does Scripted-v1 reliably sustain a functioning market?\n")
    lines.append("| config | trades/game (mean) | posts/game (mean) | validation fails/seat |")
    lines.append("|---|---:|---:|---:|")
    for cfg, s in all_summaries.items():
        m_tr, _ = _mean_std(s["n_trades_total"])
        m_po, _ = _mean_std(s["posts_per_game"])
        m_vf, _ = _mean_std(s["validation_fail_per_game"])
        lines.append(f"| {cfg} | {m_tr:.1f} | {m_po:.1f} | {m_vf:.1f} |")
    lines.append("")
    lines.append(
        "A functioning market means: nonzero trade counts, orders that "
        "actually cross, and validation failures at a low fraction of "
        "attempts. Judge by whether the above rows have trades > 0 across "
        "all foods (see the market_by_food.csv for the per-food breakdown).\n"
    )

    # Q2: reliably survives?
    lines.append("## Q2: Does at least one cadence reliably survive in the full native action space?\n")
    lines.append("| config | reaches 180s (frac) | no-survivor games (frac) | mean survivors |")
    lines.append("|---|---:|---:|---:|")
    for cfg, s in all_summaries.items():
        surv_m, _ = _mean_std(s["n_survivors"])
        lines.append(
            f"| {cfg} | {_fraction_time_limit(s):.3f} | "
            f"{sum(s['no_survivor'])/max(1,s['n_games']):.3f} | {surv_m:.3f} |"
        )
    lines.append("")

    # Q3: cadence effect
    lines.append("## Q3: How much of performance is explained purely by decision frequency?\n")
    lines.append("Per-seat survival by cadence (all seats across all configs):")
    global_cadence_surv: Dict[str, List[float]] = {c: [] for c in CADENCES}
    for cfg, s in all_summaries.items():
        for c in CADENCES:
            b = s["by_cadence"].get(c, {})
            for x in b.get("survived", []):
                global_cadence_surv[c].append(x)
    lines.append("| cadence | n seat-games | survival rate |")
    lines.append("|---|---:|---:|")
    for c in CADENCES:
        n = len(global_cadence_surv[c])
        rate = sum(global_cadence_surv[c]) / max(1, n)
        lines.append(f"| {c} | {n} | {rate:.3f} |")
    lines.append("")
    lines.append(
        "Compare with survival rates *within each mixed-cadence config* "
        "(by_cadence.csv) — if Fast seats survive more often than Slow "
        "seats in the SAME game (same producers, same initial inventory), "
        "cadence explains a lot; if survival tracks the config "
        "composition rather than the seat cadence, it's mostly a "
        "market-density effect.\n"
    )

    # Q4: dominant causes of death
    lines.append("## Q4: What are the dominant causes of death?\n")
    total_causes: Counter = Counter()
    for s in all_summaries.values():
        for cause, count in s["starvation_causes"].items():
            total_causes[cause] += count
    total_deaths = sum(total_causes.values())
    lines.append(f"Total deaths across all configs: **{total_deaths}**.\n")
    lines.append("| cause | count | share |")
    lines.append("|---|---:|---:|")
    for cause, count in total_causes.most_common():
        share = count / max(1, total_deaths)
        lines.append(f"| {cause} | {count} | {share:.3f} |")
    lines.append("")

    # Q5: assignment / food effects
    lines.append("## Q5: Are particular assignments or foods systematically causing failures?\n")
    lines.append("Per-food across all configs: fraction of games with 0 trades (dead market for that food).")
    dead_food_frac: Dict[str, float] = {}
    for f in FOOD_TYPES:
        totals_zero = 0
        totals_games = 0
        for s in all_summaries.values():
            for n_tr in s["per_food"][f]["n_trades"]:
                totals_games += 1
                if n_tr == 0:
                    totals_zero += 1
        dead_food_frac[f] = totals_zero / max(1, totals_games)
    lines.append("| food | fraction of games with 0 trades |")
    lines.append("|---|---:|")
    for f in FOOD_TYPES:
        lines.append(f"| {f} | {dead_food_frac[f]:.3f} |")
    lines.append("")
    lines.append(
        "For a fuller producer-assignment breakdown, join the per-game "
        "JSONL against seat-cadence and per-food death classification.\n"
    )

    # Q6: Fast as baseline?
    lines.append("## Q6: Is Fast suitable as the primary frozen heuristic baseline and solvability witness?\n")
    fast_only = all_summaries.get("4_fast", {})
    if fast_only:
        surv = fast_only["n_survivors"]
        no_surv = sum(fast_only["no_survivor"]) / max(1, fast_only["n_games"])
        surv_mean = sum(surv) / max(1, len(surv))
        lines.append(f"- 4_fast: mean survivors = **{surv_mean:.3f}**, no-survivor games = **{no_surv:.3f}**")
    slow_only = all_summaries.get("4_slow", {})
    if slow_only:
        surv = slow_only["n_survivors"]
        no_surv = sum(slow_only["no_survivor"]) / max(1, slow_only["n_games"])
        surv_mean = sum(surv) / max(1, len(surv))
        lines.append(f"- 4_slow: mean survivors = **{surv_mean:.3f}**, no-survivor games = **{no_surv:.3f}**")
    lines.append("")
    lines.append(
        "**Verdict for Fast as baseline:** yes if 4_fast reaches 180s "
        "with a positive mean survivor count *and* a small no-survivor "
        "fraction. If the market density (posts/game, trades/game) also "
        "looks healthy in Q1, Fast is a solvability witness.\n"
    )

    out.write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent / "results" / args.tag
    if not root.exists():
        raise SystemExit(f"no such tag directory: {root}")
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl files under {root}")

    print(f"[summarize] tag={args.tag} configs={len(files)}")
    all_summaries: Dict[str, Dict] = {}
    for f in files:
        cfg = f.stem
        print(f"  summarizing {cfg} ...", end="", flush=True)
        s = summarize_config(f)
        err_suffix = f", {s['n_errored']} errored" if s.get("n_errored") else ""
        print(f" ({s['n_games']} games{err_suffix})")
        all_summaries[cfg] = s

    write_summary_csv(all_summaries, root / "summary.csv")
    write_by_cadence_csv(all_summaries, root / "summary_by_cadence.csv")
    write_starvation_csv(all_summaries, root / "starvation_causes.csv")
    write_market_csv(all_summaries, root / "market_by_food.csv")
    write_price_distribution_csv(all_summaries, root / "price_distribution.csv")
    write_primary_questions_md(all_summaries, root / "primary_questions.md")
    print(f"[summarize] wrote 6 outputs under {root}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True)
    sys.exit(main(ap.parse_args()))
