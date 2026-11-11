"""E30 — Price formation on the heuristic ladder (full-ledger tournament).

COPY of ``workshops/eval/scripted_version_tournament.py`` extended with a
microstructure ledger.  The tournament schedule, seeds, env construction and
agent construction are byte-identical to the original run
(``workshops/results/scripted_v2_v5_tournament_small_30s/``): environment
variant ``small`` (30 s baseline), fast cadence, 100 seat-balanced four-way
games + 6x50 pairwise 2v2 games, seed_base 1_300_000.  The extension only
*reads* additional state (env.state.trades, env.state.order_books,
StepInfo.actions_applied), so per-version tournament metrics must reproduce
the published numbers exactly; the script verifies this against the original
JSON and refuses to bless the run otherwise.

Ledger per game (one JSONL line per game, one file per match):
  - every executed trade: ``dataclasses.asdict`` over ``env.state.trades``
    plus maker/taker attribution (see below),
  - per-second L1 book state: best bid / best ask / depth per food, sampled
    after every ``env.step`` (pattern from ``workshops/eval/game_runner.py``),
  - the seat->version mapping and per-seat terminal outcomes.

Maker/taker attribution (documented, from research/env/market.py semantics):
  - Every order id is created by a POST; ``StepInfo.actions_applied[seat]``
    carries ``{"kind": "post", "order_id": ...}`` in both the sequential and
    the simultaneous-clearing pathway, so an order_id -> seat registry built
    from actions_applied is complete.
  - ``Trade.taker_order_id == -1`` is the sentinel for a TAKE execution
    (immediate-or-cancel take_from_book): the maker is the owner of
    ``maker_order_id`` (the resting order); the taker is the counterparty
    player (buyer or seller, whichever is not the maker).
  - For post-crossed trades (clear_books), maker = the crossing order with
    the EARLIER sequence number; the trade prints at the maker's price.
    Both order ids are in the registry; maker seat = registry[maker_order_id],
    taker seat = the other party.
  - In every case the trade executes at the maker's quoted price, so
    "maker" = price setter / liquidity provider, "taker" = aggressor.

Outputs under ``workshops/econml/results/ladder_microstructure/``:
  ledger/*.jsonl, archetype_micro.csv, pairflow.csv, price_formation.csv,
  tournament_reproduction.{json,csv,md}, validation.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import numpy as np

from agents.cadenced_scripted import CadencedScripted
from agents.scripted_v1 import ScriptedV1Agent
from agents.scripted_v2 import ScriptedV2Agent
from agents.scripted_v3 import ScriptedV3Agent
from agents.scripted_v4 import ScriptedV4Agent
from agents.scripted_v5 import ScriptedV5Agent
from env.small_env import SCALED_ENVIRONMENTS


BOT_CLASSES = {
    "v1": ScriptedV1Agent,
    "v2": ScriptedV2Agent,
    "v3": ScriptedV3Agent,
    "v4": ScriptedV4Agent,
    "v5": ScriptedV5Agent,
}
VERSIONS = ("v2", "v3", "v4", "v5")
CSV_FIELDS = (
    "match",
    "environment_variant",
    "game_duration_seconds",
    "n_games",
    "version",
    "n_seat_games",
    "survival_rate_percent",
    "win_rate_percent",
    "mean_terminal_cash",
    "mean_survival_time_seconds",
)


# ---------------------------------------------------------------------------
# Schedule construction — UNCHANGED from the original script.
# ---------------------------------------------------------------------------


def _four_way_lineups(n_games: int, seed: int) -> list[tuple[str, ...]]:
    """Return lineups with each version occupying each seat equally often."""
    if n_games <= 0 or n_games % 4:
        raise ValueError("four-way game count must be a positive multiple of 4")
    lineups: list[tuple[str, ...]] = []
    rng = np.random.default_rng(seed)
    rotations = [
        tuple(VERSIONS[(seat + shift) % 4] for seat in range(4))
        for shift in range(4)
    ]
    for _ in range(n_games // 4):
        order = list(rng.permutation(4))
        lineups.extend(rotations[index] for index in order)
    return lineups


def _pairwise_lineups(
    left: str,
    right: str,
    n_games: int,
    seed: int,
) -> list[tuple[str, ...]]:
    """Return two-vs-two lineups with exact per-seat exposure balance."""
    if left == right or left not in BOT_CLASSES or right not in BOT_CLASSES:
        raise ValueError("pairwise match requires two distinct known versions")
    if n_games <= 0 or n_games % 2:
        raise ValueError("pairwise game count must be a positive even number")

    complementary_pairs = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    full_cycles, remainder = divmod(n_games, 6)
    left_seat_sets: list[tuple[int, int]] = [
        seats
        for _ in range(full_cycles)
        for pair in complementary_pairs
        for seats in pair
    ]
    for pair in complementary_pairs[: remainder // 2]:
        left_seat_sets.extend(pair)

    if len(left_seat_sets) != n_games:
        raise AssertionError("failed to construct the requested pairwise schedule")
    rng = np.random.default_rng(seed)
    rng.shuffle(left_seat_sets)
    return [
        tuple(left if seat in left_seats else right for seat in range(4))
        for left_seats in left_seat_sets
    ]


def _seat_exposure(lineups: Iterable[Sequence[str]]) -> dict[str, list[int]]:
    exposure = {version: [0, 0, 0, 0] for version in VERSIONS}
    for lineup in lineups:
        for seat, version in enumerate(lineup):
            exposure[version][seat] += 1
    return {version: counts for version, counts in exposure.items() if any(counts)}


# ---------------------------------------------------------------------------
# Match runner — game loop copied from the original; ledger capture added.
# The env is stepped exactly as before; everything new is read-only.
# ---------------------------------------------------------------------------


def _run_match(
    *,
    name: str,
    slug: str,
    lineups: Sequence[Sequence[str]],
    environment_variant: str,
    seed_base: int,
    ledger_dir: Path,
) -> tuple[dict, list[dict]]:
    config, environment_class = SCALED_ENVIRONMENTS[environment_variant]
    food_types = list(config.food_types)
    accum: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    ledger_records: list[dict] = []

    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / f"{slug}.jsonl"

    with ledger_path.open("w") as ledger_handle:
        for game_index, lineup in enumerate(lineups):
            game_seed = seed_base + game_index
            env = environment_class(seed=game_seed)
            observations = env.reset()
            agents = [
                CadencedScripted(
                    "fast",
                    seed=game_seed * 17 + seat,
                    inner_factory=BOT_CLASSES[version],
                )
                for seat, version in enumerate(lineup)
            ]
            for agent in agents:
                agent.reset()

            # --- ledger state (read-only additions) --------------------
            order_owner: dict[int, int] = {}  # order_id -> seat
            book_l1: list[dict] = []

            while not env.is_done():
                actions = {}
                for seat, agent in enumerate(agents):
                    legal = env.legal_actions(seat)
                    action = agent.act(observations[seat], legal)
                    if not legal[action]:
                        raise RuntimeError(
                            f"{lineup[seat]} returned illegal action {action} "
                            f"in {name}, game {game_index}, seat {seat}"
                        )
                    actions[seat] = action
                observations, _, _, step_info = env.step(actions)

                # Order-id registry from actions_applied (posts only; both
                # clearing pathways report the created order's id).
                for seat, applied in step_info.actions_applied.items():
                    if (
                        applied.get("applied")
                        and applied.get("kind") == "post"
                        and applied.get("order_id") is not None
                    ):
                        order_owner[applied["order_id"]] = seat

                # Per-second L1 book sample AFTER this step (pattern from
                # workshops/eval/game_runner.py, market-depth samples).
                sample = {"tick": int(env.state.elapsed_seconds), "books": {}}
                for food in food_types:
                    book = env.state.order_books[food]
                    best_bid = max(
                        (o.price_per_unit for o in book.bids), default=None
                    )
                    best_ask = min(
                        (o.price_per_unit for o in book.asks), default=None
                    )
                    sample["books"][food] = {
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "depth_bid": sum(o.remaining_quantity for o in book.bids),
                        "depth_ask": sum(o.remaining_quantity for o in book.asks),
                    }
                book_l1.append(sample)

            result = env.get_result()
            winners = set(result["winner_idxs"])
            for seat, version in enumerate(lineup):
                player = result["players"][seat]
                alive = player["status"] == "alive"
                survival_time = (
                    player["died_at"]
                    if player["died_at"] is not None
                    else result["elapsed_seconds"]
                )
                accum[version]["survived"].append(float(alive))
                accum[version]["won"].append(float(seat in winners))
                accum[version]["terminal_cash"].append(float(player["final_cash"]))
                accum[version]["survival_time"].append(float(survival_time))

            # --- assemble the per-game ledger record -------------------
            trades = []
            for trade in env.state.trades:
                record = asdict(trade)
                maker_idx = order_owner.get(trade.maker_order_id)
                is_take = trade.taker_order_id == -1
                if maker_idx is None:
                    taker_idx = None
                elif maker_idx == trade.buyer_idx:
                    taker_idx = trade.seller_idx
                else:
                    taker_idx = trade.buyer_idx
                record["maker_idx"] = maker_idx
                record["taker_idx"] = taker_idx
                record["is_take"] = is_take
                record["maker_version"] = (
                    lineup[maker_idx] if maker_idx is not None else None
                )
                record["taker_version"] = (
                    lineup[taker_idx] if taker_idx is not None else None
                )
                trades.append(record)

            seats = []
            for seat, version in enumerate(lineup):
                player = result["players"][seat]
                seats.append(
                    {
                        "seat": seat,
                        "version": version,
                        "produces": list(player["produces"]),
                        "status": player["status"],
                        "died_at": player["died_at"],
                        "won": seat in winners,
                        "final_cash": player["final_cash"],
                        "final_inventory": dict(player["final_inventory"]),
                        "total_bought": player["total_bought"],
                        "total_sold": player["total_sold"],
                        "cash_from_trades": player["cash_from_trades"],
                        "cash_spent_on_trades": player["cash_spent_on_trades"],
                    }
                )

            game_record = {
                "match": name,
                "match_slug": slug,
                "game_index": game_index,
                "game_seed": game_seed,
                "environment_variant": environment_variant,
                "lineup": list(lineup),
                "end_reason": result["end_reason"],
                "elapsed_seconds": result["elapsed_seconds"],
                "winner_seats": sorted(winners),
                "seats": seats,
                "trades": trades,
                "book_l1": book_l1,
            }
            ledger_handle.write(json.dumps(game_record) + "\n")
            ledger_records.append(game_record)

    per_version = {}
    for version in sorted(accum):
        values = accum[version]
        per_version[version] = {
            "n_seat_games": len(values["survived"]),
            "survival_rate_percent": 100.0 * fmean(values["survived"]),
            "win_rate_percent": 100.0 * fmean(values["won"]),
            "mean_terminal_cash": fmean(values["terminal_cash"]),
            "mean_survival_time_seconds": fmean(values["survival_time"]),
        }

    report = {
        "match": name,
        "environment_variant": environment_variant,
        "environment_class": environment_class.__name__,
        "game_duration_seconds": int(config.game_duration_seconds),
        "n_games": len(lineups),
        "seed_base": seed_base,
        "cadence": "fast",
        "seat_exposure": _seat_exposure(lineups),
        "per_version": per_version,
    }
    return report, ledger_records


# ---------------------------------------------------------------------------
# Reporting — UNCHANGED from the original script.
# ---------------------------------------------------------------------------


def _rows(matches: Sequence[dict]) -> list[dict]:
    rows = []
    for match in matches:
        for version, metrics in match["per_version"].items():
            rows.append(
                {
                    "match": match["match"],
                    "environment_variant": match["environment_variant"],
                    "game_duration_seconds": match["game_duration_seconds"],
                    "n_games": match["n_games"],
                    "version": version,
                    **metrics,
                }
            )
    return rows


def _markdown(rows: Sequence[dict]) -> str:
    lines = [
        "# Scripted bot mixed-version diagnostic",
        "",
        "All results use the 30-second baseline environment and fast cadence. "
        "Survival and win rates are per seat-game; terminal cash and survival "
        "time are averaged over every seat-game for that version.",
        "",
    ]
    current = None
    for row in rows:
        if row["match"] != current:
            if current is not None:
                lines.append("")
            current = row["match"]
            lines.extend(
                [
                    f"## {current}",
                    "",
                    "| Version | Seat-games | Survival rate | Win rate | "
                    "Terminal cash | Survival time |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
        lines.append(
            f"| {row['version'].upper()} | {row['n_seat_games']} | "
            f"{row['survival_rate_percent']:.1f}% | "
            f"{row['win_rate_percent']:.1f}% | "
            f"{row['mean_terminal_cash']:.2f} | "
            f"{row['mean_survival_time_seconds']:.2f} s |"
        )
    lines.append("")
    return "\n".join(lines)


def run_diagnostic(
    *,
    environment_variant: str,
    games_four_way: int,
    games_pairwise: int,
    seed_base: int,
    ledger_dir: Path,
) -> tuple[dict, dict[str, list[dict]]]:
    matches = []
    ledgers: dict[str, list[dict]] = {}
    four_way = _four_way_lineups(games_four_way, seed_base + 101)
    print(f"running V2 vs V3 vs V4 vs V5 ({games_four_way} games)")
    report, records = _run_match(
        name="V2 vs V3 vs V4 vs V5",
        slug="fourway",
        lineups=four_way,
        environment_variant=environment_variant,
        seed_base=seed_base,
        ledger_dir=ledger_dir,
    )
    matches.append(report)
    ledgers["fourway"] = records
    for pair_index, (left, right) in enumerate(combinations(VERSIONS, 2)):
        print(f"running {left.upper()} vs {right.upper()} ({games_pairwise} games)")
        lineups = _pairwise_lineups(
            left,
            right,
            games_pairwise,
            seed=seed_base + 1000 + pair_index,
        )
        slug = f"pair_{left}_{right}"
        report, records = _run_match(
            name=f"{left.upper()} vs {right.upper()} (2 seats each)",
            slug=slug,
            lineups=lineups,
            environment_variant=environment_variant,
            seed_base=seed_base,
            ledger_dir=ledger_dir,
        )
        matches.append(report)
        ledgers[slug] = records
    return (
        {
            "environment_variant": environment_variant,
            "games_four_way": games_four_way,
            "games_per_pairwise_match": games_pairwise,
            "total_games": games_four_way + 6 * games_pairwise,
            "seed_base": seed_base,
            "matches": matches,
        },
        ledgers,
    )


# ---------------------------------------------------------------------------
# Validation against the published tournament JSON.
# ---------------------------------------------------------------------------


def validate_against_original(report: dict, original_json: Path) -> dict:
    original = json.loads(original_json.read_text())
    mismatches: list[str] = []
    orig_matches = {m["match"]: m for m in original["matches"]}
    for match in report["matches"]:
        orig = orig_matches.get(match["match"])
        if orig is None:
            mismatches.append(f"match {match['match']!r} missing from original")
            continue
        for version, metrics in match["per_version"].items():
            orig_metrics = orig["per_version"].get(version)
            if orig_metrics is None:
                mismatches.append(f"{match['match']}/{version}: missing in original")
                continue
            for key, value in metrics.items():
                orig_value = orig_metrics.get(key)
                if value != orig_value:
                    mismatches.append(
                        f"{match['match']}/{version}/{key}: "
                        f"reproduced={value!r} original={orig_value!r}"
                    )
    return {
        "original_json": str(original_json),
        "exact_reproduction": not mismatches,
        "n_metrics_compared": sum(
            len(m["per_version"]) * 5 for m in report["matches"]
        ),
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Analyses.
# ---------------------------------------------------------------------------


def _vw_mean_std(prices: list[float], quantities: list[float]) -> tuple[float, float]:
    total = sum(quantities)
    if total <= 0:
        return float("nan"), float("nan")
    mean = sum(p * q for p, q in zip(prices, quantities)) / total
    var = sum(q * (p - mean) ** 2 for p, q in zip(prices, quantities)) / total
    return mean, math.sqrt(var)


def analyze_archetype_micro(fourway_records: list[dict]) -> list[dict]:
    """Per-archetype realized trading economics in the four-way match."""
    stats: dict[str, dict[str, float]] = {
        v: defaultdict(float) for v in VERSIONS
    }
    buy_px: dict[str, list[tuple[float, float]]] = {v: [] for v in VERSIONS}
    sell_px: dict[str, list[tuple[float, float]]] = {v: [] for v in VERSIONS}
    n_seat_games = {v: 0 for v in VERSIONS}

    for game in fourway_records:
        lineup = game["lineup"]
        for version in lineup:
            n_seat_games[version] += 1
        for trade in game["trades"]:
            buyer_v = lineup[trade["buyer_idx"]]
            seller_v = lineup[trade["seller_idx"]]
            qty = trade["quantity"]
            price = trade["price_per_unit"]
            maker_v = trade["maker_version"]
            taker_v = trade["taker_version"]

            for role_v in (buyer_v, seller_v):
                stats[role_v]["participations"] += 1
                stats[role_v]["participation_units"] += qty
            if maker_v is not None:
                stats[maker_v]["maker_trades"] += 1
                stats[maker_v]["maker_units"] += qty
            if taker_v is not None:
                stats[taker_v]["taker_trades"] += 1
                stats[taker_v]["taker_units"] += qty
                if trade["is_take"]:
                    stats[taker_v]["take_action_trades"] += 1

            stats[buyer_v]["units_bought"] += qty
            stats[buyer_v]["cash_spent"] += trade["total_price"]
            stats[seller_v]["units_sold"] += qty
            stats[seller_v]["cash_received"] += trade["total_price"]
            buy_px[buyer_v].append((price, qty))
            sell_px[seller_v].append((price, qty))

        for seat_record in game["seats"]:
            v = seat_record["version"]
            stats[v]["net_trade_cashflow"] += (
                seat_record["cash_from_trades"]
                - seat_record["cash_spent_on_trades"]
            )

    rows = []
    for v in VERSIONS:
        s = stats[v]
        n = n_seat_games[v]
        buy_mean, _ = _vw_mean_std(*map(list, zip(*buy_px[v])) if buy_px[v] else ([], []))
        sell_mean, _ = _vw_mean_std(*map(list, zip(*sell_px[v])) if sell_px[v] else ([], []))
        participations = s["participations"]
        rows.append(
            {
                "version": v,
                "n_seat_games": n,
                "trade_participations": int(participations),
                "maker_trades": int(s["maker_trades"]),
                "taker_trades": int(s["taker_trades"]),
                "maker_share_percent": (
                    100.0 * s["maker_trades"] / participations
                    if participations
                    else float("nan")
                ),
                "take_action_share_of_taker_trades_percent": (
                    100.0 * s["take_action_trades"] / s["taker_trades"]
                    if s["taker_trades"]
                    else float("nan")
                ),
                "units_bought_per_seat_game": s["units_bought"] / n,
                "units_sold_per_seat_game": s["units_sold"] / n,
                "volume_per_seat_game": (s["units_bought"] + s["units_sold"]) / n,
                "vw_mean_buy_price": buy_mean,
                "vw_mean_sell_price": sell_mean,
                "sell_minus_buy_price": sell_mean - buy_mean,
                "cash_spent_per_seat_game": s["cash_spent"] / n,
                "cash_received_per_seat_game": s["cash_received"] / n,
                "net_trade_cashflow_per_seat_game": s["net_trade_cashflow"] / n,
            }
        )
    return rows


def analyze_pairflow(
    ledgers: dict[str, list[dict]], report: dict
) -> list[dict]:
    """Per pairwise match: net cash flow between the two versions."""
    per_version_by_match = {
        m["match"]: m["per_version"] for m in report["matches"]
    }
    rows = []
    for left, right in combinations(VERSIONS, 2):
        slug = f"pair_{left}_{right}"
        records = ledgers[slug]
        match_name = f"{left.upper()} vs {right.upper()} (2 seats each)"
        flow_left_to_right = 0.0  # cash paid by LEFT buyers to RIGHT sellers
        flow_right_to_left = 0.0
        cross_units = 0
        within_left_units = 0
        within_right_units = 0
        n_games = len(records)
        for game in records:
            lineup = game["lineup"]
            for trade in game["trades"]:
                buyer_v = lineup[trade["buyer_idx"]]
                seller_v = lineup[trade["seller_idx"]]
                if buyer_v == left and seller_v == right:
                    flow_left_to_right += trade["total_price"]
                    cross_units += trade["quantity"]
                elif buyer_v == right and seller_v == left:
                    flow_right_to_left += trade["total_price"]
                    cross_units += trade["quantity"]
                elif buyer_v == left:  # seller also left
                    within_left_units += trade["quantity"]
                else:
                    within_right_units += trade["quantity"]
        net_to_left = flow_right_to_left - flow_left_to_right
        metrics = per_version_by_match[match_name]
        rows.append(
            {
                "pair": f"{left}_vs_{right}",
                "left": left,
                "right": right,
                "n_games": n_games,
                "left_win_rate_percent": metrics[left]["win_rate_percent"],
                "right_win_rate_percent": metrics[right]["win_rate_percent"],
                "left_mean_terminal_cash": metrics[left]["mean_terminal_cash"],
                "right_mean_terminal_cash": metrics[right]["mean_terminal_cash"],
                "cash_left_pays_right_total": flow_left_to_right,
                "cash_right_pays_left_total": flow_right_to_left,
                "net_trade_cash_to_left_per_game": net_to_left / n_games,
                "cross_version_units_per_game": cross_units / n_games,
                "within_left_units_per_game": within_left_units / n_games,
                "within_right_units_per_game": within_right_units / n_games,
                "net_flow_favors": (
                    left if net_to_left > 0 else right if net_to_left < 0 else "even"
                ),
            }
        )
    return rows


def analyze_price_formation(
    ledgers: dict[str, list[dict]], food_types: Sequence[str]
) -> list[dict]:
    """Per-lineup price level and dispersion; cross-experiment row last."""
    rows = []
    for slug, records in ledgers.items():
        lineup_versions = sorted(set(records[0]["lineup"]))
        prices: list[float] = []
        quantities: list[float] = []
        per_food: dict[str, list[tuple[float, float]]] = {
            f: [] for f in food_types
        }
        n_trades = 0
        n_units = 0
        for game in records:
            for trade in game["trades"]:
                prices.append(trade["price_per_unit"])
                quantities.append(trade["quantity"])
                per_food[trade["food_type"]].append(
                    (trade["price_per_unit"], trade["quantity"])
                )
                n_trades += 1
                n_units += trade["quantity"]
        vw_mean, vw_std = _vw_mean_std(prices, quantities)
        row = {
            "row_type": "lineup",
            "lineup": slug,
            "versions_present": "+".join(lineup_versions),
            "contains_v4": "v4" in lineup_versions,
            "n_games": len(records),
            "trades_per_game": n_trades / len(records),
            "units_per_game": n_units / len(records),
            "vw_mean_price": vw_mean,
            "vw_price_std": vw_std,
        }
        for food in food_types:
            pairs = per_food[food]
            f_mean, _ = _vw_mean_std(
                [p for p, _ in pairs], [q for _, q in pairs]
            )
            row[f"vw_mean_price_{food}"] = f_mean
        rows.append(row)

    # Cross-experiment comparison row (four-way lineup), formatted for later
    # comparison against the LLM-economy and HPC monoculture runs.
    fourway = next(r for r in rows if r["lineup"] == "fourway")
    rows.append(
        {
            **{k: fourway[k] for k in fourway},
            "row_type": "cross_experiment_comparison",
            "lineup": "E30_ladder_fourway_small30_fast",
        }
    )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-variant",
        choices=tuple(SCALED_ENVIRONMENTS),
        default="small",
    )
    parser.add_argument("--games-four-way", type=int, default=100)
    parser.add_argument("--games-pairwise", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=1_300_000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "ladder_microstructure",
    )
    parser.add_argument(
        "--original-json",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "scripted_v2_v5_tournament_small_30s"
        / "scripted_version_tournament.json",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = args.out_dir / "ledger"

    report, ledgers = run_diagnostic(
        environment_variant=args.environment_variant,
        games_four_way=args.games_four_way,
        games_pairwise=args.games_pairwise,
        seed_base=args.seed_base,
        ledger_dir=ledger_dir,
    )

    # Tournament reproduction outputs (same trio as the original run).
    rows = _rows(report["matches"])
    (args.out_dir / "tournament_reproduction.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.out_dir / "tournament_reproduction.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "TOURNAMENT_REPRODUCTION.md").write_text(_markdown(rows))

    # Validation against the published run.
    validation = validate_against_original(report, args.original_json)
    (args.out_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    if validation["exact_reproduction"]:
        print(
            "VALIDATION: exact reproduction of the published tournament "
            f"({validation['n_metrics_compared']} metrics compared)"
        )
    else:
        print("VALIDATION: MISMATCHES FOUND — see validation.json")
        for line in validation["mismatches"][:20]:
            print(f"  {line}")

    # Analyses.
    config, _ = SCALED_ENVIRONMENTS[args.environment_variant]
    archetype_rows = analyze_archetype_micro(ledgers["fourway"])
    pairflow_rows = analyze_pairflow(ledgers, report)
    price_rows = analyze_price_formation(ledgers, list(config.food_types))
    _write_csv(args.out_dir / "archetype_micro.csv", archetype_rows)
    _write_csv(args.out_dir / "pairflow.csv", pairflow_rows)
    _write_csv(args.out_dir / "price_formation.csv", price_rows)

    print(f"wrote results to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
