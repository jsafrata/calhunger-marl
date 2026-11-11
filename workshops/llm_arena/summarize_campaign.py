"""Aggregate the 90 s LLM campaign into one canonical CSV.

Reads every cached game under ``workshops/results/llm_arena/
campaign_90s/`` and emits ``llm_matrix.csv`` with one row per
(model, opponent) cell plus one row per model for the mixed all-LLM
tournament.  Rerunnable at any time: partially filled cells simply
report their current n (the paper cites per-cell n), so late-arriving
Gemini games upgrade rows without touching anything else.

Columns follow the cross-play schema (so LLM cells join the policy and
scripted matrices) plus the LLM reliability/cost extras, including an
estimated USD cost per cell at list prices.

Usage:
    python workshops/llm_arena/summarize_campaign.py \
        [--campaign workshops/results/llm_arena/campaign_90s]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = ("claude", "gpt", "deepseek", "gemini")
OPPONENTS = (
    "v2", "v3", "v4", "v5",
    "ippo_s0", "ippo_s2", "mappo_s1", "mappo_s3",
)

# $/MTok (input, output), list prices Aug 2026.
PRICES = {
    "llm_claude": (1.00, 5.00),
    "llm_gpt": (0.20, 1.20),
    "llm_gemini": (0.15, 1.25),
    "llm_deepseek": (0.22, 0.66),  # off-peak
}

FIELDS = (
    "model", "opponent", "n_games",
    "survival_percentage", "win_percentage",
    "win_given_survival_percentage",
    "mean_terminal_cash", "mean_terminal_cash_given_survival",
    "mean_survival_time_ticks",
    "mean_trades_as_buyer", "mean_trades_as_seller",
    "n_decisions", "malformed_percentage", "rejected_percentage",
    "wait_percentage", "post_percentage", "take_percentage",
    "cancel_percentage",
    "mean_latency_seconds", "p95_latency_seconds",
    "total_input_tokens", "total_output_tokens", "est_cost_usd",
)


def seat_rows(games: list[dict], spec_prefix: str) -> list[dict]:
    return [
        seat
        for game in games
        for seat in game["seats"]
        if seat["spec"].startswith(spec_prefix)
    ]


def summarize(games: list[dict], spec: str) -> dict:
    seats = [s for g in games for s in g["seats"] if s["spec"] == spec]
    decisions = [
        d for g in games for d in g["llm_decisions"] if d["spec"] == spec
    ]
    n = len(seats)
    survivors = sum(s["survived"] for s in seats)
    winners = sum(s["won"] for s in seats)
    cash = [s["terminal_cash"] for s in seats]
    cash_surv = [s["terminal_cash"] for s in seats if s["survived"]]
    lat = [d["latency_seconds"] for d in decisions]
    actions = defaultdict(int)
    malformed = rejected = 0
    tok_in = tok_out = 0
    for d in decisions:
        if d["malformed"]:
            malformed += 1
            actions["wait"] += 1
        else:
            actions[d["decision"]["action"] if d["decision"] else "wait"] += 1
            if d.get("mask_illegal") or (d.get("env_applied") or {}).get(
                "applied"
            ) is False:
                rejected += 1
        tok_in += d.get("input_tokens") or 0
        tok_out += d.get("output_tokens") or 0
    n_dec = len(decisions)
    price_in, price_out = PRICES[spec]
    return {
        "n_games": n,
        "survival_percentage": round(100 * survivors / n, 1),
        "win_percentage": round(100 * winners / n, 1),
        "win_given_survival_percentage": round(
            100 * winners / survivors, 1
        ) if survivors else 0.0,
        "mean_terminal_cash": round(statistics.mean(cash), 2),
        "mean_terminal_cash_given_survival": round(
            statistics.mean(cash_surv), 2
        ) if cash_surv else "",
        "mean_survival_time_ticks": round(
            statistics.mean(s["survival_time_ticks"] for s in seats), 1
        ),
        "mean_trades_as_buyer": round(
            statistics.mean(s["trades_as_buyer"] for s in seats), 2
        ),
        "mean_trades_as_seller": round(
            statistics.mean(s["trades_as_seller"] for s in seats), 2
        ),
        "n_decisions": n_dec,
        "malformed_percentage": round(100 * malformed / n_dec, 2),
        "rejected_percentage": round(100 * rejected / n_dec, 2),
        "wait_percentage": round(100 * actions["wait"] / n_dec, 1),
        "post_percentage": round(100 * actions["post"] / n_dec, 1),
        "take_percentage": round(100 * actions["take"] / n_dec, 1),
        "cancel_percentage": round(100 * actions["cancel"] / n_dec, 1),
        "mean_latency_seconds": round(statistics.mean(lat), 3),
        "p95_latency_seconds": round(sorted(lat)[int(0.95 * len(lat))], 3),
        "total_input_tokens": tok_in,
        "total_output_tokens": tok_out,
        "est_cost_usd": round(
            (tok_in * price_in + tok_out * price_out) / 1e6, 2
        ),
    }


def load_games(cell_dir: Path) -> list[dict]:
    return [
        json.loads(p.read_text())["game"]
        for p in sorted((cell_dir / "games").glob("game_*.json"))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results" / "llm_arena" / "campaign_90s",
    )
    args = parser.parse_args()

    rows = []
    for model in MODELS:
        selfplay = args.campaign / f"{model}_selfplay"
        if (selfplay / "games").is_dir():
            games = load_games(selfplay)
            if games:
                row = {"model": model, "opponent": "selfplay"}
                row.update(summarize(games, f"llm_{model}"))
                rows.append(row)
        for opp in OPPONENTS:
            cell = args.campaign / f"{model}_vs_{opp}"
            if not (cell / "games").is_dir():
                continue
            games = load_games(cell)
            if not games:
                continue
            row = {"model": model, "opponent": opp}
            row.update(summarize(games, f"llm_{model}"))
            rows.append(row)
    mixed = load_games(args.campaign / "mixed_4llm")
    if mixed:
        for model in MODELS:
            row = {"model": model, "opponent": "3llm_mix"}
            row.update(summarize(mixed, f"llm_{model}"))
            rows.append(row)

    out = args.campaign / "llm_matrix.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
