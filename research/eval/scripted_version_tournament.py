"""Compare scripted bot versions in balanced mixed-version matches.

The default diagnostic uses the 30-second baseline environment and fast
decision cadence.  It runs one four-way match (one seat per version) and every
pairwise version combination (two seats per version), balancing every version
equally across all four physical seats within each match.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def _run_match(
    *,
    name: str,
    lineups: Sequence[Sequence[str]],
    environment_variant: str,
    seed_base: int,
) -> dict:
    config, environment_class = SCALED_ENVIRONMENTS[environment_variant]
    accum: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

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
            observations, _, _, _ = env.step(actions)

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

    return {
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
) -> dict:
    matches = []
    four_way = _four_way_lineups(games_four_way, seed_base + 101)
    matches.append(
        _run_match(
            name="V2 vs V3 vs V4 vs V5",
            lineups=four_way,
            environment_variant=environment_variant,
            seed_base=seed_base,
        )
    )
    for pair_index, (left, right) in enumerate(combinations(VERSIONS, 2)):
        print(f"running {left.upper()} vs {right.upper()} ({games_pairwise} games)")
        lineups = _pairwise_lineups(
            left,
            right,
            games_pairwise,
            seed=seed_base + 1000 + pair_index,
        )
        matches.append(
            _run_match(
                name=f"{left.upper()} vs {right.upper()} (2 seats each)",
                lineups=lineups,
                environment_variant=environment_variant,
                seed_base=seed_base,
            )
        )
    return {
        "environment_variant": environment_variant,
        "games_four_way": games_four_way,
        "games_per_pairwise_match": games_pairwise,
        "total_games": games_four_way + 6 * games_pairwise,
        "seed_base": seed_base,
        "matches": matches,
    }


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
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    print(
        f"running V2 vs V3 vs V4 vs V5 ({args.games_four_way} games)"
    )
    report = run_diagnostic(
        environment_variant=args.environment_variant,
        games_four_way=args.games_four_way,
        games_pairwise=args.games_pairwise,
        seed_base=args.seed_base,
    )
    rows = _rows(report["matches"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "scripted_version_tournament.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.out_dir / "scripted_version_tournament.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "SCRIPTED_VERSION_TOURNAMENT.md").write_text(
        _markdown(rows)
    )
    print(f"wrote results to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
