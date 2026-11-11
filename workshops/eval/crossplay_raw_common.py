"""Shared loader for per-game crossplay raw JSONs (E19/E20 analyses).

Reads the ``raw/*.json`` files written by ``checkpoint_policy_crossplay`` and
exposes per-game focal outcomes as aligned numpy arrays so bootstrap analyses
can exploit the paired-seed design (game index i has the same ``game_seed``
in every cell of a campaign).

Focal identification: in a ``1v3`` cell the focal label appears in exactly
one seat per game (seat rotation moves it around); in a ``mixed`` self-play
cell all four seats share the spec and per-game values are the mean over
seats, matching how ``n_focal_seat_games`` aggregates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
SEEDS234 = RESULTS_ROOT / "seeds234"
PEER_RAW = SEEDS234 / "_peer_crossplay_100" / "raw"
# 900-game seed extension (campaign peer_crossplay_900ext, job 1791932):
# seeds 900100-900999, contiguous with the base set for a 1000-game merge.
PEER_RAW_900 = SEEDS234 / "_peer_crossplay_900ext" / "raw"
LADDER_RAW = SEEDS234 / "_ladder_validity_100" / "raw"

LABELS = [f"{a}_s{s}" for a in ("ippo", "mappo") for s in range(5)]
HEALTHY = [f"ippo_s{s}" for s in range(5)] + ["mappo_s0", "mappo_s2", "mappo_s3"]
BOTS = [f"scripted_v{i}" for i in range(1, 6)]

# Per-game focal quantities, in column order.
QUANTITIES = ("survived", "won", "terminal_cash", "survival_time_ticks")


@dataclass
class Cell:
    focal: str
    opponent: str
    lineup_mode: str
    n_games: int
    game_seed_base: int
    # (n_games, len(QUANTITIES)) — per-game focal outcome, game index == seed offset
    values: np.ndarray


def _focal_game_values(game: dict, focal: str, selfplay: bool) -> list[float]:
    seats = [s for s in game["seats"] if s["spec"] == focal]
    if not selfplay and len(seats) != 1:
        raise ValueError(
            f"expected exactly one focal seat for {focal}, got {len(seats)}"
        )
    cols = []
    for q in QUANTITIES:
        vals = [float(s[q]) for s in seats]
        cols.append(sum(vals) / len(vals))
    return cols


def load_cell(path: Path, focal: str) -> Cell:
    with open(path) as fh:
        data = json.load(fh)
    ident = data["identity"]
    games = data["games"]
    selfplay = ident["lineup_mode"] != "1v3"
    base = ident["game_seed_base"]
    values = np.empty((len(games), len(QUANTITIES)))
    for i, game in enumerate(games):
        if game["game_seed"] != base + game["game_index"] or game["game_index"] != i:
            raise ValueError(f"seed pairing broken in {path.name} at game {i}")
        values[i] = _focal_game_values(game, focal, selfplay)
    labels = [s["label"] for s in ident["slots"]]
    opponent = next((l for l in labels if l != focal), focal)
    return Cell(
        focal=focal,
        opponent=opponent,
        lineup_mode=ident["lineup_mode"],
        n_games=len(games),
        game_seed_base=base,
        values=values,
    )


def merge_cells(base: Cell, ext: Cell) -> Cell:
    """Concatenate two runs of the same cell with contiguous seed ranges."""
    if ext.game_seed_base != base.game_seed_base + base.n_games:
        raise ValueError(
            f"seed ranges not contiguous: {base.game_seed_base}+{base.n_games} "
            f"!= {ext.game_seed_base}"
        )
    if (base.focal, base.opponent, base.lineup_mode) != (
        ext.focal, ext.opponent, ext.lineup_mode
    ):
        raise ValueError("merging cells with different identities")
    return Cell(
        focal=base.focal,
        opponent=base.opponent,
        lineup_mode=base.lineup_mode,
        n_games=base.n_games + ext.n_games,
        game_seed_base=base.game_seed_base,
        values=np.concatenate([base.values, ext.values]),
    )


def load_peer_1v3(extended: bool = False) -> dict[str, list[Cell]]:
    """Per policy, its 9 focal 1v3 cells vs each other policy, opponent-sorted.

    ``extended`` merges the 900-game seed extension for 1000 games per cell.
    """
    out: dict[str, list[Cell]] = {}
    for focal in LABELS:
        cells = []
        for opp in LABELS:
            if opp == focal:
                continue
            stem = f"{focal}_vs_{opp}_1v3_stochastic"
            cell = load_cell(PEER_RAW / f"{stem}_100.json", focal)
            if extended:
                cell = merge_cells(
                    cell, load_cell(PEER_RAW_900 / f"{stem}_900.json", focal)
                )
            cells.append(cell)
        out[focal] = cells
    return out


def load_peer_selfplay(extended: bool = False) -> dict[str, Cell]:
    out = {}
    for focal in LABELS:
        stem = "mixed_" + "_".join([focal] * 4) + "_stochastic"
        cell = load_cell(PEER_RAW / f"{stem}_100.json", focal)
        if extended:
            cell = merge_cells(
                cell, load_cell(PEER_RAW_900 / f"{stem}_900.json", focal)
            )
        out[focal] = cell
    return out


def _load_seats(path: Path) -> np.ndarray:
    with open(path) as fh:
        data = json.load(fh)
    base = data["identity"]["game_seed_base"]
    games = data["games"]
    values = np.empty((len(games), 4, len(QUANTITIES)))
    for i, game in enumerate(games):
        if game["game_seed"] != base + game["game_index"]:
            raise ValueError(f"seed pairing broken in {path.name} at game {i}")
        for j, seat in enumerate(game["seats"]):
            values[i, j] = [float(seat[q]) for q in QUANTITIES]
    return values


def load_peer_selfplay_seats(extended: bool = False) -> dict[str, np.ndarray]:
    """Per policy, per-seat self-play outcomes: (n_games, 4, len(QUANTITIES)).

    Needed for survivor-conditioned metrics: the per-game seat mean of
    terminal cash in self-play is 100.0 by conservation (trading only moves
    cash between the four seats), so raw self-play cash is informative only
    at the seat level conditioned on survival.
    """
    out = {}
    for focal in LABELS:
        stem = "mixed_" + "_".join([focal] * 4) + "_stochastic"
        values = _load_seats(PEER_RAW / f"{stem}_100.json")
        if extended:
            values = np.concatenate(
                [values, _load_seats(PEER_RAW_900 / f"{stem}_900.json")]
            )
        out[focal] = values
    return out


def load_ladder() -> dict[str, list[Cell]]:
    """Per policy, its 5 cells vs 3x scripted v1..v5, rung-sorted."""
    out: dict[str, list[Cell]] = {}
    for focal in LABELS:
        cells = []
        for bot in BOTS:
            path = LADDER_RAW / f"{focal}_vs_{bot}_1v3_stochastic_100.json"
            cells.append(load_cell(path, focal))
        out[focal] = cells
    return out


def stack_values(cells_by_policy: dict[str, list[Cell]], labels: list[str]) -> np.ndarray:
    """(n_policies, n_cells, n_games, n_quantities) tensor, cell/game aligned."""
    return np.stack(
        [np.stack([c.values for c in cells_by_policy[p]]) for p in labels]
    )
