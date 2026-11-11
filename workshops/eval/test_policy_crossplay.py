"""Unit tests for the cross-play harness.

Run as:
    PYTHONPATH=research pytest workshops/eval/test_policy_crossplay.py

Scripted-only lineups are used throughout so no Ray/torch/checkpoints
are required; the policy path shares the exact same lineup, seeding,
record and summary code, which is what these tests pin down.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint_policy_crossplay as cpc  # noqa: E402

SMALL_ENV = cpc.ENVIRONMENT_SPECS["small"][0]


def _scripted(version: str) -> cpc.SeatSpec:
    return cpc.parse_spec(f"scripted:{version}")


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


def test_parse_scripted_spec() -> None:
    spec = cpc.parse_spec("scripted:v3")
    assert spec.kind == "scripted"
    assert spec.key == "scripted_v3"


def test_parse_policy_spec_requires_checkpoint_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        cpc.parse_spec(f"policy:label={tmp_path}")
    (tmp_path / "rllib_checkpoint.json").write_text("{}")
    spec = cpc.parse_spec(f"policy:label={tmp_path}@pi_0")
    assert spec.kind == "policy"
    assert spec.label == "label"
    assert spec.module_id == "pi_0"


@pytest.mark.parametrize(
    "text",
    ["scripted:v9", "policy:missing_equals", "bare_garbage"],
)
def test_parse_spec_rejects_malformed(text: str) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        cpc.parse_spec(text)


# ---------------------------------------------------------------------------
# Seat balance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_games", [4, 8, 12])
def test_rotated_lineups_balance_every_slot(n_games: int) -> None:
    lineups = cpc.rotated_lineups(n_games)
    assert len(lineups) == n_games
    for slot in range(4):
        seat_counts = Counter(
            lineup.index(slot) for lineup in lineups
        )
        assert seat_counts == {
            seat: n_games // 4 for seat in range(4)
        }, f"slot {slot} is not seat-balanced: {seat_counts}"


@pytest.mark.parametrize("n_games", [0, 3, 6, -4])
def test_rotated_lineups_reject_non_multiples_of_four(n_games: int) -> None:
    with pytest.raises(ValueError):
        cpc.rotated_lineups(n_games)


def test_1v3_focal_occupies_each_seat_equally() -> None:
    slots = cpc.lineup_slots(
        "1v3", _scripted("v2"), _scripted("v3"), None
    )
    lineups = cpc.rotated_lineups(8)
    focal_seats = Counter(lineup.index(0) for lineup in lineups)
    assert focal_seats == {0: 2, 1: 2, 2: 2, 3: 2}
    # The three opponent slots fill every other seat.
    assert all(
        sorted(
            seat
            for seat, slot in enumerate(lineup)
            if slots[slot].key == "scripted_v3"
        )
        == sorted(set(range(4)) - {lineup.index(0)})
        for lineup in lineups
    )


def test_lineup_slots_validation() -> None:
    with pytest.raises(ValueError):
        cpc.lineup_slots("1v3", None, _scripted("v2"), None)
    with pytest.raises(ValueError):
        cpc.lineup_slots("mixed", None, None, [_scripted("v2")] * 3)
    with pytest.raises(ValueError):
        cpc.lineup_slots("nope", None, None, None)


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------


def test_same_seed_base_reproduces_identical_raw_records() -> None:
    slots = cpc.lineup_slots(
        "mixed",
        None,
        None,
        [_scripted("v2"), _scripted("v3"), _scripted("v4"),
         _scripted("v5")],
    )
    kwargs = dict(
        deterministic=False,
        environment_class=SMALL_ENV,
        module_pool=None,
    )
    first = cpc.play_crossplay_games(slots, 4, 1_300_000, **kwargs)
    second = cpc.play_crossplay_games(slots, 4, 1_300_000, **kwargs)
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True
    )
    # A different seed base must actually change something.
    third = cpc.play_crossplay_games(slots, 4, 42, **kwargs)
    assert json.dumps(first, sort_keys=True) != json.dumps(
        third, sort_keys=True
    )


# ---------------------------------------------------------------------------
# Summary + CSV schema
# ---------------------------------------------------------------------------


def _tiny_run() -> tuple[list[dict], list[cpc.SeatSpec]]:
    slots = cpc.lineup_slots("1v3", _scripted("v2"), _scripted("v3"), None)
    records = cpc.play_crossplay_games(
        slots,
        4,
        1_300_000,
        deterministic=False,
        environment_class=SMALL_ENV,
        module_pool=None,
    )
    return records, slots


def test_summary_rows_schema_and_ranges() -> None:
    records, slots = _tiny_run()
    rows = cpc.summarize_records(
        records,
        slots,
        lineup_mode="1v3",
        mode="stochastic",
        game_seed_base=1_300_000,
        environment_variant="small",
        environment_class=SMALL_ENV.__name__,
        game_duration_seconds=30,
    )
    # One row per distinct spec; focal (slot 0) first.
    assert [row["focal"] for row in rows] == [
        "scripted_v2", "scripted_v3"
    ]
    assert rows[0]["opponent"] == "3xscripted_v3"
    assert rows[0]["n_focal_seat_games"] == 4  # 1 focal seat x 4 games
    assert rows[1]["n_focal_seat_games"] == 12  # 3 seats x 4 games
    for row in rows:
        assert tuple(row) == cpc.SUMMARY_FIELDS
        assert 0.0 <= row["survival_percentage"] <= 100.0
        assert 0.0 <= row["win_percentage"] <= 100.0
        assert 0.0 <= row["win_given_survival_percentage"] <= 100.0
        assert 0.0 <= row["mean_survival_time_ticks"] <= 30.0


def test_matrix_csv_schema(tmp_path) -> None:
    records, slots = _tiny_run()
    rows = cpc.summarize_records(
        records,
        slots,
        lineup_mode="1v3",
        mode="stochastic",
        game_seed_base=1_300_000,
        environment_variant="small",
        environment_class=SMALL_ENV.__name__,
        game_duration_seconds=30,
    )
    (tmp_path / "raw").mkdir()
    raw_path = cpc.raw_result_path(tmp_path, slots, "1v3", "stochastic", 4)
    raw_path.write_text(
        json.dumps({"identity": {}, "summary_rows": rows, "games": records})
    )
    matrix_rows = cpc.rebuild_matrix(tmp_path)
    assert len(matrix_rows) == len(rows)
    with (tmp_path / "crossplay_matrix.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == cpc.SUMMARY_FIELDS
        parsed = list(reader)
    assert len(parsed) == len(rows)
    assert {row["focal"] for row in parsed} == {
        "scripted_v2", "scripted_v3"
    }


# ---------------------------------------------------------------------------
# Raw-cache resumability
# ---------------------------------------------------------------------------


def test_cache_identity_mismatch_raises(tmp_path) -> None:
    slots = cpc.lineup_slots("1v3", _scripted("v2"), _scripted("v3"), None)
    identity = cpc.job_identity(
        slots,
        lineup_mode="1v3",
        mode="stochastic",
        n_games=4,
        game_seed_base=1_300_000,
        environment_variant="small",
        environment_class=SMALL_ENV.__name__,
        game_duration_seconds=30,
    )
    path = tmp_path / "job.json"
    assert cpc.load_cached_report(path, identity) is None  # no file yet
    path.write_text(json.dumps({"identity": identity, "summary_rows": []}))
    assert cpc.load_cached_report(path, identity) is not None
    changed = dict(identity, game_seed_base=999)
    with pytest.raises(ValueError):
        cpc.load_cached_report(path, changed)
