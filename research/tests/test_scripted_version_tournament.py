from collections import Counter

from eval.scripted_version_tournament import (
    VERSIONS,
    _four_way_lineups,
    _pairwise_lineups,
    _seat_exposure,
)


def test_four_way_schedule_balances_every_version_across_every_seat():
    lineups = _four_way_lineups(100, seed=123)

    assert len(lineups) == 100
    assert all(Counter(lineup) == Counter(VERSIONS) for lineup in lineups)
    assert _seat_exposure(lineups) == {
        version: [25, 25, 25, 25] for version in VERSIONS
    }


def test_pairwise_schedule_has_two_of_each_and_exact_seat_balance():
    lineups = _pairwise_lineups("v2", "v5", 50, seed=456)

    assert len(lineups) == 50
    assert all(Counter(lineup) == Counter({"v2": 2, "v5": 2}) for lineup in lineups)
    assert _seat_exposure(lineups) == {
        "v2": [25, 25, 25, 25],
        "v5": [25, 25, 25, 25],
    }
