from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("ray") is None:
    pytest.skip("ray[rllib] is not installed", allow_module_level=True)
if importlib.util.find_spec("evaluate_ppo") is None:
    # evaluate_ppo was a Phase-A helper script that is being replaced by
    # `eval.pilot_eval`. Skip until the new module lands.
    pytest.skip("evaluate_ppo module not present in this repo", allow_module_level=True)

from evaluate_ppo import _aggregate, _play_game, _rotations  # noqa: E402


def test_lineup_rotation_places_each_entry_in_each_seat():
    rotations = _rotations(["policy", "greedy", "buffer", "random"], True)

    assert len(rotations) == 4
    for agent in ("policy", "greedy", "buffer", "random"):
        assert {lineup.index(agent) for lineup in rotations} == {0, 1, 2, 3}


def test_no_rotation_keeps_one_lineup():
    lineup = ["policy", "greedy", "greedy", "noop"]
    assert _rotations(lineup, False) == [lineup]


def test_aggregate_reports_agent_and_physical_seat_metrics():
    games = [
        {
            "elapsed_seconds": 10,
            "n_trades": 2,
            "players": [
                {
                    "seat": idx,
                    "agent": name,
                    "win": idx == 0,
                    "survived": idx < 2,
                    "final_cash": 100 + idx,
                    "death_time": 10 - idx,
                }
                for idx, name in enumerate(
                    ["policy", "greedy", "buffer", "random"]
                )
            ],
        }
    ]

    summary = _aggregate(games)

    assert summary["by_agent"]["policy"]["win_rate"] == 1.0
    assert summary["by_agent"]["random"]["survival_rate"] == 0.0
    assert summary["by_seat"]["0"]["win_rate"] == 1.0


def test_mixed_game_accepts_policy_and_handcrafted_bots(fake_policy_module):
    result = _play_game(
        fake_policy_module,
        ["policy", "greedy", "buffer", "random"],
        seed=4,
        stochastic=False,
    )

    assert len(result["players"]) == 4
    assert {player["agent"] for player in result["players"]} == {
        "policy",
        "greedy",
        "buffer",
        "random",
    }


def test_deterministic_policy_evaluation_repeats_for_fixed_seed(
    fake_policy_module,
):
    first = _play_game(
        fake_policy_module,
        ["policy", "greedy", "buffer", "random"],
        seed=19,
        stochastic=False,
    )
    second = _play_game(
        fake_policy_module,
        ["policy", "greedy", "buffer", "random"],
        seed=19,
        stochastic=False,
    )

    assert first == second
