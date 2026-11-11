from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("ray") is None:
    pytest.skip("ray[rllib] is not installed", allow_module_level=True)

from env.rl.callbacks import NashHungerCallbacks


class _FakeEpisode:
    """Mimics an RLlib new-API-stack episode enough for the callback to run.

    The callback iterates over every step's infos and expects the terminal
    step to carry `episode_metrics`. We inject 5 step-infos per seat with
    `reward_dense` / `reward_terminal` values and mark the last step as
    terminal (with the aggregated `episode_metrics` payload)."""

    _AGENT_IDS = ("seat_0", "seat_1", "seat_2", "seat_3")

    def __init__(self, terminal_metrics, per_step_reward=(0.05, 0.1, 0.0, 0.05)):
        self._step_infos = []
        # Non-terminal steps: only dense reward.
        for r in per_step_reward:
            self._step_infos.append({
                agent_id: {
                    "reward_dense": r,
                    "reward_terminal": 0.0,
                    "invalid_action": False,
                }
                for agent_id in self._AGENT_IDS
            })
        # Terminal step: winner gets WIN_REWARD/k, others get either
        # SURVIVED_NON_WIN or 0.
        terminal_step: dict = {}
        for idx, agent_id in enumerate(self._AGENT_IDS):
            p = terminal_metrics["players"][idx]
            terminal_step[agent_id] = {
                "reward_dense": 0.0,
                "reward_terminal": 5.0 if p["win"] else (1.0 if p["survived"] else 0.0),
                "invalid_action": False,
                "episode_metrics": terminal_metrics,
            }
        self._step_infos.append(terminal_step)
        self.custom_metrics = {}

    def get_infos(self, *args):
        # Signature mimics both new-API (no arg) and legacy (-1 arg).
        if args and args[0] == -1:
            return self._step_infos[-1]
        return self._step_infos


def test_callback_records_terminal_and_reward_split():
    """The updated callback logs game outcomes AND separate dense vs
    terminal reward channels + invalid-action counts."""
    metrics = {
        "elapsed_seconds": 30,
        "end_reason": "single_survivor",
        "n_trades": 8,
        "players": [
            {"win": idx == 2, "survived": idx in (1, 2),
             "final_cash": 90 + idx, "death_time": 20 + idx}
            for idx in range(4)
        ],
    }
    episode = _FakeEpisode(metrics)

    NashHungerCallbacks().on_episode_end(episode=episode)

    m = episode.custom_metrics
    # Game outcomes
    assert m["game/episode_length"] == 30
    assert m["game/trades"] == 8
    assert m["game/n_survivors"] == 2.0  # seats 1 and 2 survived
    assert m["game/no_survivors"] == 0.0
    assert m["game/reached_time_limit"] == 0.0  # end_reason was single_survivor

    # Per-seat outcomes
    assert m["seat_2/win"] == 1.0
    assert m["seat_0/win"] == 0.0
    assert m["seat_0/survived"] == 0.0
    assert m["seat_2/survived"] == 1.0

    # Reward channel split — check dense per seat = mean of per_step_reward
    # (4 steps * 0.05, 0.1, 0.0, 0.05 = 0.2 per seat)
    assert m["reward/dense_per_seat"] == pytest.approx(0.2)
    # Terminal per seat: (5.0 for winner + 1.0 for one survivor + 0 + 0) / 4 = 1.5
    assert m["reward/terminal_per_seat"] == pytest.approx(1.5)
    assert m["reward/total_per_seat"] == pytest.approx(1.7)
    # Invariant: no rejected actions in the fixture
    assert m["action_rejections/per_seat_total"] == 0.0
