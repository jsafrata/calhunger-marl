from __future__ import annotations

from eval.deep_eval import play_games
from agents.scripted_v4 import ScriptedV4Agent
from agents.scripted_v5 import ScriptedV5Agent
from eval.checkpoint_scripted_diagnostics import (
    OPPONENTS,
    _restore_algorithm,
    summarize_result,
)
from env.small_env import SMALL_GAME_DURATION_SECONDS, SmallFoodTradingEnv


class _NoLearnedPolicyModule:
    def get_initial_state(self) -> dict:
        return {}


def test_checkpoint_diagnostics_registers_v4_and_v5() -> None:
    assert OPPONENTS["v4"] is ScriptedV4Agent
    assert OPPONENTS["v5"] is ScriptedV5Agent


def test_summarize_result_extracts_requested_learned_seat_metrics() -> None:
    result = {
        "terminal_metrics": {
            "outcome_counts": {"dead": 30, "alive_nonwinner": 12, "winner": 8}
        },
        "learned_seat_mean_death_tick": 146.5,
        "learned_seat_mean_final_cash": 173.25,
        "learned_seat_completed_trades_as_buyer_mean": 7.5,
        "learned_seat_completed_trades_as_seller_mean": 9.25,
        "learned_seat_survival_rate": 0.4,
    }

    summary = summarize_result(
        result,
        algorithm="ippo",
        seed=2,
        checkpoint=900,
        n_games=50,
        deterministic=False,
        game_seed_base=900_000,
    )

    assert summary["mean_survival_time_ticks"] == 146.5
    assert summary["mean_terminal_cash"] == 173.25
    assert summary["mean_learned_trade_participations"] == 16.75
    assert summary["win_percentage"] == 16.0
    assert summary["survival_percentage"] == 40.0
    assert summary["mode"] == "stochastic"


def test_play_games_can_use_the_30_second_small_environment() -> None:
    result = play_games(
        _NoLearnedPolicyModule(),
        [False, False, False, False],
        n_games=1,
        seed_base=1234,
        deterministic=True,
        environment_class=SmallFoodTradingEnv,
    )

    assert result["environment_class"] == "SmallFoodTradingEnv"
    assert result["game_duration_seconds"] == SMALL_GAME_DURATION_SECONDS == 30
    assert result["n_games"] == 1


def test_small_diagnostic_entry_point_locks_environment(monkeypatch) -> None:
    import eval.small_checkpoint_scripted_diagnostics as small_diagnostic

    captured: dict[str, object] = {}

    def fake_runner(**kwargs) -> int:
        captured.update(kwargs)
        return 17

    monkeypatch.setattr(
        small_diagnostic, "run_checkpoint_diagnostics", fake_runner
    )

    assert small_diagnostic.main() == 17
    assert captured == {
        "environment_class": SmallFoodTradingEnv,
        "environment_variant": "small",
        "required_duration_seconds": 30,
    }


def test_checkpoint_restore_retries_interrupted_directory_listing(
    monkeypatch, tmp_path
) -> None:
    sentinel = object()
    calls: list[str] = []

    def interrupted_once(path: str):
        calls.append(path)
        if len(calls) == 1:
            raise InterruptedError("transient directory listing interruption")
        return sentinel

    monkeypatch.setattr(
        "eval.checkpoint_scripted_diagnostics.Algorithm.from_checkpoint",
        interrupted_once,
    )

    checkpoint_path = tmp_path / "iter_0800"
    restored = _restore_algorithm(
        checkpoint_path,
        attempts=2,
        retry_delay_seconds=0,
    )

    assert restored is sentinel
    assert calls == [str(checkpoint_path), str(checkpoint_path)]


def test_summarize_result_update_diagnostic_extensions() -> None:
    result = {
        "terminal_metrics": {
            "outcome_counts": {"dead": 30, "alive_nonwinner": 12, "winner": 8},
            "survivor_min_required_buckets": {"frac_0_to_1": 0.35},
        },
        "learned_seat_mean_death_tick": 146.5,
        "learned_seat_mean_final_cash": 173.25,
        "learned_seat_completed_trades_as_buyer_mean": 7.5,
        "learned_seat_completed_trades_as_seller_mean": 9.25,
        "learned_seat_survival_rate": 0.4,
        "learned_seat_posted_bids_mean": 11.0,
        "learned_seat_posted_asks_mean": 4.5,
        "learned_seat_takes_mean": 2.25,
        "learned_seat_type_freq_mean": {"wait": 0.3, "post": 0.6},
    }
    head = {
        "type/mean_prob_post": 0.58,
        "type/entropy_mean": 1.02,
        "price_given_post/entropy_mean": 2.31,
    }

    summary = summarize_result(
        result,
        algorithm="mappo",
        seed=0,
        checkpoint=1200,
        n_games=100,
        deterministic=False,
        game_seed_base=900_000,
        head_diagnostics=head,
    )

    assert summary["win_given_survival_percentage"] == 40.0  # 8 of 20 survivors
    assert summary["mean_trades_as_buyer"] == 7.5
    assert summary["mean_trades_as_seller"] == 9.25
    assert summary["mean_posted_bids"] == 11.0
    assert summary["mean_posted_asks"] == 4.5
    assert summary["mean_takes"] == 2.25
    assert summary["frac_wait"] == 0.3
    assert summary["frac_post"] == 0.6
    assert summary["survivor_frac_min_required_0_to_1"] == 0.35
    assert summary["price_given_post_entropy"] == 2.31
    # Every summary key must be writable by the CSV schema.
    from eval.checkpoint_scripted_diagnostics import SUMMARY_FIELDS

    assert set(summary) <= set(SUMMARY_FIELDS)
