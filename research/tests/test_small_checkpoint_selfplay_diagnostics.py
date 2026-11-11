from __future__ import annotations

from eval.small_checkpoint_selfplay_diagnostics import summarize_selfplay_result


def test_summarize_selfplay_result_extracts_requested_metrics() -> None:
    result = {
        "environment_class": "SmallFoodTradingEnv",
        "game_duration_seconds": 30,
        "n_learned_seats": 4,
        "learned_seat_mean_death_tick": 24.75,
        "lineup_mean_trades": 18.5,
        "learned_seat_survival_rate": 0.42,
    }

    summary = summarize_selfplay_result(
        result,
        algorithm="mappo",
        seed=1,
        checkpoint=800,
        n_games=50,
        deterministic=False,
        game_seed_base=950_000,
    )

    assert summary["environment_variant"] == "small"
    assert summary["game_duration_seconds"] == 30
    assert summary["learned_seats"] == 4
    assert summary["mean_survival_time_seconds"] == 24.75
    assert summary["mean_completed_trades_per_game"] == 18.5
    assert summary["survival_percentage"] == 42.0
