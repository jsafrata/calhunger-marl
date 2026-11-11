from __future__ import annotations

from eval.checkpoint_selfplay_survival_diagnostics import summarize_result


def test_summarize_result_retains_only_requested_metrics() -> None:
    result = {
        "environment_class": "FoodTradingEnv90",
        "game_duration_seconds": 90,
        "n_learned_seats": 4,
        "learned_seat_mean_death_tick": 73.25,
        "learned_seat_survival_rate": 0.42,
        "lineup_mean_trades": 999.0,
    }

    summary = summarize_result(
        result,
        algorithm="mappo",
        seed=1,
        checkpoint=800,
        n_games=50,
        deterministic=False,
        game_seed_base=970_000,
        environment_variant="small_90",
    )

    assert summary["environment_variant"] == "small_90"
    assert summary["game_duration_seconds"] == 90
    assert summary["learned_seats"] == 4
    assert summary["average_death_tick"] == 73.25
    assert summary["survival_percentage"] == 42.0
    assert "mean_completed_trades_per_game" not in summary
