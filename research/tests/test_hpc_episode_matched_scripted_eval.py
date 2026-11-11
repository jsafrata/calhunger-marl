from __future__ import annotations

from pathlib import Path

import pytest

from env.env import FoodTradingEnv
from env.small_env import (
    FoodTradingEnv60,
    FoodTradingEnv90,
    FoodTradingEnv120,
    FoodTradingEnv150,
)
from eval.scaled_checkpoint_scripted_diagnostics import resolve_environment


ROOT = Path(__file__).resolve().parents[1]
HPC_DIR = ROOT / "hpc" / "episode_matched_scripted_eval"


@pytest.mark.parametrize(
    ("variant", "expected_class", "duration"),
    [
        ("small_60", FoodTradingEnv60, 60),
        ("small_90", FoodTradingEnv90, 90),
        ("small_120", FoodTradingEnv120, 120),
        ("small_150", FoodTradingEnv150, 150),
        ("default", FoodTradingEnv, 180),
    ],
)
def test_wrapper_selects_training_horizon(
    variant: str, expected_class: type[FoodTradingEnv], duration: int
) -> None:
    environment_class, observed_duration = resolve_environment(variant)
    assert environment_class is expected_class
    assert observed_duration == duration
    assert environment_class(seed=0).config.game_duration_seconds == duration


def test_wrapper_rejects_unknown_environment() -> None:
    with pytest.raises(ValueError, match="HUNGER_ENVIRONMENT_VARIANT"):
        resolve_environment("small_30")


def test_slurm_script_covers_four_opponents_and_all_checkpoints() -> None:
    script = (HPC_DIR / "run_array.sbatch").read_text()
    assert "#SBATCH --array=0-3%4" in script
    assert "opponents=(v2 v3 v4 v5)" in script
    assert "--algorithms ippo,mappo" in script
    assert '--seeds "${HUNGER_EVAL_SEEDS:-0,1}"' in script
    assert "--checkpoint-start 100" in script
    assert "--checkpoint-stop 1200" in script
    assert "--checkpoint-every 100" in script
    assert "--n-games 100" in script
    assert "eval.plot_checkpoint_scripted_diagnostics" not in script
    assert "MPLCONFIGDIR" not in script


def test_submission_excludes_unfinished_180_and_scales_walltimes() -> None:
    submit = (HPC_DIR / "submit_all.sh").read_text()
    assert "environments=(small_60 small_90 small_120 small_150)" in submit
    assert "walltimes=(04:00:00 06:00:00 08:00:00 10:00:00)" in submit

    submit_180 = (HPC_DIR / "submit_180.sh").read_text()
    assert "--time=05:00:00" in submit_180
    assert "HUNGER_ENVIRONMENT_VARIANT=default" in submit_180
