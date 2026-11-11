from __future__ import annotations

from pathlib import Path

import pytest

from env.env import FoodTradingEnv
from env.rl.rllib_env import NashHungerMultiAgentEnv
from env.small_env import SmallFoodTradingEnv
from hpc.small_2seed_cpu.common import (
    CAMPAIGN_DIR,
    load_campaign,
    resolve_task,
    tasks,
)
from hpc.v2_5seed_cpu.run_task import build_command


def test_campaign_maps_four_tasks_in_algorithm_then_seed_order() -> None:
    campaign = load_campaign()
    assert tasks(campaign) == [
        ("ippo", 0),
        ("ippo", 1),
        ("mappo", 0),
        ("mappo", 1),
    ]
    for task_id, expected in enumerate(tasks(campaign)):
        assert resolve_task(campaign, task_id) == expected
    with pytest.raises(ValueError):
        resolve_task(campaign, 4)


def test_small_command_is_frozen_and_collision_safe(tmp_path: Path) -> None:
    campaign = load_campaign()
    command, target, remaining = build_command(
        campaign=campaign,
        research_root=tmp_path,
        campaign_root=tmp_path / "small-campaign",
        algorithm="mappo",
        seed=1,
        resume=False,
    )
    joined = " ".join(command)
    assert remaining == 1500
    assert target == tmp_path / "small-campaign" / "mappo_seed1_headmlp"
    assert "--env_variant small" in joined
    assert "--checkpoint_every 100" in joined
    assert "--checkpoint_start 100" in joined
    assert "--entropy_coeff 0.01" in joined
    assert "--reward_kind terminal_only" in joined
    assert "--fail_if_output_exists" in command


def test_slurm_resources_and_array_match_campaign() -> None:
    campaign = load_campaign()
    script = (CAMPAIGN_DIR / "run_array.sbatch").read_text()
    resources = campaign["resources"]
    assert f"#SBATCH --partition={resources['partition']}" in script
    assert f"#SBATCH --array={resources['array']}" in script
    assert f"#SBATCH --cpus-per-task={resources['cpus_per_task']}" in script
    assert f"#SBATCH --mem={resources['memory_gb']}G" in script
    assert f"#SBATCH --time={resources['walltime']}" in script
    assert "#SBATCH --gres" not in script
    assert "hpc.small_2seed_cpu.preflight" in script
    assert "hpc.small_2seed_cpu.run_task" in script


def test_rllib_adapter_selects_small_environment_and_dealer() -> None:
    adapter = NashHungerMultiAgentEnv(
        {"seed": 17, "reward_kind": "terminal_only", "env_variant": "small"}
    )
    adapter.reset(seed=17)
    assert isinstance(adapter._env, SmallFoodTradingEnv)
    assert adapter.config.game_duration_seconds == 30
    assert adapter._env is not None
    for player in adapter._env.state.players:
        required = adapter.config.required_foods(player.produces)
        values = [player.inventory[food] for food in required]
        assert sum(player.inventory.values()) == 16
        assert sum(values) == 8
        assert all(3 <= value <= 5 for value in values)


def test_rllib_adapter_default_environment_is_unchanged() -> None:
    adapter = NashHungerMultiAgentEnv({"seed": 17})
    adapter.reset(seed=17)
    assert type(adapter._env) is FoodTradingEnv
    assert adapter.config.game_duration_seconds == 180


def test_rllib_adapter_rejects_unknown_environment_variant() -> None:
    with pytest.raises(ValueError, match="env_variant"):
        NashHungerMultiAgentEnv({"env_variant": "typo"})
