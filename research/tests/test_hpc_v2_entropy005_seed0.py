from __future__ import annotations

from pathlib import Path

from hpc.v2_5seed_cpu.run_task import build_command
from hpc.v2_entropy005_seed0.common import (
    CAMPAIGN_DIR,
    load_campaign,
    resolve_task,
)


def test_pilot_maps_ippo_then_mappo_for_seed_zero() -> None:
    campaign = load_campaign()
    assert resolve_task(campaign, 0) == ("ippo", 0)
    assert resolve_task(campaign, 1) == ("mappo", 0)


def test_pilot_command_changes_only_intended_entropy_setting(tmp_path: Path) -> None:
    campaign = load_campaign()
    command, target, remaining = build_command(
        campaign=campaign,
        research_root=tmp_path,
        campaign_root=tmp_path / "entropy005",
        algorithm="mappo",
        seed=0,
        resume=False,
    )
    joined = " ".join(command)
    assert remaining == 1500
    assert target == tmp_path / "entropy005" / "mappo_seed0_headmlp"
    assert "--entropy_coeff 0.05" in joined
    assert "--checkpoint_every 100" in joined
    assert "--checkpoint_start 100" in joined
    assert "--train_batch_size 4096" in joined
    assert "--minibatch_size 1024" in joined
    assert "--num_epochs 4" in joined
    assert "--num_env_runners 2" in joined
    assert "--reward_kind terminal_only" in joined
    assert "--fail_if_output_exists" in command


def test_slurm_resources_match_pilot_campaign() -> None:
    campaign = load_campaign()
    script = (CAMPAIGN_DIR / "run_array.sbatch").read_text()
    resources = campaign["resources"]
    assert f"#SBATCH --partition={resources['partition']}" in script
    assert f"#SBATCH --array={resources['array']}" in script
    assert f"#SBATCH --cpus-per-task={resources['cpus_per_task']}" in script
    assert f"#SBATCH --mem={resources['memory_gb']}G" in script
    assert f"#SBATCH --time={resources['walltime']}" in script
    assert "#SBATCH --gres" not in script
