from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.long_baseline import _should_save_checkpoint
from hpc.v2_5seed_cpu.common import (
    CAMPAIGN_DIR,
    load_campaign,
    resolve_task,
    run_dir,
    tasks,
)
from hpc.v2_5seed_cpu.run_task import build_command
from hpc.v2_5seed_cpu.preflight import _is_allowed_hpc_storage_path
from hpc.v2_5seed_cpu.verify_campaign import verify_run


def test_campaign_maps_ten_tasks_in_algorithm_then_seed_order() -> None:
    campaign = load_campaign()
    assert len(tasks(campaign)) == 10
    assert resolve_task(campaign, 0) == ("ippo", 0)
    assert resolve_task(campaign, 4) == ("ippo", 4)
    assert resolve_task(campaign, 5) == ("mappo", 0)
    assert resolve_task(campaign, 9) == ("mappo", 4)
    with pytest.raises(ValueError):
        resolve_task(campaign, 10)


def test_checkpoint_schedule_starts_at_100_and_always_saves_final() -> None:
    saved = [
        iteration
        for iteration in range(1, 1501)
        if _should_save_checkpoint(
            absolute_iter=iteration,
            local_iter=iteration,
            total_local_iters=1500,
            checkpoint_every=100,
            checkpoint_start=100,
        )
    ]
    assert saved == list(range(100, 1501, 100))
    assert _should_save_checkpoint(
        absolute_iter=37,
        local_iter=37,
        total_local_iters=37,
        checkpoint_every=100,
        checkpoint_start=100,
    )


def test_fresh_command_is_collision_safe_and_fully_frozen(tmp_path: Path) -> None:
    campaign = load_campaign()
    command, target, remaining = build_command(
        campaign=campaign,
        research_root=tmp_path,
        campaign_root=tmp_path / "campaign",
        algorithm="mappo",
        seed=3,
        resume=False,
    )
    joined = " ".join(command)
    assert target == tmp_path / "campaign" / "mappo_seed3_headmlp"
    assert remaining == 1500
    assert "--fail_if_output_exists" in command
    assert "--checkpoint_start 100" in joined
    assert "--checkpoint_every 100" in joined
    assert "--variant_suffix seed3_headmlp" in joined
    assert "--num_env_runners 2" in joined
    assert "--train_batch_size 4096" in joined


def test_resume_command_uses_latest_unambiguous_checkpoint(tmp_path: Path) -> None:
    campaign = load_campaign()
    target = run_dir(tmp_path, "ippo", 2)
    checkpoint = target / "checkpoints" / "iter_0500"
    checkpoint.mkdir(parents=True)
    with (target / "iters.jsonl").open("w") as fh:
        for iteration in range(1, 501):
            fh.write(json.dumps({"iter": iteration}) + "\n")

    command, command_target, remaining = build_command(
        campaign=campaign,
        research_root=tmp_path,
        campaign_root=tmp_path,
        algorithm="ippo",
        seed=2,
        resume=True,
    )
    assert command_target == target
    assert remaining == 1000
    assert "--fail_if_output_exists" not in command
    checkpoint_index = command.index("--resume_from_checkpoint") + 1
    assert command[checkpoint_index] == str(checkpoint)

    with (target / "iters.jsonl").open("a") as fh:
        fh.write(json.dumps({"iter": 501}) + "\n")
    with pytest.raises(ValueError, match="ambiguous resume"):
        build_command(
            campaign=campaign,
            research_root=tmp_path,
            campaign_root=tmp_path,
            algorithm="ippo",
            seed=2,
            resume=True,
        )


def test_slurm_resources_match_campaign_source_of_truth() -> None:
    campaign = load_campaign()
    script = (CAMPAIGN_DIR / "run_array.sbatch").read_text()
    resources = campaign["resources"]
    assert f"#SBATCH --partition={resources['partition']}" in script
    assert f"#SBATCH --array={resources['array']}" in script
    assert f"#SBATCH --cpus-per-task={resources['cpus_per_task']}" in script
    assert f"#SBATCH --mem={resources['memory_gb']}G" in script
    assert f"#SBATCH --time={resources['walltime']}" in script
    assert "#SBATCH --gres" not in script


def test_hpc_python_lock_and_module_loading_are_frozen() -> None:
    campaign = load_campaign()
    assert campaign["locked_versions"]["python_major_minor"] == "3.11"
    env_example = (CAMPAIGN_DIR / "campaign.env.example").read_text()
    assert (
        'HUNGER_PYTHON_MODULE="python/3.11.6-gcc-11.3.1-rphh4kv"'
        in env_example
    )
    for script_name in (
        "setup_env.sh",
        "validate_plan.sh",
        "smoke_test.sbatch",
        "run_array.sbatch",
    ):
        script = (CAMPAIGN_DIR / script_name).read_text()
        assert 'module load "${HUNGER_PYTHON_MODULE}"' in script


def test_preflight_allows_persistent_group_or_own_home_storage() -> None:
    assert _is_allowed_hpc_storage_path(
        Path("/resnick/home/jsafrata/hunger_runs/campaign"), "jsafrata"
    )
    assert _is_allowed_hpc_storage_path(
        Path("/resnick/groups/Hippo/jsafrata/campaign"), "jsafrata"
    )
    assert not _is_allowed_hpc_storage_path(
        Path("/resnick/home/someone_else/campaign"), "jsafrata"
    )
    assert not _is_allowed_hpc_storage_path(
        Path("/resnick/scratch/jsafrata/campaign"), "jsafrata"
    )
    assert not _is_allowed_hpc_storage_path(Path("/tmp/campaign"), "jsafrata")


def _write_complete_fake_run(run_path: Path) -> None:
    campaign = load_campaign()
    run_path.mkdir(parents=True)
    with (run_path / "iters.jsonl").open("w") as fh:
        for iteration in range(1, campaign["iterations"] + 1):
            fh.write(json.dumps({"iter": iteration}) + "\n")
    for iteration in campaign["checkpoint_iterations"]:
        checkpoint = run_path / "checkpoints" / f"iter_{iteration:04d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "rllib_checkpoint.json").write_text("{}\n")
    (run_path / "latest_checkpoint.txt").write_text("iter_1500\n")


def test_verifier_accepts_exact_complete_run_and_rejects_extra_checkpoint(
    tmp_path: Path,
) -> None:
    campaign = load_campaign()
    path = run_dir(tmp_path, "ippo", 0)
    _write_complete_fake_run(path)
    assert verify_run(path, campaign) == []

    extra = path / "checkpoints" / "iter_0450"
    extra.mkdir(parents=True)
    (extra / "rllib_checkpoint.json").write_text("{}\n")
    errors = verify_run(path, campaign)
    assert any("checkpoint mismatch" in error for error in errors)
