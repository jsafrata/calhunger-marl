from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.long_baseline import _should_save_checkpoint
from hpc.scaled_4env_2seed_cpu import preflight
from hpc.scaled_4env_2seed_cpu.common import (
    CAMPAIGN_DIR,
    campaign_for_environment,
    environment_root,
    load_campaign,
    resolve_task,
    tasks,
)
from hpc.scaled_4env_2seed_cpu.verify_campaign import verify_campaign
from hpc.v2_5seed_cpu.common import run_dir
from hpc.v2_5seed_cpu.run_task import build_command


def test_campaign_maps_all_sixteen_tasks_in_environment_algorithm_seed_order() -> None:
    campaign = load_campaign()
    expected = [
        (environment, algorithm, seed)
        for environment in ("small_60", "small_90", "small_120", "small_150")
        for algorithm in ("ippo", "mappo")
        for seed in (0, 1)
    ]
    assert tasks(campaign) == expected
    for task_id, task in enumerate(expected):
        assert resolve_task(campaign, task_id) == task
    with pytest.raises(ValueError):
        resolve_task(campaign, 16)


def test_every_task_builds_a_frozen_collision_safe_command(tmp_path: Path) -> None:
    campaign = load_campaign()
    campaign_root = tmp_path / "scaled-campaign"
    for environment, algorithm, seed in tasks(campaign):
        environment_campaign = campaign_for_environment(campaign, environment)
        output_root = environment_root(campaign_root, environment)
        command, target, remaining = build_command(
            campaign=environment_campaign,
            research_root=tmp_path,
            campaign_root=output_root,
            algorithm=algorithm,
            seed=seed,
            resume=False,
        )
        joined = " ".join(command)
        assert remaining == 1200
        assert target == output_root / f"{algorithm}_seed{seed}_headmlp"
        assert f"--env_variant {environment}" in joined
        assert "--iters 1200" in joined
        assert "--checkpoint_every 100" in joined
        assert "--checkpoint_start 100" in joined
        assert "--entropy_coeff 0.01" in joined
        assert "--reward_kind terminal_only" in joined
        assert "--fail_if_output_exists" in command


def test_checkpoint_schedule_is_exactly_100_through_1200() -> None:
    saved = [
        iteration
        for iteration in range(1, 1201)
        if _should_save_checkpoint(
            absolute_iter=iteration,
            local_iter=iteration,
            total_local_iters=1200,
            checkpoint_every=100,
            checkpoint_start=100,
        )
    ]
    assert saved == list(range(100, 1201, 100))


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
    assert "hpc.scaled_4env_2seed_cpu.preflight" in script
    assert "hpc.scaled_4env_2seed_cpu.run_task" in script
    assert not (CAMPAIGN_DIR / "smoke_test.sbatch").exists()


def test_preflight_checks_all_four_environment_dealers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        preflight,
        "inspect_base",
        lambda research_root, campaign_root, campaign: {
            "ok": True,
            "errors": [],
            "warnings": [],
            "research_root": str(research_root),
            "campaign_root": str(campaign_root),
            "git_revision": "test",
            "git_dirty": False,
            "python": "3.11.0",
            "architecture": "x86_64",
            "versions": {},
        },
    )
    report = preflight.inspect(tmp_path, tmp_path / "campaign")
    assert report["ok"] is True
    assert report["errors"] == []
    observed = report["scaled_environments"]
    assert isinstance(observed, list)
    assert [entry["variant"] for entry in observed] == [
        "small_60",
        "small_90",
        "small_120",
        "small_150",
    ]
    assert [entry["duration_seconds"] for entry in observed] == [60, 90, 120, 150]
    assert [entry["initial_units_per_player"] for entry in observed] == [
        32,
        48,
        64,
        80,
    ]


def _write_complete_run(run_path: Path, campaign: dict) -> None:
    run_path.mkdir(parents=True)
    with (run_path / "iters.jsonl").open("w") as fh:
        for iteration in range(1, int(campaign["iterations"]) + 1):
            fh.write(json.dumps({"iter": iteration}) + "\n")
    for iteration in campaign["checkpoint_iterations"]:
        checkpoint = run_path / "checkpoints" / f"iter_{iteration:04d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "rllib_checkpoint.json").write_text("{}\n")
    (run_path / "latest_checkpoint.txt").write_text("iter_1200\n")


def test_verifier_accepts_all_sixteen_complete_runs(tmp_path: Path) -> None:
    campaign = load_campaign()
    for environment, algorithm, seed in tasks(campaign):
        path = run_dir(environment_root(tmp_path, environment), algorithm, seed)
        _write_complete_run(path, campaign_for_environment(campaign, environment))

    report = verify_campaign(tmp_path)
    assert report["ok"] is True
    assert len(report["runs"]) == 16
    assert all(run["ok"] for run in report["runs"])


def test_resume_is_scoped_to_the_environment_subdirectory(tmp_path: Path) -> None:
    campaign = load_campaign()
    environment = "small_120"
    environment_campaign = campaign_for_environment(campaign, environment)
    output_root = environment_root(tmp_path, environment)
    target = run_dir(output_root, "mappo", 1)
    checkpoint = target / "checkpoints" / "iter_0600"
    checkpoint.mkdir(parents=True)
    with (target / "iters.jsonl").open("w") as fh:
        for iteration in range(1, 601):
            fh.write(json.dumps({"iter": iteration}) + "\n")

    command, command_target, remaining = build_command(
        campaign=environment_campaign,
        research_root=tmp_path,
        campaign_root=output_root,
        algorithm="mappo",
        seed=1,
        resume=True,
    )
    assert command_target == target
    assert remaining == 600
    assert command[command.index("--resume_from_checkpoint") + 1] == str(checkpoint)
    assert "--env_variant small_120" in " ".join(command)
