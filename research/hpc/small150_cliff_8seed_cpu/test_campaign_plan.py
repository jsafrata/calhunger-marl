"""Plan tests for the small_150 discovery-cliff campaign (seeds 2-9).

Run locally from the staging area with:

    PYTHONPATH=<research>:<staging dir> pytest test_campaign_plan.py

The import shim below also lets the same file run after the campaign
directory is added under ``research/hpc/`` on deploy day.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

try:
    _PKG = "hpc.small150_cliff_8seed_cpu"
    common = importlib.import_module(_PKG + ".common")
except ModuleNotFoundError:
    _PKG = "small150_cliff_8seed_cpu"
    common = importlib.import_module(_PKG + ".common")

preflight = importlib.import_module(_PKG + ".preflight")
run_task = importlib.import_module(_PKG + ".run_task")
verify_campaign_module = importlib.import_module(_PKG + ".verify_campaign")

from eval.long_baseline import _should_save_checkpoint
from hpc.v2_5seed_cpu.common import run_dir

CAMPAIGN_DIR = common.CAMPAIGN_DIR
campaign_for_environment = common.campaign_for_environment
environment_root = common.environment_root
load_campaign = common.load_campaign
resolve_task = common.resolve_task
tasks = common.tasks
verify_campaign = verify_campaign_module.verify_campaign


ENVIRONMENT = "small_150"
SEEDS = (2, 3, 4, 5, 6, 7, 8, 9)


def test_campaign_maps_sixteen_tasks_for_the_single_horizon() -> None:
    campaign = load_campaign()
    expected = [
        (ENVIRONMENT, algorithm, seed)
        for algorithm in ("ippo", "mappo")
        for seed in SEEDS
    ]
    assert tasks(campaign) == expected
    assert len(expected) == 16
    for task_id, seed in enumerate(SEEDS):
        assert resolve_task(campaign, ENVIRONMENT, task_id) == ("ippo", seed)
        assert resolve_task(campaign, ENVIRONMENT, task_id + 8) == ("mappo", seed)
    with pytest.raises(ValueError):
        resolve_task(campaign, ENVIRONMENT, 16)
    with pytest.raises(ValueError):
        resolve_task(campaign, "small_90", 0)


def test_episode_matched_batch_and_walltime_match_the_small150_arm() -> None:
    campaign = load_campaign()
    batch = campaign["training"]["episode_matched_batches"][ENVIRONMENT]
    assert batch["duration_seconds"] == 150
    assert batch["train_batch_size"] == 6750
    assert batch["minibatch_size"] == 1687
    assert campaign["training"]["games_per_update"] == 45
    walltimes = campaign["resources"]["walltime_by_environment"]
    assert walltimes == {ENVIRONMENT: "20:00:00"}


def test_every_task_builds_an_episode_matched_collision_safe_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    campaign = load_campaign()
    monkeypatch.setattr(
        run_task,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "environment_variant": ENVIRONMENT,
                "task_id": 0,
                "research_root": tmp_path,
                "campaign_root": tmp_path / "campaign",
                "execute": False,
                "resume": False,
            },
        )(),
    )
    assert run_task.main() == 0

    for environment, algorithm, seed in tasks(campaign):
        environment_campaign = campaign_for_environment(campaign, environment)
        output_root = environment_root(campaign, tmp_path, environment)
        command, target, remaining = run_task.build_command(
            campaign=environment_campaign,
            research_root=tmp_path,
            campaign_root=output_root,
            algorithm=algorithm,
            seed=seed,
            resume=False,
        )
        command.extend(["--games_per_update", "45"])
        joined = " ".join(command)
        assert remaining == 1200
        assert seed in SEEDS
        assert target == output_root / f"{algorithm}_seed{seed}_headmlp"
        assert f"--seed {seed}" in joined
        assert "--env_variant small_150" in joined
        assert "--train_batch_size 6750" in joined
        assert "--minibatch_size 1687" in joined
        assert "--games_per_update 45" in joined
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


def test_slurm_array_and_submission_walltime_match_campaign() -> None:
    campaign = load_campaign()
    script = (CAMPAIGN_DIR / "run_array.sbatch").read_text()
    submit = (CAMPAIGN_DIR / "submit_all.sh").read_text()
    resources = campaign["resources"]
    assert resources["array"] == "0-15%16"
    assert f"#SBATCH --partition={resources['partition']}" in script
    assert f"#SBATCH --array={resources['array']}" in script
    assert f"#SBATCH --cpus-per-task={resources['cpus_per_task']}" in script
    assert f"#SBATCH --mem={resources['memory_gb']}G" in script
    assert "#SBATCH --time=20:00:00" in script
    assert "hpc.small150_cliff_8seed_cpu.preflight" in script
    assert "hpc.small150_cliff_8seed_cpu.run_task" in script
    # PYTHONHASHSEED must equal the actual campaign seed (2 + task % 8).
    assert 'campaign_seed="$((2 + SLURM_ARRAY_TASK_ID % 8))"' in script
    assert "small_150" in submit
    assert "20:00:00" in submit
    assert "hpc/small150_cliff_8seed_cpu/run_array.sbatch" in submit
    validate = (CAMPAIGN_DIR / "validate_plan.sh").read_text()
    assert "seq 0 15" in validate
    assert not (CAMPAIGN_DIR / "smoke_test.sbatch").exists()


def test_preflight_checks_the_single_horizon(
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
    observed = report["episode_matched_environments"]
    assert isinstance(observed, list)
    assert [entry["variant"] for entry in observed] == [ENVIRONMENT]
    assert [entry["duration_seconds"] for entry in observed] == [150]
    assert [entry["train_batch_size"] for entry in observed] == [6750]
    assert [entry["minibatch_size"] for entry in observed] == [1687]


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
        path = run_dir(
            environment_root(campaign, tmp_path, environment), algorithm, seed
        )
        _write_complete_run(path, campaign_for_environment(campaign, environment))

    report = verify_campaign(tmp_path)
    assert report["ok"] is True
    assert len(report["runs"]) == 16
    assert all(run["ok"] for run in report["runs"])
