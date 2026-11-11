"""Plan tests for the best-config MLP seeds 2-4 extension campaign.

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
    _PKG = "hpc.best_config_mlp_seeds234_cpu"
    common = importlib.import_module(_PKG + ".common")
except ModuleNotFoundError:
    _PKG = "best_config_mlp_seeds234_cpu"
    common = importlib.import_module(_PKG + ".common")

run_task = importlib.import_module(_PKG + ".run_task")
verify_campaign_module = importlib.import_module(_PKG + ".verify_campaign")

from eval.long_baseline import _should_save_checkpoint
from hpc.v2_5seed_cpu.common import run_dir

CAMPAIGN_DIR = common.CAMPAIGN_DIR
arm_by_name = common.arm_by_name
arm_root = common.arm_root
campaign_for_arm = common.campaign_for_arm
load_campaign = common.load_campaign
resolve_task = common.resolve_task
tasks = common.tasks
verify_campaign = verify_campaign_module.verify_campaign


ARMS = ("small_90_mlp", "default_180_mlp")
SEEDS = (2, 3, 4)


def test_campaign_maps_six_tasks_per_arm_and_twelve_total() -> None:
    campaign = load_campaign()
    expected = [
        (arm, algorithm, seed)
        for arm in ARMS
        for algorithm in ("ippo", "mappo")
        for seed in SEEDS
    ]
    assert tasks(campaign) == expected
    assert len(expected) == 12
    for arm in ARMS:
        assert resolve_task(campaign, arm, 0) == ("ippo", 2)
        assert resolve_task(campaign, arm, 1) == ("ippo", 3)
        assert resolve_task(campaign, arm, 2) == ("ippo", 4)
        assert resolve_task(campaign, arm, 3) == ("mappo", 2)
        assert resolve_task(campaign, arm, 4) == ("mappo", 3)
        assert resolve_task(campaign, arm, 5) == ("mappo", 4)
        with pytest.raises(ValueError):
            resolve_task(campaign, arm, 6)
    with pytest.raises(ValueError):
        arm_by_name(campaign, "small_90_lstm")
    with pytest.raises(ValueError):
        arm_by_name(campaign, "default_180_lstm")


def test_arms_are_mlp_only_episode_matched_and_bundle_flags_are_frozen() -> None:
    campaign = load_campaign()
    training = campaign["training"]
    assert training["include_survival_features"] is True
    assert training["include_trade_history"] is True
    assert training["entropy_coeff"] == 0.01
    assert training["entropy_anneal_to"] == 0.001
    assert training["lr_anneal_to"] == 0.0001
    by_name = {arm["name"]: arm for arm in campaign["arms"]}
    assert set(by_name) == set(ARMS)
    assert by_name["small_90_mlp"]["train_batch_size"] == 4050
    assert by_name["small_90_mlp"]["walltime"] == "14:00:00"
    assert by_name["default_180_mlp"]["train_batch_size"] == 8100
    assert by_name["default_180_mlp"]["walltime"] == "30:00:00"
    for arm in campaign["arms"]:
        assert arm["torso"] == "mlp"
        assert arm["minibatch_size"] == arm["train_batch_size"] // 4
        assert arm["name"].endswith(arm["torso"])


def test_every_task_builds_the_full_bundle_command(
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
                "arm": "small_90_mlp",
                "task_id": 0,
                "research_root": tmp_path,
                "campaign_root": tmp_path / "campaign",
                "execute": False,
                "resume": False,
            },
        )(),
    )
    assert run_task.main() == 0

    for arm_name, algorithm, seed in tasks(campaign):
        arm = arm_by_name(campaign, arm_name)
        arm_campaign = campaign_for_arm(campaign, arm_name)
        output_root = arm_root(tmp_path, arm_name)
        command, target, remaining = run_task.build_command(
            campaign=arm_campaign,
            research_root=tmp_path,
            campaign_root=output_root,
            algorithm=algorithm,
            seed=seed,
            resume=False,
        )
        command = run_task.extend_command(command, campaign, arm)
        joined = " ".join(command)
        assert remaining == 1200
        assert seed in SEEDS
        assert target == output_root / f"{algorithm}_seed{seed}_headmlp"
        assert f"--seed {seed}" in joined
        assert f"--env_variant {arm['env_variant']}" in joined
        assert "--torso mlp" in joined
        assert "--torso lstm" not in joined
        assert "--games_per_update 45" in joined
        assert "--entropy_anneal_to 0.001" in joined
        assert "--lr_anneal_to 0.0001" in joined
        assert "--include_survival_features" in command
        assert "--include_trade_history" in command
        assert f"--train_batch_size {arm['train_batch_size']}" in joined
        assert f"--minibatch_size {arm['minibatch_size']}" in joined
        assert "--iters 1200" in joined
        assert "--checkpoint_every 100" in joined
        assert "--checkpoint_start 100" in joined
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


def test_slurm_scripts_cover_mlp_arms_only_with_their_walltimes() -> None:
    campaign = load_campaign()
    script = (CAMPAIGN_DIR / "run_array.sbatch").read_text()
    submit = (CAMPAIGN_DIR / "submit_all.sh").read_text()
    resources = campaign["resources"]
    assert resources["array"] == "0-5%6"
    assert f"#SBATCH --partition={resources['partition']}" in script
    assert f"#SBATCH --array={resources['array']}" in script
    assert f"#SBATCH --cpus-per-task={resources['cpus_per_task']}" in script
    assert f"#SBATCH --mem={resources['memory_gb']}G" in script
    assert "hpc.best_config_mlp_seeds234_cpu.preflight" in script
    assert "hpc.best_config_mlp_seeds234_cpu.run_task" in script
    # PYTHONHASHSEED must equal the actual campaign seed (2 + task % 3).
    assert 'campaign_seed="$((2 + SLURM_ARRAY_TASK_ID % 3))"' in script
    for arm in campaign["arms"]:
        assert arm["name"] in submit
        assert arm["walltime"] in submit
    assert "lstm" not in submit
    validate = (CAMPAIGN_DIR / "validate_plan.sh").read_text()
    assert "lstm" not in validate
    assert "0 1 2 3 4 5" in validate


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


def test_verifier_accepts_all_twelve_complete_runs(tmp_path: Path) -> None:
    campaign = load_campaign()
    for arm_name, algorithm, seed in tasks(campaign):
        path = run_dir(arm_root(tmp_path, arm_name), algorithm, seed)
        _write_complete_run(path, campaign_for_arm(campaign, arm_name))

    report = verify_campaign(tmp_path)
    assert report["ok"] is True
    assert len(report["runs"]) == 12
    assert all(run["ok"] for run in report["runs"])
