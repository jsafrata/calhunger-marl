from __future__ import annotations

import json
from pathlib import Path

import pytest

from hpc.best_config_2seed_cpu import run_task
from hpc.best_config_2seed_cpu.common import (
    CAMPAIGN_DIR,
    arm_by_name,
    arm_root,
    campaign_for_arm,
    load_campaign,
    resolve_task,
    tasks,
)
from hpc.best_config_2seed_cpu.verify_campaign import verify_campaign
from hpc.v2_5seed_cpu.common import run_dir


ARMS = ("small_90_mlp", "small_90_lstm", "default_180_mlp", "default_180_lstm")


def test_campaign_maps_four_tasks_per_arm_and_sixteen_total() -> None:
    campaign = load_campaign()
    expected = [
        (arm, algorithm, seed)
        for arm in ARMS
        for algorithm in ("ippo", "mappo")
        for seed in (0, 1)
    ]
    assert tasks(campaign) == expected
    for arm in ARMS:
        assert resolve_task(campaign, arm, 0) == ("ippo", 0)
        assert resolve_task(campaign, arm, 3) == ("mappo", 1)
        with pytest.raises(ValueError):
            resolve_task(campaign, arm, 4)


def test_arms_are_episode_matched_and_bundle_flags_are_frozen() -> None:
    campaign = load_campaign()
    training = campaign["training"]
    assert training["include_survival_features"] is True
    assert training["include_trade_history"] is True
    assert training["entropy_coeff"] == 0.01
    assert training["entropy_anneal_to"] == 0.001
    assert training["lr_anneal_to"] == 0.0001
    by_name = {arm["name"]: arm for arm in campaign["arms"]}
    assert by_name["small_90_mlp"]["train_batch_size"] == 4050
    assert by_name["small_90_lstm"]["train_batch_size"] == 4050
    assert by_name["default_180_mlp"]["train_batch_size"] == 8100
    assert by_name["default_180_lstm"]["train_batch_size"] == 8100
    for arm in campaign["arms"]:
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
                "arm": "small_90_lstm",
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
        assert target == output_root / f"{algorithm}_seed{seed}_headmlp"
        assert f"--env_variant {arm['env_variant']}" in joined
        assert f"--torso {arm['torso']}" in joined
        assert "--games_per_update 45" in joined
        assert "--entropy_anneal_to 0.001" in joined
        assert "--lr_anneal_to 0.0001" in joined
        assert "--include_survival_features" in command
        assert "--include_trade_history" in command
        assert f"--train_batch_size {arm['train_batch_size']}" in joined
        assert "--iters 1200" in joined
        assert "--checkpoint_start 100" in joined
        assert "--fail_if_output_exists" in command


def test_slurm_scripts_cover_all_arms_and_walltimes() -> None:
    campaign = load_campaign()
    script = (CAMPAIGN_DIR / "run_array.sbatch").read_text()
    submit = (CAMPAIGN_DIR / "submit_all.sh").read_text()
    resources = campaign["resources"]
    assert f"#SBATCH --partition={resources['partition']}" in script
    assert f"#SBATCH --array={resources['array']}" in script
    assert "hpc.best_config_2seed_cpu.preflight" in script
    assert "hpc.best_config_2seed_cpu.run_task" in script
    for arm in campaign["arms"]:
        assert arm["name"] in submit
        assert arm["walltime"] in submit

    eval_dir = CAMPAIGN_DIR.parent / "best_config_eval"
    scripted = (eval_dir / "run_array_scripted.sbatch").read_text()
    selfplay = (eval_dir / "run_array_selfplay.sbatch").read_text()
    for arm in campaign["arms"]:
        assert arm["name"] in scripted
        assert arm["name"] in selfplay
    assert "--array=0-15%16" in scripted
    assert "--array=0-3%4" in selfplay


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
    for arm_name, algorithm, seed in tasks(campaign):
        path = run_dir(arm_root(tmp_path, arm_name), algorithm, seed)
        _write_complete_run(path, campaign_for_arm(campaign, arm_name))

    report = verify_campaign(tmp_path)
    assert report["ok"] is True
    assert len(report["runs"]) == 16
    assert all(run["ok"] for run in report["runs"])
