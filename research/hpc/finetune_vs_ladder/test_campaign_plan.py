"""Campaign-plan and env-contract tests for finetune_vs_ladder.

Run with:  PYTHONPATH=<research root> python -m pytest \
    hpc/finetune_vs_ladder/test_campaign_plan.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from env.rl.rllib_config import episode_matched_batch  # noqa: E402
from hpc.finetune_vs_ladder import campaign  # noqa: E402
from hpc.finetune_vs_ladder.scripted_opponent_env import (  # noqa: E402
    ARCHETYPES,
    ScriptedOpponentEnv,
)

CAMPAIGN_DIR = Path(__file__).resolve().parent
SBATCH = (CAMPAIGN_DIR / "run_array.sbatch").read_text()


# ---------------------------------------------------------------------------
# Task mapping + sbatch plan
# ---------------------------------------------------------------------------


def test_task_table() -> None:
    assert len(campaign.TASKS) == 8
    assert campaign.TASKS[:5] == tuple(("ippo", s) for s in range(5))
    assert campaign.TASKS[5:] == (("mappo", 0), ("mappo", 2), ("mappo", 3))


def test_task_checkpoint_roots() -> None:
    for algorithm, seed in campaign.TASKS:
        var = campaign.ckpt_root_env_var(seed)
        assert var == (
            "HUNGER_CKPT_ROOT_2SEED" if seed <= 1 else "HUNGER_CKPT_ROOT_S234"
        )
        assert campaign.checkpoint_relpath(algorithm, seed) == (
            f"{algorithm}_seed{seed}_headmlp/checkpoints/iter_1200"
        )


def test_sbatch_array_bounds_and_resources() -> None:
    assert f"#SBATCH --array={campaign.ARRAY_SPEC}\n" in SBATCH
    assert "#SBATCH --cpus-per-task=6" in SBATCH
    assert "#SBATCH --mem=16G" in SBATCH
    assert "#SBATCH --time=12:00:00" in SBATCH
    assert "#SBATCH --partition=expansion" in SBATCH


def test_sbatch_task_arrays_match_campaign() -> None:
    algos = re.search(r"^algos=\(([^)]*)\)", SBATCH, re.M).group(1).split()
    seeds = re.search(r"^seeds=\(([^)]*)\)", SBATCH, re.M).group(1).split()
    assert [
        (algo, int(seed)) for algo, seed in zip(algos, seeds, strict=True)
    ] == list(campaign.TASKS)
    lo, hi = campaign.ARRAY_SPEC.split("-")
    assert int(lo) == 0 and int(hi) == len(campaign.TASKS) - 1


def test_sbatch_env_and_flags() -> None:
    for required in (
        "export PYTHONHASHSEED=\"${seed}\"",
        "export HG_MAX_QTY=5",
        "export HG_MAX_PRICE=50",
        "export PYTHONNOUSERSITE=1",
        "export OMP_NUM_THREADS=1",
        "--include-survival-features",
        "--include-trade-history",
        "--iterations 300",
        "--checkpoint-every 50",
        "iter_1200",
    ):
        assert required in SBATCH, f"missing from run_array.sbatch: {required}"


def test_batch_math_matches_best_config() -> None:
    train, mini = episode_matched_batch(
        campaign.GAME_DURATION_SECONDS,
        games_per_update=campaign.GAMES_PER_UPDATE,
    )
    assert (train, mini) == (8100, 2025)
    assert (campaign.TRAIN_BATCH_SIZE, campaign.MINIBATCH_SIZE) == (8100, 2025)


def test_checkpoint_numbering_continues_from_1200() -> None:
    ticks = campaign.checkpoint_iterations()
    assert ticks == list(range(1250, 1501, 50))
    # The exact series eval/checkpoint_scripted_diagnostics.py builds from
    # --checkpoint-start 1250 --checkpoint-stop 1500 --checkpoint-every 50.
    assert ticks == list(range(1250, 1500 + 1, 50))


# ---------------------------------------------------------------------------
# Env contract
# ---------------------------------------------------------------------------


def _random_legal_action(rng: np.random.Generator, obs: dict) -> int:
    legal = np.flatnonzero(obs["action_mask"] > 0)
    assert legal.size > 0
    return int(rng.choice(legal))


def _play_episode(env: ScriptedOpponentEnv, rng: np.random.Generator):
    """Play one full episode; return (steps, reward trace, terminal info)."""
    obs, infos = env.reset()
    assert len(obs) == 1 and len(infos) == 1
    (agent_id,) = obs
    rewards_trace: list[float] = []
    steps = 0
    terminal_info: dict = {}
    while True:
        action = _random_legal_action(rng, obs[agent_id])
        obs, rewards, terminated, truncated, infos = env.step(
            {agent_id: action}
        )
        steps += 1
        # Exactly one RLlib agent in every step dict.
        assert set(rewards) == {agent_id}
        assert set(infos) == {agent_id}
        assert set(terminated) == {agent_id, "__all__"}
        assert set(truncated) == {agent_id, "__all__"}
        assert not truncated["__all__"]
        rewards_trace.append(float(rewards[agent_id]))
        if terminated["__all__"]:
            assert terminated[agent_id]
            terminal_info = infos[agent_id]
            break
        assert set(obs) == {agent_id}
        assert steps <= env.config.game_duration_seconds + 1
    return steps, rewards_trace, terminal_info


def test_reset_contract_default_flags() -> None:
    env = ScriptedOpponentEnv({"seed": 0})
    obs, infos = env.reset()
    assert len(obs) == 1
    ((agent_id, agent_obs),) = obs.items()
    assert re.fullmatch(r"seat_[0-3]", agent_id)
    assert set(agent_obs) == {
        "observations", "action_mask", "critic_observations"
    }
    assert agent_obs["observations"].shape == (171,)
    assert agent_obs["critic_observations"].shape == (305,)
    info = infos[agent_id]
    assert info["opponent_archetype"] in ARCHETYPES
    assert info["focal_seat"] == int(agent_id.removeprefix("seat_"))


def test_reset_contract_best_config_flags() -> None:
    """Best-config observation contract: actor 176-D, MAPPO critic 401-D."""
    env = ScriptedOpponentEnv(
        {
            "seed": 0,
            "include_survival_features": True,
            "include_trade_history": True,
        }
    )
    obs, _ = env.reset()
    ((agent_id, agent_obs),) = obs.items()
    assert agent_obs["observations"].shape == (176,)
    assert agent_obs["critic_observations"].shape == (401,)
    space = env.observation_spaces[agent_id]
    assert space["observations"].shape == (176,)
    assert space["critic_observations"].shape == (401,)


def test_episode_terminal_only_reward_and_single_agent() -> None:
    env = ScriptedOpponentEnv({"seed": 7})
    rng = np.random.default_rng(7)
    for _ in range(3):
        steps, trace, terminal_info = _play_episode(env, rng)
        # Terminal-only: zero everywhere except (possibly) the final step.
        assert all(r == 0.0 for r in trace[:-1])
        assert trace[-1] in (-1.0, 0.0, 1.0)
        assert steps <= env.config.game_duration_seconds
        # Terminal info carries the focal diagnostics.
        assert "focal_survived" in terminal_info
        assert terminal_info["opponent_archetype"] in ARCHETYPES
        # Death <-> -1 consistency (v2 terminal-only semantics).
        if terminal_info["died"]:
            assert trace[-1] == -1.0
            assert not terminal_info["focal_survived"]
        else:
            assert steps == env.config.game_duration_seconds
            assert trace[-1] in (0.0, 1.0)
            assert terminal_info["focal_survived"]


def test_scripted_seats_act_and_trade() -> None:
    """The three scripted seats must actually trade among themselves.

    The focal seat always WAITs (action 0 is Wait and always legal), so any
    executed trade is scripted-vs-scripted activity inside step().
    """
    env = ScriptedOpponentEnv({"seed": 11})
    total_trades = 0
    for _ in range(3):
        obs, _ = env.reset()
        (agent_id,) = obs
        while True:
            _, _, terminated, _, _ = env.step({agent_id: 0})
            if terminated["__all__"]:
                break
        total_trades += len(env._inner._env.state.trades)
    assert total_trades > 0


def test_archetype_and_seat_uniformity() -> None:
    env = ScriptedOpponentEnv({"seed": 3})
    archetype_counts: dict[str, int] = {}
    seat_counts: dict[int, int] = {}
    n_episodes = 48
    for _ in range(n_episodes):
        _, infos = env.reset()
        ((_, info),) = infos.items()
        archetype_counts[info["opponent_archetype"]] = (
            archetype_counts.get(info["opponent_archetype"], 0) + 1
        )
        seat_counts[info["focal_seat"]] = (
            seat_counts.get(info["focal_seat"], 0) + 1
        )
    assert set(archetype_counts) == set(ARCHETYPES)
    for archetype, count in archetype_counts.items():
        assert count >= 4, (archetype, archetype_counts)
    assert set(seat_counts) == {0, 1, 2, 3}, seat_counts


def test_rotate_focal_selection_cycles() -> None:
    env = ScriptedOpponentEnv({"seed": 5, "focal_selection": "rotate"})
    seats = []
    for _ in range(8):
        _, infos = env.reset()
        ((_, info),) = infos.items()
        seats.append(info["focal_seat"])
    assert seats == [0, 1, 2, 3, 0, 1, 2, 3]


def test_reset_stream_is_reproducible() -> None:
    def draw(n: int) -> list[tuple[int, str]]:
        env = ScriptedOpponentEnv({"seed": 42})
        out = []
        for _ in range(n):
            _, infos = env.reset()
            ((_, info),) = infos.items()
            out.append((info["focal_seat"], info["opponent_archetype"]))
        return out

    assert draw(6) == draw(6)


def test_rejects_actions_for_scripted_seats() -> None:
    env = ScriptedOpponentEnv({"seed": 0})
    obs, _ = env.reset()
    (agent_id,) = obs
    other = next(
        f"seat_{i}" for i in range(4) if f"seat_{i}" != agent_id
    )
    with pytest.raises(ValueError):
        env.step({agent_id: 0, other: 0})
