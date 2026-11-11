"""Shared RLlib configuration for training and evaluation."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from .callbacks import NashHungerCallbacks
from .rllib_env import NashHungerMultiAgentEnv
from .rllib_module import (
    MAX_SEQ_LEN,
    NashHungerPPOTorchRLModule,
)

ENV_NAME = "nash-hunger-multi-agent-v0"
SHARED_POLICY_ID = "shared_policy"
# v3 Condition C — 4 independent policies, one per seat. Names must match
# the mapping seat_i → pi_i used in `independent_policy_mapping_fn`.
INDEPENDENT_POLICY_IDS = ("pi_0", "pi_1", "pi_2", "pi_3")


def register_nash_hunger_env() -> None:
    register_env(
        ENV_NAME,
        lambda env_config: NashHungerMultiAgentEnv(env_config),
    )


def shared_policy_mapping_fn(agent_id, *args, **kwargs) -> str:
    del agent_id, args, kwargs
    return SHARED_POLICY_ID


def independent_policy_mapping_fn(agent_id, *args, **kwargs) -> str:
    """seat_i → pi_i. Used for Condition C (4 independent policies)."""
    del args, kwargs
    if not agent_id.startswith("seat_"):
        raise ValueError(f"expected seat_i agent_id, got {agent_id!r}")
    seat = int(agent_id.removeprefix("seat_"))
    if not (0 <= seat < len(INDEPENDENT_POLICY_IDS)):
        raise ValueError(f"seat {seat} out of range for independent policies")
    return INDEPENDENT_POLICY_IDS[seat]


def episode_matched_batch(
    game_duration_seconds: int,
    *,
    games_per_update: int,
    num_minibatches: int = 4,
) -> tuple[int, int]:
    """(train_batch_size, minibatch_size) holding games-per-update constant.

    With terminal-only rewards and lambda=1, every step of a game shares
    that game's single Monte-Carlo return, so the effective number of
    independent samples in a PPO update is the number of *games* in the
    batch — not the number of env steps. A fixed step-count batch therefore
    silently degrades gradient quality as the horizon grows (4096 steps is
    ~68 games at 60s but ~27 at 150s), confounding "longer horizons are
    harder" with "longer horizons get noisier updates". Sizing the batch as
    `games x duration` env steps removes the confound.

    The env advances one step per game-second and batch sizes count env
    steps (each carrying 4 agent transitions), so a game contributes
    exactly `game_duration_seconds` env steps.
    """
    if game_duration_seconds <= 0:
        raise ValueError("game_duration_seconds must be positive")
    if games_per_update <= 0:
        raise ValueError("games_per_update must be positive")
    if num_minibatches <= 0:
        raise ValueError("num_minibatches must be positive")
    train_batch_size = int(games_per_update) * int(game_duration_seconds)
    minibatch_size = max(1, train_batch_size // int(num_minibatches))
    return train_batch_size, minibatch_size


def linear_anneal_schedule(
    initial: float,
    final: float,
    total_env_steps: int,
) -> list[list[float]]:
    """RLlib schedule annealing linearly from `initial` to `final`.

    Format is the `[[timestep, value], ...]` list that Ray >= 2.x's
    `Scheduler` accepts for `lr` and `entropy_coeff`; timesteps count env
    steps sampled, and the value holds at `final` beyond the last entry.
    Callers computing `total_env_steps` from an iteration budget should use
    `iters * train_batch_size`.
    """
    if total_env_steps <= 0:
        raise ValueError("total_env_steps must be positive")
    return [[0, float(initial)], [int(total_env_steps), float(final)]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_gpu_request(num_gpus: float) -> None:
    if num_gpus > 0 and not torch.cuda.is_available():
        raise RuntimeError(
            f"{num_gpus} GPU(s) requested, but PyTorch cannot access CUDA"
        )


def build_ppo_config(
    *,
    seed: int = 0,
    num_env_runners: int = 1,
    num_gpus: float = 0.0,
    evaluation_interval: int | None = None,
    evaluation_episodes: int = 20,
    max_seq_len: int = MAX_SEQ_LEN,
    train_batch_size: int = 16_384,
    minibatch_size: int = 2_048,
    num_epochs: int = 4,
    env_config: dict[str, Any] | None = None,
    algo: str = "mappo",
    torso: str = "mlp",
    head_kind: str = "mlp",
    entropy_coeff: float | list = 0.01,
    lr: float | list = 3e-4,
    gamma: float = 1.0,
    lambda_: float = 1.0,
    parameterization: str = "shared",
    include_agent_id: bool = False,
    include_survival_features: bool = False,
    include_trade_history: bool = False,
) -> PPOConfig:
    """Build the single-policy, four-seat PPO configuration.

    Args:
        algo: `"ippo"` uses local observations for the critic; `"mappo"` uses
            the centralized global state. Actor and everything else are
            identical between the two — this is the only structural switch.
        torso: `"mlp"` (default) is the feed-forward baseline; `"lstm"`
            enables recurrence.
        head_kind: `"mlp"` (v2 default, one hidden layer per head with
            LayerNorm + SiLU) or `"linear"` (v1 baseline, single linear
            projection). Preserved on the module and stored in
            checkpoint metadata via RLlib's model_config.
        entropy_coeff: Constant float, or a `[[timestep, value], ...]`
            schedule (timesteps in env steps sampled) — see
            `linear_anneal_schedule` for the annealed form.
        lr: Constant float (v2 default 3e-4), or a schedule in the same
            format as `entropy_coeff`.
        gamma: PPO discount. v2 default = 1.0 — a fixed 180-step horizon
            with terminal-only rewards; we want the terminal signal to
            reach the first tick undiscounted.
        lambda_: GAE lambda. v2 default = 1.0 — same rationale.
        parameterization: `"shared"` (v2 default, one policy for all 4
            seats, gradients pooled) or `"independent"` (v3 Condition C,
            four separate policies pi_0..pi_3, each trained only from its
            own seat trajectories, no parameter or gradient sharing).
        include_agent_id: Append a seat one-hot to local and centralized
            observations.  Used only by v3 Condition B; False exactly
            preserves the v2 observation contract.
        include_survival_features: Append per-food starvation cover and
            consumption-cycle phase to the *local* observation (v4).
            False exactly preserves the prior observation contract.
        include_trade_history: Append the previous tick's executed-trade
            matrices to the *centralized* observation (v4), making the
            MAPPO critic's input a superset of the actor's market
            information. False exactly preserves the prior contract.
    """
    if parameterization not in ("shared", "independent"):
        raise ValueError(
            f"parameterization must be 'shared' or 'independent', got "
            f"{parameterization!r}"
        )
    validate_gpu_request(num_gpus)
    register_nash_hunger_env()
    model_config = {
        "algo": algo,
        "torso": torso,
        "head_kind": head_kind,
        "max_seq_len": max_seq_len,
        "include_agent_id": include_agent_id,
        "include_survival_features": include_survival_features,
        "include_trade_history": include_trade_history,
    }
    if parameterization == "shared":
        policies_set = {SHARED_POLICY_ID}
        policy_mapping_fn = shared_policy_mapping_fn
        policies_to_train = [SHARED_POLICY_ID]
        rl_module_spec = RLModuleSpec(
            module_class=NashHungerPPOTorchRLModule,
            model_config=model_config,
        )
    else:  # "independent"
        policies_set = set(INDEPENDENT_POLICY_IDS)
        policy_mapping_fn = independent_policy_mapping_fn
        policies_to_train = list(INDEPENDENT_POLICY_IDS)
        rl_module_spec = MultiRLModuleSpec(
            rl_module_specs={
                pid: RLModuleSpec(
                    module_class=NashHungerPPOTorchRLModule,
                    model_config=model_config,
                )
                for pid in INDEPENDENT_POLICY_IDS
            }
        )

    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env=ENV_NAME,
            env_config={
                "seed": seed,
                **(env_config or {}),
                "include_agent_id": include_agent_id,
                "include_survival_features": include_survival_features,
                "include_trade_history": include_trade_history,
            },
        )
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length="auto",
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=num_gpus,
        )
        .training(
            gamma=gamma,
            lambda_=lambda_,
            lr=lr,
            clip_param=0.2,
            use_kl_loss=False,
            kl_coeff=0.0,
            entropy_coeff=entropy_coeff,
            vf_loss_coeff=0.5,
            # Deliberately never triggers: terminal-only returns live in
            # [-1, 1], so 1000 disables value-loss clipping. The policy-
            # ratio clip (clip_param) and grad-norm clip stay active.
            vf_clip_param=1_000.0,
            grad_clip=0.5,
            train_batch_size_per_learner=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
        )
        .multi_agent(
            policies=policies_set,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
        .rl_module(rl_module_spec=rl_module_spec)
        .callbacks(NashHungerCallbacks)
        .debugging(seed=seed)
    )
    if evaluation_interval:
        config = config.evaluation(
            evaluation_interval=evaluation_interval,
            evaluation_duration=evaluation_episodes,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False},
        )
    return config
