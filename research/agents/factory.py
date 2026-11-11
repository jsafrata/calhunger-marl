from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List

from . import AGENT_REGISTRY
from .base import Agent


def agent_help() -> str:
    return (
        "comma-separated list of 4 agent names from: "
        + ", ".join(AGENT_REGISTRY)
        + ", or policy:NAME"
    )


def add_checkpoint_agent_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="checkpoint for a policy:NAME agent; may be passed more than once",
    )
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="sample checkpoint policy actions instead of taking argmax",
    )


def parse_agent_names(spec: str | Iterable[str]) -> list[str]:
    if isinstance(spec, str):
        names = [part.strip() for part in spec.split(",")]
    else:
        names = []
        for item in spec:
            names.extend(part.strip() for part in item.split(","))
    return [name for name in names if name]


def parse_checkpoint_specs(specs: Iterable[str] | None) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for spec in specs or ():
        if "=" not in spec:
            raise SystemExit(
                f"checkpoint spec must be NAME=PATH, got {spec!r}"
            )
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise SystemExit(
                f"checkpoint spec must be NAME=PATH, got {spec!r}"
            )
        checkpoints[label] = Path(path)
    return checkpoints


def validate_agent_specs(
    spec: str | Iterable[str],
    checkpoint_specs: Iterable[str] | None = None,
) -> list[str]:
    names = parse_agent_names(spec)
    if len(names) != 4:
        raise SystemExit(f"need 4 agent names, got {len(names)}")

    checkpoints = parse_checkpoint_specs(checkpoint_specs)
    for name in names:
        if name.startswith("policy:"):
            label = name.split(":", 1)[1].strip()
            if not label:
                raise SystemExit("policy agents must be written as policy:NAME")
            if label not in checkpoints:
                raise SystemExit(
                    f"missing checkpoint for policy:{label}; pass "
                    f"--checkpoint {label}=PATH"
                )
            continue

        if name not in AGENT_REGISTRY:
            choices = [*AGENT_REGISTRY, "policy:NAME"]
            raise SystemExit(f"unknown agent '{name}'. choices: {choices}")
    return names


def build_agents(
    spec: str | Iterable[str],
    *,
    base_seed: int,
    checkpoint_specs: Iterable[str] | None = None,
    stochastic_policy: bool = False,
) -> List[Agent]:
    names = validate_agent_specs(spec, checkpoint_specs)
    checkpoints = parse_checkpoint_specs(checkpoint_specs)
    agents: list[Agent] = []
    for idx, name in enumerate(names):
        if name.startswith("policy:"):
            label = name.split(":", 1)[1].strip()
            if not label:
                raise SystemExit("policy agents must be written as policy:NAME")
            checkpoint_path = checkpoints.get(label)
            if checkpoint_path is None:
                raise SystemExit(
                    f"missing checkpoint for policy:{label}; pass "
                    f"--checkpoint {label}=PATH"
                )
            from .checkpoint_policy_agent import CheckpointPolicyAgent

            agents.append(
                CheckpointPolicyAgent(
                    label=label,
                    checkpoint_path=checkpoint_path,
                    stochastic=stochastic_policy,
                    seed=base_seed * 1000 + idx,
                )
            )
            continue

        if name not in AGENT_REGISTRY:
            choices = [*AGENT_REGISTRY, "policy:NAME"]
            raise SystemExit(f"unknown agent '{name}'. choices: {choices}")
        cls = AGENT_REGISTRY[name]
        try:
            agents.append(cls(seed=base_seed * 1000 + idx))
        except TypeError:
            agents.append(cls())
    return agents


def cleanup_checkpoint_policy_cache() -> None:
    module = sys.modules.get("agents.checkpoint_policy_agent")
    if module is not None:
        module.cleanup_checkpoint_policy_cache()
