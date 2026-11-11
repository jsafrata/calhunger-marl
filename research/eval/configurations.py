"""The 7 configurations we run Scripted-v1 through, plus tools for
enumerating paired-seed pools across them.

Every configuration is a tuple of 4 cadences (one per seat). The seat
assignment is randomized per game — the `Configuration` here only fixes
the *multiset* of cadences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Configuration:
    """A multiset of cadences that populates the 4 seats of a game."""

    name: str
    cadences: Tuple[str, str, str, str]

    def __post_init__(self) -> None:
        if len(self.cadences) != 4:
            raise ValueError("Configuration must specify exactly 4 cadences")
        for c in self.cadences:
            if c not in ("fast", "medium", "slow"):
                raise ValueError(f"unknown cadence: {c!r}")


CONFIGURATIONS: List[Configuration] = [
    Configuration("4_fast",         ("fast",   "fast",   "fast",   "fast"  )),
    Configuration("4_medium",       ("medium", "medium", "medium", "medium")),
    Configuration("4_slow",         ("slow",   "slow",   "slow",   "slow"  )),
    Configuration("2fast_2medium",  ("fast",   "fast",   "medium", "medium")),
    Configuration("2fast_2slow",    ("fast",   "fast",   "slow",   "slow"  )),
    Configuration("1fast_3slow",    ("fast",   "slow",   "slow",   "slow"  )),
    Configuration("1fast_1medium_2slow", ("fast", "medium", "slow", "slow")),
]


def config_by_name(name: str) -> Configuration:
    for c in CONFIGURATIONS:
        if c.name == name:
            return c
    raise KeyError(f"unknown configuration: {name!r}")


def seat_permutation_seed(shared_seed: int, config_idx: int) -> int:
    """Derive a per-config seat-permutation seed from a shared base seed.

    The shared seed drives env-side randomization (producers, initial
    inventory). By deriving the seat-permutation seed from it we can
    guarantee that "game #k of config A" and "game #k of config B" share
    the same environmental setup, differing only in which cadences sit
    where — the "paired seeds" property the spec asks for.
    """
    return (shared_seed * 1_000_003) ^ (config_idx * 2_147_483_647)
