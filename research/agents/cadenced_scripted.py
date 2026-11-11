"""Cadence wrapper around ScriptedV1.

The live-game bot fires on its own `setInterval` with a per-difficulty
range of intervals. In our env time only advances at `env.step()`
granularity (1 game-second), so we schedule the *underlying* bot's
`act()` at sub-second granularity and gate each call by cadence:

    Fast:    inter-decision interval U(0.5, 1.0)  seconds
    Medium:  inter-decision interval U(1.5, 2.5)  seconds
    Slow:    inter-decision interval U(3.5, 5.5)  seconds

Each wrapper keeps a `next_action_time_s` (float, in game-seconds). On
each env step the wrapper looks at the current tick time (an integer
count of seconds from `obs["tick"]`) and:

    - If `next_action_time_s <= tick`: delegate to the underlying bot
      and schedule the next fire.
    - Otherwise: return WAIT.

The initial `next_action_time_s` is jittered on `reset()` so bots don't
all fire in lock-step at t=0.

The wrapper is agnostic about *which* scripted bot it gates: pass
`inner_factory` to wrap ScriptedV2 / ScriptedV3 instead of the default
ScriptedV1. Cadence is a property of the live game's `setInterval`, not of any
one bot's strategy, so all three baselines are compared at the same firing
rates.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from env.actions import ACTION_WAIT

from .base import Agent
from .scripted_v1 import ScriptedV1Agent


CADENCE_PRESETS: Dict[str, Tuple[float, float]] = {
    "fast": (0.5, 1.0),
    "medium": (1.5, 2.5),
    "slow": (3.5, 5.5),
}


class CadencedScripted(Agent):
    """ScriptedV1 gated by a per-difficulty decision cadence.

    Args:
        cadence: one of {"fast", "medium", "slow"}.
        seed: RNG seed for both the scheduler and the underlying scripted
            bot's coin-flip / jitter.
        inner_factory: callable taking `seed=` and returning the gated bot.
            Defaults to `ScriptedV1Agent`, so existing call sites that just
            pass `(cadence, seed)` are unaffected.
    """

    def __init__(
        self,
        cadence: str = "medium",
        seed: int = 0,
        inner_factory: Optional[Callable[..., Agent]] = None,
    ) -> None:
        if cadence not in CADENCE_PRESETS:
            raise ValueError(
                f"cadence must be one of {sorted(CADENCE_PRESETS)}, got {cadence!r}"
            )
        self.cadence = cadence
        self.min_interval, self.max_interval = CADENCE_PRESETS[cadence]
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # Underlying bot gets a distinct-but-derived seed so we don't tie
        # its coin flips to the scheduler's jitter.
        factory = inner_factory if inner_factory is not None else ScriptedV1Agent
        self._inner = factory(seed=seed ^ 0xC0DEC0DE)
        # Deferred to reset()
        self._next_action_time_s: float = 0.0
        # Diagnostic counters — used by the harness.
        self.n_decisions_taken: int = 0
        self.n_decisions_skipped: int = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self._inner.name}[{self.cadence}]"

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._inner.reset()
        # Initial jittered offset in [0, max_interval) so bots don't sync.
        self._next_action_time_s = float(self._rng.random() * self.max_interval)
        self.n_decisions_taken = 0
        self.n_decisions_skipped = 0

    def act(self, obs: Dict[str, Any], legal_mask: np.ndarray) -> int:
        # obs["tick"] is the current elapsed_seconds (integer). Treat it as
        # the "time now" for scheduling. The scheduler resolution is thus
        # 1 second — matching the env's step granularity.
        now = float(obs.get("tick", 0))
        if now < self._next_action_time_s:
            self.n_decisions_skipped += 1
            return ACTION_WAIT
        self.n_decisions_taken += 1
        # Schedule next fire — U(min, max) added to the current time so a
        # slow bot that misses several ticks doesn't get penalized further.
        interval = float(
            self._rng.uniform(self.min_interval, self.max_interval)
        )
        self._next_action_time_s = now + interval
        return self._inner.act(obs, legal_mask)
