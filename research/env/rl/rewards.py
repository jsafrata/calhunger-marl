"""Shared reward accounting for RL environment adapters.

Two trackers live here.

**v2 default — `TerminalOnlyRewardTracker`:**

  Nonterminal reward is exactly 0. On the terminal tick each seat gets:

      dead-before-180  →  -1
      alive-nonwinner  →   0
      winner (each)    →  +1  (a k-way tie awards +1 to every winner)

  Under the v2 corrected termination (game ends at t=180 or at zero
  alive), "everyone dies early" gives every player -1. There is no
  dense shaping, no buffer bonus, and no rank scaling: the optimizer
  sees only whether the seat is alive at 180s and whether it is a
  winner. This aligns the training signal with the true CalHunger
  objective (survive, then be the strongest survivor).

**Legacy — `ShapedRewardTracker` (retained for archive-parity tests):**

  Dense buffer shaping at consumption ticks + terminal (win/survive/dead).
  Used by every v1 run in `research/eval/results/archive/`. Not the
  default under the new env config.
"""
from __future__ import annotations

from typing import Mapping

from ..config import GameConfig
from ..env import FoodTradingEnv, StepInfo

# Dense shaping magnitude per (consumption tick × unit-of-clipped-buffer).
SURVIVAL_REWARD_SCALE = 0.01
# Max buffer that contributes to shaping. Beyond this, extra stock earns
# nothing — discourages runaway hoarding while still rewarding a healthy
# survival margin.
BUFFER_CAP = 20
# Default terminal reward magnitudes (per-player, not shared/split).
# These are the values from the original baseline; alternative training
# variants override them via the tracker's constructor.
WIN_REWARD = 5.0
SURVIVED_NON_WIN_REWARD = 1.0
DEATH_PENALTY = 0.0  # baseline: no explicit death penalty (dead → 0)
# Historical: the baseline split WIN_REWARD across k-way winners so a
# solo winner earned +5 and a 4-way tie earned +1.25 each. Alternative
# variants may prefer per-winner payouts; the tracker exposes this as a
# constructor flag.
WIN_SPLIT_ACROSS_TIES = True


class ShapedRewardTracker:
    """Stateful reward interface shared by Gymnasium and RLlib adapters.

    Terminal magnitudes are per-instance so we can run controlled A/B
    experiments that change ONLY the terminal signal without touching any
    other file. `dense_*` settings drive the buffer shaping (unchanged
    across all currently planned variants).
    """

    def __init__(
        self,
        config: GameConfig,
        *,
        dense_reward: bool = True,
        win_reward: float = WIN_REWARD,
        survived_non_win_reward: float = SURVIVED_NON_WIN_REWARD,
        death_penalty: float = DEATH_PENALTY,
        win_split_across_ties: bool = WIN_SPLIT_ACROSS_TIES,
    ) -> None:
        self.config = config
        self.dense_reward = dense_reward
        # Terminal magnitudes — swap for A/B experiments.
        self.win_reward = float(win_reward)
        self.survived_non_win_reward = float(survived_non_win_reward)
        self.death_penalty = float(death_penalty)
        self.win_split_across_ties = bool(win_split_across_ties)
        self._was_reset = False
        # Populated by compute() — most-recent per-player {dense, terminal, total}.
        self.last_breakdown: dict[int, dict[str, float]] = {}

    def reset(self, env: FoodTradingEnv) -> None:
        self._was_reset = True

    def _required_food_score(self, agent_obs: dict) -> float:
        """Return the min (across required foods) of *clipped* available
        inventory. Clipping to BUFFER_CAP caps the maximum shaping any
        strategy can earn."""
        required = agent_obs["required_foods"]
        if not required:
            return 0.0
        return min(
            min(max(float(agent_obs["available_inventory"][food]), 0.0), BUFFER_CAP)
            for food in required
        )

    def _survival_score(self, agent_obs: dict) -> float:
        return self._required_food_score(agent_obs)

    def _is_consumption_tick(self, info: StepInfo) -> bool:
        """The tick that just fired is a consumption tick iff env's
        elapsed_seconds is a positive multiple of the consumption interval.
        `info.tick` is the elapsed_seconds AFTER the step, matching the
        env's own convention (see env/tick.py)."""
        interval = self.config.consumption_interval_seconds
        return info.tick > 0 and info.tick % interval == 0

    def compute(
        self,
        env: FoodTradingEnv,
        observations: Mapping[int, dict],
        *,
        done: bool,
        info: StepInfo,
    ) -> dict[int, float]:
        """Return per-player total reward for this step.

        Also caches a per-player breakdown into `self.last_breakdown` so
        callers (env adapters + RLlib callbacks) can log dense and terminal
        contributions separately. This satisfies the requirement that the
        two channels never get mixed up in the training telemetry.
        """
        if not self._was_reset:
            raise RuntimeError("reward tracker must be reset before compute()")

        rewards: dict[int, float] = {}
        breakdown: dict[int, dict[str, float]] = {}
        emit_dense = self.dense_reward and self._is_consumption_tick(info)

        for idx in range(self.config.player_count):
            player = env.state.players[idx]
            agent_obs = observations[idx]
            dense = 0.0
            terminal = 0.0

            if emit_dense and player.status == "alive":
                dense = (
                    SURVIVAL_REWARD_SCALE
                    * self._survival_score(agent_obs)
                )

            if done:
                winners = env.state.winner_idxs
                if idx in winners:
                    if self.win_split_across_ties and len(winners) > 0:
                        terminal = self.win_reward / len(winners)
                    else:
                        terminal = self.win_reward
                elif player.status == "alive":
                    terminal = self.survived_non_win_reward
                else:
                    # Dead. Baseline uses 0; alternative variants use a
                    # negative penalty here.
                    terminal = self.death_penalty

            total = dense + terminal
            rewards[idx] = float(total)
            breakdown[idx] = {
                "dense": float(dense),
                "terminal": float(terminal),
                "total": float(total),
            }

        # Cache the per-player split so callers can plumb it into `infos`.
        self.last_breakdown = breakdown
        return rewards


class TerminalOnlyRewardTracker:
    """v2 reward tracker. Every non-death, non-game-done step returns 0.

    A seat's -1/0/+1 reward is delivered on its **per-seat terminal step**:
    - a seat that dies mid-episode gets `-1` on the tick it dies (this is
      that seat's terminal step from PPO's POV — the multi-agent env
      removes it from the active set immediately after);
    - a seat still alive at game-done gets `+1` if it is (or shares) the
      winner, else `0`;
    - a seat that had already died gets `-1` re-emitted on game-done, but
      the RLlib adapter never delivers that value because the seat is not
      in `active_before_step` (it was terminated in its death step). The
      re-emission is harmless bookkeeping and keeps `last_breakdown`
      consistent for callbacks that enumerate all four seats.

    Winner selection is delegated to the env via `env.state.winner_idxs`.

    The tracker mirrors `ShapedRewardTracker`'s interface (reset / compute
    / last_breakdown) so the RLlib and gym adapters can swap trackers
    transparently.
    """

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        # Kept for interface parity with ShapedRewardTracker so callbacks
        # can log dense=0.0 without a special case. `terminal` carries the
        # -1/0/+1 signal on that seat's terminal step and 0 everywhere else.
        self.last_breakdown: dict[int, dict[str, float]] = {}
        self._was_reset = False
        # Track which seats already received their -1 death reward so a
        # re-emission on game_done is idempotent for logging purposes.
        self._death_reward_emitted: set[int] = set()

    def reset(self, env: FoodTradingEnv) -> None:
        self._was_reset = True
        self._death_reward_emitted = set()

    def compute(
        self,
        env: FoodTradingEnv,
        observations: Mapping[int, dict],
        *,
        done: bool,
        info: StepInfo,
    ) -> dict[int, float]:
        if not self._was_reset:
            raise RuntimeError("reward tracker must be reset before compute()")
        rewards: dict[int, float] = {}
        breakdown: dict[int, dict[str, float]] = {}
        deaths_this_tick = set(info.deaths_this_tick) if info else set()

        if not done:
            # Non-terminal-game step. Seats that died this tick get -1 on
            # their per-seat terminal step; everyone else gets 0.
            for idx in range(self.config.player_count):
                if idx in deaths_this_tick:
                    r = -1.0
                    self._death_reward_emitted.add(idx)
                else:
                    r = 0.0
                rewards[idx] = r
                breakdown[idx] = {"dense": 0.0, "terminal": r, "total": r}
            self.last_breakdown = breakdown
            return rewards

        # Game-done step. Winners get +1, alive-nonwinners 0, dead -1.
        # Seats that already got their -1 in an earlier death step have it
        # re-emitted here for logging consistency; the RLlib env drops those
        # from active_before_step so it isn't double-delivered to PPO.
        winners = set(env.state.winner_idxs)
        for idx in range(self.config.player_count):
            player = env.state.players[idx]
            if idx in winners:
                r = 1.0
            elif player.status == "alive":
                r = 0.0
            else:
                r = -1.0
            rewards[idx] = float(r)
            breakdown[idx] = {"dense": 0.0, "terminal": float(r), "total": float(r)}
        self.last_breakdown = breakdown
        return rewards
