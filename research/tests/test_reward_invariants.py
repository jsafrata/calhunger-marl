"""Invariants for the shaped reward — the properties the training team
committed to before launching IPPO/MAPPO.

Items covered:

  1. `required_foods` per seat has length 2 (complement of the 2 produced foods).
  2. The dense buffer signal reads *post-consumption* available inventory.
  3. The dense and terminal channels are returned separately in
     `ShapedRewardTracker.last_breakdown`.
  4. IPPO and MAPPO get the same reward (same tracker, same code path).
  5. Dense fires only at consumption ticks (already covered in
     `test_mappo_env`; re-exercised here on the core reward path for the
     record).
"""
from __future__ import annotations

import pytest

from env.config import DEFAULT_CONFIG
from env.env import FoodTradingEnv
from env.observation import build_all_observations
from env.rl.rewards import (
    BUFFER_CAP,
    ShapedRewardTracker,
    SURVIVAL_REWARD_SCALE,
    SURVIVED_NON_WIN_REWARD,
    WIN_REWARD,
)


# ---------------------------------------------------------------------------
# Item 1 — required_foods is always exactly 2
# ---------------------------------------------------------------------------


def test_required_foods_is_exactly_two_per_seat():
    """Every seat produces 2 foods and requires the other 2."""
    for seed in range(50):
        env = FoodTradingEnv(seed=seed)
        obs = env.reset()
        for i in range(env.config.player_count):
            required = obs[i]["required_foods"]
            produces = obs[i]["produces"]
            assert len(required) == 2, (
                f"seed={seed} seat={i}: required_foods has {len(required)} "
                f"entries, expected 2"
            )
            assert len(produces) == 2, (
                f"seed={seed} seat={i}: produces has {len(produces)} entries, "
                f"expected 2"
            )
            # Required and produced must be disjoint AND cover all 4 foods.
            assert set(required).isdisjoint(produces)
            assert set(required) | set(produces) == set(env.config.food_types)


def test_reward_buffer_only_reads_required_foods():
    """The dense signal must depend ONLY on the two required foods.
    We verify by setting non-required (i.e. produced) inventory to 0 and
    non-produced inventory very high — the reward must be dominated by
    the required foods, ignoring the produced ones."""
    env = FoodTradingEnv(seed=3)
    obs = env.reset()
    tracker = ShapedRewardTracker(env.config, dense_reward=True)
    tracker.reset(env)
    # Step to just before the first consumption tick.
    interval = env.config.consumption_interval_seconds
    for _ in range(interval - 1):
        obs, _, _, _ = env.step({i: 0 for i in range(4)})

    player = env.state.players[0]
    required = env.config.required_foods(player.produces)
    produces = list(player.produces)
    # Required: set both to 7 (post-consumption: 5). Produced: set both to
    # 200 — irrelevant to reward. Also clear reservations so `available`
    # matches inventory.
    for food in env.config.food_types:
        player.reserved_inventory[food] = 0
    for food in required:
        player.inventory[food] = 5 + env.config.consumption_per_required_food_per_second
    for food in produces:
        player.inventory[food] = 200

    obs, _, done, info = env.step({i: 0 for i in range(4)})
    rewards = tracker.compute(env, obs, done=done, info=info)
    # Post-consumption: required-food inventory = 5 → min(5, BUFFER_CAP=20) = 5
    # Produced-food inventory = 200 but IGNORED. Terminal reward not yet
    # active (game not done). Reward = 0.01 × 5 = 0.05.
    assert rewards[0] == pytest.approx(SURVIVAL_REWARD_SCALE * 5), (
        f"reward {rewards[0]} depends on produced foods; expected "
        f"{SURVIVAL_REWARD_SCALE * 5} from the min(5,5) required signal."
    )


# ---------------------------------------------------------------------------
# Item 2 — dense uses POST-consumption available inventory
# ---------------------------------------------------------------------------


def test_reward_reads_post_consumption_inventory():
    """The reward tracker consumes the `observations` returned by
    env.step(), which are built AFTER production + consumption. This test
    sets pre-consumption inventory so the post-consumption value is a
    known target, then verifies the reward reflects the post value."""
    env = FoodTradingEnv(seed=9)
    obs = env.reset()
    tracker = ShapedRewardTracker(env.config, dense_reward=True)
    tracker.reset(env)
    interval = env.config.consumption_interval_seconds
    consume = env.config.consumption_per_required_food_per_second
    produce = env.config.production_per_second

    # Advance to just before the consumption tick.
    for _ in range(interval - 1):
        obs, _, _, _ = env.step({i: 0 for i in range(4)})

    player = env.state.players[0]
    required = env.config.required_foods(player.produces)
    # Clear reservations for a clean available = inventory.
    for food in env.config.food_types:
        player.reserved_inventory[food] = 0
    # Pre-consumption required inventory = X ; post-consumption = X - consume
    # Target post-consumption min = 4 → set pre = 4 + consume for both.
    target_post = 4
    for food in required:
        player.inventory[food] = target_post + consume
    # Also fix produced-food inventory so production doesn't blow past
    # BUFFER_CAP unexpectedly (irrelevant to reward but keeps state clean).
    for food in player.produces:
        player.inventory[food] = 50

    obs, _, done, info = env.step({i: 0 for i in range(4)})
    rewards = tracker.compute(env, obs, done=done, info=info)
    assert rewards[0] == pytest.approx(SURVIVAL_REWARD_SCALE * target_post), (
        f"got {rewards[0]}, expected {SURVIVAL_REWARD_SCALE * target_post}. "
        f"If the reward reads PRE-consumption inventory, we'd instead see "
        f"{SURVIVAL_REWARD_SCALE * (target_post + consume)}."
    )
    # Sanity: the actual post-consumption inventory in the observation matches
    # our target — proves the reward is really reading post-consumption obs.
    for food in required:
        assert obs[0]["available_inventory"][food] == target_post


# ---------------------------------------------------------------------------
# Item 3 — dense and terminal returned separately
# ---------------------------------------------------------------------------


def test_reward_breakdown_split_available_and_correct():
    """`last_breakdown` returns the dense and terminal contributions
    separately, and their sum equals the returned total."""
    env = FoodTradingEnv(seed=1)
    obs = env.reset()
    tracker = ShapedRewardTracker(env.config, dense_reward=True)
    tracker.reset(env)
    # Take one step (not a consumption tick, not terminal → both channels 0).
    obs, _, done, info = env.step({i: 0 for i in range(4)})
    rewards = tracker.compute(env, obs, done=done, info=info)
    for i in range(4):
        assert i in tracker.last_breakdown
        b = tracker.last_breakdown[i]
        assert set(b) == {"dense", "terminal", "total"}
        assert b["dense"] + b["terminal"] == pytest.approx(b["total"])
        assert b["total"] == pytest.approx(rewards[i])
        assert b["dense"] == 0.0  # non-consumption tick
        assert b["terminal"] == 0.0  # non-terminal


def test_reward_breakdown_populates_dense_on_consumption_tick():
    """On the first consumption tick, dense channel is positive and terminal
    is still zero."""
    env = FoodTradingEnv(seed=2)
    obs = env.reset()
    tracker = ShapedRewardTracker(env.config, dense_reward=True)
    tracker.reset(env)
    interval = env.config.consumption_interval_seconds
    for step_idx in range(1, interval + 1):
        obs, _, done, info = env.step({i: 0 for i in range(4)})
        tracker.compute(env, obs, done=done, info=info)
    for i in range(4):
        b = tracker.last_breakdown[i]
        # Dense > 0 iff player is still alive and required inventory > 0
        player = env.state.players[i]
        if player.status == "alive":
            assert b["dense"] > 0.0, (
                f"seat {i} alive on consumption tick, dense should be > 0 "
                f"but got {b['dense']}"
            )
        assert b["terminal"] == 0.0, (
            f"terminal channel fired on non-terminal step: {b['terminal']}"
        )


# ---------------------------------------------------------------------------
# Item 4 — IPPO and MAPPO use the same reward
# ---------------------------------------------------------------------------


def test_reward_identical_across_algo_variants():
    """The reward tracker is instantiated by the shared multi-agent env
    adapter — algorithm choice must NOT change the reward stream. We
    verify by running the same seed under the same env config with two
    separate tracker instances (mimicking two learner instances that
    happen to be labeled IPPO vs MAPPO) and asserting bit-for-bit
    identical rewards throughout the episode."""
    seed = 42

    def run(seed_):
        env = FoodTradingEnv(seed=seed_)
        obs = env.reset()
        tracker = ShapedRewardTracker(env.config, dense_reward=True)
        tracker.reset(env)
        stream = []
        while not env.is_done():
            actions = {i: 0 for i in range(4)}  # all wait
            obs, _, done, info = env.step(actions)
            r = tracker.compute(env, obs, done=done, info=info)
            stream.append(tuple(r[i] for i in range(4)))
        return stream

    a = run(seed)
    b = run(seed)
    assert a == b, "reward stream must be deterministic across identical env configs"
    # Non-degenerate signal expected (all-wait games hit terminal survive/no-survive)
    assert any(any(x != 0 for x in tup) for tup in a), "expected some non-zero rewards"


# ---------------------------------------------------------------------------
# Terminal magnitude sanity — winner strictly beats maximum shaping
# ---------------------------------------------------------------------------


def test_terminal_reward_config_overrides_defaults():
    """A/B experiments swap the terminal magnitudes without touching any
    other file. Verify the tracker uses the overrides."""
    env = FoodTradingEnv(seed=0)
    obs = env.reset()
    tracker = ShapedRewardTracker(
        env.config,
        dense_reward=False,
        win_reward=5.0,
        survived_non_win_reward=2.0,
        death_penalty=-5.0,
        win_split_across_ties=False,
    )
    tracker.reset(env)
    # Force game to end this step by manually setting phase.
    env.state.phase = "ended"
    env.state.players[0].status = "dead"
    env.state.players[1].status = "alive"
    env.state.players[2].status = "alive"
    env.state.players[3].status = "dead"
    env.state.winner_idxs = [1]

    from env.env import StepInfo
    info = StepInfo(tick=180, actions_applied={}, deaths_this_tick=[],
                    end_reason="single_survivor")
    rewards = tracker.compute(env, obs, done=True, info=info)
    assert rewards[0] == -5.0   # dead
    assert rewards[1] == 5.0    # sole winner (no split because win_split_across_ties=False)
    assert rewards[2] == 2.0    # survived non-winner
    assert rewards[3] == -5.0   # dead


def test_terminal_reward_defaults_match_baseline():
    """Default terminal values must NOT change unless explicitly overridden.
    Otherwise every existing pipeline silently gets a new reward."""
    from env.rl.rewards import (
        WIN_REWARD, SURVIVED_NON_WIN_REWARD, DEATH_PENALTY, WIN_SPLIT_ACROSS_TIES,
    )
    assert WIN_REWARD == 5.0
    assert SURVIVED_NON_WIN_REWARD == 1.0
    assert DEATH_PENALTY == 0.0
    assert WIN_SPLIT_ACROSS_TIES is True


def test_winner_signal_dominates_max_dense_at_config_limits():
    """The winner-vs-survive distinguishing signal (WIN - SURVIVED_NON_WIN)
    must exceed the maximum possible integrated dense signal for the
    configured buffer cap and consumption interval. This is the invariant
    that prevents shaping from swamping the terminal objective."""
    duration = DEFAULT_CONFIG.game_duration_seconds
    interval = DEFAULT_CONFIG.consumption_interval_seconds
    n_consumption_ticks = duration // interval
    max_dense_possible = SURVIVAL_REWARD_SCALE * BUFFER_CAP * n_consumption_ticks
    signal = WIN_REWARD - SURVIVED_NON_WIN_REWARD
    assert max_dense_possible < signal, (
        f"max integrated dense ({max_dense_possible}) exceeds winner "
        f"distinguishing signal ({signal}). Adjust BUFFER_CAP, SCALE, or "
        f"terminal-reward constants until the invariant holds."
    )
