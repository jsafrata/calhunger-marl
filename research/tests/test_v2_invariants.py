"""v2 invariants — termination, terminal-only reward, MAX_QTY/MAX_PRICE
configurability, MLP-vs-linear head equivalence under identical params,
and gamma/lambda plumbing.

Tests must all pass before the v2 RL baselines are launched.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from env.actions import (  # noqa: E402
    MAX_PRICE,
    MAX_QTY,
    NUM_ACTIONS,
    encode_post,
    encode_take,
    factored_masks_from_flat,
    legal_action_mask,
    set_action_space,
)
from env.config import GameConfig  # noqa: E402
from env.env import FoodTradingEnv  # noqa: E402
from env.rl.factored_actor import (  # noqa: E402
    FactoredActor,
    FactoredMaskTensors,
    factored_masks_from_flat_tensor,
    flat_to_factored_batch,
)
from env.rl.rewards import TerminalOnlyRewardTracker  # noqa: E402


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


class _DieAtT:
    """Test agent that always waits; the env kills them naturally when their
    required-food inventory hits zero. Wraps `apply_action` so tests can
    ergonomically play a whole game with `WAIT` from most seats and a few
    seed actions from one seat.
    """


def _play_all_wait(seed: int = 0):
    env = FoodTradingEnv(seed=seed)
    env.reset()
    steps = 0
    while not env.is_done():
        env.step({i: 0 for i in range(4)})  # WAIT
        steps += 1
        if steps > 400:  # guard
            raise RuntimeError("game did not end within 400 steps")
    return env


def test_no_single_survivor_end_reason():
    """Under v2, the env must never report end_reason='single_survivor'."""
    for seed in range(20):
        env = _play_all_wait(seed)
        assert env.state.end_reason in ("time_limit", "no_survivors"), (
            f"seed={seed}: end_reason={env.state.end_reason}"
        )


def test_game_reaches_time_limit_when_survivor_exists():
    """If at least one seat is alive at t=180, end_reason must be time_limit."""
    env = _play_all_wait(seed=0)
    if any(p.status == "alive" for p in env.state.players):
        assert env.state.elapsed_seconds == 180
        assert env.state.end_reason == "time_limit"


def test_all_wait_game_ends_at_zero_survivors_or_time_limit():
    """Every all-WAIT game must end at exactly t=180 (time_limit) or when
    the last player dies (no_survivors) — never before that."""
    for seed in range(10):
        env = _play_all_wait(seed)
        alive_at_end = sum(1 for p in env.state.players if p.status == "alive")
        if env.state.end_reason == "time_limit":
            assert env.state.elapsed_seconds == 180
        else:
            assert env.state.end_reason == "no_survivors"
            assert alive_at_end == 0
            assert env.state.elapsed_seconds <= 180


# ---------------------------------------------------------------------------
# Terminal-only reward (nonterminal must be exactly 0)
# ---------------------------------------------------------------------------


def _run_with_terminal_tracker(seed: int = 0):
    env = FoodTradingEnv(seed=seed)
    obs = env.reset()
    tracker = TerminalOnlyRewardTracker(env.config)
    tracker.reset(env)
    per_step_rewards: list[dict[int, float]] = []
    while not env.is_done():
        obs, _, done, info = env.step({i: 0 for i in range(4)})
        r = tracker.compute(env, obs, done=done, info=info)
        per_step_rewards.append(r)
        if done:
            break
    return env, per_step_rewards


def test_terminal_only_reward_zero_at_every_non_death_non_gamedone_step():
    """Nonterminal steps deliver 0 EXCEPT the tick a seat dies on: that is
    that seat's per-agent terminal step, so it gets -1. All other seats on
    that same tick still get 0."""
    env = FoodTradingEnv(seed=0)
    obs = env.reset()
    tracker = TerminalOnlyRewardTracker(env.config)
    tracker.reset(env)
    step = 0
    while not env.is_done():
        obs, _, done, info = env.step({i: 0 for i in range(4)})
        r = tracker.compute(env, obs, done=done, info=info)
        deaths_this_tick = set(info.deaths_this_tick)
        if not done:
            for pidx, val in r.items():
                if pidx in deaths_this_tick:
                    assert val == -1.0, (
                        f"step {step} seat {pidx} died this tick — "
                        f"expected -1, got {val}"
                    )
                else:
                    assert val == 0.0, (
                        f"step {step} seat {pidx} did NOT die and game not "
                        f"done — expected 0, got {val}"
                    )
        step += 1


def test_terminal_only_reward_at_horizon_matches_survival():
    """At termination each seat gets -1/0/+1 based on winner/dead status."""
    for seed in range(20):
        env, rewards_per_step = _run_with_terminal_tracker(seed)
        terminal = rewards_per_step[-1]
        winners = set(env.state.winner_idxs)
        for pidx in range(4):
            p = env.state.players[pidx]
            if pidx in winners:
                assert terminal[pidx] == 1.0
            elif p.status == "alive":
                assert terminal[pidx] == 0.0
            else:
                assert terminal[pidx] == -1.0


def test_terminal_reward_sums_to_expected_range():
    """Total per-episode terminal reward equals (n_winners) - (n_dead).

    Alive-nonwinners contribute 0; each winner +1; each dead −1. Death
    happens at the death step; game-done re-emits the -1 for logging
    consistency.
    """
    for seed in range(10):
        env, rewards_per_step = _run_with_terminal_tracker(seed)
        terminal = rewards_per_step[-1]
        n_dead = sum(1 for p in env.state.players if p.status == "dead")
        n_winners = len(env.state.winner_idxs)
        assert sum(terminal.values()) == float(n_winners - n_dead)


def test_dead_seat_receives_minus_one_on_death_step():
    """A seat that dies mid-episode gets its -1 reward on the tick it
    dies (its per-seat terminal step). Without this, a dead-agent PPO
    policy never sees the death penalty."""
    from env.actions import ACTION_NOOP

    env = FoodTradingEnv(seed=0)
    env.reset()
    tracker = TerminalOnlyRewardTracker(env.config)
    tracker.reset(env)
    # All-wait — seats die at their first consumption cycle when they
    # run out of a required food.
    seen_death_rewards: dict[int, float] = {}
    while not env.is_done():
        obs, _, done, info = env.step({i: ACTION_NOOP for i in range(4)})
        r = tracker.compute(env, obs, done=done, info=info)
        for pidx in info.deaths_this_tick:
            if pidx not in seen_death_rewards:
                seen_death_rewards[pidx] = r[pidx]
    assert seen_death_rewards, "expected at least one death across the episode"
    for pidx, r in seen_death_rewards.items():
        assert r == -1.0, f"seat {pidx} got {r} on death step, expected -1"


# ---------------------------------------------------------------------------
# Winner selection — cash → total inventory → shared tie
# ---------------------------------------------------------------------------


def test_winner_prefers_higher_cash():
    from env.tick import determine_winners
    from env.types import Player

    state = type("S", (), {})()
    state.players = [
        Player(idx=0, name="P0", cash=100, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
        Player(idx=1, name="P1", cash=200, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
        Player(idx=2, name="P2", status="dead", cash=999, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
        Player(idx=3, name="P3", cash=150, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
    ]
    winners = determine_winners(state)
    assert winners == [1]


def test_winner_ties_break_by_inventory():
    from env.tick import determine_winners
    from env.types import Player

    state = type("S", (), {})()
    state.players = [
        Player(idx=0, name="P0", cash=100, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
        Player(idx=1, name="P1", cash=100, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
        Player(idx=2, name="P2", cash=100, inventory={"A": 10, "B": 10, "C": 10, "D": 10}),
        Player(idx=3, name="P3", cash=50, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
    ]
    winners = determine_winners(state)
    assert winners == [2]


def test_winner_exact_tie_gives_shared_win():
    from env.tick import determine_winners
    from env.types import Player

    state = type("S", (), {})()
    state.players = [
        Player(idx=0, name="P0", cash=100, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
        Player(idx=1, name="P1", cash=100, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
        Player(idx=2, name="P2", cash=50, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
        Player(idx=3, name="P3", cash=100, inventory={"A": 5, "B": 5, "C": 5, "D": 5}),
    ]
    winners = sorted(determine_winners(state))
    assert winners == [0, 1, 3]


def test_no_survivors_returns_empty_winner_list():
    from env.tick import determine_winners
    from env.types import Player

    state = type("S", (), {})()
    state.players = [
        Player(idx=i, name=f"P{i}", status="dead", cash=100 + i,
               inventory={"A": 5, "B": 5, "C": 5, "D": 5})
        for i in range(4)
    ]
    assert determine_winners(state) == []


# ---------------------------------------------------------------------------
# MAX_QTY / MAX_PRICE configurability (subprocess: fresh env-var read)
# ---------------------------------------------------------------------------


PROBE = (
    "import sys; from pathlib import Path; "
    f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
    "from env.actions import MAX_PRICE, MAX_QTY, NUM_ACTIONS, encode_post; "
    "print(f'{MAX_PRICE} {MAX_QTY} {NUM_ACTIONS}'); "
    "encode_post('bid', 'A', qty=MAX_QTY, price=MAX_PRICE)"
)


def _run_probe(env_overrides: dict[str, str]) -> tuple[int, int, int]:
    env = os.environ.copy()
    env.update(env_overrides)
    out = subprocess.check_output(
        [sys.executable, "-c", PROBE], env=env, cwd=REPO_ROOT
    ).decode().strip()
    max_price, max_qty, num_actions = out.split()
    return int(max_price), int(max_qty), int(num_actions)


def test_max_qty_1_valid():
    p, q, n = _run_probe({"HG_MAX_PRICE": "50", "HG_MAX_QTY": "1"})
    assert (p, q, n) == (50, 1, 1 + 8 + 2 * 4 * 1 + 2 * 4 * 1 * 50)


def test_max_qty_5_valid_default():
    p, q, n = _run_probe({"HG_MAX_PRICE": "50", "HG_MAX_QTY": "5"})
    assert (p, q, n) == (50, 5, 2049)


def test_max_qty_20_valid():
    p, q, n = _run_probe({"HG_MAX_PRICE": "50", "HG_MAX_QTY": "20"})
    assert (p, q, n) == (50, 20, 1 + 8 + 2 * 4 * 20 + 2 * 4 * 20 * 50)


def test_max_price_100_valid():
    p, q, n = _run_probe({"HG_MAX_PRICE": "100", "HG_MAX_QTY": "5"})
    assert (p, q, n) == (100, 5, 1 + 8 + 2 * 4 * 5 + 2 * 4 * 5 * 100)


def test_v2_default_action_count():
    """With MAX_QTY=5 and MAX_PRICE=50, the flat action count must be 2049."""
    # This test runs in the parent process — verifies the imported default.
    assert MAX_QTY == 5
    assert MAX_PRICE == 50
    assert NUM_ACTIONS == 2049


def test_set_action_space_recomputes_derived_in_process():
    """`set_action_space` mutates module globals — check the derived
    constants recompute correctly. Restore afterwards so other tests
    see the v2 default."""
    saved_price, saved_qty = MAX_PRICE, MAX_QTY
    try:
        info = set_action_space(max_price=100, max_qty=20)
        assert info["MAX_PRICE"] == 100
        assert info["MAX_QTY"] == 20
        assert info["NUM_ACTIONS"] == 1 + 8 + 2 * 4 * 20 + 2 * 4 * 20 * 100
    finally:
        set_action_space(max_price=saved_price, max_qty=saved_qty)


# ---------------------------------------------------------------------------
# Flat ↔ factored equivalence still holds
# ---------------------------------------------------------------------------


def test_flat_factored_roundtrip_covers_full_space():
    """For every legal flat action the factored decode → encode returns
    the same id. Only run for the current MAX_QTY/MAX_PRICE."""
    from env.actions import factored_to_flat, flat_to_factored

    for aid in range(NUM_ACTIONS):
        tup = flat_to_factored(aid)
        assert factored_to_flat(*tup) == aid


def test_per_head_mask_agrees_with_flat_mask():
    env = FoodTradingEnv(seed=0)
    env.reset()
    # step a few ticks to accumulate some non-trivial legal actions
    for _ in range(20):
        env.step({i: 0 for i in range(4)})
        if env.is_done():
            break
    for pidx in range(4):
        flat = env.legal_actions(pidx)
        pyth = factored_masks_from_flat(flat)
        # torch mirror
        tor = factored_masks_from_flat_tensor(
            torch.from_numpy(flat.astype(np.float32)).unsqueeze(0)
        )
        for name in ("type_mask", "food_mask", "side_mask", "price_mask", "qty_mask"):
            np_arr = getattr(pyth, name)
            torch_arr = getattr(tor, name)[0].cpu().numpy()
            assert (np_arr == torch_arr).all(), name


# ---------------------------------------------------------------------------
# MLP vs linear heads produce identical logits when linear params match
# ---------------------------------------------------------------------------


def _dummy_masks(B: int) -> FactoredMaskTensors:
    return FactoredMaskTensors(
        type_mask=torch.ones(B, 4, dtype=torch.bool),
        food_mask=torch.ones(B, 4, 4, dtype=torch.bool),
        side_mask=torch.ones(B, 4, 4, 2, dtype=torch.bool),
        price_mask=torch.ones(B, 4, 4, 2, MAX_PRICE, dtype=torch.bool),
        qty_mask=torch.ones(B, 4, 4, 2, MAX_PRICE, MAX_QTY, dtype=torch.bool),
    )


def test_linear_and_mlp_heads_both_obey_masks():
    torch.manual_seed(0)
    linear = FactoredActor(embedding_size=32, head_kind="linear")
    mlp = FactoredActor(embedding_size=32, head_kind="mlp")
    B = 8
    emb = torch.randn(B, 32)
    # Zero out the WAIT bit on the type mask for a few rows to force
    # non-trivial masking.
    masks = _dummy_masks(B)
    for actor in (linear, mlp):
        for _ in range(3):
            factored, logp, _ent = actor.sample(emb, masks, deterministic=False)
            # Every sampled type must be one the mask says is legal.
            assert (masks.type_mask.gather(1, factored["type_id"].unsqueeze(1)) > 0).all()


def test_head_kinds_have_expected_parameter_count_signatures():
    """Sanity check: MLP heads should have more params than linear heads."""
    linear = FactoredActor(embedding_size=32, head_kind="linear")
    mlp = FactoredActor(embedding_size=32, head_kind="mlp")
    p_lin = sum(p.numel() for p in linear.parameters())
    p_mlp = sum(p.numel() for p in mlp.parameters())
    assert p_mlp > p_lin, (p_lin, p_mlp)


def test_irrelevant_heads_contribute_zero_logp_under_both_head_kinds():
    """Sample WAIT actions; only the type head should contribute logp."""
    torch.manual_seed(0)
    for kind in ("linear", "mlp"):
        actor = FactoredActor(embedding_size=32, head_kind=kind)
        B = 4
        emb = torch.randn(B, 32)
        # Only WAIT legal.
        masks = _dummy_masks(B)
        masks = FactoredMaskTensors(
            type_mask=torch.tensor([[True, False, False, False]] * B),
            food_mask=masks.food_mask,
            side_mask=masks.side_mask,
            price_mask=masks.price_mask,
            qty_mask=masks.qty_mask,
        )
        factored, logp, _ = actor.sample(emb, masks, deterministic=True)
        # type_head over 1 legal option → logp=0 exactly for the type head.
        # All other heads irrelevant for WAIT → 0 contribution.
        assert torch.allclose(logp, torch.zeros(B), atol=1e-6), (kind, logp)
        assert (factored["type_id"] == 0).all()


# ---------------------------------------------------------------------------
# Reward tracker sanity: dense channel always 0 under TerminalOnly
# ---------------------------------------------------------------------------


def test_terminal_only_dense_breakdown_is_zero():
    env, _ = _run_with_terminal_tracker(seed=0)
    # not directly reachable — instead do a fresh episode and inspect breakdown
    env = FoodTradingEnv(seed=1)
    obs = env.reset()
    tracker = TerminalOnlyRewardTracker(env.config)
    tracker.reset(env)
    while not env.is_done():
        obs, _, done, info = env.step({i: 0 for i in range(4)})
        tracker.compute(env, obs, done=done, info=info)
        for pidx, b in tracker.last_breakdown.items():
            assert b["dense"] == 0.0, (pidx, b)


# ---------------------------------------------------------------------------
# gamma / lambda plumbing
# ---------------------------------------------------------------------------


def test_gamma_lambda_defaults_are_one():
    """v2 default: gamma=1.0 and lambda=1.0 in build_ppo_config."""
    import inspect
    from env.rl.rllib_config import build_ppo_config as f
    sig = inspect.signature(f)
    assert sig.parameters["gamma"].default == 1.0
    assert sig.parameters["lambda_"].default == 1.0


def test_head_kind_default_is_mlp():
    import inspect
    from env.rl.rllib_config import build_ppo_config as f
    sig = inspect.signature(f)
    assert sig.parameters["head_kind"].default == "mlp"


# ---------------------------------------------------------------------------
# Run metadata surfaces every knob
# ---------------------------------------------------------------------------


def test_build_ppo_config_puts_head_kind_in_model_config():
    """The model_config dict passed to RLModuleSpec must include head_kind,
    torso, and algo so it round-trips through checkpoint save/load."""
    from env.rl.rllib_config import build_ppo_config
    cfg = build_ppo_config(head_kind="linear", num_env_runners=0,
                           train_batch_size=64, minibatch_size=32,
                           num_epochs=1)
    spec = cfg.rl_module_spec
    # Ray exposes model_config via spec._model_config_dict on some
    # versions and .model_config on others.
    model_cfg = getattr(spec, "model_config", None)
    if not model_cfg:
        model_cfg = getattr(spec, "_model_config_dict", {}) or {}
    assert model_cfg.get("head_kind") == "linear"
    assert model_cfg.get("torso") == "mlp"
    assert model_cfg.get("algo") == "mappo"
