"""Shared harness for the ScriptedV2 / ScriptedV3 test suites."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.cadenced_scripted import CadencedScripted
from agents.scripted_common import CONSUME_PER_CYCLE
from env.env import FoodTradingEnv
from env.small_env import SmallFoodTradingEnv

CADENCES = ("fast", "medium", "slow")

# Both bots derive their whole schedule from the observation, so they must work
# unchanged on the 30-second variant. Its three-cycle horizon is the case where
# an unclamped planning window would buy food the game has no time to consume.
ENV_CLASSES = (FoodTradingEnv, SmallFoodTradingEnv)


def build_agents(inner_factory, seed: int, cadence: str) -> List[CadencedScripted]:
    agents = [
        CadencedScripted(cadence, seed=seed * 17 + i, inner_factory=inner_factory)
        for i in range(4)
    ]
    for a in agents:
        a.reset()
    return agents


def assert_no_reserved_starvation(env: FoodTradingEnv) -> None:
    """No player may reserve itself below the consumption threshold.

    `tick.consume_and_kill` reads `inventory - reserved_inventory`, so a
    resting, unfilled ask on a required food starves you exactly like a
    completed sale. This is the invariant `scripted_common.safe_ask_qty`
    exists to guarantee: if you *hold* enough food to survive, you must not
    have reserved so much of it that you no longer do.
    """
    config = env.config
    for p in env.state.players:
        if p.status != "alive":
            continue
        for food in config.required_foods(p.produces):
            held = p.inventory[food]
            available = held - p.reserved_inventory[food]
            if held >= CONSUME_PER_CYCLE:
                assert available >= CONSUME_PER_CYCLE, (
                    f"P{p.idx} reserved itself into starvation on {food} at "
                    f"t={env.state.elapsed_seconds}: held={held} "
                    f"reserved={p.reserved_inventory[food]} available={available}"
                )


def play_game(
    inner_factory,
    seed: int,
    cadence: str = "fast",
    *,
    check_invariants: bool = True,
    env_class: type = FoodTradingEnv,
) -> Dict[str, Any]:
    """Run one full self-play game, asserting per-step invariants as it goes."""
    env = env_class(seed=seed)
    obs = env.reset()
    agents = build_agents(inner_factory, seed, cadence)
    action_log: List[Tuple[int, ...]] = []

    while not env.is_done():
        actions = {}
        for i in range(4):
            mask = env.legal_actions(i)
            action = agents[i].act(obs[i], mask)
            if check_invariants:
                assert mask[action], (
                    f"P{i} returned illegal action {action} at "
                    f"t={obs[i]['tick']} (cadence={cadence}, seed={seed})"
                )
            actions[i] = action
        action_log.append(tuple(actions.values()))
        obs, _, _, _ = env.step(actions)
        if check_invariants:
            assert_no_reserved_starvation(env)

    result = env.get_result()
    result["action_log"] = action_log
    result["survivors"] = sum(1 for p in result["players"] if p["status"] == "alive")
    return result


def obs_with_inventory(
    seed: int = 0,
    *,
    player_idx: int = 0,
    inventory: Optional[Dict[str, int]] = None,
    tick: int = 0,
    env_class: type = FoodTradingEnv,
) -> Dict[str, Any]:
    """A real observation with the acting player's inventory forced."""
    env = env_class(seed=seed)
    env.reset()
    env.state.elapsed_seconds = tick
    if inventory is not None:
        for food, units in inventory.items():
            env.state.players[player_idx].inventory[food] = units
    return env.observe(player_idx)
