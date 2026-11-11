"""The v2 prompt derives every game constant from the env being played."""
from env.actions import MAX_PRICE, MAX_QTY
from env.env import FoodTradingEnv
from env.small_env import FoodTradingEnv60, SmallFoodTradingEnv

from llm_arena.prompt import (
    STATE_KEYS,
    build_prompt,
    build_rules,
    build_state_block,
    minimal_game_state,
    prompt_template_text,
)


def _obs(env_class, seed=3):
    env = env_class(seed=seed)
    obs = env.reset()
    return env, obs


def test_rules_track_the_environment_config():
    env_default, obs_default = _obs(FoodTradingEnv)
    env_small, obs_small = _obs(SmallFoodTradingEnv)
    default_rules = build_rules(env_default.config, obs_default[0])
    small_rules = build_rules(env_small.config, obs_small[0])
    assert "after second 180" in default_rules
    assert "after second 30" in small_rules
    assert f"1..{MAX_PRICE}" in default_rules
    assert f"1..{MAX_QTY}" in default_rules
    # Seat role is spelled out in the rules.
    produces = list(obs_default[0]["produces"])
    assert f"You produce {produces[0]} and {produces[1]}" in default_rules


def test_state_block_reports_feeding_arithmetic():
    env, obs = _obs(FoodTradingEnv60, seed=5)
    block = build_state_block(obs[1], env.config)
    assert "next feeding at second 10, 10s away" in block
    assert "6 feedings left" in block
    for f in obs[1]["required_foods"]:
        cover = obs[1]["available_inventory"][f] // 2
        assert f"survives {cover} feeding" in block


def test_starvation_warning_appears_when_below_consumption():
    env, obs = _obs(FoodTradingEnv60, seed=5)
    seat_obs = dict(obs[0])
    starved = dict(seat_obs["available_inventory"])
    starved[seat_obs["required_foods"][0]] = 1
    seat_obs["available_inventory"] = starved
    block = build_state_block(seat_obs, env.config)
    assert "YOU DIE at second 10" in block


def test_prompt_is_deterministic_and_advice_free():
    env, obs = _obs(FoodTradingEnv60, seed=7)
    first = build_prompt(obs[2], env.config)
    second = build_prompt(env.observe(2), env.config)
    assert first == second
    for heading in ("SURVIVAL RULES", "MARKET RULES", "WINNING", "YOUR STATE",
                    "MARKET (", "OUTPUT"):
        assert heading in first
    assert "HOW TO PLAY WELL" not in first


def test_template_text_is_stable_for_identity_hashing():
    config = FoodTradingEnv60(seed=0).config
    assert prompt_template_text(config) == prompt_template_text(config)
    assert prompt_template_text(config) != prompt_template_text(
        FoodTradingEnv(seed=0).config
    )


def test_minimal_state_exposes_exactly_the_allowed_keys():
    env, obs = _obs(SmallFoodTradingEnv)
    state = minimal_game_state(obs[0])
    assert set(state) == STATE_KEYS
    assert "players" not in state
    assert state["cash"] == env.config.initial_cash


def test_own_orders_are_excluded_from_market_block():
    from env.actions import encode_post

    from llm_arena.prompt import build_market_block

    env = SmallFoodTradingEnv(seed=11)
    env.reset()
    food = env.state.players[0].produces[0]
    actions = {0: encode_post("ask", food, 1, 3), 1: 0, 2: 0, 3: 0}
    obs, _, _, _ = env.step(actions)
    own_orders = obs[0]["own_orders"]
    if any(o["food"] == food and o["side"] == "ask" for o in own_orders):
        poster_view = build_market_block(obs[0], env.config)
        other_view = build_market_block(obs[1], env.config)
        poster_line = next(
            line for line in poster_view.splitlines() if f"- {food} " in line
        )
        other_line = next(
            line for line in other_view.splitlines() if f"- {food} " in line
        )
        assert "1@3" not in poster_line.split("|")[0]
        assert "1@3" in other_line.split("|")[0]


def test_legal_block_matches_the_env_mask():
    """The LEGAL block's caps must agree with env.legal_actions exactly."""
    from agents.cadenced_scripted import CadencedScripted
    from agents.scripted_v2 import ScriptedV2Agent
    from env.actions import MAX_QTY, encode_cancel, encode_post, encode_take

    env = FoodTradingEnv60(seed=23)
    obs = env.reset()
    agents = {
        s: CadencedScripted("fast", seed=23 * 7 + s, inner_factory=ScriptedV2Agent)
        for s in range(4)
    }
    for a in agents.values():
        a.reset()

    def other_orders(o, food, key):
        return [x for x in o["order_books"][food][key]
                if x["player_idx"] != o["player_idx"]]

    checked = 0
    while not env.is_done():
        if env.state.elapsed_seconds % 7 == 0:
            for seat in range(4):
                if env.state.players[seat].status != "alive":
                    continue
                o = obs[seat]
                mask = env.legal_actions(seat)
                cash = o["available_cash"]
                avail = o["available_inventory"]
                for f in "ABCD":
                    asks = other_orders(o, f, "ask_orders")
                    derived_buy = (
                        min(MAX_QTY, cash // min(x["price"] for x in asks))
                        if asks else 0
                    )
                    mask_buy = max(
                        (q for q in range(1, MAX_QTY + 1)
                         if mask[encode_take("bid", f, q)]),
                        default=0,
                    )
                    assert derived_buy == mask_buy, (seat, f, "take-buy")
                    bids = other_orders(o, f, "bid_orders")
                    derived_sell = min(MAX_QTY, avail[f]) if bids and avail[f] >= 1 else 0
                    mask_sell = max(
                        (q for q in range(1, MAX_QTY + 1)
                         if mask[encode_take("ask", f, q)]),
                        default=0,
                    )
                    assert derived_sell == mask_sell, (seat, f, "take-sell")
                    derived_post_ask = min(MAX_QTY, avail[f])
                    mask_post_ask = max(
                        (q for q in range(1, MAX_QTY + 1)
                         if mask[encode_post("ask", f, q, 1)]),
                        default=0,
                    )
                    assert derived_post_ask == mask_post_ask, (seat, f, "post-ask")
                    mask_post_bid_q1 = mask[encode_post("bid", f, 1, 1)]
                    assert bool(mask_post_bid_q1) == (cash >= 1), (seat, f, "post-bid")
                    own = {(x["side"], x["food"]) for x in o["own_orders"]}
                    for side in ("bid", "ask"):
                        assert bool(mask[encode_cancel(side, f)]) == ((side, f) in own), (
                            seat, f, side, "cancel")
                    checked += 1
        actions = {s: agents[s].act(obs[s], env.legal_actions(s)) for s in range(4)}
        obs, _, _, _ = env.step(actions)
    assert checked >= 20
