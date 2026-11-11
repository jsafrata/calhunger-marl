"""End-to-end arena orchestration with the mock provider (no tokens).

Mirrors the invariants the paper's cross-play harness enforces: exact
seat balance, fixed paired seeds, per-game caching, and summary rows
that join the existing matrices — plus the LLM-only trajectory records.
"""
import asyncio
import json

import pytest

from llm_arena.arena import (
    LLMSpec,
    job_identity,
    parse_arena_spec,
    play_arena_games,
    summarize_games,
)
from llm_arena.providers import ProviderRuntime
import checkpoint_policy_crossplay as crossplay

ENV_VARIANT = "small"
ENV_CLASS = crossplay.ENVIRONMENT_SPECS[ENV_VARIANT][0]
SEED_BASE = 900_000
N_GAMES = 4


def _slots():
    focal = parse_arena_spec("llm:mm=mock/market-maker")
    opponent = parse_arena_spec("scripted:v1")
    return [focal, opponent, opponent, opponent]


def _run(tmp_path, overwrite=False):
    slots = _slots()
    identity = job_identity(
        slots,
        lineup_mode="1v3",
        mode="stochastic",
        n_games=N_GAMES,
        game_seed_base=SEED_BASE,
        environment_variant=ENV_VARIANT,
        environment_class=ENV_CLASS,
        llm_attempts=5,
        on_malformed="wait",
    )
    records = asyncio.run(
        play_arena_games(
            slots,
            N_GAMES,
            SEED_BASE,
            deterministic=False,
            environment_class=ENV_CLASS,
            identity=identity,
            games_dir=tmp_path / "games",
            llm_runtime=ProviderRuntime(),
            overwrite=overwrite,
        )
    )
    return slots, identity, records


def test_spec_grammar():
    llm = parse_arena_spec("llm:haiku=anthropic_api/claude-haiku-4-5")
    assert isinstance(llm, LLMSpec)
    assert llm.key == "llm_haiku"
    assert llm.config.provider == "anthropic_api"
    scripted = parse_arena_spec("scripted:v4")
    assert scripted.kind == "scripted"
    with pytest.raises(ValueError):
        parse_arena_spec("llm:nolabel")
    with pytest.raises(ValueError):
        parse_arena_spec("llm:x=providernomodel")
    with pytest.raises(ValueError):
        parse_arena_spec("robot:v1")


def test_full_mock_game_run(tmp_path):
    slots, _, records = _run(tmp_path)
    assert len(records) == N_GAMES

    # Exact seat balance: the focal slot occupies each physical seat once.
    focal_seats = sorted(
        record["lineup"].index("llm_mm") for record in records
    )
    assert focal_seats == [0, 1, 2, 3]

    # Paired seeds: game g always plays seed base + g.
    assert [r["game_seed"] for r in records] == [
        SEED_BASE + g for g in range(N_GAMES)
    ]

    for record in records:
        assert len(record["seats"]) == 4
        decisions = record["llm_decisions"]
        assert decisions, "the mock seat must have produced decisions"
        duration = ENV_CLASS(seed=0).config.game_duration_seconds
        assert len(decisions) <= duration
        for decision in decisions:
            assert decision["spec"] == "llm_mm"
            assert decision["attempts"] == 1
            assert decision["malformed"] is False
            # The env recorded a verdict for every submitted action.
            assert decision["env_applied"] is not None


def test_cache_resume_returns_identical_records(tmp_path):
    _, _, first = _run(tmp_path)
    game_files = sorted((tmp_path / "games").glob("game_*.json"))
    assert len(game_files) == N_GAMES
    _, _, second = _run(tmp_path)  # must load every game from cache
    assert first == second


def test_cache_identity_mismatch_is_refused(tmp_path):
    _run(tmp_path)
    path = tmp_path / "games" / "game_0000.json"
    payload = json.loads(path.read_text())
    payload["identity"]["game_seed_base"] = 123
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match"):
        _run(tmp_path)


def test_summary_rows_join_the_crossplay_schema(tmp_path):
    slots, _, records = _run(tmp_path)
    rows = summarize_games(
        records,
        slots,
        lineup_mode="1v3",
        mode="stochastic",
        game_seed_base=SEED_BASE,
        environment_variant=ENV_VARIANT,
        environment_class=ENV_CLASS.__name__,
        game_duration_seconds=30,
    )
    by_focal = {row["focal"]: row for row in rows}
    assert set(by_focal) == {"llm_mm", "scripted_v1"}

    llm_row = by_focal["llm_mm"]
    for name in crossplay.SUMMARY_FIELDS:
        assert name in llm_row
    assert llm_row["n_focal_seat_games"] == N_GAMES
    assert llm_row["n_llm_decisions"] > 0
    assert llm_row["malformed_decision_percentage"] == 0.0
    assert 0.0 <= llm_row["survival_percentage"] <= 100.0

    scripted_row = by_focal["scripted_v1"]
    assert scripted_row["n_llm_decisions"] == ""
    assert scripted_row["n_focal_seat_games"] == 3 * N_GAMES


def test_malformed_fallback_records_wait(tmp_path):
    from llm_arena.arena import LLMSeat
    from llm_arena.providers import DecisionProvider, ProviderReply

    class GarbageProvider(DecisionProvider):
        async def decide(self, prompt, state=None):
            return ProviderReply("not json at all", "mock", "garbage")

    spec = parse_arena_spec("llm:bad=mock/garbage")
    seat = LLMSeat(
        spec,
        ProviderRuntime(),
        attempts=2,
        on_malformed="wait",
        provider=GarbageProvider(spec.config, ProviderRuntime()),
    )
    env = ENV_CLASS(seed=1)
    obs = env.reset()
    seat.begin_game(1, [0])
    actions, records = asyncio.run(seat.act_async(env, obs))
    assert actions == {0: 0}
    assert records[0].malformed is True
    assert records[0].attempts == 2


def test_mask_illegal_action_is_replaced_with_wait(tmp_path):
    from llm_arena.arena import LLMSeat
    from llm_arena.providers import DecisionProvider, ProviderReply

    class TakerProvider(DecisionProvider):
        # take on an empty book is well-formed but mask-illegal
        async def decide(self, prompt, state=None):
            return ProviderReply(
                '{"action":"take","side":"bid","food":"A",'
                '"quantity":1,"price":null}',
                "mock",
                "taker",
            )

    spec = parse_arena_spec("llm:taker=mock/taker")
    seat = LLMSeat(
        spec,
        ProviderRuntime(),
        attempts=2,
        on_malformed="wait",
        provider=TakerProvider(spec.config, ProviderRuntime()),
    )
    env = ENV_CLASS(seed=1)
    obs = env.reset()  # books empty: nothing to take
    seat.begin_game(1, [0])
    actions, records = asyncio.run(seat.act_async(env, obs))
    assert actions == {0: 0}  # WAIT submitted instead
    assert records[0].mask_illegal is True
    assert records[0].malformed is False
    assert records[0].decision == {
        "action": "take", "side": "bid", "food": "A", "quantity": 1,
    }
