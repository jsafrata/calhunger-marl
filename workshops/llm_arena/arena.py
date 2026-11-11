"""Seat-balanced, seed-paired games with LLM, policy, and scripted seats.

This is the paper's evaluation protocol (``workshops/eval/
checkpoint_policy_crossplay.py``) extended with a third seat kind:

- ``llm:LABEL=PROVIDER/MODEL`` — a language model queried once per
  simulated second over the strict JSON wire protocol;
- ``policy:LABEL=CHECKPOINT[@MODULE]`` — a trained RLlib checkpoint
  (reuses the cross-play harness's ``PolicySeat``/``ModulePool``);
- ``scripted:v1``..``scripted:v5`` — the calibrated heuristic ladder
  behind the standard fast cadence wrapper.

Protocol conventions inherited unchanged from the cross-play harness:
cyclic seat rotation (exact seat balance), fixed game-seed base (paired
games across lineups), ``seed_everything`` before policy sampling, and
the mandatory self-play diagonal is expressible as a uniform lineup.

LLM-specific measurement (the workshop's trajectory-level axis): every
decision is recorded with raw response, parse attempts, latency, token
usage, and the env's verdict on the parsed action (applied vs rejected).
A model that exhausts its parse attempts falls back to WAIT by default
(``on_malformed="wait"``) so reliability is measured rather than fatal;
``on_malformed="abort"`` reproduces the preliminary arena's fail-closed
exhibition behavior.

Games are cached one file each under ``<output>/games/``, so an
interrupted (or rate-limited) run resumes at game granularity.  Note:
when a run with policy seats resumes, the torch sampling stream restarts
at the job seed for the remaining games (game seeds and lineups are
unaffected).
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ._paths import bootstrap

bootstrap()

import numpy as np  # noqa: E402

import checkpoint_policy_crossplay as crossplay  # noqa: E402
from env.actions import ACTION_WAIT  # noqa: E402
from env.env import FoodTradingEnv  # noqa: E402

from .action_protocol import DecisionError, parse_decision  # noqa: E402
from .prompt import (  # noqa: E402
    PROTOCOL_VERSION,
    build_prompt,
    minimal_game_state,
    prompt_template_text,
)
from .providers import (  # noqa: E402
    DecisionProvider,
    LLMSeatConfig,
    ProviderError,
    ProviderRuntime,
    make_provider,
)

# Environment registry shared with the cross-play harness (default 180 s
# plus the small/30 .. small_150 scaled family).
ENVIRONMENT_SPECS = crossplay.ENVIRONMENT_SPECS

LLM_SUMMARY_EXTRAS = (
    "n_llm_decisions",
    "mean_decision_latency_seconds",
    "mean_attempts_per_decision",
    "malformed_decision_percentage",
    "rejected_action_percentage",
    "wait_action_percentage",
    "total_input_tokens",
    "total_output_tokens",
    "total_cached_input_tokens",
)

SUMMARY_FIELDS = crossplay.SUMMARY_FIELDS + LLM_SUMMARY_EXTRAS


class ArenaAbort(RuntimeError):
    """A seat failed hard (provider error / fail-closed malformed)."""


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMSpec:
    """One seat driven by a language model."""

    label: str
    config: LLMSeatConfig
    kind: str = "llm"

    @property
    def key(self) -> str:
        return f"llm_{self.label}"

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "llm",
            "label": self.label,
            "provider": self.config.provider,
            "model": self.config.model,
        }


def parse_arena_spec(
    text: str,
    llm_overrides: dict[str, dict[str, str]] | None = None,
) -> Any:
    """Parse ``llm:LABEL=PROVIDER/MODEL`` or delegate to the cross-play
    grammar (``policy:...`` / ``scripted:...``).

    ``llm_overrides`` maps an LLM label to optional provider settings
    (``api_key_env``, ``base_url``, ``executable``) from ``--llm-config``.
    """
    if not text.startswith("llm:"):
        return crossplay.parse_spec(text)
    body = text.removeprefix("llm:")
    label, sep, rest = body.partition("=")
    if not sep or not label or not rest:
        raise ValueError(
            f"llm spec must look like llm:LABEL=PROVIDER/MODEL, got {text!r}"
        )
    provider, sep, model = rest.partition("/")
    if not sep or not provider or not model:
        raise ValueError(
            f"llm spec must look like llm:LABEL=PROVIDER/MODEL, got {text!r}"
        )
    overrides = (llm_overrides or {}).get(label, {})
    unknown = set(overrides) - {"api_key_env", "base_url", "executable"}
    if unknown:
        raise ValueError(
            f"unknown llm-config keys for {label!r}: {sorted(unknown)}"
        )
    return LLMSpec(
        label=label,
        config=LLMSeatConfig(
            label=label,
            provider=provider,
            model=model,
            api_key_env=overrides.get("api_key_env"),
            base_url=overrides.get("base_url"),
            executable=overrides.get("executable"),
        ),
    )


# ---------------------------------------------------------------------------
# LLM seat runner
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """One LLM decision: the full trajectory-level audit record."""

    tick: int
    seat: int
    spec: str
    provider: str
    model: str
    prompt_sha256: str
    decision: dict[str, Any] | None
    action_index: int
    raw_response: str
    request_id: str | None
    attempts: int
    malformed: bool
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    # A well-formed action the legal mask forbids (e.g. selling food the
    # seat does not have available).  A WAIT is submitted in its place —
    # the same constraint policy/scripted seats live under by
    # construction — and the violation is scored via
    # ``rejected_action_percentage``.  Also required for env integrity:
    # the simultaneous-clearing pipeline assumes mask-legal actions, and
    # unmasked replacement posts can crash it (validated pre-tick, then
    # invalidated by a same-tick take before queuing).
    mask_illegal: bool = False
    env_applied: dict[str, Any] | None = None


class LLMSeat:
    """Provider-backed seats sharing one spec; async, one call per seat.

    Mirrors the preliminary arena's retry discipline: a malformed
    response is discarded and the byte-identical prompt is retried, up
    to ``attempts`` total; no repair instruction is ever appended.
    Provider hard errors (auth, network, exhausted HTTP retries) abort
    the game — a missing answer cannot be scored fairly.
    """

    def __init__(
        self,
        spec: LLMSpec,
        runtime: ProviderRuntime,
        *,
        attempts: int,
        on_malformed: str,
        provider: DecisionProvider | None = None,
    ) -> None:
        if on_malformed not in {"wait", "abort"}:
            raise ValueError("on_malformed must be 'wait' or 'abort'")
        self.spec = spec
        self.provider = provider or make_provider(spec.config, runtime)
        self.attempts = attempts
        self.on_malformed = on_malformed
        self._seats: list[int] = []

    def preflight(self) -> list[str]:
        return self.provider.preflight()

    def begin_game(self, game_seed: int, seats: Sequence[int]) -> None:
        del game_seed  # LLM sampling happens provider-side
        self._seats = list(seats)

    async def act_async(
        self, env: FoodTradingEnv, obs_raw: dict
    ) -> tuple[dict[int, int], list[DecisionRecord]]:
        actions: dict[int, int] = {}
        pending: list[int] = []
        for seat in self._seats:
            if env.state.players[seat].status != "alive":
                # Dead seats wait without an API call; the env would
                # no-op any action from them anyway.
                actions[seat] = ACTION_WAIT
            else:
                pending.append(seat)
        records = await asyncio.gather(
            *(self._one_decision(env, obs_raw[seat], seat) for seat in pending)
        )
        for record in records:
            actions[record.seat] = record.action_index
        return actions, list(records)

    async def _one_decision(
        self, env: FoodTradingEnv, observation: dict, seat: int
    ) -> DecisionRecord:
        prompt = build_prompt(observation, env.config)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        started = time.monotonic()
        errors: list[str] = []
        raw_text = ""
        request_id: str | None = None
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        saw_usage = False

        structured_state = minimal_game_state(observation)
        for attempt in range(1, self.attempts + 1):
            try:
                reply = await self.provider.decide(prompt, structured_state)
            except ProviderError as exc:
                raise ArenaAbort(
                    f"seat {seat} ({self.spec.key}) provider request failed: {exc}"
                ) from None
            raw_text = reply.text
            request_id = reply.request_id
            if reply.input_tokens is not None:
                input_tokens += reply.input_tokens
                saw_usage = True
            if reply.output_tokens is not None:
                output_tokens += reply.output_tokens
                saw_usage = True
            if reply.cached_input_tokens is not None:
                cached_tokens += reply.cached_input_tokens
            try:
                parsed = parse_decision(reply.text)
            except DecisionError as exc:
                errors.append(f"attempt {attempt}: {exc}")
                continue
            mask_illegal = not bool(
                env.legal_actions(seat)[parsed.action_index]
            )
            return DecisionRecord(
                tick=env.state.elapsed_seconds,
                seat=seat,
                spec=self.spec.key,
                provider=reply.provider,
                model=reply.model,
                prompt_sha256=prompt_sha,
                decision=parsed.value,
                action_index=ACTION_WAIT if mask_illegal else parsed.action_index,
                mask_illegal=mask_illegal,
                raw_response=reply.text,
                request_id=request_id,
                attempts=attempt,
                malformed=False,
                latency_seconds=round(time.monotonic() - started, 6),
                input_tokens=input_tokens if saw_usage else None,
                output_tokens=output_tokens if saw_usage else None,
                cached_input_tokens=cached_tokens if saw_usage else None,
            )

        if self.on_malformed == "abort":
            raise ArenaAbort(
                f"seat {seat} ({self.spec.key}) produced no correctly formatted "
                f"decision after {self.attempts} attempts: " + " | ".join(errors)
            )
        return DecisionRecord(
            tick=env.state.elapsed_seconds,
            seat=seat,
            spec=self.spec.key,
            provider=self.spec.config.provider,
            model=self.spec.config.model,
            prompt_sha256=prompt_sha,
            decision=None,
            action_index=ACTION_WAIT,
            raw_response=raw_text,
            request_id=request_id,
            attempts=self.attempts,
            malformed=True,
            latency_seconds=round(time.monotonic() - started, 6),
            input_tokens=input_tokens if saw_usage else None,
            output_tokens=output_tokens if saw_usage else None,
            cached_input_tokens=cached_tokens if saw_usage else None,
        )


# ---------------------------------------------------------------------------
# Job identity / caching
# ---------------------------------------------------------------------------


def prompt_template_sha(environment_class: type[FoodTradingEnv]) -> str:
    config = environment_class(seed=0).config
    return hashlib.sha256(
        prompt_template_text(config).encode("utf-8")
    ).hexdigest()


def job_identity(
    slots: Sequence[Any],
    *,
    lineup_mode: str,
    mode: str,
    n_games: int,
    game_seed_base: int,
    environment_variant: str,
    environment_class: type[FoodTradingEnv],
    llm_attempts: int,
    on_malformed: str,
) -> dict[str, Any]:
    return {
        "harness": "llm_arena",
        "protocol_version": PROTOCOL_VERSION,
        "prompt_template_sha256": prompt_template_sha(environment_class),
        "lineup_mode": lineup_mode,
        "slots": [spec.identity() for spec in slots],
        "mode": mode,
        "n_games": n_games,
        "game_seed_base": game_seed_base,
        "environment_variant": environment_variant,
        "environment_class": environment_class.__name__,
        "game_duration_seconds": int(
            environment_class(seed=0).config.game_duration_seconds
        ),
        "llm_attempts": llm_attempts,
        "on_malformed": on_malformed,
    }


def _game_path(games_dir: Path, game_index: int) -> Path:
    return games_dir / f"game_{game_index:04d}.json"


def _load_cached_game(
    path: Path, identity: dict[str, Any]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("identity") != identity:
        raise ValueError(
            f"cached game metadata does not match this run: {path}; "
            "use --overwrite to replace it"
        )
    return payload["game"]


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


def _build_runners(
    slots: Sequence[Any],
    *,
    deterministic: bool,
    module_pool: crossplay.ModulePool | None,
    llm_runtime: ProviderRuntime,
    llm_attempts: int,
    on_malformed: str,
    llm_providers: dict[str, DecisionProvider] | None = None,
) -> dict[str, Any]:
    """One runner per distinct spec key (shared module / shared provider)."""
    runners: dict[str, Any] = {}
    for spec in slots:
        if spec.key in runners:
            continue
        if spec.kind == "scripted":
            runners[spec.key] = crossplay.ScriptedSeat(spec)
        elif spec.kind == "policy":
            if module_pool is None:
                raise ValueError("policy specs require a ModulePool")
            runners[spec.key] = crossplay.PolicySeat(
                module_pool.module_for(spec), deterministic
            )
        else:
            runners[spec.key] = LLMSeat(
                spec,
                llm_runtime,
                attempts=llm_attempts,
                on_malformed=on_malformed,
                provider=(llm_providers or {}).get(spec.key),
            )
    return runners


async def _play_one_game(
    slots: Sequence[Any],
    lineup: Sequence[Any],
    runners: dict[str, Any],
    *,
    game_index: int,
    game_seed: int,
    environment_class: type[FoodTradingEnv],
    print_progress: bool,
) -> dict[str, Any]:
    env = environment_class(seed=game_seed)
    obs_raw = env.reset()

    seats_by_key: dict[str, list[int]] = {}
    for seat, spec in enumerate(lineup):
        seats_by_key.setdefault(spec.key, []).append(seat)
    ordered_keys = sorted(seats_by_key, key=lambda key: seats_by_key[key][0])
    for key in ordered_keys:
        runners[key].begin_game(game_seed, seats_by_key[key])
    sync_keys = [k for k in ordered_keys if not isinstance(runners[k], LLMSeat)]
    llm_keys = [k for k in ordered_keys if isinstance(runners[k], LLMSeat)]

    trades_seen = 0
    buyer_trades = [0, 0, 0, 0]
    seller_trades = [0, 0, 0, 0]
    decisions: list[DecisionRecord] = []
    interval = env.config.consumption_interval_seconds

    while not env.is_done():
        actions: dict[int, int] = {}
        for key in sync_keys:
            actions.update(runners[key].act(env, obs_raw))
        # Per-step mask-legality invariant for trained/scripted seats
        # only — an LLM's illegal action is a measurement, not a bug.
        for seat, action in actions.items():
            if not env.legal_actions(seat)[action]:
                raise RuntimeError(
                    f"{lineup[seat].key} returned illegal action "
                    f"{action} in game {game_index}, seat {seat}"
                )
        tick_records: list[DecisionRecord] = []
        if llm_keys:
            results = await asyncio.gather(
                *(runners[key].act_async(env, obs_raw) for key in llm_keys)
            )
            for llm_actions, llm_records in results:
                actions.update(llm_actions)
                tick_records.extend(llm_records)

        obs_raw, _, _, info = env.step(actions)

        for record in tick_records:
            record.env_applied = info.actions_applied.get(record.seat)
        decisions.extend(tick_records)

        for trade in env.state.trades[trades_seen:]:
            buyer_trades[trade.buyer_idx] += 1
            seller_trades[trade.seller_idx] += 1
        trades_seen = len(env.state.trades)

        if print_progress and llm_keys and (
            info.tick == 1 or info.tick % interval == 0 or env.is_done()
        ):
            alive = env.state.alive_indices()
            print(
                f"  game {game_index}: tick {info.tick:3d}/"
                f"{env.config.game_duration_seconds} | alive={alive} | "
                f"trades={len(env.state.trades)}",
                flush=True,
            )

    result = env.get_result()
    winners = set(result["winner_idxs"])
    seat_records = []
    for seat, spec in enumerate(lineup):
        player = result["players"][seat]
        alive = player["status"] == "alive"
        survival_time = (
            player["died_at"]
            if player["died_at"] is not None
            else result["elapsed_seconds"]
        )
        seat_records.append(
            {
                "seat": seat,
                "spec": spec.key,
                "survived": bool(alive),
                "won": seat in winners,
                "terminal_cash": float(player["final_cash"]),
                "survival_time_ticks": float(survival_time),
                "trades_as_buyer": buyer_trades[seat],
                "trades_as_seller": seller_trades[seat],
            }
        )
    return {
        "game_index": game_index,
        "game_seed": game_seed,
        "lineup": [spec.key for spec in lineup],
        "end_reason": result["end_reason"],
        "n_trades": result["n_trades"],
        "seats": seat_records,
        "llm_decisions": [dataclasses.asdict(r) for r in decisions],
    }


async def play_arena_games(
    slots: Sequence[Any],
    n_games: int,
    game_seed_base: int,
    *,
    deterministic: bool,
    environment_class: type[FoodTradingEnv],
    identity: dict[str, Any],
    games_dir: Path,
    module_pool: crossplay.ModulePool | None = None,
    llm_runtime: ProviderRuntime | None = None,
    llm_attempts: int = 5,
    on_malformed: str = "wait",
    llm_providers: dict[str, DecisionProvider] | None = None,
    overwrite: bool = False,
    print_progress: bool = False,
) -> list[dict[str, Any]]:
    """Play (or resume) seat-balanced games; return one record per game.

    Every game ``g`` plays with seed ``game_seed_base + g`` and the
    cyclic rotation from the cross-play harness, and is cached as one
    JSON file, so an interrupted run resumes exactly where it stopped.

    Seat balance is exact only when ``n_games`` is a multiple of 4;
    other counts are allowed for debugging runs.
    """
    if n_games <= 0:
        raise ValueError("n_games must be positive")
    lineups = [
        tuple((seat - game) % 4 for seat in range(4))
        for game in range(n_games)
    ]
    games_dir.mkdir(parents=True, exist_ok=True)

    runners = _build_runners(
        slots,
        deterministic=deterministic,
        module_pool=module_pool,
        llm_runtime=llm_runtime or ProviderRuntime(),
        llm_attempts=llm_attempts,
        on_malformed=on_malformed,
        llm_providers=llm_providers,
    )

    problems: dict[str, list[str]] = {
        key: runner.preflight()
        for key, runner in runners.items()
        if isinstance(runner, LLMSeat)
    }
    failures = {key: msgs for key, msgs in problems.items() if msgs}
    if failures:
        rendered = "; ".join(
            f"{key}: {', '.join(msgs)}" for key, msgs in failures.items()
        )
        raise ProviderError("preflight failed: " + rendered)

    if any(spec.kind == "policy" for spec in slots):
        from env.rl.rllib_config import seed_everything

        seed_everything(game_seed_base)

    records: list[dict[str, Any]] = []
    for game_index, slot_of_seat in enumerate(lineups):
        path = _game_path(games_dir, game_index)
        cached = None if overwrite else _load_cached_game(path, identity)
        if cached is not None:
            records.append(cached)
            continue
        lineup = [slots[slot] for slot in slot_of_seat]
        game = await _play_one_game(
            slots,
            lineup,
            runners,
            game_index=game_index,
            game_seed=game_seed_base + game_index,
            environment_class=environment_class,
            print_progress=print_progress,
        )
        path.write_text(
            json.dumps({"identity": identity, "game": game}) + "\n"
        )
        records.append(game)
    return records


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize_games(
    records: Sequence[dict[str, Any]],
    slots: Sequence[Any],
    *,
    lineup_mode: str,
    mode: str,
    game_seed_base: int,
    environment_variant: str,
    environment_class: str,
    game_duration_seconds: int,
) -> list[dict[str, Any]]:
    """Cross-play summary rows plus LLM reliability/cost columns.

    The first block of columns matches ``checkpoint_policy_crossplay``
    exactly, so LLM cells can be joined into the same matrices as
    policy and scripted cells.
    """
    rows = crossplay.summarize_records(
        records,
        slots,
        lineup_mode=lineup_mode,
        mode=mode,
        game_seed_base=game_seed_base,
        environment_variant=environment_variant,
        environment_class=environment_class,
        game_duration_seconds=game_duration_seconds,
    )

    llm_stats: dict[str, dict[str, list[float]]] = {}
    for record in records:
        for decision in record.get("llm_decisions", ()):
            values = llm_stats.setdefault(
                decision["spec"],
                {
                    "latency": [],
                    "attempts": [],
                    "malformed": [],
                    "rejected": [],
                    "wait": [],
                    "input_tokens": [],
                    "output_tokens": [],
                    "cached_input_tokens": [],
                },
            )
            values["latency"].append(float(decision["latency_seconds"]))
            values["attempts"].append(float(decision["attempts"]))
            values["malformed"].append(float(decision["malformed"]))
            applied = decision.get("env_applied") or {}
            values["rejected"].append(
                float(
                    not decision["malformed"]
                    and (
                        decision.get("mask_illegal", False)
                        or applied.get("applied") is False
                    )
                )
            )
            values["wait"].append(
                float(decision["action_index"] == ACTION_WAIT)
            )
            if decision.get("input_tokens") is not None:
                values["input_tokens"].append(float(decision["input_tokens"]))
            if decision.get("output_tokens") is not None:
                values["output_tokens"].append(float(decision["output_tokens"]))
            if decision.get("cached_input_tokens") is not None:
                values["cached_input_tokens"].append(
                    float(decision["cached_input_tokens"])
                )

    for row in rows:
        stats = llm_stats.get(row["focal"])
        if stats is None:
            row.update({name: "" for name in LLM_SUMMARY_EXTRAS})
            continue
        n = len(stats["latency"])
        row.update(
            {
                "n_llm_decisions": n,
                "mean_decision_latency_seconds": float(np.mean(stats["latency"])),
                "mean_attempts_per_decision": float(np.mean(stats["attempts"])),
                "malformed_decision_percentage": 100.0 * float(np.mean(stats["malformed"])),
                "rejected_action_percentage": 100.0 * float(np.mean(stats["rejected"])),
                "wait_action_percentage": 100.0 * float(np.mean(stats["wait"])),
                "total_input_tokens": int(sum(stats["input_tokens"])),
                "total_output_tokens": int(sum(stats["output_tokens"])),
                "total_cached_input_tokens": int(
                    sum(stats["cached_input_tokens"])
                ),
            }
        )
    return rows
