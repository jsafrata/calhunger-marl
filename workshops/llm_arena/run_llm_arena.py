"""CLI for the CalHunger LLM arena.

Seat-balanced, seed-paired games mixing LLM seats, trained RLlib
checkpoints, and the scripted heuristic ladder in one lineup.

Examples
--------
# Smoke test end to end without spending tokens (mock LLM vs the ladder):
python workshops/llm_arena/run_llm_arena.py \
    --mode 1v3 --focal "llm:mm=mock/market-maker" --opponent scripted:v1 \
    --environment-variant small --n-games 4 --name smoke

# Claude Haiku vs three JIT bots on the 30-second variant:
python workshops/llm_arena/run_llm_arena.py \
    --mode 1v3 \
    --focal "llm:haiku=anthropic_api/claude-haiku-4-5-20251001" \
    --opponent scripted:v3 \
    --environment-variant small --n-games 20 --name haiku_vs_v3

# An LLM seated with a trained policy population:
python workshops/llm_arena/run_llm_arena.py \
    --mode 1v3 \
    --focal "llm:haiku=anthropic_api/claude-haiku-4-5-20251001" \
    --opponent "policy:ippo_s0=/path/to/checkpoints/iter_1200" \
    --environment-variant small --n-games 20 --name haiku_vs_ippo

# Four different models against each other (rotated for seat balance):
python workshops/llm_arena/run_llm_arena.py \
    --mode mixed \
    --seat "llm:gpt=openai_api/gpt-5.2" \
    --seat "llm:claude=anthropic_api/claude-haiku-4-5-20251001" \
    --seat "llm:gemini=gemini_api/gemini-3.5-flash-lite" \
    --seat "llm:deepseek=deepseek_api/deepseek-v4-flash" \
    --environment-variant small --n-games 8 --name four_llms

API keys come from the environment or ``workshops/llm_arena/.env``
(KEY=VALUE lines; never committed).  ``--preflight`` validates keys and
executables without spending tokens; ``--probe N`` requests exactly one
live decision for lineup slot N.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_arena._paths import ARENA_ROOT, WORKSHOPS_ROOT, bootstrap  # noqa: E402

bootstrap()

import checkpoint_policy_crossplay as crossplay  # noqa: E402

from llm_arena.arena import (  # noqa: E402
    ENVIRONMENT_SPECS,
    SUMMARY_FIELDS,
    LLMSeat,
    LLMSpec,
    job_identity,
    parse_arena_spec,
    play_arena_games,
    summarize_games,
)
from llm_arena.providers import ProviderRuntime, load_env_file  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seat-balanced games mixing LLM, policy, and scripted seats"
    )
    parser.add_argument("--mode", dest="lineup_mode", default="1v3",
                        choices=("1v3", "mixed"))
    parser.add_argument(
        "--focal",
        help="1v3 focal slot spec (llm:LABEL=PROVIDER/MODEL, "
        "policy:LABEL=PATH[@MODULE], or scripted:vN)",
    )
    parser.add_argument(
        "--opponent",
        help="1v3 opponent spec, occupies the other three seats",
    )
    parser.add_argument(
        "--seat",
        action="append",
        dest="seats",
        help="mixed-mode per-slot spec; pass exactly four times",
    )
    parser.add_argument("--n-games", type=int, default=8,
                        help="multiple of 4 (seat balance)")
    parser.add_argument("--game-seed-base", type=int, default=900_000)
    parser.add_argument(
        "--environment-variant",
        default="small",
        choices=sorted(ENVIRONMENT_SPECS),
        help="default 'small' (30 s): 30 LLM decisions per seat per game",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="argmax actions for policy seats (LLM sampling is provider-side)",
    )
    parser.add_argument("--llm-attempts", type=int, default=5,
                        help="parse attempts per decision (byte-identical retry)")
    parser.add_argument("--llm-timeout", type=float, default=120.0,
                        help="per-request timeout in seconds")
    parser.add_argument("--http-attempts", type=int, default=3,
                        help="transient-HTTP retries per request (429/5xx)")
    parser.add_argument(
        "--on-malformed",
        default="wait",
        choices=("wait", "abort"),
        help="after exhausted parse attempts: score a WAIT and record the "
        "violation (default), or abort the game (preliminary-arena behavior)",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        help="optional TOML with [providers.LABEL] api_key_env/base_url/"
        "executable overrides",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="replace every LLM seat's provider with the deterministic mock "
        "(no tokens spent; identity records provider 'mock')",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="check keys/executables for all LLM seats and exit",
    )
    parser.add_argument(
        "--probe",
        type=int,
        metavar="SLOT",
        help="request one live decision for lineup slot SLOT and exit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace cached games instead of resuming from them",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress per-tick progress"
    )
    parser.add_argument("--name", default="run")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output dir (default workshops/results/llm_arena/<name>)",
    )
    return parser.parse_args(argv)


def _load_llm_overrides(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    providers = raw.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError(f"{path}: [providers] must be a table of labels")
    return {
        str(label): {str(k): str(v) for k, v in value.items()}
        for label, value in providers.items()
    }


def _mock_spec(spec: Any) -> Any:
    if isinstance(spec, LLMSpec):
        return parse_arena_spec(f"llm:{spec.label}=mock/{spec.config.model}")
    return spec


def _resolve_slots(args: argparse.Namespace) -> list[Any]:
    overrides = _load_llm_overrides(args.llm_config)
    if args.lineup_mode == "1v3":
        if not args.focal or not args.opponent:
            raise ValueError("1v3 mode requires --focal and --opponent")
        focal = parse_arena_spec(args.focal, overrides)
        opponent = parse_arena_spec(args.opponent, overrides)
        slots = [focal, opponent, opponent, opponent]
    else:
        if not args.seats or len(args.seats) != 4:
            raise ValueError("mixed mode requires exactly four --seat specs")
        slots = [parse_arena_spec(text, overrides) for text in args.seats]
    if args.mock_llm:
        slots = [_mock_spec(spec) for spec in slots]
    return slots


def _probe(
    slots: list[Any], slot: int, runtime: ProviderRuntime, args: argparse.Namespace
) -> int:
    spec = slots[slot]
    if not isinstance(spec, LLMSpec):
        raise ValueError(f"slot {slot} ({spec.key}) is not an LLM seat")
    environment_class, _ = ENVIRONMENT_SPECS[args.environment_variant]
    env = environment_class(seed=args.game_seed_base)
    obs = env.reset()
    seat_runner = LLMSeat(
        spec, runtime, attempts=args.llm_attempts, on_malformed="abort"
    )
    problems = seat_runner.preflight()
    if problems:
        raise SystemExit(f"preflight failed for {spec.key}: {'; '.join(problems)}")
    record = asyncio.run(seat_runner._one_decision(env, obs[0], 0))
    print(json.dumps(dataclasses.asdict(record), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file(ARENA_ROOT / ".env")
    slots = _resolve_slots(args)
    runtime = ProviderRuntime(
        decision_timeout_seconds=args.llm_timeout,
        http_attempts=args.http_attempts,
        project_root=ARENA_ROOT,
    )

    if args.probe is not None:
        return _probe(slots, args.probe, runtime, args)

    if args.preflight:
        failures = 0
        for spec in {spec.key: spec for spec in slots}.values():
            if not isinstance(spec, LLMSpec):
                print(f"  {spec.key}: not an LLM seat, nothing to check")
                continue
            seat_runner = LLMSeat(
                spec, runtime, attempts=args.llm_attempts, on_malformed="wait"
            )
            problems = seat_runner.preflight()
            status = "ok" if not problems else "; ".join(problems)
            print(f"  {spec.key} ({spec.config.provider}/{spec.config.model}): {status}")
            failures += bool(problems)
        return 1 if failures else 0

    environment_class, expected_duration = ENVIRONMENT_SPECS[
        args.environment_variant
    ]
    game_duration_seconds = int(
        environment_class(seed=0).config.game_duration_seconds
    )
    if game_duration_seconds != expected_duration:
        raise ValueError(
            f"{environment_class.__name__} advertises {game_duration_seconds}s; "
            f"registry expected {expected_duration}s"
        )

    output_dir = (
        args.output
        if args.output is not None
        else WORKSHOPS_ROOT / "results" / "llm_arena" / args.name
    ).resolve()
    games_dir = output_dir / "games"
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "deterministic" if args.deterministic else "stochastic"
    identity = job_identity(
        slots,
        lineup_mode=args.lineup_mode,
        mode=mode,
        n_games=args.n_games,
        game_seed_base=args.game_seed_base,
        environment_variant=args.environment_variant,
        environment_class=environment_class,
        llm_attempts=args.llm_attempts,
        on_malformed=args.on_malformed,
    )
    (output_dir / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")

    lineup_text = (
        f"{slots[0].key} vs 3x {slots[1].key}"
        if args.lineup_mode == "1v3"
        else " | ".join(spec.key for spec in slots)
    )
    print(
        f"playing {args.n_games} games ({mode}, {args.environment_variant}, "
        f"seed base {args.game_seed_base}): {lineup_text}",
        flush=True,
    )

    needs_pool = any(spec.kind == "policy" for spec in slots)
    pool = crossplay.ModulePool() if needs_pool else None
    started = time.time()
    try:
        records = asyncio.run(
            play_arena_games(
                slots,
                args.n_games,
                args.game_seed_base,
                deterministic=args.deterministic,
                environment_class=environment_class,
                identity=identity,
                games_dir=games_dir,
                module_pool=pool,
                llm_runtime=runtime,
                llm_attempts=args.llm_attempts,
                on_malformed=args.on_malformed,
                overwrite=args.overwrite,
                print_progress=not args.quiet,
            )
        )
    finally:
        if pool is not None:
            pool.close()
            crossplay._shutdown_ray()
    elapsed = time.time() - started

    rows = summarize_games(
        records,
        slots,
        lineup_mode=args.lineup_mode,
        mode=mode,
        game_seed_base=args.game_seed_base,
        environment_variant=args.environment_variant,
        environment_class=environment_class.__name__,
        game_duration_seconds=game_duration_seconds,
    )
    csv_path = output_dir / "arena_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "arena_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    for row in rows:
        extra = ""
        if row["n_llm_decisions"] != "":
            extra = (
                f"  malformed={row['malformed_decision_percentage']:.1f}%"
                f"  rejected={row['rejected_action_percentage']:.1f}%"
                f"  latency={row['mean_decision_latency_seconds']:.2f}s"
            )
        print(
            f"  {row['focal']:>24} vs {row['opponent']:<24} "
            f"survival={row['survival_percentage']:5.1f}%  "
            f"win={row['win_percentage']:5.1f}%  "
            f"cash={row['mean_terminal_cash']:8.2f}{extra}"
        )
    print(
        f"{args.n_games} games in {elapsed:.1f}s; summary in {csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
