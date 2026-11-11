"""Evaluate checkpoint survival in four-seat learned self-play.

Each checkpoint controls all four seats for the same seeded games.  Only the
two requested aggregates are retained: learned-seat survival percentage and
average death tick.  A seat that survives contributes the full environment
horizon to the average death tick, matching the existing self-play diagnostic
convention.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import ray

from env.config import DEFAULT_CONFIG
from env.env import FoodTradingEnv
from env.rl.rllib_config import (
    SHARED_POLICY_ID,
    register_nash_hunger_env,
    seed_everything,
)
from env.small_env import SCALED_ENVIRONMENTS
from eval.checkpoint_scripted_diagnostics import (
    _expected_jobs,
    _parse_csv_values,
    _restore_algorithm,
    _validate_checkpoints,
)
from eval.deep_eval import play_games, sample_head_diagnostics


# The scaled variants plus the full-length default environment, so the same
# CLI can sweep every horizon a campaign trained (including default_180).
DIAGNOSTIC_ENVIRONMENTS = {
    **SCALED_ENVIRONMENTS,
    "default": (DEFAULT_CONFIG, FoodTradingEnv),
}

SUMMARY_FIELDS = (
    "algorithm",
    "seed",
    "checkpoint",
    "n_games",
    "mode",
    "game_seed_base",
    "environment_variant",
    "environment_class",
    "game_duration_seconds",
    "learned_seats",
    "average_death_tick",
    "survival_percentage",
    # Head-level policy diagnostics (see deep_eval.sample_head_diagnostics).
    # `price_given_post_entropy` is the signal the entropy-annealing /
    # per-head-normalization decision hinges on; the full payload is stored
    # in each raw report under "head_diagnostics".
    "type_prob_post_mean",
    "type_entropy_mean",
    "price_given_post_entropy",
)


def summarize_result(
    result: dict[str, Any],
    *,
    algorithm: str,
    seed: int,
    checkpoint: int,
    n_games: int,
    deterministic: bool,
    game_seed_base: int,
    environment_variant: str,
    head_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    head = head_diagnostics or {}
    return {
        "algorithm": algorithm,
        "seed": seed,
        "checkpoint": checkpoint,
        "n_games": n_games,
        "mode": "deterministic" if deterministic else "stochastic",
        "game_seed_base": game_seed_base,
        "environment_variant": environment_variant,
        "environment_class": str(result["environment_class"]),
        "game_duration_seconds": int(result["game_duration_seconds"]),
        "learned_seats": int(result["n_learned_seats"]),
        "average_death_tick": float(result["learned_seat_mean_death_tick"]),
        "survival_percentage": 100.0
        * float(result["learned_seat_survival_rate"]),
        "type_prob_post_mean": head.get("type/mean_prob_post"),
        "type_entropy_mean": head.get("type/entropy_mean"),
        "price_given_post_entropy": head.get("price_given_post/entropy_mean"),
    }


def _raw_path(
    output_dir: Path,
    algorithm: str,
    seed: int,
    checkpoint: int,
    mode: str,
    n_games: int,
) -> Path:
    return output_dir / "raw" / (
        f"{algorithm}_seed{seed}_iter{checkpoint:04d}_"
        f"4learned_selfplay_survival_{mode}_{n_games}.json"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--environment-variant",
        choices=tuple(DIAGNOSTIC_ENVIRONMENTS),
        required=True,
    )
    parser.add_argument("--algorithms", default="ippo,mappo")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--checkpoint-start", type=int, default=600)
    parser.add_argument("--checkpoint-stop", type=int, default=1200)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--n-games", type=int, default=50)
    parser.add_argument("--game-seed-base", type=int, default=970_000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_games <= 0:
        raise ValueError("--n-games must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.checkpoint_stop < args.checkpoint_start:
        raise ValueError("checkpoint stop must be at least checkpoint start")

    algorithms = _parse_csv_values(args.algorithms, str)
    seeds = _parse_csv_values(args.seeds, int)
    checkpoints = list(
        range(
            args.checkpoint_start,
            args.checkpoint_stop + 1,
            args.checkpoint_every,
        )
    )
    campaign_root = args.campaign_root.resolve()
    output_dir = args.output_dir.resolve()
    mode = "deterministic" if args.deterministic else "stochastic"
    config, environment_class = DIAGNOSTIC_ENVIRONMENTS[args.environment_variant]
    jobs = _expected_jobs(campaign_root, algorithms, seeds, checkpoints)
    _validate_checkpoints(jobs)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    ray.init(
        ignore_reinit_error=True,
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
    )
    register_nash_hunger_env()
    try:
        for job_number, (algorithm, seed, checkpoint, checkpoint_path) in enumerate(
            jobs, start=1
        ):
            raw_path = _raw_path(
                output_dir,
                algorithm,
                seed,
                checkpoint,
                mode,
                args.n_games,
            )
            expected = {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "n_games": args.n_games,
                "mode": mode,
                "game_seed_base": args.game_seed_base,
                "environment_variant": args.environment_variant,
                "environment_class": environment_class.__name__,
                "game_duration_seconds": int(config.game_duration_seconds),
                "learned_seats": [0, 1, 2, 3],
            }
            if raw_path.is_file() and not args.overwrite:
                report = json.loads(raw_path.read_text())
                if any(report.get(key) != value for key, value in expected.items()):
                    raise ValueError(
                        f"cached result metadata mismatch: {raw_path}; "
                        "use --overwrite to replace it"
                    )
                rows.append(report["summary"])
                print(
                    f"[{job_number}/{len(jobs)}] cached "
                    f"{algorithm} seed{seed} iter{checkpoint}",
                    flush=True,
                )
                continue

            print(
                f"[{job_number}/{len(jobs)}] evaluating {args.environment_variant} "
                f"self-play {algorithm} seed{seed} iter{checkpoint}: "
                f"{args.n_games} games, {mode}",
                flush=True,
            )
            algorithm_obj = _restore_algorithm(checkpoint_path)
            try:
                module = algorithm_obj.get_module(SHARED_POLICY_ID)
                seed_everything(args.game_seed_base)
                result = play_games(
                    module,
                    [True, True, True, True],
                    args.n_games,
                    args.game_seed_base,
                    deterministic=args.deterministic,
                    environment_class=environment_class,
                )
                # Per-head policy diagnostics on this checkpoint's own
                # self-play state distribution — the price-head entropy
                # trajectory that the annealing/normalization decision needs.
                head_diag_sample = result.pop("_head_diag_sample", None)
                head_diagnostics = (
                    sample_head_diagnostics(module, head_diag_sample)
                    if head_diag_sample
                    else None
                )
            finally:
                algorithm_obj.stop()

            if result["game_duration_seconds"] != config.game_duration_seconds:
                raise ValueError(
                    "self-play rollout used the wrong environment duration: "
                    f"{result['game_duration_seconds']} instead of "
                    f"{config.game_duration_seconds}"
                )
            summary = summarize_result(
                result,
                algorithm=algorithm,
                seed=seed,
                checkpoint=checkpoint,
                n_games=args.n_games,
                deterministic=args.deterministic,
                game_seed_base=args.game_seed_base,
                environment_variant=args.environment_variant,
                head_diagnostics=head_diagnostics,
            )
            report = {
                **expected,
                "summary": summary,
                "head_diagnostics": head_diagnostics,
            }
            raw_path.write_text(json.dumps(report, indent=2) + "\n")
            rows.append(summary)
            _write_csv(output_dir / "selfplay_survival_diagnostics.csv", rows)
    finally:
        ray.shutdown()

    rows.sort(key=lambda row: (row["algorithm"], row["seed"], row["checkpoint"]))
    _write_csv(output_dir / "selfplay_survival_diagnostics.csv", rows)
    (output_dir / "selfplay_survival_diagnostics.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(f"wrote {len(rows)} self-play survival summaries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
