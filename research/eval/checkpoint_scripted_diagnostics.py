"""Evaluate checkpoint progress against scripted opponent generations.

The learned policy always occupies seat 0.  Every checkpoint is evaluated on
the same game seeds, which makes progression comparisons less noisy.  Policy
actions are sampled stochastically by default; pass ``--deterministic`` for
argmax actions.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import ray
from ray.rllib.algorithms.algorithm import Algorithm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval._warnings import silence_ray_deprecations  # noqa: E402

silence_ray_deprecations()

from agents.scripted_v1 import ScriptedV1Agent  # noqa: E402
from agents.scripted_v2 import ScriptedV2Agent  # noqa: E402
from agents.scripted_v3 import ScriptedV3Agent  # noqa: E402
from agents.scripted_v4 import ScriptedV4Agent  # noqa: E402
from agents.scripted_v5 import ScriptedV5Agent  # noqa: E402
from eval.deep_eval import play_games, sample_head_diagnostics  # noqa: E402
from env.env import FoodTradingEnv  # noqa: E402

# Scripted opponent archetypes the learned seat can be scored against.
# v1 is the frozen port of the live game's bot and stays the default so
# existing result sets remain comparable.
OPPONENTS = {
    "v1": ScriptedV1Agent,
    "v2": ScriptedV2Agent,
    "v3": ScriptedV3Agent,
    "v4": ScriptedV4Agent,
    "v5": ScriptedV5Agent,
}
from env.rl.rllib_config import (  # noqa: E402
    SHARED_POLICY_ID,
    register_nash_hunger_env,
    seed_everything,
)


SUMMARY_FIELDS = (
    "algorithm",
    "seed",
    "checkpoint",
    "opponent",
    "n_games",
    "mode",
    "game_seed_base",
    "environment_variant",
    "environment_class",
    "game_duration_seconds",
    "mean_survival_time_ticks",
    "mean_terminal_cash",
    "mean_learned_trade_participations",
    "win_percentage",
    "survival_percentage",
    # ---- update-diagnostic extensions (v4/best-config campaigns) ----
    # Headroom: how often surviving converts to winning — the metric the
    # whole "learns to survive but not to win" story is anchored on.
    "win_given_survival_percentage",
    # Trade split: selling is the profit channel; a seat that only buys
    # can survive but cannot out-cash a scripted trader.
    "mean_trades_as_buyer",
    "mean_trades_as_seller",
    # Order-flow fingerprint of the learned policy.
    "mean_posted_bids",
    "mean_posted_asks",
    "mean_takes",
    "frac_wait",
    "frac_post",
    # Survival margin: fraction of surviving seats that end within one
    # consumption tick of death (0-1 units of their scarcest required
    # food). The survival features exist to push this down.
    "survivor_frac_min_required_0_to_1",
    # Per-head policy diagnostics — price noise is the entropy-annealing
    # target; P(post) tracks the entropy-bonus bias toward posting.
    "type_prob_post_mean",
    "type_entropy_mean",
    "price_given_post_entropy",
)


def _parse_csv_values(value: str, cast) -> list[Any]:
    parsed = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return parsed


def summarize_result(
    result: dict[str, Any],
    *,
    algorithm: str,
    seed: int,
    checkpoint: int,
    n_games: int,
    deterministic: bool,
    game_seed_base: int,
    opponent: str = "v1",
    head_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the five standard learned-seat statistics plus the
    update-diagnostic extensions (see SUMMARY_FIELDS for rationale)."""
    outcomes = result["terminal_metrics"]["outcome_counts"]
    buyer = float(result["learned_seat_completed_trades_as_buyer_mean"])
    seller = float(result["learned_seat_completed_trades_as_seller_mean"])
    winners = float(outcomes["winner"])
    survivors = winners + float(outcomes["alive_nonwinner"])
    type_freq = result.get("learned_seat_type_freq_mean", {})
    buckets = result.get("terminal_metrics", {}).get(
        "survivor_min_required_buckets", {}
    )
    head = head_diagnostics or {}
    return {
        "algorithm": algorithm,
        "seed": seed,
        "checkpoint": checkpoint,
        "opponent": opponent,
        "n_games": n_games,
        "mode": "deterministic" if deterministic else "stochastic",
        "game_seed_base": game_seed_base,
        "environment_variant": str(result.get("environment_variant", "default")),
        "environment_class": str(
            result.get("environment_class", FoodTradingEnv.__name__)
        ),
        "game_duration_seconds": int(
            result.get("game_duration_seconds", 180)
        ),
        # Surviving through the game contributes the full configured horizon.
        "mean_survival_time_ticks": float(
            result["learned_seat_mean_death_tick"]
        ),
        "mean_terminal_cash": float(result["learned_seat_mean_final_cash"]),
        # With one learned seat and no self-trades, buyer + seller counts each
        # completed trade involving the learned seat exactly once.
        "mean_learned_trade_participations": buyer + seller,
        "win_percentage": 100.0 * float(outcomes["winner"]) / n_games,
        "survival_percentage": 100.0
        * float(result["learned_seat_survival_rate"]),
        "win_given_survival_percentage": (
            100.0 * winners / survivors if survivors else 0.0
        ),
        "mean_trades_as_buyer": buyer,
        "mean_trades_as_seller": seller,
        "mean_posted_bids": float(
            result.get("learned_seat_posted_bids_mean", 0.0)
        ),
        "mean_posted_asks": float(
            result.get("learned_seat_posted_asks_mean", 0.0)
        ),
        "mean_takes": float(result.get("learned_seat_takes_mean", 0.0)),
        "frac_wait": float(type_freq.get("wait", 0.0)),
        "frac_post": float(type_freq.get("post", 0.0)),
        "survivor_frac_min_required_0_to_1": buckets.get("frac_0_to_1"),
        "type_prob_post_mean": head.get("type/mean_prob_post"),
        "type_entropy_mean": head.get("type/entropy_mean"),
        "price_given_post_entropy": head.get("price_given_post/entropy_mean"),
    }


def _expected_jobs(
    campaign_root: Path,
    algorithms: list[str],
    seeds: list[int],
    checkpoints: list[int],
) -> list[tuple[str, int, int, Path]]:
    return [
        (
            algorithm,
            seed,
            checkpoint,
            campaign_root
            / f"{algorithm}_seed{seed}_headmlp"
            / "checkpoints"
            / f"iter_{checkpoint:04d}",
        )
        for algorithm in algorithms
        for seed in seeds
        for checkpoint in checkpoints
    ]


def _validate_checkpoints(jobs: list[tuple[str, int, int, Path]]) -> None:
    missing = [
        path
        for _, _, _, path in jobs
        if not (path / "rllib_checkpoint.json").is_file()
    ]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"missing {len(missing)} required checkpoint(s):\n{preview}{suffix}"
        )


def _raw_result_path(
    output_dir: Path,
    algorithm: str,
    seed: int,
    checkpoint: int,
    mode: str,
    n_games: int,
    opponent: str = "v1",
) -> Path:
    # The opponent is part of the identity of a result: the same checkpoint
    # scored against v2 and against v3 are different measurements and must not
    # share a cache entry. v1 keeps its original filename so previously
    # generated result sets still resume.
    suffix = "" if opponent == "v1" else f"_{opponent}"
    return output_dir / "raw" / (
        f"{algorithm}_seed{seed}_iter{checkpoint:04d}_"
        f"1learned_3scripted{suffix}_{mode}_{n_games}.json"
    )


def _load_cached_summary(
    path: Path,
    *,
    checkpoint_path: Path,
    n_games: int,
    mode: str,
    game_seed_base: int,
    environment_variant: str,
    environment_class: str,
    game_duration_seconds: int,
    opponent: str = "v1",
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    expected = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "n_games": n_games,
        "mode": mode,
        "game_seed_base": game_seed_base,
        "environment_variant": environment_variant,
        "environment_class": environment_class,
        "game_duration_seconds": game_duration_seconds,
        "opponent": opponent,
    }
    actual = {
        # Results predating the multi-opponent flag were all played against v1.
        "opponent": report.get("opponent", "v1"),
        **report,
        # Results created before environment metadata was introduced used the
        # default 180-second environment.
        "environment_variant": report.get("environment_variant", "default"),
        "environment_class": report.get(
            "environment_class", FoodTradingEnv.__name__
        ),
        "game_duration_seconds": report.get("game_duration_seconds", 180),
    }
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"cached result metadata does not match this run: {path}; "
            "use --overwrite to replace it"
        )
    return report["summary"]


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _restore_algorithm(
    checkpoint_path: Path,
    *,
    attempts: int = 3,
    retry_delay_seconds: float = 0.5,
):
    """Restore an RLlib checkpoint, retrying only interrupted filesystem calls."""
    if attempts < 1:
        raise ValueError("restore attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return Algorithm.from_checkpoint(str(checkpoint_path))
        except InterruptedError:
            if attempt == attempts:
                raise
            delay = retry_delay_seconds * attempt
            print(
                f"checkpoint listing was interrupted; retrying "
                f"({attempt}/{attempts}) in {delay:.1f}s: {checkpoint_path}",
                flush=True,
            )
            time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run focused checkpoint diagnostics against scripted bots"
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--algorithms", default="ippo,mappo")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--checkpoint-start", type=int, default=500)
    parser.add_argument("--checkpoint-stop", type=int, default=1500)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--n-games", type=int, default=50)
    parser.add_argument("--game-seed-base", type=int, default=900_000)
    parser.add_argument(
        "--opponent",
        default="v1",
        choices=sorted(OPPONENTS),
        help=(
            "Scripted archetype filling seats 1-3: v1 (frozen live-game bot), "
            "v2 (inventory-skewed market maker), v3 (deadline-aware procurement)."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions instead of stochastic training-time sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace matching per-checkpoint JSON files instead of resuming.",
    )
    return parser.parse_args()


def main(
    *,
    environment_class: type[FoodTradingEnv] = FoodTradingEnv,
    environment_variant: str = "default",
    required_duration_seconds: int | None = None,
) -> int:
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
    environment_name = environment_class.__name__
    environment_probe = environment_class(seed=0)
    game_duration_seconds = int(environment_probe.config.game_duration_seconds)
    if (
        required_duration_seconds is not None
        and game_duration_seconds != required_duration_seconds
    ):
        raise ValueError(
            f"{environment_name} has a {game_duration_seconds}-second horizon; "
            f"this diagnostic requires {required_duration_seconds} seconds"
        )
    jobs = _expected_jobs(campaign_root, algorithms, seeds, checkpoints)

    # Fail before starting Ray or playing games if even one requested
    # checkpoint is absent. This prevents silently producing incomplete plots.
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
            raw_path = _raw_result_path(
                output_dir,
                algorithm,
                seed,
                checkpoint,
                mode,
                args.n_games,
                args.opponent,
            )
            if not args.overwrite:
                cached = _load_cached_summary(
                    raw_path,
                    checkpoint_path=checkpoint_path,
                    n_games=args.n_games,
                    mode=mode,
                    game_seed_base=args.game_seed_base,
                    environment_variant=environment_variant,
                    environment_class=environment_name,
                    game_duration_seconds=game_duration_seconds,
                    opponent=args.opponent,
                )
                if cached is not None:
                    rows.append(cached)
                    print(
                        f"[{job_number}/{len(jobs)}] cached "
                        f"{algorithm} seed{seed} iter{checkpoint}"
                    )
                    continue

            print(
                f"[{job_number}/{len(jobs)}] evaluating {algorithm} seed{seed} "
                f"iter{checkpoint} vs {args.opponent}: {args.n_games} games, {mode}",
                flush=True,
            )
            algorithm_obj = _restore_algorithm(checkpoint_path)
            try:
                module = algorithm_obj.get_module(SHARED_POLICY_ID)
                # Reset stochastic-policy RNG after checkpoint restoration so
                # every policy sees the same game seeds and action-sampling stream.
                seed_everything(args.game_seed_base)
                result = play_games(
                    module,
                    [True, False, False, False],
                    args.n_games,
                    args.game_seed_base,
                    deterministic=args.deterministic,
                    environment_class=environment_class,
                    opponent_factory=OPPONENTS[args.opponent],
                )
                result["environment_variant"] = environment_variant
                if result["game_duration_seconds"] != game_duration_seconds:
                    raise ValueError(
                        "rollout duration does not match the selected environment: "
                        f"{result['game_duration_seconds']} != {game_duration_seconds}"
                    )
                # Per-head policy diagnostics (type-head probabilities and,
                # crucially, the conditional price-head entropy) on real
                # masked observations from these games. This is the signal
                # that decides whether entropy annealing alone brings price
                # noise down or per-head normalization is needed — the
                # joint entropy scalar in the training logs cannot tell.
                head_diag_sample = result.pop("_head_diag_sample", None)
                head_diagnostics = (
                    sample_head_diagnostics(module, head_diag_sample)
                    if head_diag_sample
                    else None
                )
            finally:
                algorithm_obj.stop()

            summary = summarize_result(
                result,
                algorithm=algorithm,
                seed=seed,
                checkpoint=checkpoint,
                n_games=args.n_games,
                deterministic=args.deterministic,
                game_seed_base=args.game_seed_base,
                opponent=args.opponent,
                head_diagnostics=head_diagnostics,
            )
            report = {
                "checkpoint_path": str(checkpoint_path),
                "n_games": args.n_games,
                "mode": mode,
                "game_seed_base": args.game_seed_base,
                "opponent": args.opponent,
                "environment_variant": environment_variant,
                "environment_class": environment_name,
                "game_duration_seconds": game_duration_seconds,
                "learned_seat": 0,
                "scripted_seats": [1, 2, 3],
                "summary": summary,
                "head_diagnostics": head_diagnostics,
                "raw_result": result,
            }
            raw_path.write_text(json.dumps(report, indent=2) + "\n")
            rows.append(summary)
            _write_summary_csv(output_dir / "checkpoint_diagnostics.csv", rows)

    finally:
        ray.shutdown()

    rows.sort(key=lambda row: (row["algorithm"], row["seed"], row["checkpoint"]))
    _write_summary_csv(output_dir / "checkpoint_diagnostics.csv", rows)
    (output_dir / "checkpoint_diagnostics.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(f"wrote {len(rows)} checkpoint summaries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
