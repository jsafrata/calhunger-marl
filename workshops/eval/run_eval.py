"""Systematic evaluation of Scripted-v1 across the 7 configurations.

Usage:
    python -m eval.run_eval --n_games 100  --tag smoke
    python -m eval.run_eval --n_games 1000 --tag main

Writes one JSONL file per configuration under
    eval/results/<tag>/<config_name>.jsonl

Each line is a full game log (see `game_runner.run_game`). Games are
"paired" across configurations: the k-th game of every configuration
uses the same `game_seed` (drives producers + initial inventory) and a
per-config-derived `seat_permutation_seed` (drives which seat gets which
cadence). This lets us compare configurations under identical
environmental setups.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from eval.configurations import CONFIGURATIONS, seat_permutation_seed  # noqa: E402
from eval.game_runner import run_game  # noqa: E402


def _json_default(obj):
    """JSON serializer that unwraps numpy scalars/arrays — the harness
    passes them through in a few places (seat permutations, sampled
    intervals) that would otherwise blow up json.dumps."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"unserializable object: {type(obj).__name__}")


def _dumps(o) -> str:
    return json.dumps(o, default=_json_default)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Scripted-v1 evaluation.")
    p.add_argument("--n_games", type=int, required=True)
    p.add_argument("--tag", type=str, required=True, help="output subdir name")
    p.add_argument("--seed_base", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--configurations",
        type=str,
        default="",
        help="comma-separated subset of config names; empty = all",
    )
    return p.parse_args()


def _run_one(job: Dict) -> Dict:
    """Worker fn — pickled across processes. Never raises; on failure it
    returns a stub log so the batch keeps moving and we can spot the
    offending seed in postprocessing."""
    try:
        log = run_game(
            game_seed=job["game_seed"],
            cadences=job["cadences"],
            seat_permutation_seed=job["seat_permutation_seed"],
        )
    except Exception as exc:  # noqa: BLE001 — deliberately broad
        import traceback
        log = {
            "game_seed": job["game_seed"],
            "seat_permutation_seed": job["seat_permutation_seed"],
            "cadences_configuration_order": list(job["cadences"]),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    log["_job_config_name"] = job["config_name"]
    return log


def main(args: argparse.Namespace) -> int:
    out_root = Path(__file__).resolve().parent / "results" / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    if args.configurations:
        wanted = {c.strip() for c in args.configurations.split(",")}
        configs = [c for c in CONFIGURATIONS if c.name in wanted]
        missing = wanted - {c.name for c in configs}
        if missing:
            raise SystemExit(f"unknown configuration(s): {sorted(missing)}")
    else:
        configs = list(CONFIGURATIONS)

    print(
        f"[eval] running {args.n_games} games × {len(configs)} configs = "
        f"{args.n_games * len(configs)} total games; writing to {out_root}"
    )

    # Build the whole job list at once so we can fan out to a pool.
    jobs: List[Dict] = []
    for cfg_idx, cfg in enumerate(configs):
        for k in range(args.n_games):
            shared_seed = args.seed_base + k
            perm_seed = seat_permutation_seed(shared_seed, cfg_idx)
            jobs.append(
                {
                    "config_name": cfg.name,
                    "cadences": cfg.cadences,
                    "game_seed": shared_seed,
                    "seat_permutation_seed": perm_seed,
                }
            )

    # Open output files (one per config).
    out_paths: Dict[str, Path] = {c.name: out_root / f"{c.name}.jsonl" for c in configs}
    for p in out_paths.values():
        p.write_text("")  # truncate

    t0 = time.time()
    n_done = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one, j): j for j in jobs}
            handles = {name: open(out_paths[name], "a") for name in out_paths}
            try:
                for fut in as_completed(futures):
                    log = fut.result()
                    name = log.pop("_job_config_name")
                    handles[name].write(_dumps(log) + "\n")
                    n_done += 1
                    if n_done % max(1, len(jobs) // 20) == 0:
                        elapsed = time.time() - t0
                        print(
                            f"[eval] {n_done}/{len(jobs)} games "
                            f"({100*n_done/len(jobs):.1f}%) in {elapsed:.1f}s"
                        )
            finally:
                for h in handles.values():
                    h.close()
    else:
        handles = {name: open(out_paths[name], "a") for name in out_paths}
        try:
            for job in jobs:
                log = _run_one(job)
                name = log.pop("_job_config_name")
                handles[name].write(_dumps(log) + "\n")
                n_done += 1
                if n_done % max(1, len(jobs) // 20) == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[eval] {n_done}/{len(jobs)} games "
                        f"({100*n_done/len(jobs):.1f}%) in {elapsed:.1f}s"
                    )
        finally:
            for h in handles.values():
                h.close()

    elapsed = time.time() - t0
    print(
        f"[eval] done: {n_done} games in {elapsed:.1f}s "
        f"({elapsed/n_done*1000:.1f} ms/game) → {out_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
