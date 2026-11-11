"""Driver: v3 final checkpoints vs the scripted ladder + mixed populations (Tier 0 items 3/6).

Jobs (default 180 s env, 100 games, stochastic):
  A. 1v3 vs ladder: each v3 policy (shared s1/s2, shared+ID s0/s1 @ iter 1100, plus
     shared_s1 @ 1400) vs 3x each scripted archetype v2-v5  -> 20 jobs.
     Gives the parameter-sharing-axis cross-play table at the paper's headline horizon.
  B. Mixed populations: each @1100 policy in a seat with one-of-each scripted
     (v2+v3+v4) and (v2+v4+v5)  -> 8 jobs.
     Tests sensitivity of conclusions to population composition (TAE angle).

Output: workshops/results/policy_crossplay/v3_vs_scripted_180s/ (accumulating matrix CSV).

Usage:  python workshops/eval/run_v3_scripted_evals.py [--n-games 100]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
HARNESS = ROOT / "workshops" / "eval" / "checkpoint_policy_crossplay.py"
OUT = ROOT / "workshops" / "results" / "policy_crossplay" / "v3_vs_scripted_180s"

V3ROOT = RESEARCH / "eval/results/v3_shared_vs_independent"
POLICIES = {
    "shared_s1": V3ROOT / "ippo_shared_seed1/checkpoints/iter_1100",
    "shared_s2": V3ROOT / "ippo_shared_seed2/checkpoints/iter_1100",
    "sharedid_s0": V3ROOT / "ippo_shared_id_seed0/checkpoints/iter_1100",
    "sharedid_s1": V3ROOT / "ippo_shared_id_seed1/checkpoints/iter_1100",
    "shared_s1_1400": V3ROOT / "ippo_shared_seed1/checkpoints/iter_1400",
}
LADDER = ["v2", "v3", "v4", "v5"]
MIXES = [("mix_v2v3v4", ["v2", "v3", "v4"]), ("mix_v2v4v5", ["v2", "v4", "v5"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=100)
    args = ap.parse_args()

    jobs: list[list[str]] = []
    for label, path in POLICIES.items():
        for opp in LADDER:
            jobs.append(["--mode", "1v3", "--focal", f"policy:{label}={path}",
                         "--opponent", f"scripted:{opp}"])
    for label, path in POLICIES.items():
        if label == "shared_s1_1400":
            continue
        for _, bots in MIXES:
            job = ["--mode", "mixed", "--seat", f"policy:{label}={path}"]
            for b in bots:
                job += ["--seat", f"scripted:{b}"]
            jobs.append(job)

    print(f"DRIVER START {len(jobs)} jobs, n_games={args.n_games}", flush=True)
    for i, job in enumerate(jobs, 1):
        t0 = time.time()
        cmd = [sys.executable, str(HARNESS), "--n-games", str(args.n_games),
               "--environment-variant", "default", "--output", str(OUT)] + job
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env={"PYTHONPATH": str(RESEARCH), "PATH": "/usr/bin:/bin:/usr/local/bin",
                                "HOME": str(Path.home())})
        tag = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
        print(f"[{i}/{len(jobs)}] {tag} {time.time()-t0:.0f}s :: " + " ".join(job[:5]), flush=True)
        if r.returncode != 0:
            print("  " + "\n  ".join((r.stderr or r.stdout).strip().splitlines()[-8:]), flush=True)
    print("DRIVER DONE", flush=True)


if __name__ == "__main__":
    main()
