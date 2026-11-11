"""Driver: run the full local policy-vs-policy cross-play matrices sequentially.

Order (fast -> slow):
  1. small 30s matrix   — small_2seed_partial @ iter_1000, 4 policies,
                          all 12 ordered 1v3 pairs + 4 self, stochastic + deterministic.
  2. v3 180s matrix     — v3_shared_vs_independent @ iter_1100, 4 complete policies
                          (ippo shared s1/s2, shared_id s0/s1), same structure; plus the
                          rich-and-dead endpoint ippo_shared_seed1 @ iter_1400 as an extra
                          focal row (vs each peer + its own self-play) to test whether the
                          cross-play collapse seen vs scripted also holds vs peers.
  3. hpc_v2 180s matrix — hpc_v2_5seed @ iter_1500, 10 policies, all 90 ordered pairs +
                          10 self, stochastic only (deterministic: self rows only).

Each job is a subprocess invocation of checkpoint_policy_crossplay.py (fresh Ray per job;
cache-resumable, so re-running this driver skips completed jobs). Progress lines go to
stdout (the monitor tails them).

Usage:  PYTHONPATH=<research> python workshops/eval/run_crossplay_matrices.py [--n-games 100]
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # Hunger_games_2/
RESEARCH = ROOT / "research"
HARNESS = ROOT / "workshops" / "eval" / "checkpoint_policy_crossplay.py"
OUT = ROOT / "workshops" / "results" / "policy_crossplay"

SMALL = {
    f"{algo}_s{seed}": RESEARCH
    / f"eval/results/small_2seed_partial/{algo}_seed{seed}_headmlp/checkpoints/iter_1000"
    for algo in ("ippo", "mappo")
    for seed in (0, 1)
}
V3 = {
    "shared_s1": RESEARCH
    / "eval/results/v3_shared_vs_independent/ippo_shared_seed1/checkpoints/iter_1100",
    "shared_s2": RESEARCH
    / "eval/results/v3_shared_vs_independent/ippo_shared_seed2/checkpoints/iter_1100",
    "sharedid_s0": RESEARCH
    / "eval/results/v3_shared_vs_independent/ippo_shared_id_seed0/checkpoints/iter_1100",
    "sharedid_s1": RESEARCH
    / "eval/results/v3_shared_vs_independent/ippo_shared_id_seed1/checkpoints/iter_1100",
}
V3_1400 = {
    "shared_s1_1400": RESEARCH
    / "eval/results/v3_shared_vs_independent/ippo_shared_seed1/checkpoints/iter_1400",
}
HPC_V2 = {
    f"{algo}_s{seed}": RESEARCH
    / f"eval/results/hpc_v2_5seed_ippo_mappo_1500/{algo}_seed{seed}_headmlp/checkpoints/iter_1500"
    for algo in ("ippo", "mappo")
    for seed in range(5)
}


def spec(label: str, pols: dict) -> str:
    return f"policy:{label}={pols[label]}"


def jobs_for_matrix(pols: dict, env: str, out: Path, modes: tuple[str, ...],
                    det_self_only: bool = False) -> list[list[str]]:
    jobs = []
    labels = sorted(pols)
    for mode in modes:
        det = ["--deterministic"] if mode == "det" else []
        for a in labels:  # self-play diagonal (mandatory: separates weakness from incompatibility)
            jobs.append(["--mode", "mixed"] + [x for _ in range(4) for x in ("--seat", spec(a, pols))]
                        + det + ["--output", str(out)])
        if mode == "det" and det_self_only:
            continue
        for a, b in itertools.permutations(labels, 2):
            jobs.append(["--mode", "1v3", "--focal", spec(a, pols), "--opponent", spec(b, pols)]
                        + det + ["--output", str(out)])
    for j in jobs:
        j += ["--environment-variant", env]
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=100)
    args = ap.parse_args()

    phases: list[tuple[str, list[list[str]]]] = []
    phases.append(("small30", jobs_for_matrix(SMALL, "small", OUT / "small30_iter1000",
                                              modes=("stoch", "det"))))
    v3_jobs = jobs_for_matrix(V3, "default", OUT / "v3_180s_iter1100", modes=("stoch",))
    both = {**V3, **V3_1400}
    v3_jobs.append(["--mode", "mixed"] + [x for _ in range(4) for x in ("--seat", spec("shared_s1_1400", both))]
                   + ["--output", str(OUT / "v3_180s_iter1100"), "--environment-variant", "default"])
    for peer in sorted(V3):
        v3_jobs.append(["--mode", "1v3", "--focal", spec("shared_s1_1400", both),
                        "--opponent", spec(peer, both),
                        "--output", str(OUT / "v3_180s_iter1100"), "--environment-variant", "default"])
    phases.append(("v3_180s", v3_jobs))
    phases.append(("hpc_v2_180s", jobs_for_matrix(HPC_V2, "default", OUT / "hpc_v2_180s_iter1500",
                                                  modes=("stoch", "det"), det_self_only=True)))

    total = sum(len(j) for _, j in phases)
    done = 0
    print(f"DRIVER START {total} jobs, n_games={args.n_games}", flush=True)
    for name, jobs in phases:
        print(f"PHASE {name}: {len(jobs)} jobs", flush=True)
        for job in jobs:
            done += 1
            t0 = time.time()
            cmd = [sys.executable, str(HARNESS), "--n-games", str(args.n_games)] + job
            r = subprocess.run(cmd, capture_output=True, text=True,
                               env={"PYTHONPATH": str(RESEARCH), "PATH": "/usr/bin:/bin:/usr/local/bin",
                                    "HOME": str(Path.home())})
            tag = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
            desc = " ".join(x for x in job if not x.startswith("--") and "=" not in x)[:60]
            print(f"[{done}/{total}] {name} {tag} {time.time()-t0:.0f}s :: "
                  + " ".join(job[:5]), flush=True)
            if r.returncode != 0:
                tail = (r.stderr or r.stdout).strip().splitlines()[-8:]
                print("  " + "\n  ".join(tail), flush=True)
    print("DRIVER DONE", flush=True)


if __name__ == "__main__":
    main()
