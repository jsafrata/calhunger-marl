"""E22: evaluation-budget prescription for the peer cross-play protocol (TAE).

How many games per cell does the protocol need to DECIDE each healthy
pairwise verdict? From the paired-seed bootstrap at the full merged budget
(n=1000 games/cell), estimate the SD of each pairwise score difference;
SD scales as 1/sqrt(n) (verified empirically against the n=100 bootstrap),
so the budget needed for a pair with true gap |delta| is

  n_alpha  = n0 * (z_alpha/2 * sd0 / delta)^2          (50% power: the
             observed gap sits exactly at the decision threshold)
  n_power  = n0 * ((z_alpha/2 + z_beta) * sd0 / delta)^2   (80% power)

with n0=1000, alpha=0.05 (z=1.96), beta=0.20 (z=0.84). The ranking is
certified when every ADJACENT pair in the reference ordering is decided,
so the certification budget is the max over adjacent pairs.

Usage (repo outer root):
    python3 workshops/eval/analyze_evaluation_budget.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from crossplay_raw_common import (
    HEALTHY,
    LABELS,
    RESULTS_ROOT,
    load_peer_1v3,
    stack_values,
)

OUT_DIR = RESULTS_ROOT / "evaluation_budget"
B = 2000
RNG_SEED = 20260829
Z_ALPHA = 1.96
Z_BETA = 0.84
N_CAP = 10_000_000  # report anything above this as effectively undecidable


def paired_scores(surv: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(P, C, G) survival + shared idx (B, n) -> (B, P) macro scores."""
    b = idx.shape[0]
    p, c, g = surv.shape
    cell_means = np.empty((b, p, c))
    for ci in range(c):
        for pi in range(p):
            cell_means[:, pi, ci] = surv[pi, ci][idx].mean(axis=1)
    return cell_means.mean(axis=2)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    peer = stack_values(load_peer_1v3(extended=True), LABELS)
    surv = peer[..., 0]  # (10, 9, 1000)
    n0 = surv.shape[2]
    healthy_ix = np.array([LABELS.index(p) for p in HEALTHY])

    full = surv.mean(axis=2).mean(axis=1)  # (10,)
    boot_1000 = paired_scores(surv, rng.integers(0, n0, size=(B, n0)))
    boot_100 = paired_scores(surv, rng.integers(0, n0, size=(B, 100)))

    order = sorted(range(len(HEALTHY)), key=lambda k: -full[healthy_ix][k])
    adjacent = {
        frozenset((HEALTHY[order[k]], HEALTHY[order[k + 1]]))
        for k in range(len(order) - 1)
    }

    pairs = []
    scaling_ratios = []
    for a, b_ in itertools.combinations(range(len(HEALTHY)), 2):
        i, j = healthy_ix[a], healthy_ix[b_]
        delta = abs(float(full[i] - full[j]))
        sd0 = float(np.std(boot_1000[:, i] - boot_1000[:, j]))
        sd100 = float(np.std(boot_100[:, i] - boot_100[:, j]))
        scaling_ratios.append(sd100 / sd0)
        if delta == 0:
            n_dec = n_pow = float("inf")
        else:
            n_dec = n0 * (Z_ALPHA * sd0 / delta) ** 2
            n_pow = n0 * ((Z_ALPHA + Z_BETA) * sd0 / delta) ** 2
        pairs.append(
            {
                "pair": [HEALTHY[a], HEALTHY[b_]],
                "adjacent_in_ranking": frozenset((HEALTHY[a], HEALTHY[b_]))
                in adjacent,
                "delta_pct_points": delta * 100,
                "sd_at_n1000_pct_points": sd0 * 100,
                "decided_at_n1000": delta > Z_ALPHA * sd0,
                "n_decide_50pct_power": n_dec,
                "n_decide_80pct_power": n_pow,
            }
        )

    adj = [p for p in pairs if p["adjacent_in_ranking"]]
    certify_n = max(p["n_decide_80pct_power"] for p in adj)
    results = {
        "config": {"B": B, "rng_seed": RNG_SEED, "n0": n0,
                   "z_alpha": Z_ALPHA, "z_beta": Z_BETA, "healthy": HEALTHY},
        "reference_order": [HEALTHY[k] for k in order],
        "sd_scaling_check": {
            "mean_sd100_over_sd1000": float(np.mean(scaling_ratios)),
            "expected_sqrt10": float(np.sqrt(10)),
        },
        "pairs": [
            {**p, "n_decide_50pct_power": (None if not np.isfinite(
                p["n_decide_50pct_power"]) else float(p["n_decide_50pct_power"])),
             "n_decide_80pct_power": (None if not np.isfinite(
                p["n_decide_80pct_power"]) else float(p["n_decide_80pct_power"]))}
            for p in pairs
        ],
        "certification_budget_games_per_cell_80pct": float(certify_n),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "evaluation_budget.json", "w") as fh:
        json.dump(results, fh, indent=1)

    def fmt_n(v: float) -> str:
        if not np.isfinite(v) or v > N_CAP:
            return f">{N_CAP:,}"
        return f"{v:,.0f}"

    lines = [
        "# E22: games-per-cell budget to decide each healthy pairwise verdict",
        "",
        f"SDs from the paired-seed bootstrap at n0={n0} (B={B}, seed {RNG_SEED});",
        "1/sqrt(n) scaling verified empirically: mean sd(100)/sd(1000) = "
        f"{results['sd_scaling_check']['mean_sd100_over_sd1000']:.2f} "
        f"(expected sqrt(10) = 3.16). Survival metric, macro over 9 cells.",
        "",
        f"Reference order: {' > '.join(results['reference_order'])}",
        "",
        "| pair | adjacent | gap (pts) | sd@1000 (pts) | decided@1000 | n (50% power) | n (80% power) |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in sorted(pairs, key=lambda x: -x["delta_pct_points"]):
        lines.append(
            f"| {p['pair'][0]} vs {p['pair'][1]} | "
            f"{'Y' if p['adjacent_in_ranking'] else ''} | "
            f"{p['delta_pct_points']:.2f} | {p['sd_at_n1000_pct_points']:.2f} | "
            f"{'yes' if p['decided_at_n1000'] else 'NO'} | "
            f"{fmt_n(p['n_decide_50pct_power'])} | "
            f"{fmt_n(p['n_decide_80pct_power'])} |"
        )
    n_undecided = sum(not p["decided_at_n1000"] for p in pairs)
    lines += [
        "",
        "## Summary",
        "",
        f"- {n_undecided}/{len(pairs)} healthy pairs remain UNDECIDED at n=1000",
        f"- certifying the full healthy-8 ranking (all adjacent pairs, 80% power) "
        f"needs **{fmt_n(certify_n)} games per cell** "
        f"(x90 cells = {fmt_n(certify_n * 90)} games total)",
        "- the original 100-game protocol could only ever decide pairs with gaps "
        "of roughly >= "
        f"{np.mean([p['sd_at_n1000_pct_points'] for p in pairs]) * np.sqrt(10) * Z_ALPHA:.1f} "
        "points (1.96 x sd at n=100)",
        "",
    ]
    with open(OUT_DIR / "evaluation_budget_summary.md", "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
