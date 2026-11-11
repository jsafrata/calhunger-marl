"""E33: what "certifying the full ranking costs ~18,700 games/cell" means.

Camera-ready addition (TAE): an 80%-power calculation for one difficult
pairwise comparison is not an 80% guarantee for all orderings simultaneously;
give the calculation behind the estimate (error criterion, dependence
assumptions, sensitivity to the estimated gaps) and distinguish individual
from familywise claims.

This script reproduces the E22 number and then stress-tests it four ways on
the same paired-seed bootstrap of the merged 1000-game peer matrix:

  (1) PLUG-IN, PER PAIR (E22 as published). For adjacent pair k in the
      reference survival ordering with observed gap d_k and bootstrap SD
      sd_k at n0 = 1000 games/cell, the games/cell needed to reject
      "d_k = 0" at two-sided alpha = .05 with power 1 - beta = .80, under
      the empirically verified 1/sqrt(n) scaling, is
          n_k = n0 * ((z_{.975} + z_{.80}) * sd_k / d_k)^2 .
      "Certifying the ranking" = deciding every adjacent pair; the budget is
      max_k n_k. The SDs come from a PAIRED-seed bootstrap of macro scores,
      so cross-cell dependence induced by shared game seeds is in sd_k.
  (2) GAP UNCERTAINTY. The tightest gaps (~0.5 pts) carry SEs of ~0.7 pts,
      so the plug-in budget is itself noisy: recompute max_k n_k on each
      bootstrap replicate's gaps and report the distribution (and the budget
      when the binding gap is moved +-1 SE).
  (3) FAMILYWISE PER-PAIR BUDGET. Bonferroni over the 7 adjacent comparisons
      (two-sided alpha/7 each) -- the budget at which each adjacent verdict
      is individually powered at 80% while the familywise type-I error over
      the 7 comparisons is held at 5%.
  (4) JOINT POWER. Simulating the 7 adjacent differences from a multivariate
      normal with the bootstrap covariance (scaled by n0/n), the probability
      that ALL seven are decided in the correct direction at budget n; and
      the budget at which that joint probability reaches 80% (per-pair and
      Bonferroni thresholds).

Usage (repo outer root):
    python3 workshops/eval/analyze_evaluation_budget_sensitivity.py
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.stats import norm

from crossplay_raw_common import (
    HEALTHY,
    LABELS,
    RESULTS_ROOT,
    load_peer_1v3,
    stack_values,
)

OUT_DIR = RESULTS_ROOT / "evaluation_budget_sensitivity"
B = 2000
RNG_SEED = 20260829  # same as E22 so the SDs reproduce
ALPHA = 0.05
POWER = 0.80
Z_A = norm.ppf(1 - ALPHA / 2)
Z_B = norm.ppf(POWER)
N_MC = 200_000
N_CAP = 10_000_000


def paired_scores(surv: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(P, C, G) survival + shared idx (B, n) -> (B, P) macro scores."""
    b = idx.shape[0]
    p, c, g = surv.shape
    cell_means = np.empty((b, p, c))
    for ci in range(c):
        for pi in range(p):
            cell_means[:, pi, ci] = surv[pi, ci][idx].mean(axis=1)
    return cell_means.mean(axis=2)


def budget(gap: np.ndarray, sd: np.ndarray, n0: int, z_a: float, z_b: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return n0 * ((z_a + z_b) * sd / np.abs(gap)) ** 2


def joint_power(n: float, gaps: np.ndarray, cov0: np.ndarray, n0: int, z_a: float,
                rng: np.random.Generator) -> float:
    """P(all adjacent differences significant in the correct direction) at n."""
    cov = cov0 * (n0 / n)
    sd = np.sqrt(np.diag(cov))
    draws = rng.multivariate_normal(gaps, cov, size=N_MC)
    ok = (np.sign(draws) == np.sign(gaps)) & (np.abs(draws) / sd > z_a)
    return float(ok.all(axis=1).mean())


def solve_joint(gaps, cov0, n0, z_a, rng, target=POWER) -> float:
    lo, hi = 100.0, 1e7
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if joint_power(mid, gaps, cov0, n0, z_a, rng) < target:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    peer = stack_values(load_peer_1v3(extended=True), LABELS)
    surv = peer[..., 0]
    n0 = surv.shape[2]
    hix = np.array([LABELS.index(p) for p in HEALTHY])

    full = surv.mean(axis=2).mean(axis=1)[hix] * 100  # pts
    boot = paired_scores(surv, rng.integers(0, n0, size=(B, n0)))[:, hix] * 100

    order = np.argsort(-full)
    adj = [(order[k], order[k + 1]) for k in range(len(order) - 1)]
    names = [f"{HEALTHY[i]} > {HEALTHY[j]}" for i, j in adj]
    gaps = np.array([full[i] - full[j] for i, j in adj])
    diffs = np.stack([boot[:, i] - boot[:, j] for i, j in adj], axis=1)  # (B, 7)
    sd0 = diffs.std(axis=0, ddof=1)
    cov0 = np.cov(diffs, rowvar=False)
    corr0 = np.corrcoef(diffs, rowvar=False)

    # (1) plug-in per pair, as published
    n_pair = budget(gaps, sd0, n0, Z_A, Z_B)
    k_bind = int(np.argmax(n_pair))

    # (2) gap uncertainty: budget recomputed on each replicate's gaps
    n_rep = budget(diffs, sd0[None, :], n0, Z_A, Z_B).max(axis=1)
    finite = n_rep[np.isfinite(n_rep)]
    pct = np.percentile(finite, [5, 25, 50, 75, 95])
    frac_over_cap = float((~np.isfinite(n_rep) | (n_rep > N_CAP)).mean())
    g, s = gaps[k_bind], sd0[k_bind]
    pm = {
        "gap_minus_1se": None if g - s <= 0 else float(budget(np.array([g - s]), np.array([s]), n0, Z_A, Z_B)[0]),
        "gap_plus_1se": float(budget(np.array([g + s]), np.array([s]), n0, Z_A, Z_B)[0]),
    }

    # (3) Bonferroni over the 7 adjacent comparisons
    z_a_bonf = norm.ppf(1 - ALPHA / (2 * len(adj)))
    n_pair_bonf = budget(gaps, sd0, n0, z_a_bonf, Z_B)

    # (4) joint power
    jp_at_published = joint_power(n_pair.max(), gaps, cov0, n0, Z_A, rng)
    n_joint = solve_joint(gaps, cov0, n0, Z_A, rng)
    n_joint_bonf = solve_joint(gaps, cov0, n0, z_a_bonf, rng)
    jp_curve = {int(n): joint_power(n, gaps, cov0, n0, Z_A, rng)
                for n in (100, 1000, 5000, 10000, 18681, 30000, 50000, 100000)}

    results = {
        "config": {"B": B, "rng_seed": RNG_SEED, "n0": n0, "alpha": ALPHA, "power": POWER,
                   "z_alpha_half": float(Z_A), "z_beta": float(Z_B),
                   "z_alpha_bonferroni_7": float(z_a_bonf), "n_mc": N_MC},
        "criterion": ("two-sided test of a zero pairwise macro-survival difference at alpha=.05 "
                      "with 80% power, per adjacent pair of the reference survival ordering; "
                      "SD from the paired-seed bootstrap at n0 (shared game seeds across cells, "
                      "so cross-cell dependence is inside sd); 1/sqrt(n) scaling (verified "
                      "empirically, sd(100)/sd(1000)=3.17 vs 3.16); ranking certified when every "
                      "adjacent pair is decided; budget = max over adjacent pairs"),
        "reference_order": [HEALTHY[i] for i in order],
        "adjacent_pairs": [
            {"pair": names[k], "gap_pts": float(gaps[k]), "sd_at_n0_pts": float(sd0[k]),
             "n_80pct_power": float(n_pair[k]), "n_80pct_power_bonferroni7": float(n_pair_bonf[k])}
            for k in range(len(adj))
        ],
        "adjacent_diff_correlation": corr0.round(3).tolist(),
        "published_budget_games_per_cell": float(n_pair.max()),
        "binding_pair": names[k_bind],
        "gap_uncertainty": {
            "binding_gap_pts": float(g), "binding_gap_se_pts": float(s),
            "budget_if_gap_minus_1se": pm["gap_minus_1se"],
            "budget_if_gap_plus_1se": pm["gap_plus_1se"],
            "bootstrap_budget_percentiles_5_25_50_75_95": [float(x) for x in pct],
            "fraction_replicates_budget_over_10M_or_infinite": frac_over_cap,
        },
        "bonferroni7_budget_games_per_cell": float(n_pair_bonf.max()),
        "joint_power": {
            "at_published_budget": jp_at_published,
            "budget_for_80pct_joint_power": n_joint,
            "budget_for_80pct_joint_power_bonferroni7": n_joint_bonf,
            "curve_per_pair_threshold": jp_curve,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "evaluation_budget_sensitivity.json").write_text(json.dumps(results, indent=1))

    def f(v):
        return "inf" if v is None or not np.isfinite(v) else f"{v:,.0f}"

    lines = [
        "# E33: the certification budget, made explicit",
        "",
        f"Criterion: {results['criterion']}.",
        "",
        f"Reference order: {' > '.join(results['reference_order'])}",
        "",
        "| adjacent pair | gap (pts) | sd@1000 (pts) | n, 80% power, alpha .05 | n, 80% power, Bonferroni(7) |",
        "|---|---|---|---|---|",
    ]
    for r in results["adjacent_pairs"]:
        lines.append(f"| {r['pair']} | {r['gap_pts']:.2f} | {r['sd_at_n0_pts']:.2f} | "
                     f"{f(r['n_80pct_power'])} | {f(r['n_80pct_power_bonferroni7'])} |")
    gu = results["gap_uncertainty"]
    lines += [
        "",
        f"- published plug-in budget (max over adjacent pairs): **{f(results['published_budget_games_per_cell'])}** games/cell, binding pair {results['binding_pair']}",
        f"- binding gap {gu['binding_gap_pts']:.2f} +- {gu['binding_gap_se_pts']:.2f} pts (bootstrap SE): budget {f(gu['budget_if_gap_plus_1se'])} at gap+1SE, {f(gu['budget_if_gap_minus_1se'])} at gap-1SE",
        f"- budget recomputed on each bootstrap replicate's gaps: 5/25/50/75/95th pct = {', '.join(f(x) for x in gu['bootstrap_budget_percentiles_5_25_50_75_95'])}; "
        f"{100*gu['fraction_replicates_budget_over_10M_or_infinite']:.1f}% of replicates exceed 10M (a replicate gap at ~0)",
        f"- Bonferroni over the 7 adjacent comparisons (per-pair 80% power, FWER 5%): **{f(results['bonferroni7_budget_games_per_cell'])}** games/cell",
        f"- joint power (all 7 adjacent verdicts correct and significant) at the published budget: **{100*results['joint_power']['at_published_budget']:.0f}%**",
        f"- budget for 80% JOINT power: **{f(results['joint_power']['budget_for_80pct_joint_power'])}** (per-pair alpha .05) / {f(results['joint_power']['budget_for_80pct_joint_power_bonferroni7'])} (Bonferroni 7)",
        "- joint-power curve (per-pair threshold): " + ", ".join(f"n={k}: {100*v:.0f}%" for k, v in results['joint_power']['curve_per_pair_threshold'].items()),
        "",
        "Adjacent-difference correlation matrix (paired bootstrap):",
        "",
    ]
    for row in results["adjacent_diff_correlation"]:
        lines.append("    " + "  ".join(f"{x:+.2f}" for x in row))
    lines.append("")
    (OUT_DIR / "evaluation_budget_sensitivity_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
