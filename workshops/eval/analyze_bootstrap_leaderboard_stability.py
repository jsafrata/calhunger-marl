"""E19: bootstrap stability of the peer cross-play leaderboard (TAE).

Question (TAE "uncertainty and robustness"): how stable are the evaluation's
conclusions — leaderboard order, top-1 identity, pairwise verdicts, and the
E17 ladder-validity correlation — under sampling variation, as a function of
the evaluation budget n (games per cell)?

Design
------
Data: ``results/seeds234/_peer_crossplay_100`` (90 ordered 1v3 cells, 100
paired games each, seed base 900000) and ``_ladder_validity_100`` (50 cells).
A policy's score is the macro-mean over its 9 focal 1v3 cells of the per-cell
game mean (identical to E12/E17's "peer off-diagonal" aggregation).

Bootstrap: B=2000 replicates at n in {5, 10, 20, 50, 100} games per cell,
under two schemes:

- ``independent``: each cell resamples n game indices independently — what a
  naive protocol without seed pairing delivers;
- ``paired``: one index vector shared by ALL cells per replicate — respects
  the campaign's paired-seed design (game i has the same seed everywhere).

Reported per (metric, scheme, n): mean/5th-pct Spearman rho of the replicate
leaderboard vs the full-data (n=100) leaderboard, top-1 stability, pairwise
verdict flip probabilities (fraction of replicates where sign(S_A - S_B)
disagrees with the full data), computed for all 10 policies and the healthy 8.
Conclusion-level: the bootstrap distribution of E17's headline Spearman
(healthy-8 mean vs-v2..v5 ladder survival vs peer off-diagonal survival).

Usage (repo outer root):
    python3 workshops/eval/analyze_bootstrap_leaderboard_stability.py
    python3 workshops/eval/analyze_bootstrap_leaderboard_stability.py --extended
        # merge the 900-game seed extension (campaign peer_crossplay_900ext)
        # for 1000 games/cell; budgets up to n=1000; output dir suffixed _1000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import rankdata, spearmanr

from crossplay_raw_common import (
    HEALTHY,
    LABELS,
    QUANTITIES,
    RESULTS_ROOT,
    load_ladder,
    load_peer_1v3,
    stack_values,
)

OUT_DIR = RESULTS_ROOT / "leaderboard_stability"  # overridden in main()
B = 2000
SUBSAMPLE_SIZES = (5, 10, 20, 50, 100)
SUBSAMPLE_SIZES_EXTENDED = (5, 10, 20, 50, 100, 200, 500, 1000)
METRICS = {"survival": 0, "win": 1, "cash": 2}
RNG_SEED = 20260827


def spearman_rows(sample_scores: np.ndarray, full_scores: np.ndarray) -> np.ndarray:
    """Spearman rho of each replicate row of (B, P) against full (P,)."""
    r_full = rankdata(full_scores)
    r_samp = rankdata(sample_scores, axis=1)
    r_full = (r_full - r_full.mean()) / r_full.std()
    with np.errstate(invalid="ignore", divide="ignore"):
        r_samp = (r_samp - r_samp.mean(axis=1, keepdims=True)) / r_samp.std(
            axis=1, keepdims=True
        )
        return (r_samp * r_full).mean(axis=1)  # all-tied replicates -> nan


def bootstrap_scores(
    values: np.ndarray, idx: np.ndarray, paired: bool
) -> np.ndarray:
    """Replicate macro-mean scores.

    values: (P, C, G, Q); idx: (B, n) if paired else (B, C, n).
    Returns (B, P, Q).
    """
    p, c, g, q = values.shape
    b = idx.shape[0]
    # gather per (cell, policy) to keep memory bounded at large n
    cell_means = np.empty((b, p, c, q))
    for ci in range(c):
        cell_idx = idx if paired else idx[:, ci, :]
        for pi in range(p):
            sub = values[pi, ci][cell_idx, :]  # (B, n, Q)
            cell_means[:, pi, ci, :] = sub.mean(axis=1)
    return cell_means.mean(axis=2)


def flip_stats(sample_scores: np.ndarray, full_scores: np.ndarray) -> dict:
    """Pairwise verdict flip probabilities vs the full-data verdicts."""
    p = full_scores.shape[0]
    iu = np.triu_indices(p, k=1)
    full_sign = np.sign(full_scores[iu[0]] - full_scores[iu[1]])
    diff = sample_scores[:, iu[0]] - sample_scores[:, iu[1]]
    flipped = np.sign(diff) != full_sign  # ties count as flips
    prob = flipped.mean(axis=0)
    order = np.argsort(-full_scores)
    adjacent = [(order[k], order[k + 1]) for k in range(p - 1)]
    pair_index = {(min(a, b), max(a, b)): j for j, (a, b) in enumerate(zip(*iu))}
    adj_prob = [float(prob[pair_index[(min(a, b), max(a, b))]]) for a, b in adjacent]
    return {
        "mean_pair_flip_prob": float(prob.mean()),
        "max_pair_flip_prob": float(prob.max()),
        "n_pairs": int(prob.size),
        "n_pairs_flip_ge_5pct": int((prob >= 0.05).sum()),
        "n_pairs_flip_ge_25pct": int((prob >= 0.25).sum()),
        "mean_adjacent_flip_prob": float(np.mean(adj_prob)),
        "adjacent_flip_probs": adj_prob,
    }


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true",
                        help="merge the 900-game seed extension (1000 games/cell)")
    args = parser.parse_args()
    sizes = SUBSAMPLE_SIZES_EXTENDED if args.extended else SUBSAMPLE_SIZES
    if args.extended:
        OUT_DIR = RESULTS_ROOT / "leaderboard_stability_1000"

    rng = np.random.default_rng(RNG_SEED)
    peer = stack_values(load_peer_1v3(extended=args.extended), LABELS)
    ladder = stack_values(load_ladder(), LABELS)  # (10, 5, 100, Q) — no extension
    n_cells, n_games = peer.shape[1], peer.shape[2]
    healthy_ix = np.array([LABELS.index(p) for p in HEALTHY])

    full_scores = peer.mean(axis=2).mean(axis=1)  # (10, Q)
    results: dict = {
        "config": {
            "B": B,
            "subsample_sizes": list(sizes),
            "extended": args.extended,
            "n_games_per_cell": int(n_games),
            "rng_seed": RNG_SEED,
            "labels": LABELS,
            "healthy": HEALTHY,
        },
        "full_scores": {
            m: {p: float(full_scores[i, q]) for i, p in enumerate(LABELS)}
            for m, q in METRICS.items()
        },
        "stability": [],
    }

    for n in sizes:
        idx_paired = rng.integers(0, n_games, size=(B, n))
        idx_indep = rng.integers(0, n_games, size=(B, n_cells, n))
        for scheme, scores in (
            ("paired", bootstrap_scores(peer, idx_paired, paired=True)),
            ("independent", bootstrap_scores(peer, idx_indep, paired=False)),
        ):
            for metric, q in METRICS.items():
                for group, ix in (("all10", np.arange(len(LABELS))), ("healthy8", healthy_ix)):
                    samp = scores[:, ix, q]
                    full = full_scores[ix, q]
                    rho = spearman_rows(samp, full)
                    top1 = samp.argmax(axis=1) == full.argmax()
                    entry = {
                        "n_games_per_cell": n,
                        "scheme": scheme,
                        "metric": metric,
                        "group": group,
                        "mean_spearman_vs_full": float(np.nanmean(rho)),
                        "p5_spearman_vs_full": float(np.nanpercentile(rho, 5)),
                        "top1_stability": float(top1.mean()),
                        **flip_stats(samp, full),
                    }
                    results["stability"].append(entry)
        print(f"n={n} done")

    # Conclusion-level: E17 headline Spearman under paired resampling at the
    # full budget of each side (peer: n_games; ladder: its own 100 games —
    # not extended, so its resampling uses a separate index space).
    idx = rng.integers(0, n_games, size=(B, n_games))
    ladder_n = ladder.shape[2]
    idx_ladder = rng.integers(0, ladder_n, size=(B, ladder_n))
    peer_boot = bootstrap_scores(peer, idx, paired=True)[:, :, METRICS["survival"]]
    ladder_v25 = ladder[:, 1:, :, METRICS["survival"]]  # drop v1, keep v2..v5
    ladder_boot = (
        ladder_v25[:, :, idx_ladder].mean(axis=3).mean(axis=1).transpose(1, 0)
    )  # (B, 10)
    rhos = np.array(
        [
            spearmanr(ladder_boot[b_][healthy_ix], peer_boot[b_][healthy_ix]).statistic
            for b_ in range(B)
        ]
    )
    point = spearmanr(
        ladder.mean(axis=2)[:, 1:, METRICS["survival"]].mean(axis=1)[healthy_ix],
        full_scores[healthy_ix, METRICS["survival"]],
    )
    results["e17_ladder_validity_rho"] = {
        "point_estimate": float(point.statistic),
        "point_pvalue": float(point.pvalue),
        "bootstrap_mean": float(rhos.mean()),
        "ci95": [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
        "prob_rho_le_0": float((rhos <= 0).mean()),
        "prob_rho_ge_significance_threshold_0.74": float((rhos >= 0.74).mean()),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "bootstrap_stability.json", "w") as fh:
        json.dump(results, fh, indent=1)
    write_summary(results)
    print(f"wrote {OUT_DIR}")


def write_summary(results: dict) -> None:
    lines = [
        "# E19: bootstrap stability of the peer cross-play leaderboard",
        "",
        f"B={results['config']['B']} replicates; seed {results['config']['rng_seed']}; "
        "score = macro-mean over the 9 focal 1v3 cells; full data = "
        f"{results['config']['n_games_per_cell']} paired games/cell"
        + (" (merged 100+900 seed extension)" if results["config"]["extended"] else "")
        + ".",
        "",
        "## Leaderboard stability vs evaluation budget (survival metric)",
        "",
        "| n/cell | scheme | group | mean rho | p5 rho | top-1 stable | mean flip | pairs ≥5% flip | adj. flip |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in results["stability"]:
        if e["metric"] != "survival":
            continue
        lines.append(
            f"| {e['n_games_per_cell']} | {e['scheme']} | {e['group']} | "
            f"{e['mean_spearman_vs_full']:.3f} | {e['p5_spearman_vs_full']:.3f} | "
            f"{e['top1_stability']*100:.0f}% | {e['mean_pair_flip_prob']*100:.1f}% | "
            f"{e['n_pairs_flip_ge_5pct']}/{e['n_pairs']} | "
            f"{e['mean_adjacent_flip_prob']*100:.1f}% |"
        )
    lines += ["", "## Same, win-rate metric", ""]
    lines += [
        "| n/cell | scheme | group | mean rho | p5 rho | top-1 stable | mean flip | pairs ≥5% flip | adj. flip |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in results["stability"]:
        if e["metric"] != "win":
            continue
        lines.append(
            f"| {e['n_games_per_cell']} | {e['scheme']} | {e['group']} | "
            f"{e['mean_spearman_vs_full']:.3f} | {e['p5_spearman_vs_full']:.3f} | "
            f"{e['top1_stability']*100:.0f}% | {e['mean_pair_flip_prob']*100:.1f}% | "
            f"{e['n_pairs_flip_ge_5pct']}/{e['n_pairs']} | "
            f"{e['mean_adjacent_flip_prob']*100:.1f}% |"
        )
    e17 = results["e17_ladder_validity_rho"]
    lines += [
        "",
        "## E17 conclusion under resampling (healthy 8, ladder mean v2-v5 vs peer survival)",
        "",
        f"- point estimate rho = {e17['point_estimate']:.3f} (p = {e17['point_pvalue']:.3f})",
        f"- bootstrap 95% CI: [{e17['ci95'][0]:.3f}, {e17['ci95'][1]:.3f}] (n=100 games/cell, paired)",
        f"- P(rho <= 0) = {e17['prob_rho_le_0']*100:.1f}%",
        f"- P(rho >= 0.74, the n=8 significance threshold) = "
        f"{e17['prob_rho_ge_significance_threshold_0.74']*100:.1f}%",
        "",
    ]
    with open(OUT_DIR / "bootstrap_stability_summary.md", "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
