"""E34: metric inversions under familywise error control.

Camera-ready addition (TAE): the "15 of 28 significant inversions"
involve searching across metrics and policy pairs; distinguish individual
confidence intervals from a familywise claim.

Same data and estimator as E20 (merged 1000-game peer matrix, six peer
metrics per policy as macro-means over its nine 1v3 cells, paired-seed
bootstrap). A healthy pair is a STRICT INVERSION if its survival verdict is
significant and some alternative peer metric's verdict is significant with
the opposite sign. We count strict inversions under four significance rules:

  per-test 95% / 99% (E20 as published; percentile bootstrap CIs),
  Bonferroni over the family of all 28 pairs x 6 metrics = 168 tests,
  Holm step-down over the same 168 tests,
  and the same with the six metrics collapsed to three families
  (survival, win, cash; 84 tests).

p-values are two-sided normal approximations z = delta / sd_boot (the
percentile bootstrap cannot resolve p < 1/B); the percentile version is
reported alongside where it is resolvable.

Usage (repo outer root):
    python3 workshops/eval/analyze_inversions_familywise.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from scipy.stats import norm

from analyze_metric_disagreement import PEER_METRICS, peer_scores
from crossplay_raw_common import HEALTHY, LABELS, RESULTS_ROOT, load_peer_1v3, stack_values

OUT_DIR = RESULTS_ROOT / "inversions_familywise"
B = 4000
RNG_SEED = 20260828
ALPHA = 0.05
FAMILIES = {"survival": ["survival", "survival_time"],
            "win": ["win", "win_given_survival"],
            "cash": ["cash", "cash_given_survival"]}


def holm_reject(p: np.ndarray, alpha: float) -> np.ndarray:
    """Holm step-down: boolean rejections for a flat vector of p-values."""
    m = p.size
    order = np.argsort(p)
    rej = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    peer = stack_values(load_peer_1v3(extended=True), LABELS)
    n_games = peer.shape[2]
    hix = [LABELS.index(p) for p in HEALTHY]
    full = peer_scores(peer, None)[hix]  # (8, 6)
    boot = np.concatenate(
        [peer_scores(peer, rng.integers(0, n_games, size=(b, n_games)))[:, hix, :]
         for b in [B // 4] * 4], axis=0)  # (B, 8, 6), chunked for memory
    pairs = list(itertools.combinations(range(len(HEALTHY)), 2))
    M = len(PEER_METRICS)
    delta = np.array([[full[i, m] - full[j, m] for m in range(M)] for i, j in pairs])
    dboot = np.stack([boot[:, i, :] - boot[:, j, :] for i, j in pairs], axis=1)  # (B, 28, 6)
    sd = dboot.std(axis=0, ddof=1)
    z = delta / sd
    p_norm = 2 * norm.sf(np.abs(z))
    lo95, hi95 = np.percentile(dboot, [2.5, 97.5], axis=0)
    lo99, hi99 = np.percentile(dboot, [0.5, 99.5], axis=0)
    dec95 = (lo95 > 0) | (hi95 < 0)
    dec99 = (lo99 > 0) | (hi99 < 0)
    sign = np.sign(delta)

    def count_strict(decided: np.ndarray, survival_col: int = 0) -> tuple[int, list]:
        out = []
        for k, (i, j) in enumerate(pairs):
            if not decided[k, survival_col]:
                continue
            alts = [m for m in range(M) if m != survival_col
                    and decided[k, m] and sign[k, m] != sign[k, survival_col]]
            if alts:
                out.append([HEALTHY[i], HEALTHY[j], [PEER_METRICS[m] for m in alts]])
        return len(out), out

    rules = {}
    rules["per_test_95_percentile"] = count_strict(dec95)
    rules["per_test_99_percentile"] = count_strict(dec99)
    rules["per_test_95_normal"] = count_strict(p_norm < ALPHA)
    rules["bonferroni_168_normal"] = count_strict(p_norm < ALPHA / p_norm.size)
    rules["holm_168_normal"] = count_strict(holm_reject(p_norm.ravel(), ALPHA).reshape(p_norm.shape))

    # family-collapsed: representative = each family's most significant metric per pair
    fam_names = list(FAMILIES)
    fam_cols = [[PEER_METRICS.index(m) for m in ms] for ms in FAMILIES.values()]
    p_fam = np.stack([p_norm[:, cols].min(axis=1) for cols in fam_cols], axis=1)  # (28, 3)
    sign_fam = np.stack([sign[np.arange(len(pairs)), np.array(cols)[np.argmin(p_norm[:, cols], axis=1)]]
                         for cols in fam_cols], axis=1)

    def count_strict_fam(decided: np.ndarray) -> tuple[int, list]:
        out = []
        for k, (i, j) in enumerate(pairs):
            if not decided[k, 0]:
                continue
            alts = [fam_names[f] for f in (1, 2) if decided[k, f] and sign_fam[k, f] != sign_fam[k, 0]]
            if alts:
                out.append([HEALTHY[i], HEALTHY[j], alts])
        return len(out), out

    rules["families_per_test_95_normal"] = count_strict_fam(p_fam < ALPHA)
    rules["families_bonferroni_84_normal"] = count_strict_fam(p_fam < ALPHA / p_fam.size)
    rules["families_holm_84_normal"] = count_strict_fam(holm_reject(p_fam.ravel(), ALPHA).reshape(p_fam.shape))

    # smallest Bonferroni-adjusted p among the strict-inversion pairs' binding tests
    results = {
        "config": {"B": B, "rng_seed": RNG_SEED, "n_games_per_cell": int(n_games), "alpha": ALPHA,
                   "metrics": PEER_METRICS, "n_pairs": len(pairs), "n_tests": int(p_norm.size),
                   "min_percentile_p_resolvable": 1.0 / B},
        "counts": {k: v[0] for k, v in rules.items()},
        "strict_pairs": {k: v[1] for k, v in rules.items()},
        "per_pair": [
            {"pair": [HEALTHY[i], HEALTHY[j]],
             **{PEER_METRICS[m]: {"delta": float(delta[k, m]), "sd": float(sd[k, m]),
                                  "z": float(z[k, m]), "p_normal": float(p_norm[k, m]),
                                  "p_bonferroni_168": float(min(1.0, p_norm[k, m] * p_norm.size))}
                for m in range(M)}}
            for k, (i, j) in enumerate(pairs)
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "inversions_familywise.json").write_text(json.dumps(results, indent=1))
    lines = [
        "# E34: strict metric inversions (healthy 8, n=1000 paired games/cell) under familywise control",
        "",
        f"Family = {len(pairs)} pairs x {M} metrics = {p_norm.size} tests (normal-approx p from the paired bootstrap SD, B={B}).",
        "Strict inversion = survival verdict significant AND an alternative-metric verdict significant with the opposite sign.",
        "",
        "| rule | strict inversions / 28 |",
        "|---|---|",
    ]
    for k, v in results["counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "Pairs under Holm(168):", ""]
    for p_ in results["strict_pairs"]["holm_168_normal"]:
        lines.append(f"- {p_[0]} vs {p_[1]}: {', '.join(p_[2])}")
    lines.append("")
    (OUT_DIR / "inversions_familywise_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
