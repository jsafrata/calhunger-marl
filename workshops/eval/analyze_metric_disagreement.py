"""E20: metric-portfolio disagreement audit on the peer cross-play games (TAE).

Question (TAE "measurement and causal validity"): on the SAME underlying
games, how often do pairwise verdicts ("A beats B") and leaderboard orderings
invert across defensible metric choices?

Design
------
Data: ``results/seeds234/_peer_crossplay_100`` — per-game focal outcomes of
the 90 ordered 1v3 peer cells plus the 10 self-play diagonal cells.

Peer metrics per policy (macro-mean over its 9 focal cells):
  survival, win, cash (mean terminal, incl. the dead), cash|survival,
  survival_time (ticks), win|survival.
Self-play instruments (diagonal): selfplay_survival, selfplay_cash.

Analyses:
  A. leaderboard per metric + metric x metric Spearman (all 10 / healthy 8);
  B. pairwise verdict flips between every metric pair;
  C. significance filter: paired-seed bootstrap (B=2000, n=100) 95% CI on
     each pairwise difference; a verdict is "decided" if the CI excludes 0;
     a STRICT inversion = decided under both metrics in opposite directions.

Usage (repo outer root):
    python3 workshops/eval/analyze_metric_disagreement.py
    python3 workshops/eval/analyze_metric_disagreement.py --extended
        # merge the 900-game seed extension (1000 games/cell); output _1000
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from scipy.stats import spearmanr

from crossplay_raw_common import (
    HEALTHY,
    LABELS,
    RESULTS_ROOT,
    load_peer_1v3,
    load_peer_selfplay_seats,
    stack_values,
)

OUT_DIR = RESULTS_ROOT / "metric_disagreement"  # overridden in main()
B = 2000
RNG_SEED = 20260828

PEER_METRICS = ["survival", "win", "cash", "cash_given_survival", "survival_time", "win_given_survival"]
# NB: raw self-play mean cash is 100.0 by conservation (cash only moves between
# the four seats of the same policy), so the non-trivial self-play cash
# instrument is necessarily survivor-conditioned.
SELF_METRICS = ["selfplay_survival", "selfplay_cash_given_survival"]
ALL_METRICS = PEER_METRICS + SELF_METRICS
# Quantity columns: survived=0, won=1, cash=2, ticks=3.


def cell_metrics(sub: np.ndarray) -> np.ndarray:
    """Per-cell metric aggregates. sub: (..., n_games, Q) -> (..., 6)."""
    survived = sub[..., 0]
    won = sub[..., 1]
    cash = sub[..., 2]
    ticks = sub[..., 3]
    n_surv = survived.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.stack(
            [
                survived.mean(axis=-1),
                won.mean(axis=-1),
                cash.mean(axis=-1),
                np.where(n_surv > 0, (cash * survived).sum(axis=-1) / n_surv, np.nan),
                ticks.mean(axis=-1),
                np.where(n_surv > 0, (won * survived).sum(axis=-1) / n_surv, np.nan),
            ],
            axis=-1,
        )
    return out


def peer_scores(values: np.ndarray, idx: np.ndarray | None) -> np.ndarray:
    """Macro-mean peer metrics. values (P,C,G,Q); idx (B,n) paired or None.

    Returns (P, 6) without idx, else (B, P, 6).
    """
    if idx is None:
        return np.nanmean(cell_metrics(values), axis=1)
    p, c, g, q = values.shape
    b = idx.shape[0]
    cms = np.empty((b, p, c, len(PEER_METRICS)))
    for ci in range(c):
        for pi in range(p):
            sub = values[pi, ci][idx, :]  # (B, n, Q)
            cms[:, pi, ci, :] = cell_metrics(sub)
    return np.nanmean(cms, axis=2)


def selfplay_scores(values: np.ndarray, idx: np.ndarray | None) -> np.ndarray:
    """Self-play survival and survivor cash. values (P,G,4,Q) per seat.

    Returns (P,2) without idx, else (B,P,2).
    """
    def agg(vals: np.ndarray) -> np.ndarray:
        survived = vals[..., 0]
        cash = vals[..., 2]
        n_surv = survived.sum(axis=(-1, -2))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.stack(
                [
                    survived.mean(axis=(-1, -2)),
                    np.where(
                        n_surv > 0,
                        (cash * survived).sum(axis=(-1, -2)) / n_surv,
                        np.nan,
                    ),
                ],
                axis=-1,
            )

    if idx is None:
        return agg(values)
    # gather per policy to keep memory bounded at large n
    out = np.empty((idx.shape[0], values.shape[0], 2))
    for pi in range(values.shape[0]):
        out[:, pi, :] = agg(values[pi][idx])  # (B, n, 4, Q) -> (B, 2)
    return out


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true",
                        help="merge the 900-game seed extension (1000 games/cell)")
    args = parser.parse_args()
    if args.extended:
        OUT_DIR = RESULTS_ROOT / "metric_disagreement_1000"

    rng = np.random.default_rng(RNG_SEED)
    peer = stack_values(load_peer_1v3(extended=args.extended), LABELS)
    selfp = np.stack(
        [load_peer_selfplay_seats(extended=args.extended)[p] for p in LABELS]
    )
    n_games = peer.shape[2]
    healthy_ix = np.array([LABELS.index(p) for p in HEALTHY])

    full = np.concatenate([peer_scores(peer, None), selfplay_scores(selfp, None)], axis=1)
    idx = rng.integers(0, n_games, size=(B, n_games))
    boot = np.concatenate(
        [peer_scores(peer, idx), selfplay_scores(selfp, idx)], axis=2
    )  # (B, 10, 8)

    results: dict = {
        "config": {"B": B, "rng_seed": RNG_SEED, "metrics": ALL_METRICS,
                   "extended": args.extended, "n_games_per_cell": int(n_games),
                   "labels": LABELS, "healthy": HEALTHY},
        "full_scores": {
            m: {p: float(full[i, mi]) for i, p in enumerate(LABELS)}
            for mi, m in enumerate(ALL_METRICS)
        },
    }

    # A. metric x metric leaderboard correlation
    for group, ix in (("all10", np.arange(len(LABELS))), ("healthy8", healthy_ix)):
        mat = {}
        for (ai, a), (bi, b) in itertools.combinations(enumerate(ALL_METRICS), 2):
            rho = spearmanr(full[ix, ai], full[ix, bi])
            mat[f"{a}~{b}"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}
        results[f"metric_spearman_{group}"] = mat

    # B + C. pairwise verdicts with bootstrap significance
    pairs = list(itertools.combinations(range(len(LABELS)), 2))
    verdicts = []  # per (pair, metric): sign, ci, decided
    for i, j in pairs:
        row: dict = {"pair": [LABELS[i], LABELS[j]],
                     "healthy_pair": bool(i in healthy_ix and j in healthy_ix)}
        for mi, m in enumerate(ALL_METRICS):
            d_full = full[i, mi] - full[j, mi]
            d_boot = boot[:, i, mi] - boot[:, j, mi]
            if np.isnan(d_full) or np.all(np.isnan(d_boot)):
                row[m] = {"delta": None, "ci95": None, "ci99": None,
                          "decided": False, "decided99": False, "sign": 0}
                continue
            lo, hi = np.nanpercentile(d_boot, [2.5, 97.5])
            lo99, hi99 = np.nanpercentile(d_boot, [0.5, 99.5])
            row[m] = {
                "delta": float(d_full),
                "ci95": [float(lo), float(hi)],
                "ci99": [float(lo99), float(hi99)],
                "decided": bool(lo > 0 or hi < 0),
                "decided99": bool(lo99 > 0 or hi99 < 0),
                "sign": int(np.sign(d_full)),
            }
        verdicts.append(row)
    results["pairwise"] = verdicts

    # summarize flips / strict inversions per metric pair
    flip_summary = []
    for a, b in itertools.combinations(ALL_METRICS, 2):
        for group_name, restrict in (("all10", None), ("healthy8", True)):
            rows = [r for r in verdicts if restrict is None or r["healthy_pair"]]
            n_pairs = len(rows)
            flips = [r for r in rows if r[a]["sign"] != r[b]["sign"]]
            strict = [
                r for r in flips if r[a]["decided"] and r[b]["decided"]
            ]
            strict99 = [
                r for r in flips if r[a]["decided99"] and r[b]["decided99"]
            ]
            flip_summary.append(
                {
                    "metric_a": a,
                    "metric_b": b,
                    "group": group_name,
                    "n_pairs": n_pairs,
                    "n_sign_flips": len(flips),
                    "n_strict_inversions": len(strict),
                    "n_strict_inversions_99": len(strict99),
                    "strict_pairs": [r["pair"] for r in strict],
                }
            )
    results["flip_summary"] = flip_summary

    # Metric-family collapse: a pair counts once per FAMILY pair, immune to
    # "you counted near-duplicate metrics" (cash~cash|survival rho=0.98).
    families = {
        "survival_family": ["survival", "survival_time"],
        "win_family": ["win", "win_given_survival"],
        "cash_family": ["cash", "cash_given_survival"],
    }
    fam_summary = []
    for (fa, ms_a), (fb, ms_b) in itertools.combinations(families.items(), 2):
        for group_name, restrict in (("all10", None), ("healthy8", True)):
            rows = [r for r in verdicts if restrict is None or r["healthy_pair"]]
            strict, strict99 = [], []
            for r in rows:
                combos = [
                    (r[a], r[b]) for a in ms_a for b in ms_b
                    if r[a]["sign"] != r[b]["sign"]
                ]
                if any(va["decided"] and vb["decided"] for va, vb in combos):
                    strict.append(r["pair"])
                if any(va["decided99"] and vb["decided99"] for va, vb in combos):
                    strict99.append(r["pair"])
            fam_summary.append(
                {
                    "family_a": fa,
                    "family_b": fb,
                    "group": group_name,
                    "n_pairs": len(rows),
                    "n_strict_inversions": len(strict),
                    "n_strict_inversions_99": len(strict99),
                    "strict_pairs_99": strict99,
                }
            )
    results["family_summary"] = fam_summary

    # headline: verdicts that flip under >=1 alternative peer metric vs survival
    base = "survival"
    for group_name, restrict in (("all10", None), ("healthy8", True)):
        rows = [r for r in verdicts if restrict is None or r["healthy_pair"]]
        n_decided = [r for r in rows if r[base]["decided"]]
        any_flip = [
            r
            for r in rows
            if any(r[base]["sign"] != r[m]["sign"] for m in PEER_METRICS if m != base)
        ]
        any_strict = [
            r
            for r in n_decided
            if any(
                r[base]["sign"] != r[m]["sign"] and r[m]["decided"]
                for m in PEER_METRICS
                if m != base
            )
        ]
        any_strict99 = [
            r
            for r in rows
            if r[base]["decided99"]
            and any(
                r[base]["sign"] != r[m]["sign"] and r[m]["decided99"]
                for m in PEER_METRICS
                if m != base
            )
        ]
        results[f"headline_{group_name}"] = {
            "n_pairs": len(rows),
            "n_flip_vs_survival_any_peer_metric": len(any_flip),
            "n_strict_inversions_vs_survival": len(any_strict),
            "n_strict_inversions_vs_survival_99": len(any_strict99),
            "strict_pairs": [r["pair"] for r in any_strict],
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "metric_disagreement.json", "w") as fh:
        json.dump(results, fh, indent=1)
    write_summary(results)
    print(f"wrote {OUT_DIR}")


def write_summary(results: dict) -> None:
    n = results["config"]["n_games_per_cell"]
    lines = [
        f"# E20: metric-portfolio disagreement audit (peer cross-play, n={n} paired games/cell)",
        "",
        f"B={results['config']['B']} paired-seed bootstrap replicates; 'decided' = 95% CI of",
        "the pairwise difference excludes 0; strict inversion = decided under both metrics",
        "with opposite signs.",
        "",
        "## Leaderboards (full data)",
        "",
        "| policy | " + " | ".join(ALL_METRICS) + " |",
        "|---|" + "---|" * len(ALL_METRICS),
    ]
    fs = results["full_scores"]
    for p in results["config"]["labels"]:
        cells = []
        for m in ALL_METRICS:
            v = fs[m][p]
            scale = 100.0 if m in ("survival", "win", "win_given_survival",
                                    "selfplay_survival") else 1.0
            cells.append(f"{v*scale:.1f}")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")

    for group in ("all10", "healthy8"):
        lines += ["", f"## Metric x metric leaderboard Spearman ({group})", ""]
        for k, v in results[f"metric_spearman_{group}"].items():
            lines.append(f"- {k}: rho={v['rho']:+.2f} (p={v['p']:.3f})")

    lines += ["", "## Sign flips and strict inversions per metric pair", "",
              "| metric A | metric B | group | sign flips | strict @95 | strict @99 |",
              "|---|---|---|---|---|---|"]
    for e in results["flip_summary"]:
        lines.append(
            f"| {e['metric_a']} | {e['metric_b']} | {e['group']} | "
            f"{e['n_sign_flips']}/{e['n_pairs']} | {e['n_strict_inversions']} | "
            f"{e['n_strict_inversions_99']} |"
        )

    lines += ["", "## Metric-family collapse (cross-family strict inversions)", "",
              "| family A | family B | group | strict @95 | strict @99 |",
              "|---|---|---|---|---|"]
    for e in results["family_summary"]:
        lines.append(
            f"| {e['family_a']} | {e['family_b']} | {e['group']} | "
            f"{e['n_strict_inversions']}/{e['n_pairs']} | "
            f"{e['n_strict_inversions_99']}/{e['n_pairs']} |"
        )

    lines += ["", "## Headline (vs the survival verdict)", ""]
    for group in ("all10", "healthy8"):
        h = results[f"headline_{group}"]
        lines.append(
            f"- {group}: {h['n_flip_vs_survival_any_peer_metric']}/{h['n_pairs']} pairwise"
            f" verdicts flip under >=1 alternative peer metric; "
            f"{h['n_strict_inversions_vs_survival']} are STRICT inversions at 95% "
            f"({h['n_strict_inversions_vs_survival_99']} survive at 99%)"
            f" ({', '.join(' vs '.join(p) for p in h['strict_pairs'])})"
        )
    lines.append("")
    with open(OUT_DIR / "metric_disagreement_summary.md", "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
