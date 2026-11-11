"""E12 rebase: population sensitivity on the best-config peer matrix (TAE).

Replaces the deprecated v3-campaign E12 exhibit ("same policy, different
opponent population, verdict flips") with the same claim computed on the
best-config architecture and the merged 1000-game matrix (E19 extension):
for each policy, its survival/win/cash across the 10 populations it can
face (self-play + nine single-peer 1v3 contexts), and the rank it would
receive if the evaluation had used each single population alone.

Usage (repo outer root):
    python3 workshops/eval/analyze_population_sensitivity.py
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import rankdata, spearmanr

from crossplay_raw_common import (
    HEALTHY,
    LABELS,
    RESULTS_ROOT,
    load_peer_1v3,
    load_peer_selfplay,
    stack_values,
)

OUT_DIR = RESULTS_ROOT / "population_sensitivity_1000"


def main() -> None:
    peer_cells = load_peer_1v3(extended=True)
    selfp = load_peer_selfplay(extended=True)
    peer = stack_values(peer_cells, LABELS)  # (10, 9, 1000, Q)
    opp_names = {p: [c.opponent for c in peer_cells[p]] for p in LABELS}

    # per policy x context: survival %, win %, mean cash (focal side)
    per_policy = {}
    for pi, p in enumerate(LABELS):
        rows = {}
        for ci, opp in enumerate(opp_names[p]):
            v = peer[pi, ci]
            rows[f"vs 3x{opp}"] = {
                "survival_pct": float(v[:, 0].mean() * 100),
                "win_pct": float(v[:, 1].mean() * 100),
                "mean_cash": float(v[:, 2].mean()),
            }
        sp = selfp[p].values
        rows["selfplay"] = {
            "survival_pct": float(sp[:, 0].mean() * 100),
            "win_pct": float(sp[:, 1].mean() * 100),
            "mean_cash": float(sp[:, 2].mean()),
        }
        surv = [r["survival_pct"] for k, r in rows.items() if k != "selfplay"]
        wins = [r["win_pct"] for k, r in rows.items() if k != "selfplay"]
        per_policy[p] = {
            "contexts": rows,
            "peer_survival_min": float(min(surv)),
            "peer_survival_max": float(max(surv)),
            "peer_win_min": float(min(wins)),
            "peer_win_max": float(max(wins)),
        }

    # rank a policy would receive if the evaluation used ONE opponent
    # population: for stranger S, rank all focals F != S by survival vs 3xS.
    context_ranks = {p: [] for p in LABELS}
    context_leaderboards = {}
    for s in LABELS:
        focals = [f for f in LABELS if f != s]
        scores = []
        for f in focals:
            ci = opp_names[f].index(s)
            scores.append(peer[LABELS.index(f), ci, :, 0].mean())
        ranks = rankdata([-x for x in scores])
        context_leaderboards[s] = {
            f: {"survival_pct": float(x * 100), "rank": int(r)}
            for f, x, r in zip(focals, scores, ranks)
        }
        for f, r in zip(focals, ranks):
            context_ranks[f].append(int(r))

    # agreement between single-context leaderboards (healthy focals only)
    healthy_set = set(HEALTHY)
    rhos = []
    hs = [s for s in HEALTHY]
    for a_i in range(len(hs)):
        for b_i in range(a_i + 1, len(hs)):
            sa, sb = hs[a_i], hs[b_i]
            common = [f for f in LABELS if f not in (sa, sb) and f in healthy_set]
            xa = [context_leaderboards[sa][f]["survival_pct"] for f in common]
            xb = [context_leaderboards[sb][f]["survival_pct"] for f in common]
            rhos.append(spearmanr(xa, xb).statistic)

    results = {
        "config": {"n_games_per_cell": int(peer.shape[2]), "labels": LABELS,
                   "healthy": HEALTHY, "source": "merged 1000-game peer matrix"},
        "per_policy": per_policy,
        "context_rank_range": {
            p: [int(min(r)), int(max(r))] for p, r in context_ranks.items()
        },
        "context_leaderboard_spearman_healthy": {
            "mean": float(np.mean(rhos)),
            "min": float(np.min(rhos)),
            "max": float(np.max(rhos)),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "population_sensitivity.json", "w") as fh:
        json.dump(results, fh, indent=1)

    lines = [
        "# E12 rebase: population sensitivity, best-config peer matrix (1000 games/cell)",
        "",
        "## Same policy, different population: survival / win spread",
        "",
        "| policy | selfplay surv | peer surv min-max | peer win min-max |",
        "|---|---|---|---|",
    ]
    for p in LABELS:
        d = per_policy[p]
        sp = d["contexts"]["selfplay"]["survival_pct"]
        lines.append(
            f"| {p} | {sp:.1f} | {d['peer_survival_min']:.1f} - "
            f"{d['peer_survival_max']:.1f} | {d['peer_win_min']:.1f} - "
            f"{d['peer_win_max']:.1f} |"
        )
    lines += [
        "",
        "## Rank a policy receives depending on which single population evaluates it",
        "",
        "| policy | best rank | worst rank |",
        "|---|---|---|",
    ]
    for p in LABELS:
        lo, hi = results["context_rank_range"][p]
        lines.append(f"| {p} | {lo} | {hi} |")
    cr = results["context_leaderboard_spearman_healthy"]
    lines += [
        "",
        "## Do two defensible single-population evaluations agree? (healthy strangers)",
        "",
        f"Spearman between single-context leaderboards over shared healthy focals: "
        f"mean {cr['mean']:.2f}, min {cr['min']:.2f}, max {cr['max']:.2f}",
        "",
    ]
    with open(OUT_DIR / "population_sensitivity_summary.md", "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
