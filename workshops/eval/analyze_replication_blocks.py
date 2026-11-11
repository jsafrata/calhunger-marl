"""E21: ten independent replications of the peer cross-play protocol (TAE).

The merged 1000-game peer matrix (E19 extension) splits into TEN DISJOINT
100-game seed blocks (games 0-99 = the original campaign at seed base
900000; games 100k..100k+99 = seeds 900000+100k..). Each block is a
genuinely independent run of the original evaluation protocol — same
budget, same cells, fresh seeds — not a bootstrap simulation.

Per block we re-derive the protocol's published conclusions and ask which
replicate:
  1. leaderboard (healthy-8 macro-mean survival): top-1 identity, and
     Spearman vs the merged 1000-game reference ranking;
  2. pairwise verdicts: agreement of each of the 28 healthy-pair signs
     with the 1000-game reference sign;
  3. E17 ladder validity: Spearman of the (fixed, 100-game) ladder mean
     v2-v5 survival vs the block's peer off-diagonal survival, healthy 8 —
     does any block reach n=8 significance (|rho| >= 0.738)?

Usage (repo outer root):
    python3 workshops/eval/analyze_replication_blocks.py
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import rankdata, spearmanr

from crossplay_raw_common import (
    HEALTHY,
    LABELS,
    RESULTS_ROOT,
    load_ladder,
    load_peer_1v3,
    stack_values,
)

OUT_DIR = RESULTS_ROOT / "replication_blocks"
BLOCK = 100
SIG_RHO_N8 = 0.738  # two-tailed 5% critical value for Spearman at n=8


def main() -> None:
    peer = stack_values(load_peer_1v3(extended=True), LABELS)  # (10, 9, 1000, Q)
    ladder = stack_values(load_ladder(), LABELS)  # (10, 5, 100, Q)
    n_games = peer.shape[2]
    assert n_games % BLOCK == 0
    n_blocks = n_games // BLOCK
    healthy_ix = np.array([LABELS.index(p) for p in HEALTHY])

    surv = peer[..., 0]  # (10, 9, 1000)
    ref_scores = surv.mean(axis=2).mean(axis=1)  # merged 1000-game reference
    ref_rank_h = rankdata(-ref_scores[healthy_ix])
    ref_top1 = HEALTHY[int(np.argmax(ref_scores[healthy_ix]))]
    iu = np.triu_indices(len(healthy_ix), k=1)
    ref_sign = np.sign(
        ref_scores[healthy_ix][iu[0]] - ref_scores[healthy_ix][iu[1]]
    )

    ladder_score = ladder[..., 0].mean(axis=2)[:, 1:].mean(axis=1)  # v2-v5 mean

    blocks = []
    for b in range(n_blocks):
        sl = surv[:, :, b * BLOCK : (b + 1) * BLOCK]
        scores = sl.mean(axis=2).mean(axis=1)
        h = scores[healthy_ix]
        top1 = HEALTHY[int(np.argmax(h))]
        rho_ref = spearmanr(rankdata(-h), ref_rank_h).statistic
        agree = int((np.sign(h[iu[0]] - h[iu[1]]) == ref_sign).sum())
        e17 = spearmanr(ladder_score[healthy_ix], h)
        blocks.append(
            {
                "block": b,
                "seed_range": [900000 + b * BLOCK, 900000 + (b + 1) * BLOCK - 1],
                "is_original_campaign": b == 0,
                "top1": top1,
                "top1_matches_reference": top1 == ref_top1,
                "spearman_vs_reference": float(rho_ref),
                "pairwise_agreement": f"{agree}/{len(ref_sign)}",
                "e17_rho": float(e17.statistic),
                "e17_significant": bool(abs(e17.statistic) >= SIG_RHO_N8),
                "healthy_scores": {p: float(v) for p, v in zip(HEALTHY, h)},
            }
        )

    top1_hits = sum(x["top1_matches_reference"] for x in blocks)
    e17_sig = sum(x["e17_significant"] for x in blocks)
    e17_rhos = [x["e17_rho"] for x in blocks]
    pair_agree = [int(x["pairwise_agreement"].split("/")[0]) for x in blocks]
    results = {
        "config": {"n_blocks": n_blocks, "block_size": BLOCK,
                   "reference": "merged 1000-game leaderboard",
                   "labels": LABELS, "healthy": HEALTHY},
        "reference_top1": ref_top1,
        "blocks": blocks,
        "summary": {
            "top1_reproduced": f"{top1_hits}/{n_blocks}",
            "mean_spearman_vs_reference": float(
                np.mean([x["spearman_vs_reference"] for x in blocks])
            ),
            "min_spearman_vs_reference": float(
                np.min([x["spearman_vs_reference"] for x in blocks])
            ),
            "mean_pairwise_agreement": float(np.mean(pair_agree)),
            "min_pairwise_agreement": int(np.min(pair_agree)),
            "e17_rho_range": [float(np.min(e17_rhos)), float(np.max(e17_rhos))],
            "e17_significant_blocks": f"{e17_sig}/{n_blocks}",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "replication_blocks.json", "w") as fh:
        json.dump(results, fh, indent=1)

    s = results["summary"]
    lines = [
        "# E21: ten independent 100-game replications of the peer protocol",
        "",
        "Blocks are disjoint 100-seed slices of the merged 1000-game matrix; block 0",
        "is the original campaign. Reference = merged 1000-game leaderboard "
        f"(top-1 {ref_top1}). Healthy-8 survival metric throughout.",
        "",
        "| block | seeds | top-1 | =ref? | rho vs ref | pair agree | E17 rho | E17 sig? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in blocks:
        tag = " (orig)" if x["is_original_campaign"] else ""
        lines.append(
            f"| {x['block']}{tag} | {x['seed_range'][0]}-{x['seed_range'][1]} | "
            f"{x['top1']} | {'Y' if x['top1_matches_reference'] else 'N'} | "
            f"{x['spearman_vs_reference']:.3f} | {x['pairwise_agreement']} | "
            f"{x['e17_rho']:+.2f} | {'YES' if x['e17_significant'] else 'no'} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- top-1 reproduced in {s['top1_reproduced']} replications",
        f"- Spearman vs reference: mean {s['mean_spearman_vs_reference']:.3f}, "
        f"min {s['min_spearman_vs_reference']:.3f}",
        f"- pairwise verdict agreement with reference: mean "
        f"{s['mean_pairwise_agreement']:.1f}/28, min {s['min_pairwise_agreement']}/28",
        f"- E17 ladder-validity rho across replications: "
        f"[{s['e17_rho_range'][0]:+.2f}, {s['e17_rho_range'][1]:+.2f}]; significant in "
        f"{s['e17_significant_blocks']} (threshold |rho|>=0.738 at n=8)",
        "",
    ]
    with open(OUT_DIR / "replication_blocks_summary.md", "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
