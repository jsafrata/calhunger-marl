"""E27: leaderboards as social choice — aggregation & manipulability (EconML).

Question: if each evaluation population (and each metric) is a *voter* over
policies, what do social-choice tools say about the leaderboard?

Voters
------
- Population-voters (E12 convention, ``analyze_population_sensitivity.py``):
  judge j = the population "3x j"; it ranks every focal policy i != j by
  survival M[i, j] on the merged 1000-game peer matrix. The self-play
  diagonal is NOT a voter (E12's context leaderboards exclude it), so each
  judge casts a partial ballot omitting itself. Two judge universes:
  all-10 judges (mean over them == the status-quo 9-cell macro-mean
  leaderboard exactly) and healthy-8 judges.
- Metric-voters (E20 convention, ``analyze_metric_disagreement.py``): the 6
  peer metrics (survival, win, cash, cash|survival, survival_time,
  win|survival), each a complete ballot over the candidates via its
  macro-mean over the 9 focal 1v3 cells.

Rules
-----
mean-score (status quo; population-voters only), Borda (pairwise-support
form, sum_b P[a,b] — exact Borda for complete ballots, its natural extension
for partial ones), Copeland (wins + 0.5*ties), plurality (each voter's top-1
among the candidates it ranks; argmax ties broken by LABELS order), Kemeny
(exact via Held-Karp DP over subsets, ties between optimal orders broken
deterministically by candidate index).

Bootstrap (paired scheme of ``analyze_bootstrap_leaderboard_stability.py``):
B=2000 replicates, one n=1000 seed-index vector shared by ALL cells per
replicate, rng seed 20260830. Per rule x voter-set: top-1 stability (winner
== full-data winner) and self-Spearman of the replicate ranking vs the
full-data ranking of the same rule.

Sanity checks: the mean-score leaderboard is compared elementwise against
``results/leaderboard_stability_1000/bootstrap_stability.json``; crownability
ranks are compared against
``results/population_sensitivity_1000/population_sensitivity.json``.

Usage (repo outer root):
    python3 workshops/econml/analyze_social_choice.py
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))

from crossplay_raw_common import (  # noqa: E402
    HEALTHY,
    LABELS,
    RESULTS_ROOT,
    load_peer_1v3,
    stack_values,
)
from analyze_metric_disagreement import PEER_METRICS, peer_scores  # noqa: E402

OUT_DIR = HERE / "results" / "social_choice"
B = 2000
RNG_SEED = 20260830
N_POL = len(LABELS)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build_matrices():
    """Return (peer_surv, col_of, M) from the merged 1000-game matrix.

    peer_surv: (10, 9, 1000) per-game focal survival, cell order as loaded;
    col_of[i][j]: cell index of focal i vs 3x LABELS[j];
    M: (10, 10) survival matrix, M[i, j] = mean survival of focal i vs 3x j,
    diagonal NaN (self-play excluded, E12 convention).
    """
    cells = load_peer_1v3(extended=True)
    peer = stack_values(cells, LABELS)  # (10, 9, 1000, Q)
    peer_surv = peer[:, :, :, 0]
    col_of = {}
    M = np.full((N_POL, N_POL), np.nan)
    for i, p in enumerate(LABELS):
        col_of[i] = {}
        for ci, cell in enumerate(cells[p]):
            j = LABELS.index(cell.opponent)
            col_of[i][j] = ci
            M[i, j] = cell.values[:, 0].mean()
    return peer, peer_surv, col_of, M


def boot_matrix(peer_surv: np.ndarray, col_of, idx: np.ndarray) -> np.ndarray:
    """Paired-resample survival matrices: (B, 10, 10), diagonal NaN."""
    b = idx.shape[0]
    Mb = np.full((b, N_POL, N_POL), np.nan)
    for i in range(N_POL):
        for j, ci in col_of[i].items():
            Mb[:, i, j] = peer_surv[i, ci][idx].mean(axis=1)
    return Mb


# ---------------------------------------------------------------------------
# ballots -> pairwise preference counts
# ---------------------------------------------------------------------------

def pop_pref_counts(M: np.ndarray, cand: list[int], judges: list[int]):
    """P[..., a, b] = #judges (excluding a, b themselves) with M[a,j] > M[b,j].

    Also returns n_valid[a, b] = #judges eligible to compare the pair
    (judges j not in {cand a, cand b}); ties count for neither side.
    """
    n = len(cand)
    lead = M.shape[:-2]
    P = np.zeros(lead + (n, n))
    n_valid = np.zeros((n, n), dtype=int)
    for ai, bi in itertools.combinations(range(n), 2):
        a, b = cand[ai], cand[bi]
        js = [j for j in judges if j not in (a, b)]
        n_valid[ai, bi] = n_valid[bi, ai] = len(js)
        va = M[..., a, js]
        vb = M[..., b, js]
        P[..., ai, bi] = (va > vb).sum(axis=-1)
        P[..., bi, ai] = (vb > va).sum(axis=-1)
    return P, n_valid


def metric_pref_counts(S: np.ndarray, cand: list[int]):
    """S: (..., 10, n_metrics); every metric ranks every candidate."""
    n = len(cand)
    lead = S.shape[:-2]
    P = np.zeros(lead + (n, n))
    n_valid = np.full((n, n), S.shape[-1], dtype=int)
    np.fill_diagonal(n_valid, 0)
    for ai, bi in itertools.combinations(range(n), 2):
        va = S[..., cand[ai], :]
        vb = S[..., cand[bi], :]
        # NaN-safe: NaN comparisons are False on both sides == tie
        with np.errstate(invalid="ignore"):
            P[..., ai, bi] = (va > vb).sum(axis=-1)
            P[..., bi, ai] = (vb > va).sum(axis=-1)
    return P, n_valid


# ---------------------------------------------------------------------------
# rules (scores: higher = better; all vectorized over leading axes)
# ---------------------------------------------------------------------------

def borda_scores(P: np.ndarray) -> np.ndarray:
    return P.sum(axis=-1)


def copeland_scores(P: np.ndarray) -> np.ndarray:
    Pt = np.swapaxes(P, -1, -2)
    wins = (P > Pt).sum(axis=-1).astype(float)
    ties = ((P == Pt).sum(axis=-1) - 1).astype(float)  # minus the diagonal
    return wins + 0.5 * ties


def pop_plurality_scores(M: np.ndarray, cand: list[int], judges: list[int]):
    """Top-1 votes; each judge votes for its best candidate != itself."""
    lead = M.shape[:-2]
    counts = np.zeros(lead + (len(cand),))
    for j in judges:
        elig = [ci for ci, c in enumerate(cand) if c != j]
        vals = M[..., [cand[ci] for ci in elig], j]
        top = np.argmax(vals, axis=-1)  # ties -> first (LABELS order)
        pick = np.asarray(np.take(np.array(elig), top)).reshape(-1)
        np.add.at(counts.reshape(-1, len(cand)),
                  (np.arange(pick.size), pick), 1.0)
    return counts


def metric_plurality_scores(S: np.ndarray, cand: list[int]):
    lead = S.shape[:-2]
    counts = np.zeros(lead + (len(cand),))
    vals = S[..., cand, :]  # (..., n_cand, n_metrics)
    with np.errstate(invalid="ignore"):
        top = np.nanargmax(vals, axis=-2)  # (..., n_metrics)
    flat = top.reshape(-1, vals.shape[-1])
    cflat = counts.reshape(-1, len(cand))
    for m in range(vals.shape[-1]):
        np.add.at(cflat, (np.arange(flat.shape[0]), flat[:, m]), 1.0)
    return counts


def pop_mean_scores(M: np.ndarray, cand: list[int], judges: list[int]):
    """Mean survival over the judge columns (self excluded via NaN)."""
    cols = M[..., cand, :][..., :, judges]
    return np.nanmean(cols, axis=-1)


def kemeny_orders(P: np.ndarray):
    """Exact Kemeny via Held-Karp DP over subsets.

    P: (B, n, n) or (n, n) pairwise preference counts. Returns
    (orders, cost): orders (B, n) candidate indices best-to-worst,
    cost (B,) the Kemeny score (total weighted disagreements).
    Deterministic tie-break: lowest candidate index wins the DP argmin.
    """
    single = P.ndim == 2
    if single:
        P = P[None]
    b, n, _ = P.shape
    nf = 1 << n
    totcol = P.sum(axis=1)  # (b, n): sum_d P[d, c]
    dp = np.full((nf, b), np.inf)
    dp[0] = 0.0
    parent = np.zeros((nf, b), dtype=np.int8)
    colsum = np.zeros((nf, b, n))  # sum_{d in S} P[d, c]
    for S in range(1, nf):
        c0 = (S & -S).bit_length() - 1
        S0 = S ^ (1 << c0)
        colsum[S] = colsum[S0] + P[:, c0, :]
        rem = S
        while rem:
            c = (rem & -rem).bit_length() - 1
            rem ^= 1 << c
            R = S ^ (1 << c)
            cost = dp[R] + totcol[:, c] - colsum[S, :, c]
            better = cost < dp[S]
            dp[S][better] = cost[better]
            parent[S][better] = c
    orders = np.empty((b, n), dtype=int)
    for bi in range(b):
        S = nf - 1
        for pos in range(n - 1, -1, -1):
            c = int(parent[S, bi])
            orders[bi, pos] = c
            S ^= 1 << c
    cost = dp[nf - 1]
    if single:
        return orders[0], float(cost[0])
    return orders, cost


# ---------------------------------------------------------------------------
# majority structure
# ---------------------------------------------------------------------------

def majority_structure(P: np.ndarray, n_valid: np.ndarray, names: list[str]):
    n = len(names)
    beats = P > np.swapaxes(P, -1, -2)  # strict majority
    tied = (P == np.swapaxes(P, -1, -2))
    edges, ties = [], []
    for ai, bi in itertools.combinations(range(n), 2):
        rec = {
            "pair": [names[ai], names[bi]],
            "votes": [int(P[ai, bi]), int(P[bi, ai])],
            "n_judges": int(n_valid[ai, bi]),
        }
        if beats[ai, bi]:
            edges.append({**rec, "winner": names[ai]})
        elif beats[bi, ai]:
            edges.append({**rec, "winner": names[bi]})
        else:
            ties.append(rec)
    condorcet = None
    weak = []
    for ai in range(n):
        if all(beats[ai, bi] for bi in range(n) if bi != ai):
            condorcet = names[ai]
        if all(beats[ai, bi] or tied[ai, bi] for bi in range(n) if bi != ai):
            weak.append(names[ai])  # loses no pairwise majority
    cycles = []
    for ai, bi, ci in itertools.combinations(range(n), 3):
        tri = [(ai, bi), (bi, ci), (ci, ai)]
        fwd = all(beats[x, y] for x, y in tri)
        bwd = all(beats[y, x] for x, y in tri)
        if fwd or bwd:
            cycles.append([names[ai], names[bi], names[ci]])
    return {
        "condorcet_winner": condorcet,
        "weak_condorcet_winners": weak,
        "n_cycle_triples": len(cycles),
        "n_triples": n * (n - 1) * (n - 2) // 6,
        "cycle_triples": cycles,
        "n_majority_ties": len(ties),
        "majority_ties": ties,
        "majority_edges": edges,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ranking_from_scores(scores: np.ndarray, names: list[str]):
    order = np.argsort(-scores, kind="stable")
    return [names[i] for i in order], order


def spearman_rows(sample_scores: np.ndarray, full_scores: np.ndarray) -> np.ndarray:
    """Spearman rho of each replicate row (B, P) against full (P,). (E19)"""
    r_full = rankdata(full_scores)
    r_samp = rankdata(sample_scores, axis=1)
    r_full = (r_full - r_full.mean()) / r_full.std()
    with np.errstate(invalid="ignore", divide="ignore"):
        r_samp = (r_samp - r_samp.mean(axis=1, keepdims=True)) / r_samp.std(
            axis=1, keepdims=True
        )
        return (r_samp * r_full).mean(axis=1)


def order_to_scores(orders: np.ndarray, n: int) -> np.ndarray:
    """Kemeny orders (B, n) -> score arrays (higher = better)."""
    b = orders.shape[0]
    scores = np.empty((b, n))
    pos = np.arange(n)
    for bi in range(b):
        scores[bi, orders[bi]] = n - pos
    return scores


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    peer, peer_surv, col_of, M = build_matrices()
    n_games = peer.shape[2]
    all10 = list(range(N_POL))
    healthy_ix = [LABELS.index(p) for p in HEALTHY]

    # metric-voter scores (macro-mean over the 9 focal cells; E20 convention)
    S_full = peer_scores(peer, None)  # (10, 6)

    # ---------------- sanity check 1: mean-score leaderboard ---------------
    ref_path = RESULTS_ROOT / "leaderboard_stability_1000" / "bootstrap_stability.json"
    with open(ref_path) as fh:
        ref = json.load(fh)["full_scores"]["survival"]
    mean_all10 = pop_mean_scores(M, all10, all10)  # == 9-cell macro-mean
    max_abs_err = max(abs(mean_all10[i] - ref[p]) for i, p in enumerate(LABELS))
    ref_order = [p for p, _ in sorted(ref.items(), key=lambda kv: -kv[1])]
    my_order, _ = ranking_from_scores(mean_all10, LABELS)
    sanity_mean = {
        "reference": str(ref_path.relative_to(RESULTS_ROOT.parent)),
        "max_abs_score_diff": float(max_abs_err),
        "ordering_matches": my_order == ref_order,
        "ordering": my_order,
    }
    assert max_abs_err < 1e-9 and my_order == ref_order, sanity_mean

    # ---------------- crownability (E12 convention) ------------------------
    # all-10: judge s ranks the 9 focals f != s by survival (rankdata on -x,
    # exactly as analyze_population_sensitivity.py).
    context_ranks = {p: [] for p in LABELS}
    crowns_all10 = {p: [] for p in LABELS}
    for j in all10:
        focals = [f for f in all10 if f != j]
        scores = M[focals, j]
        ranks = rankdata(-scores)
        for f, r in zip(focals, ranks):
            context_ranks[LABELS[f]].append(int(r))
            if r == 1:
                crowns_all10[LABELS[f]].append(LABELS[j])
    my_rank_range = {p: [min(r), max(r)] for p, r in context_ranks.items()}
    ps_path = RESULTS_ROOT / "population_sensitivity_1000" / "population_sensitivity.json"
    with open(ps_path) as fh:
        ps_ref = json.load(fh)["context_rank_range"]
    sanity_crown = {
        "reference": str(ps_path.relative_to(RESULTS_ROOT.parent)),
        "rank_range_matches": {p: my_rank_range[p] == ps_ref[p] for p in LABELS},
        "all_match": all(my_rank_range[p] == ps_ref[p] for p in LABELS),
    }
    assert sanity_crown["all_match"], sanity_crown

    # healthy-restricted: healthy judges, healthy candidates
    crowns_h8 = {p: [] for p in HEALTHY}
    ranks_h8 = {p: [] for p in HEALTHY}
    for j in healthy_ix:
        focals = [f for f in healthy_ix if f != j]
        scores = M[focals, j]
        ranks = rankdata(-scores)
        for f, r in zip(focals, ranks):
            ranks_h8[LABELS[f]].append(int(r))
            if r == 1:
                crowns_h8[LABELS[f]].append(LABELS[j])

    with open(OUT_DIR / "crownable.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "policy", "crownable", "crowned_by",
                    "best_rank", "worst_rank", "n_judges"])
        for p in LABELS:
            w.writerow(["all10", p, bool(crowns_all10[p]),
                        ";".join(crowns_all10[p]),
                        min(context_ranks[p]), max(context_ranks[p]), 9])
        for p in HEALTHY:
            w.writerow(["healthy8", p, bool(crowns_h8[p]),
                        ";".join(crowns_h8[p]),
                        min(ranks_h8[p]), max(ranks_h8[p]), 7])

    # ---------------- voter sets -------------------------------------------
    voter_sets = {}  # name -> dict with P, n_valid, names, extra score fns
    specs = [
        ("pop_all10judges_healthy8", healthy_ix, "pop", all10),
        ("pop_healthy8judges_healthy8", healthy_ix, "pop", healthy_ix),
        ("pop_all10judges_all10", all10, "pop", all10),
        ("metrics6_healthy8", healthy_ix, "metric", None),
        ("metrics6_all10", all10, "metric", None),
    ]
    for name, cand, kind, judges in specs:
        cnames = [LABELS[c] for c in cand]
        if kind == "pop":
            P, n_valid = pop_pref_counts(M, cand, judges)
        else:
            P, n_valid = metric_pref_counts(S_full, cand)
        voter_sets[name] = {
            "cand": cand, "names": cnames, "kind": kind, "judges": judges,
            "P": P, "n_valid": n_valid,
        }

    # ---------------- majority structure -----------------------------------
    majority = {
        "config": {
            "n_games_per_cell": int(n_games),
            "source": "merged 1000-game peer matrix (load_peer_1v3(extended=True))",
            "labels": LABELS, "healthy": HEALTHY,
            "population_voter_convention": (
                "judge j ranks focal i != j by survival M[i,j]; self-play "
                "diagonal excluded (analyze_population_sensitivity.py convention)"
            ),
            "metric_voters": PEER_METRICS,
        },
        "survival_matrix": {
            LABELS[i]: {LABELS[j]: (None if i == j else float(M[i, j]))
                        for j in all10} for i in all10
        },
        "voter_sets": {},
    }
    for name, vs in voter_sets.items():
        majority["voter_sets"][name] = majority_structure(
            vs["P"], vs["n_valid"], vs["names"]
        )

    with open(OUT_DIR / "pairwise_majority.json", "w") as fh:
        json.dump(majority, fh, indent=1)

    # ---------------- rules on full data -----------------------------------
    def full_rule_scores(vs):
        out = {}
        out["borda"] = borda_scores(vs["P"])
        out["copeland"] = copeland_scores(vs["P"])
        if vs["kind"] == "pop":
            out["plurality"] = pop_plurality_scores(M, vs["cand"], vs["judges"])
            out["mean_score"] = pop_mean_scores(M, vs["cand"], vs["judges"])
        else:
            out["plurality"] = metric_plurality_scores(S_full, vs["cand"])
        order, cost = kemeny_orders(vs["P"])
        out["kemeny"] = order_to_scores(order[None], len(vs["cand"]))[0]
        out["_kemeny_cost"] = cost
        return out

    rules_rows = []
    full_rankings = {}  # (voter_set, rule) -> (names ranked, scores)
    for name, vs in voter_sets.items():
        scores = full_rule_scores(vs)
        for rule, sc in scores.items():
            if rule.startswith("_"):
                continue
            ranked, order = ranking_from_scores(np.asarray(sc, float), vs["names"])
            full_rankings[(name, rule)] = (ranked, np.asarray(sc, float))
            for pos, ci in enumerate(order):
                rules_rows.append({
                    "voter_set": name, "rule": rule, "rank": pos + 1,
                    "policy": vs["names"][ci], "score": float(sc[ci]),
                })

    with open(OUT_DIR / "rules_ranking.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["voter_set", "rule", "rank",
                                           "policy", "score"])
        w.writeheader()
        w.writerows(rules_rows)

    # ---------------- bootstrap rule stability -----------------------------
    idx = rng.integers(0, n_games, size=(B, n_games))  # paired (E19 scheme)
    Mb = boot_matrix(peer_surv, col_of, idx)  # (B, 10, 10)
    Sb = peer_scores(peer, idx)  # (B, 10, 6) — metric voters under resampling

    boot_sets = [
        "pop_all10judges_healthy8",
        "pop_healthy8judges_healthy8",
        "metrics6_healthy8",
    ]
    stab_rows = []
    for name in boot_sets:
        vs = voter_sets[name]
        cand, judges, kind = vs["cand"], vs["judges"], vs["kind"]
        n = len(cand)
        if kind == "pop":
            Pb, _ = pop_pref_counts(Mb, cand, judges)
        else:
            Pb, _ = metric_pref_counts(Sb, cand)
        boot_scores = {
            "borda": borda_scores(Pb),
            "copeland": copeland_scores(Pb),
        }
        if kind == "pop":
            boot_scores["plurality"] = pop_plurality_scores(Mb, cand, judges)
            boot_scores["mean_score"] = pop_mean_scores(Mb, cand, judges)
        else:
            boot_scores["plurality"] = metric_plurality_scores(Sb, cand)
        orders_b, _ = kemeny_orders(Pb)
        boot_scores["kemeny"] = order_to_scores(orders_b, n)
        for rule, sb in boot_scores.items():
            full_names, full_sc = full_rankings[(name, rule)]
            full_winner_ix = int(np.argmax(full_sc))
            top1 = (np.argmax(sb, axis=1) == full_winner_ix).mean()
            rho = spearman_rows(sb, full_sc)
            stab_rows.append({
                "voter_set": name, "rule": rule,
                "B": B, "n_games_per_cell": n_games, "scheme": "paired",
                "top1_stability": float(top1),
                "mean_self_spearman": float(np.nanmean(rho)),
                "p5_self_spearman": float(np.nanpercentile(rho, 5)),
                "full_winner": full_names[0],
            })
        print(f"bootstrap done: {name}")

    with open(OUT_DIR / "rule_stability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["voter_set", "rule", "B",
                                           "n_games_per_cell", "scheme",
                                           "top1_stability",
                                           "mean_self_spearman",
                                           "p5_self_spearman", "full_winner"])
        w.writeheader()
        w.writerows(stab_rows)

    # ---------------- findings ---------------------------------------------
    write_findings(majority, full_rankings, stab_rows, crowns_all10, crowns_h8,
                   context_ranks, sanity_mean, sanity_crown, n_games)
    print(f"wrote {OUT_DIR}")


def write_findings(majority, full_rankings, stab_rows, crowns_all10, crowns_h8,
                   context_ranks, sanity_mean, sanity_crown, n_games) -> None:
    rules = ["mean_score", "borda", "copeland", "plurality", "kemeny"]
    lines = [
        "# E27. Leaderboards as social choice — aggregation & manipulability "
        "[EconML theme 1]",
        "",
        f"Merged 1000-game peer matrix (n={n_games}/cell); population-voters = "
        "single-opponent populations judging focals by survival (self-play "
        "diagonal excluded, E12 convention); metric-voters = the 6 E20 peer "
        "metrics. Rules: mean-score (status quo), Borda (pairwise-support), "
        "Copeland, plurality, exact Kemeny (Held-Karp). Bootstrap: paired "
        f"scheme, B={B}, n={n_games}, rng {RNG_SEED}.",
        "",
        "## Condorcet structure",
        "",
        "| voter set | Condorcet winner | weak CW (loses no majority) | "
        "cycle triples | majority ties |",
        "|---|---|---|---|---|",
    ]
    for name, m in majority["voter_sets"].items():
        lines.append(
            f"| {name} | {m['condorcet_winner'] or 'NONE'} | "
            f"{', '.join(m['weak_condorcet_winners']) or 'NONE'} | "
            f"{m['n_cycle_triples']}/{m['n_triples']} | {m['n_majority_ties']} |"
        )
        if m["cycle_triples"]:
            for t in m["cycle_triples"]:
                lines.append(f"|  | cycle: {' > '.join(t)} > {t[0]} | | | |")

    lines += ["", "## Rule dependence (full rankings, best to worst)", ""]
    for vs_name in majority["voter_sets"]:
        lines += [f"### {vs_name}", "", "| rule | ranking |", "|---|---|"]
        winners = set()
        for rule in rules:
            key = (vs_name, rule)
            if key not in full_rankings:
                continue
            ranked, _ = full_rankings[key]
            winners.add(ranked[0])
            lines.append(f"| {rule} | {' > '.join(ranked)} |")
        lines += ["", f"Distinct winners across rules: {sorted(winners)}", ""]

    lines += [
        "## Manipulability: which policies can some single-population judge crown #1?",
        "",
        "| group | crownable | who crowns whom |",
        "|---|---|---|",
    ]
    ca = {p: v for p, v in crowns_all10.items() if v}
    ch = {p: v for p, v in crowns_h8.items() if v}
    lines.append(
        f"| all10 (9 candidates/judge) | {len(ca)}/10 | "
        + "; ".join(f"{p} by {'+'.join(v)}" for p, v in ca.items()) + " |"
    )
    lines.append(
        f"| healthy8 (7 candidates/judge) | {len(ch)}/8 | "
        + "; ".join(f"{p} by {'+'.join(v)}" for p, v in ch.items()) + " |"
    )
    lines += [
        "",
        "Cross-check vs `results/population_sensitivity_1000/`: rank ranges "
        f"match exactly = {sanity_crown['all_match']} (e.g. mappo_s2 spans "
        f"{min(context_ranks['mappo_s2'])}-{max(context_ranks['mappo_s2'])}, "
        "matching E12's 1-9).",
        "",
        "## Rule stability under paired bootstrap (B=2000, n=1000)",
        "",
        "| voter set | rule | top-1 stable | mean self-Spearman | p5 | winner |",
        "|---|---|---|---|---|---|",
    ]
    for r in stab_rows:
        lines.append(
            f"| {r['voter_set']} | {r['rule']} | {r['top1_stability']*100:.0f}% | "
            f"{r['mean_self_spearman']:.3f} | {r['p5_self_spearman']:.3f} | "
            f"{r['full_winner']} |"
        )
    lines += [
        "",
        "## Interpretation (EconML: preference aggregation in model evaluation)",
        "",
        "A leaderboard is a social-choice object: evaluation populations (and "
        "metrics) are voters, policies are candidates, and the published "
        "ranking is an aggregation rule applied to their preferences. Here no "
        "voter set yields a strict Condorcet winner. Under the defensible "
        "healthy-8 judge universe the failure is not a technicality: the top "
        "three policies form a genuine Condorcet cycle (ippo_s2 maj-beats "
        "ippo_s1 4-2, mappo_s3 maj-beats ippo_s2 4-2, ippo_s1 maj-beats "
        "mappo_s3 4-2, and each beats all five others), so WHICH of the three "
        "is champion is decided purely by the aggregation rule: mean-score and "
        "Copeland crown ippo_s1, Borda crowns ippo_s2, Kemeny crowns mappo_s3, "
        "plurality crowns ippo_s1 — three distinct champions from the same "
        "ballots. Under the all-10 judge universe ippo_s2 is a weak Condorcet "
        "winner (loses no majority, blocked only by a 4-4 tie with mappo_s3), "
        "yet plurality still elects ippo_s1. Metric-voters split the crown "
        "between the two weak co-winners ippo_s2 (Borda, Copeland) and "
        "mappo_s2 (plurality, Kemeny) — the E20 metric disagreement expressed "
        "as an electoral outcome. Manipulability is first-order: "
        "5/8 healthy policies can be crowned #1 by choosing a single "
        "sympathetic judge population (mappo_s2 spans ranks 1-9 all-10, "
        "matching E12), so an evaluator free to pick the opponent population "
        "can strategically elect most of the field. Bootstrap: the status-quo "
        "mean-score is the MOST stable rule on its own terms (top-1 90-96%), "
        "while the rules with better axiomatic credentials are fragile "
        "(Kemeny 41%, plurality 38% top-1 under all-10 judges) — the cycle "
        "sits inside sampling noise, so rule choice and sampling interact: "
        "the leaderboard is an artifact of both.",
        "",
        "## Sanity checks",
        "",
        f"- mean-score leaderboard reproduces "
        f"`results/leaderboard_stability_1000/` exactly: ordering match = "
        f"{sanity_mean['ordering_matches']}, max |score diff| = "
        f"{sanity_mean['max_abs_score_diff']:.2e}.",
        f"- crownability rank ranges match `population_sensitivity_1000` for "
        f"all 10 policies: {sanity_crown['all_match']}.",
        "",
        "Data: `workshops/econml/results/social_choice/` "
        "(pairwise_majority.json, rules_ranking.csv, crownable.csv, "
        "rule_stability.csv); script `workshops/econml/analyze_social_choice.py` "
        f"(rng seed {RNG_SEED}).",
        "",
    ]
    with open(OUT_DIR / "FINDINGS.md", "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
