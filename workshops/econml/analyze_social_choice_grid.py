"""E27-grid: the full measurement portfolio as an electorate (EconML).

Follow-up to E27 (``analyze_social_choice.py``). There, population-voters and
metric-voters were two separate electorates. Here we build the combined voter
GRID: each (evaluation-population judge j, peer metric m) pair is ONE voter
that ranks the healthy-8 candidate policies by metric m of the 1v3 cell
"focal i vs 3x j" on the merged 1000-game peer matrix.

Voter grid
----------
Judges = the single-opponent populations (E12 convention: judge j never ranks
itself, so every candidate is judged by its 9 single-population 1v3 contexts;
the judge universe pinned by the cross-check below is the 10 populations).
Metrics = E20's six peer metrics (survival, win, cash, cash|survival,
survival_time, win|survival). Grid cells = 10 x 6 = 60 potential voters
(secondary voter set: healthy-8 judges x 6 = 48). A voter cell is DROPPED if
its whole ballot is undefined (all-NaN, e.g. cash|survival when no games
survive in any of its cells); an undefined single candidate value (NaN) makes
that ballot partial — the voter abstains on comparisons involving that
candidate (the NaN-tie convention of ``metric_pref_counts``).

Rules: mean-of-normalized-scores is ill-defined across heterogeneous metric
units, so RANK-BASED rules only — Borda (pairwise-support form), Copeland,
plurality (argmax ties -> first in LABELS order), exact Kemeny (Held-Karp,
n=8). Manipulability is now 2-D: a policy is crownable if ANY (population,
metric) voter ranks it strictly #1.

Bootstrap: identical paired scheme to E27 / E19 (one n=1000 seed-index vector
shared by ALL cells per replicate), B=2000, rng seed 20260903.

Cross-check (asserted): the grid restricted to (all populations, survival)
reproduces E27's pop_all10judges_healthy8 voter set EXACTLY — same pairwise
preference counts, same majority structure (vs the stored
``pairwise_majority.json``), same Borda/Copeland/plurality/Kemeny rankings
(vs the stored ``rules_ranking.csv``).

Hard rule honoured: writes ONLY new files (grid_*.csv/json, FINDINGS_grid.md).

Usage (repo outer root):
    python3 workshops/econml/analyze_social_choice_grid.py
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))

import analyze_social_choice as sc  # noqa: E402  (E27 module: loaders + rules)
from crossplay_raw_common import HEALTHY, LABELS  # noqa: E402
from analyze_metric_disagreement import PEER_METRICS, cell_metrics  # noqa: E402

OUT_DIR = HERE / "results" / "social_choice"
B = 2000
RNG_SEED = 20260903
N_POL = len(LABELS)
N_MET = len(PEER_METRICS)
RULES = ["borda", "copeland", "plurality", "kemeny"]

E27_HEALTHY8_CROWNABLE = 5  # E27 FINDINGS.md: healthy8 group, 5/8 crownable


# ---------------------------------------------------------------------------
# grid construction
# ---------------------------------------------------------------------------

def build_cell_metric_tensor(peer: np.ndarray, col_of) -> np.ndarray:
    """C[i, j, m] = metric m of cell (focal i vs 3x j); diagonal NaN."""
    C = np.full((N_POL, N_POL, N_MET), np.nan)
    for i in range(N_POL):
        for j, ci in col_of[i].items():
            C[i, j, :] = cell_metrics(peer[i, ci])
    return C


def boot_cell_metric_tensor(peer: np.ndarray, col_of, idx: np.ndarray) -> np.ndarray:
    """Paired-resampled tensor: (B, 10, 10, 6), diagonal NaN."""
    b = idx.shape[0]
    Cb = np.full((b, N_POL, N_POL, N_MET), np.nan)
    for i in range(N_POL):
        for j, ci in col_of[i].items():
            Cb[:, i, j, :] = cell_metrics(peer[i, ci][idx, :])
    return Cb


def grid_values(C: np.ndarray, cand: list[int], judges: list[int],
                metric_ix: list[int]) -> np.ndarray:
    """Voter-value tensor V[..., v, c]: voter v=(judge, metric) j-major.

    C: (..., 10, 10, 6). Returns (..., n_judges*n_metrics, n_cand).
    """
    sub = C[..., cand, :, :]            # (..., n_cand, 10, 6)
    sub = sub[..., :, judges, :]        # (..., n_cand, n_j, 6)
    sub = sub[..., :, :, metric_ix]     # (..., n_cand, n_j, n_m)
    V = np.moveaxis(sub, -3, -1)        # (..., n_j, n_m, n_cand)
    lead = V.shape[:-3]
    return V.reshape(lead + (len(judges) * len(metric_ix), len(cand)))


def grid_voter_names(judges: list[int], metric_ix: list[int]) -> list[str]:
    return [f"{LABELS[j]}|{PEER_METRICS[m]}" for j in judges for m in metric_ix]


def grid_pref_counts(V: np.ndarray):
    """P[..., a, b] = #voters with v_a > v_b (NaN comparisons abstain).

    For 2-D V also returns n_valid[a, b] = #voters ranking BOTH a and b
    (both values defined); ties count for neither side, matching E27.
    """
    n = V.shape[-1]
    lead = V.shape[:-2]
    P = np.zeros(lead + (n, n))
    for ai, bi in itertools.combinations(range(n), 2):
        va = V[..., :, ai]
        vb = V[..., :, bi]
        with np.errstate(invalid="ignore"):
            P[..., ai, bi] = (va > vb).sum(axis=-1)
            P[..., bi, ai] = (vb > va).sum(axis=-1)
    n_valid = None
    if V.ndim == 2:
        ok = (~np.isnan(V)).astype(int)  # (n_voters, n_cand)
        n_valid = ok.T @ ok
        np.fill_diagonal(n_valid, 0)
    return P, n_valid


def grid_plurality_scores(V: np.ndarray) -> np.ndarray:
    """Each voter votes its top candidate among those it ranks.

    NaN -> -inf; argmax ties -> first index (LABELS order, E27 convention).
    Voters with an all-undefined ballot (possible in a bootstrap replicate)
    cast no vote.
    """
    n = V.shape[-1]
    lead = V.shape[:-2]
    Vf = np.where(np.isnan(V), -np.inf, V)
    top = np.argmax(Vf, axis=-1)                      # (..., n_voters)
    has = np.isfinite(np.max(Vf, axis=-1))            # voter ranks someone
    counts = np.zeros(lead + (n,))
    cflat = counts.reshape(-1, n)
    tflat = top.reshape(-1, top.shape[-1])
    hflat = has.reshape(-1, top.shape[-1])
    rows = np.arange(cflat.shape[0])
    for v in range(tflat.shape[1]):
        np.add.at(cflat, (rows, tflat[:, v]), hflat[:, v].astype(float))
    return counts


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    peer, peer_surv, col_of, M = sc.build_matrices()
    n_games = peer.shape[2]
    all10 = list(range(N_POL))
    healthy_ix = [LABELS.index(p) for p in HEALTHY]
    all_metrics = list(range(N_MET))
    cnames = list(HEALTHY)

    C = build_cell_metric_tensor(peer, col_of)

    # ---------------- voter sets (the grid) --------------------------------
    specs = {
        "grid_pop10xmetrics6_healthy8": (healthy_ix, all10, all_metrics),
        "grid_pop8xmetrics6_healthy8": (healthy_ix, healthy_ix, all_metrics),
    }
    voter_sets = {}
    for name, (cand, judges, mix) in specs.items():
        V = grid_values(C, cand, judges, mix)         # (n_voters, n_cand)
        vnames = grid_voter_names(judges, mix)
        # drop voter cells whose entire ballot is undefined
        defined = ~np.all(np.isnan(V), axis=-1)
        dropped = [vnames[i] for i in np.where(~defined)[0]]
        keep = np.where(defined)[0]
        V = V[keep]
        vnames = [vnames[i] for i in keep]
        # partial-ballot bookkeeping: NaN candidate values beyond the
        # structural self-exclusion (judge j never ranks candidate j)
        selfmask = np.array(
            [[cand[c] == judges[v // len(mix)] for c in range(len(cand))]
             for v in keep], dtype=bool)
        nan_cells = int((np.isnan(V) & ~selfmask).sum())
        P, n_valid = grid_pref_counts(V)
        voter_sets[name] = {
            "cand": cand, "judges": judges, "metric_ix": mix,
            "names": cnames if cand == healthy_ix else [LABELS[c] for c in cand],
            "V": V, "voter_names": vnames, "P": P, "n_valid": n_valid,
            "n_potential_voters": len(judges) * len(mix),
            "n_voters": len(vnames), "dropped_voters": dropped,
            "n_undefined_cells": nan_cells,
        }

    # ---------------- cross-check vs E27 (asserted) ------------------------
    # grid restricted to (all populations, survival) == pop_all10judges_healthy8
    surv_ix = [PEER_METRICS.index("survival")]
    V_surv = grid_values(C, healthy_ix, all10, surv_ix)
    P_surv, nv_surv = grid_pref_counts(V_surv)
    P_pop, nv_pop = sc.pop_pref_counts(M, healthy_ix, all10)
    assert np.array_equal(P_surv, P_pop), "pref counts differ from E27"
    assert np.array_equal(nv_surv, nv_pop), "n_valid differs from E27"

    ms_surv = sc.majority_structure(P_surv, nv_surv, cnames)
    with open(OUT_DIR / "pairwise_majority.json") as fh:
        ref_ms = json.load(fh)["voter_sets"]["pop_all10judges_healthy8"]
    ms_surv_json = json.loads(json.dumps(ms_surv))
    assert ms_surv_json == ref_ms, "majority structure differs from stored E27"

    # rule rankings on the restricted grid vs stored E27 rules_ranking.csv
    ref_rules: dict[str, list[str]] = {}
    with open(OUT_DIR / "rules_ranking.csv") as fh:
        for row in csv.DictReader(fh):
            if row["voter_set"] == "pop_all10judges_healthy8" and row["rule"] in RULES:
                ref_rules.setdefault(row["rule"], []).append(row["policy"])
    surv_rank = {}
    surv_rank["borda"], _ = sc.ranking_from_scores(sc.borda_scores(P_surv), cnames)
    surv_rank["copeland"], _ = sc.ranking_from_scores(sc.copeland_scores(P_surv), cnames)
    surv_rank["plurality"], _ = sc.ranking_from_scores(
        grid_plurality_scores(V_surv), cnames)
    ko, _ = sc.kemeny_orders(P_surv)
    surv_rank["kemeny"], _ = sc.ranking_from_scores(
        sc.order_to_scores(ko[None], len(cnames))[0], cnames)
    rules_match = {r: surv_rank[r] == ref_rules[r] for r in RULES}
    assert all(rules_match.values()), f"restricted-grid rules differ: {rules_match}"
    crosscheck = {
        "restriction": "(all 10 populations, survival) column of the grid",
        "reference": "E27 pop_all10judges_healthy8",
        "pref_counts_equal": True,
        "majority_structure_equal": True,
        "rule_rankings_equal": rules_match,
    }
    print("cross-check vs E27 passed (pref counts, majority structure, "
          f"rules {sorted(rules_match)})")

    # ---------------- majority structure on the grid -----------------------
    majority = {
        "config": {
            "n_games_per_cell": int(n_games),
            "source": "merged 1000-game peer matrix (load_peer_1v3(extended=True))",
            "labels": LABELS, "healthy": HEALTHY,
            "grid_voter_convention": (
                "voter (j, m) ranks healthy candidates i != j by metric m of "
                "cell (i vs 3x j); self-play diagonal excluded (E12); NaN "
                "candidate values (e.g. cash|survival with 0 survivors) make "
                "the ballot partial; all-NaN ballots are dropped"
            ),
            "metrics": PEER_METRICS,
            "judge_universe_note": (
                "the cross-check pins the judge universe to the 10 "
                "populations (each candidate is judged by its 9 "
                "single-population 1v3 contexts); healthy-8-judges grid "
                "reported as a secondary voter set"
            ),
        },
        "crosscheck_vs_E27": crosscheck,
        "voter_sets": {},
    }
    for name, vs in voter_sets.items():
        entry = sc.majority_structure(vs["P"], vs["n_valid"], vs["names"])
        entry["n_potential_voters"] = vs["n_potential_voters"]
        entry["n_voters"] = vs["n_voters"]
        entry["dropped_voters"] = vs["dropped_voters"]
        entry["n_undefined_candidate_cells"] = vs["n_undefined_cells"]
        majority["voter_sets"][name] = entry
    majority["voter_sets"]["grid_restricted_pop10_survival"] = ms_surv_json

    with open(OUT_DIR / "grid_pairwise_majority.json", "w") as fh:
        json.dump(majority, fh, indent=1)

    # ---------------- rules on full data -----------------------------------
    rules_rows = []
    full_rankings = {}
    for name, vs in voter_sets.items():
        scores = {
            "borda": sc.borda_scores(vs["P"]),
            "copeland": sc.copeland_scores(vs["P"]),
            "plurality": grid_plurality_scores(vs["V"]),
        }
        order, _cost = sc.kemeny_orders(vs["P"])
        scores["kemeny"] = sc.order_to_scores(order[None], len(vs["names"]))[0]
        for rule in RULES:
            sc_arr = np.asarray(scores[rule], float)
            ranked, order_ix = sc.ranking_from_scores(sc_arr, vs["names"])
            full_rankings[(name, rule)] = (ranked, sc_arr)
            for pos, ci in enumerate(order_ix):
                rules_rows.append({
                    "voter_set": name, "rule": rule, "rank": pos + 1,
                    "policy": vs["names"][ci], "score": float(sc_arr[ci]),
                })

    with open(OUT_DIR / "grid_rules_ranking.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["voter_set", "rule", "rank",
                                           "policy", "score"])
        w.writeheader()
        w.writerows(rules_rows)

    # ---------------- 2-D crownability -------------------------------------
    # main grid: does ANY (population, metric) voter rank policy p strictly #1?
    vs_main = voter_sets["grid_pop10xmetrics6_healthy8"]
    crowns = {p: [] for p in cnames}
    rank_range = {p: [np.inf, -np.inf] for p in cnames}
    for vi, vname in enumerate(vs_main["voter_names"]):
        vals = vs_main["V"][vi]
        ok = ~np.isnan(vals)
        r = rankdata(-vals[ok])
        for c, rr in zip(np.where(ok)[0], r):
            p = cnames[c]
            rank_range[p][0] = min(rank_range[p][0], rr)
            rank_range[p][1] = max(rank_range[p][1], rr)
            if rr == 1:
                crowns[p].append(vname)

    with open(OUT_DIR / "grid_crownable.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "crownable", "n_crowning_voters",
                    "crowning_judges", "crowning_metrics", "crowned_by",
                    "best_rank", "worst_rank"])
        for p in cnames:
            judges_p = sorted({v.split("|")[0] for v in crowns[p]})
            mets_p = sorted({v.split("|")[1] for v in crowns[p]})
            w.writerow([p, bool(crowns[p]), len(crowns[p]),
                        ";".join(judges_p), ";".join(mets_p),
                        ";".join(crowns[p]),
                        rank_range[p][0], rank_range[p][1]])

    crownable = [p for p in cnames if crowns[p]]
    surv_crowned = [p for p in cnames
                    if any(v.endswith("|survival") for v in crowns[p])]
    print(f"crownable on the grid: {len(crownable)}/8 = {crownable}")

    # ---------------- bootstrap rule stability -----------------------------
    idx = rng.integers(0, n_games, size=(B, n_games))  # paired (E19 scheme)
    Cb = boot_cell_metric_tensor(peer, col_of, idx)    # (B, 10, 10, 6)

    stab_rows = []
    for name, vs in voter_sets.items():
        Vb = grid_values(Cb, vs["cand"], vs["judges"], vs["metric_ix"])
        # keep the same voter cells as the full-data grid (dropped ones out)
        keep = [i for i, vn in enumerate(
            grid_voter_names(vs["judges"], vs["metric_ix"]))
            if vn in set(vs["voter_names"])]
        Vb = Vb[:, keep, :]
        Pb, _ = grid_pref_counts(Vb)
        boot_scores = {
            "borda": sc.borda_scores(Pb),
            "copeland": sc.copeland_scores(Pb),
            "plurality": grid_plurality_scores(Vb),
        }
        orders_b, _ = sc.kemeny_orders(Pb)
        boot_scores["kemeny"] = sc.order_to_scores(orders_b, len(vs["names"]))
        for rule in RULES:
            sb = boot_scores[rule]
            full_names, full_sc = full_rankings[(name, rule)]
            full_winner_ix = int(np.argmax(full_sc))
            boot_win = np.argmax(sb, axis=1)
            top1 = (boot_win == full_winner_ix).mean()
            wc = np.bincount(boot_win, minlength=len(vs["names"]))
            alt = ";".join(
                f"{vs['names'][i]}:{wc[i] / B * 100:.1f}%"
                for i in np.argsort(-wc)
                if wc[i] > 0 and i != full_winner_ix)
            rho = sc.spearman_rows(sb, full_sc)
            stab_rows.append({
                "voter_set": name, "rule": rule,
                "B": B, "n_games_per_cell": n_games, "scheme": "paired",
                "top1_stability": float(top1),
                "mean_self_spearman": float(np.nanmean(rho)),
                "p5_self_spearman": float(np.nanpercentile(rho, 5)),
                "full_winner": full_names[0],
                "alt_winners": alt,
            })
        print(f"bootstrap done: {name}")

    with open(OUT_DIR / "grid_rule_stability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["voter_set", "rule", "B",
                                           "n_games_per_cell", "scheme",
                                           "top1_stability",
                                           "mean_self_spearman",
                                           "p5_self_spearman", "full_winner",
                                           "alt_winners"])
        w.writeheader()
        w.writerows(stab_rows)

    # ---------------- findings ---------------------------------------------
    write_findings(majority, full_rankings, stab_rows, crowns, rank_range,
                   crownable, surv_crowned, voter_sets, n_games)
    print(f"wrote {OUT_DIR} (grid_* files)")


def write_findings(majority, full_rankings, stab_rows, crowns, rank_range,
                   crownable, surv_crowned, voter_sets, n_games) -> None:
    cnames = list(HEALTHY)
    main_name = "grid_pop10xmetrics6_healthy8"
    vs_main = voter_sets[main_name]
    m_main = majority["voter_sets"][main_name]

    winners = {}
    for name in voter_sets:
        winners[name] = {r: full_rankings[(name, r)][0][0] for r in RULES}
    distinct_main = sorted(set(winners[main_name].values()))
    added = [p for p in crownable if p not in
             ["ippo_s1", "ippo_s2", "ippo_s4", "mappo_s2", "mappo_s3"]]
    stab_main = {r["rule"]: r for r in stab_rows if r["voter_set"] == main_name}

    lines = [
        "# E27-grid. The full measurement portfolio as an electorate — "
        "combined (population x metric) voter grid [EconML theme 1]",
        "",
        f"Merged 1000-game peer matrix (n={n_games}/cell). Voter = one "
        "(population judge, metric) grid cell: judge j x metric m ranks the "
        "healthy-8 candidates i != j by metric m of cell (i vs 3x j) — the "
        "E12 population convention crossed with E20's six peer metrics "
        "(survival, win, cash, cash|survival, survival_time, win|survival). "
        "Judge universe = the 10 populations (pinned by the E27 cross-check; "
        "each candidate is judged by its 9 single-population 1v3 contexts), "
        "i.e. 60 grid voter cells; healthy-8-judges grid (48 cells) as a "
        "secondary voter set. Mean-of-normalized-scores is ill-defined "
        "across metric units, so rank-based rules only: Borda, Copeland, "
        "plurality, exact Kemeny (Held-Karp, n=8). Bootstrap: paired scheme, "
        f"B={B}, n={n_games}, rng {RNG_SEED}.",
        "",
        "## Cross-check vs E27 (asserted in-script)",
        "",
        "The grid restricted to (all 10 populations, survival) reproduces "
        "E27's `pop_all10judges_healthy8` EXACTLY: identical pairwise "
        "preference counts, identical majority structure (vs the stored "
        "`pairwise_majority.json`: no CW, weak CW ippo_s2, 0 cycles, 4 "
        "majority ties), identical Borda/Copeland/plurality/Kemeny rankings "
        "(vs the stored `rules_ranking.csv`).",
        "",
        "## Voter accounting",
        "",
        "| voter set | potential | dropped (all-NaN ballot) | seated | "
        "partial-ballot NaN cells |",
        "|---|---|---|---|---|",
    ]
    for name, vs in voter_sets.items():
        lines.append(
            f"| {name} | {vs['n_potential_voters']} | "
            f"{len(vs['dropped_voters'])}"
            + (f" ({', '.join(vs['dropped_voters'])})" if vs["dropped_voters"] else "")
            + f" | {vs['n_voters']} | {vs['n_undefined_cells']} |"
        )

    lines += [
        "",
        "## Condorcet structure on the grid",
        "",
        "| voter set | Condorcet winner | weak CW (loses no majority) | "
        "cycle triples | majority ties |",
        "|---|---|---|---|---|",
    ]
    for name in list(voter_sets) + ["grid_restricted_pop10_survival"]:
        m = majority["voter_sets"][name]
        lines.append(
            f"| {name} | {m['condorcet_winner'] or 'NONE'} | "
            f"{', '.join(m['weak_condorcet_winners']) or 'NONE'} | "
            f"{m['n_cycle_triples']}/{m['n_triples']} | {m['n_majority_ties']} |"
        )
        for t in m["cycle_triples"][:10]:
            lines.append(f"|  | cycle: {' > '.join(t)} > {t[0]} | | | |")
        if len(m["cycle_triples"]) > 10:
            lines.append(f"|  | ... {len(m['cycle_triples']) - 10} more cycle "
                         "triples in grid_pairwise_majority.json | | | |")

    lines += ["", "## Rule dependence (full rankings, best to worst)", ""]
    for name in voter_sets:
        lines += [f"### {name}", "", "| rule | ranking |", "|---|---|"]
        for rule in RULES:
            ranked, _ = full_rankings[(name, rule)]
            lines.append(f"| {rule} | {' > '.join(ranked)} |")
        lines += ["", "Distinct winners across rules: "
                  f"{sorted(set(winners[name].values()))}", ""]

    n_cr = len(crownable)
    lines += [
        "## 2-D manipulability: which policies can some (population, metric) "
        "voter crown #1?",
        "",
        "| policy | crownable | #crowning voters | crowning metrics | "
        "rank range |",
        "|---|---|---|---|---|",
    ]
    for p in cnames:
        mets_p = sorted({v.split("|")[1] for v in crowns[p]})
        lines.append(
            f"| {p} | {bool(crowns[p])} | {len(crowns[p])} | "
            f"{', '.join(mets_p) or '-'} | "
            f"{rank_range[p][0]:g}-{rank_range[p][1]:g} |"
        )
    lines += [
        "",
        f"Crownable set on the grid: **{n_cr}/8** "
        f"({', '.join(crownable)}) vs E27's 5/8 under population-voters "
        "alone (survival metric, healthy judges) — adding the metric "
        f"dimension {'adds ' + ', '.join(added) if added else 'adds no new policy'}"
        f" ({E27_HEALTHY8_CROWNABLE}/8 -> {n_cr}/8). Survival-column-only "
        f"crownable under the 10-judge universe: {len(surv_crowned)}/8 "
        f"({', '.join(surv_crowned)}).",
        "",
        "## Rule stability under paired bootstrap "
        f"(B={B}, n={n_games}, rng {RNG_SEED})",
        "",
        "| voter set | rule | top-1 stable | mean self-Spearman | p5 | winner "
        "| alternate winners |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in stab_rows:
        lines.append(
            f"| {r['voter_set']} | {r['rule']} | {r['top1_stability']*100:.0f}% | "
            f"{r['mean_self_spearman']:.3f} | {r['p5_self_spearman']:.3f} | "
            f"{r['full_winner']} | {r['alt_winners'].replace(';', ', ') or '-'} |"
        )

    cw = m_main["condorcet_winner"]
    weak = m_main["weak_condorcet_winners"]
    verdict = (f"a STRICT Condorcet winner: {cw}" if cw else
               ("no strict Condorcet winner; weak CW (loses no majority): "
                + ", ".join(weak) if weak else
                "no Condorcet winner, strict or weak"))
    # closest pairwise race involving the top-2 Borda candidates
    borda_top2 = full_rankings[(main_name, "borda")][0][:2]
    race = next((e for e in m_main["majority_edges"]
                 if set(e["pair"]) == set(borda_top2)), None)
    race_txt = ""
    if race:
        race_txt = (f" (it majority-beats runner-up {borda_top2[1]} "
                    f"{max(race['votes'])}-{min(race['votes'])} among the "
                    f"{race['n_judges']} voters ranking both)")
    lines += [
        "",
        "## Interpretation (EconML: the measurement portfolio as one electorate)",
        "",
        "E27 asked what happens when evaluation populations vote, and "
        "separately when metrics vote. Here the full measurement portfolio "
        "votes at once: every (population, metric) pair one has defensible "
        "grounds to publish is a voter, so the grid is the electorate an "
        "evaluator implicitly samples from when they pick 'the' leaderboard. "
        f"The headline inverts E27: on the {vs_main['n_voters']}-voter grid "
        f"the majority relation yields {verdict}{race_txt}, with "
        f"{m_main['n_cycle_triples']}/{m_main['n_triples']} cycle triples "
        f"and {m_main['n_majority_ties']} majority ties — E27's Condorcet "
        "cycle among survival-voters and the metric-voters' split crown both "
        "dissolve once the portfolio votes jointly, and every rule agrees: "
        + "; ".join(f"{r} -> {winners[main_name][r]}" for r in RULES)
        + f" ({len(distinct_main)} distinct champion"
        + ("s" if len(distinct_main) != 1 else "") + " from identical "
        "ballots, vs 2-3 in E27). The consensus champion is mappo_s2 — the "
        "policy the status-quo survival mean ranks 7th of 8 — because "
        "breadth across the portfolio (top-1 for 26/60 voter cells, strong "
        "on the cash and win channels in nearly every population context) "
        "beats depth on the single survival channel. Yet aggregation only "
        "relocates the evaluator's discretion, it does not remove it: "
        "manipulability now has two dimensions, and with both a population "
        f"AND a metric free to choose, {n_cr}/8 policies can be crowned #1 "
        "by some single defensible voter cell (vs 5/8 with the population "
        "dimension alone; only ippo_s3 is never crownable) — the two "
        "degrees of evaluator freedom compose. Bootstrap: the Condorcet "
        "verdict is solid — "
        + "; ".join(
            f"{r} top-1 {stab_main[r]['top1_stability']*100:.0f}%"
            for r in RULES)
        + " on the main grid — with the one soft spot that Borda's "
        "positional scores put the runner-up within sampling noise "
        f"({stab_main['borda']['alt_winners'].replace(';', ', ')}) while the "
        "majority-relation rules (Copeland, Kemeny) flip in 0/2000 "
        "replicates and plurality in 1/2000. Widening the electorate to the "
        "full measurement portfolio "
        "thus buys BOTH a Condorcet-consistent, rule-robust, "
        "sampling-stable champion AND strictly more room for single-cell "
        "cherry-picking: the choice being made is no longer which rule, but "
        "which electorate.",
        "",
        "## Sanity checks",
        "",
        "- restricted-grid (all populations, survival) == E27 "
        "pop_all10judges_healthy8: pref counts, majority structure, and all "
        "four rule rankings match exactly (asserted).",
        f"- dropped voter cells: "
        f"{len(vs_main['dropped_voters'])} of "
        f"{vs_main['n_potential_voters']} on the main grid"
        + (f" ({', '.join(vs_main['dropped_voters'])})"
           if vs_main["dropped_voters"] else "")
        + f"; partial-ballot NaN candidate cells: "
        f"{vs_main['n_undefined_cells']}.",
        "",
        "Data: `workshops/econml/results/social_choice/` "
        "(grid_pairwise_majority.json, grid_rules_ranking.csv, "
        "grid_crownable.csv, grid_rule_stability.csv); script "
        "`workshops/econml/analyze_social_choice_grid.py` "
        f"(rng seed {RNG_SEED}).",
        "",
    ]
    with open(OUT_DIR / "FINDINGS_grid.md", "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
