"""E28 follow-up: alpha-rank as a third solution concept (EconML theme 1).

Question: E28 showed Nash averaging crowns the spoiler mappo_s1 on the
survival meta-game and is LESS ranking-stable than the macro-mean. Does
alpha-rank (Omidshafiei et al. 2019, "alpha-Rank: Multi-Agent Evaluation by
Evolution") — an *evolutionary* solution concept that does not presume a
zero-sum game — side with the Nash equilibrium (reward the spoiler) or with
the mean leaderboard? And is it any more stable?

Data: identical to E28 — merged 1000-game peer cross-play matrix (E19 seed
extension), loader ``workshops/eval/crossplay_raw_common.load_peer_1v3(
extended=True)`` + ``load_peer_selfplay``; meta-game M[i, j] = mean focal-i
outcome (survival, or win) vs 3x j, self-play diagonal from the mixed cells;
all cells paired on the same seed range (asserted).

Alpha-rank, exactly as implemented here
---------------------------------------
Single-population, m-individual finite-population model over the pure
strategies (monomorphic population states). A resident-sigma population is
invaded by a single tau-mutant; under the logit (Fermi) selection model with
intensity alpha the mutant fixates with the paper's closed-form probability

    rho_{sigma -> tau} = (1 - exp(-alpha * D)) / (1 - exp(-m * alpha * D)),
    D = M[tau, sigma] - M[sigma, tau]     (rho = 1/m when D = 0),

i.e. the payoff comparison is the *mirrored-matchup difference* — the same
antisymmetric information A = (M - M^T)/2 that drives Nash averaging (the
self-play diagonal never enters). Transition matrix of the response-graph
Markov chain: C[sigma, tau] = rho_{sigma->tau} / (n - 1) for tau != sigma
(each of the n-1 mutants equally likely to appear), diagonal = remainder.
The alpha-rank score of strategy i is the stationary mass pi_i of C.

Population size m = 50. Alpha sweep: {0.01, 0.1, 1, 10, 100, inf}. The
"inf" chain is the exact large-alpha limit (rho -> 1 if D > 0, 1/m if D = 0,
0 if D < 0), mixed with a 1e-12 uniform perturbation for irreducibility
(the limit chain can be reducible; the perturbation concentrates the
stationary mass on the sink components, as in the paper's infinite-alpha
regime). Numerics: fixation probabilities computed via expm1 with a
log-space branch for large negative alpha*D (no overflow); stationary
distribution by direct linear solve (replace-one-row), verified to residual
||pi C - pi||_inf <= 1e-10, eigen-decomposition fallback (counted).

Self-validation (hard assertions, run before any analysis):
  * rock-paper-scissors gives the uniform stationary distribution at every
    swept alpha and at the inf limit (max deviation < 1e-9);
  * a strictly dominant strategy absorbs (pi_dom > 0.999 at alpha >= 10 and
    at inf);
  * every stationary solve satisfies the residual check.

Ranking-stabilization ("the paper's recommended large-alpha regime"): per
metric/group, rankings (average ranks of stationary masses snapped to 1e-7)
are compared across the sweep; reported: Spearman of each alpha's ranking vs
the inf-limit ranking, the top-1 at each alpha, and the smallest swept alpha
whose snapped ranking already equals the inf-limit ranking.

Three-concept comparison (same data): macro-mean scores (E19 definition,
recomputed from the tensor and cross-checked against
``results/nash_averaging/nash_ratings.csv`` to 1e-6), Nash-averaging ratings
(read from that same CSV — the E28 point estimates), alpha-rank stationary
mass (inf limit). Reported per metric/group: pairwise Spearman between the
three rankings and each concept's #1 — in particular whether alpha-rank
joins Nash in crowning the spoiler mappo_s1 on survival.

Stability: alpha-rank pushed through the SAME paired game-level bootstrap as
E19/E28 — one index vector shared by all 100 cells per replicate, n = 1000,
B = 2000, rng seed 20260902 (E28 used 20260831; the macro-mean/Nash
reference rows are copied verbatim from
``results/nash_averaging/nash_vs_mean_stability.csv`` rather than re-run, as
specified). Two alpha-rank estimator rows per combo: the inf-limit chain
(headline, matches the paper's regime) and alpha = 10 (graded masses,
robustness). Tie handling identical to E28's Nash convention: scores snapped
to the 1e-7 grid, tie-aware top-1 (point argmax within the replicate's
maximal set), pair flips vs the strict point verdict, average-rank Spearman.

Outputs (workshops/econml/results/alpha_rank/):
  alpha_rank_ratings.csv, alpha_sweep.csv, three_concepts_stability.csv,
  alpha_rank_summary.json  (FINDINGS.md written separately)

Usage:  python3 workshops/econml/analyze_alpha_rank.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
from crossplay_raw_common import (  # noqa: E402
    HEALTHY,
    LABELS,
    QUANTITIES,
    load_peer_1v3,
    load_peer_selfplay,
)

OUT_DIR = HERE / "results" / "alpha_rank"
NASH_DIR = HERE / "results" / "nash_averaging"

RNG_SEED = 20260902
B = 2000
M_POP = 50                      # finite-population size m
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)   # + inf limit, handled separately
BOOT_ALPHA = 10.0               # graded-mass robustness estimator
TIE_TOL = 1e-7
STAT_TOL = 1e-10                # stationarity residual requirement
INF_MIX = 1e-12                 # uniform perturbation for the inf-limit chain
METRICS = {"survival": QUANTITIES.index("survived"), "win": QUANTITIES.index("won")}
GROUPS = ("all10", "healthy8")

_eig_fallbacks = 0


# ---------------------------------------------------------------- data

def build_tensor() -> np.ndarray:
    """V[i, j, g, q]: per-game outcome of focal LABELS[i] vs 3x LABELS[j].

    Identical construction to analyze_nash_averaging.build_tensor (paired-seed
    alignment asserted across all 100 cells).
    """
    peer = load_peer_1v3(extended=True)
    selfp = load_peer_selfplay(extended=True)
    n = len(LABELS)
    ref = selfp[LABELS[0]]
    n_games, base = ref.n_games, ref.game_seed_base
    V = np.empty((n, n, n_games, len(QUANTITIES)))
    for i, focal in enumerate(LABELS):
        cells = {c.opponent: c for c in peer[focal]}
        for j, opp in enumerate(LABELS):
            cell = selfp[focal] if i == j else cells[opp]
            if cell.n_games != n_games or cell.game_seed_base != base:
                raise ValueError(
                    f"seed alignment broken: {focal} vs {opp} has "
                    f"base={cell.game_seed_base}, n={cell.n_games}"
                )
            V[i, j] = cell.values
    return V


def bootstrap_cell_means(V: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(B, n, n, Q) resampled meta-games; idx (B, n_games) shared (paired)."""
    n = V.shape[0]
    b = idx.shape[0]
    out = np.empty((b, n, n, V.shape[3]))
    for i in range(n):
        for j in range(n):
            out[:, i, j, :] = V[i, j][idx, :].mean(axis=1)
    return out


# ---------------------------------------------------------------- alpha-rank

def fixation_probs(D: np.ndarray, alpha: float, m: int) -> np.ndarray:
    """rho[s, t] = fixation prob of a t-mutant in an s-resident population.

    D[s, t] = M[t, s] - M[s, t] (mutant-vs-resident payoff difference).
    Paper formula rho = (1 - e^{-alpha D}) / (1 - e^{-m alpha D}) = 1/m at
    D = 0; computed overflow-free (log-space branch for alpha*D << 0).
    """
    z = alpha * D
    rho = np.empty_like(z)
    tiny = np.abs(z) < 1e-12
    rho[tiny] = 1.0 / m
    big_neg = (~tiny) & (z < -500.0 / (m - 1))
    # e^{-z} >> 1 and e^{-mz} >> 1: rho ~= e^{-z}/e^{-mz} = e^{z(m-1)}
    rho[big_neg] = np.exp(z[big_neg] * (m - 1))
    rest = ~(tiny | big_neg)
    with np.errstate(over="ignore"):
        rho[rest] = np.expm1(-z[rest]) / np.expm1(-m * z[rest])
    return rho


def fixation_probs_inf(D: np.ndarray, m: int) -> np.ndarray:
    """Exact alpha -> inf limit: 1 if D > 0, 1/m if D = 0, 0 if D < 0."""
    rho = np.where(D > 1e-12, 1.0, np.where(D < -1e-12, 0.0, 1.0 / m))
    return rho.astype(float)


def transition_matrix(M: np.ndarray, alpha: float | None, m: int) -> np.ndarray:
    """Response-graph Markov chain over monomorphic states.

    alpha=None means the inf limit (with the 1e-12 uniform mixing).
    """
    n = M.shape[0]
    D = M.T - M                                   # D[s, t] = M[t,s] - M[s,t]
    rho = fixation_probs_inf(D, m) if alpha is None else fixation_probs(D, alpha, m)
    C = rho / (n - 1)
    np.fill_diagonal(C, 0.0)
    np.fill_diagonal(C, 1.0 - C.sum(axis=1))
    if alpha is None:
        C = (1.0 - INF_MIX) * C + INF_MIX / n
    return C


def stationary(C: np.ndarray) -> np.ndarray:
    """Stationary distribution pi = pi C; solve + verify, eigen fallback."""
    global _eig_fallbacks
    n = C.shape[0]
    A = C.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi = np.full(n, np.nan)
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if s > 0:
        pi = pi / s
    resid = float(np.max(np.abs(pi @ C - pi))) if np.all(np.isfinite(pi)) else np.inf
    if resid > STAT_TOL:
        _eig_fallbacks += 1
        w, v = np.linalg.eig(C.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        pi = np.clip(pi, 0.0, None)
        pi = pi / pi.sum()
        resid = float(np.max(np.abs(pi @ C - pi)))
    if resid > 1e-8:
        raise RuntimeError(f"stationary residual {resid:.2e} > 1e-8")
    return pi


def alpharank(M: np.ndarray, alpha: float | None, m: int = M_POP) -> np.ndarray:
    return stationary(transition_matrix(M, alpha, m))


def sink_states(M: np.ndarray) -> np.ndarray:
    """is_sink[s]: no mutant strictly invades s (M[t,s] <= M[s,t] for all t)."""
    D = M.T - M
    np.fill_diagonal(D, 0.0)
    return (D <= 1e-12).all(axis=1)


# ---------------------------------------------------------------- validation

def self_validate() -> None:
    # 1. rock-paper-scissors: uniform at every alpha and at the inf limit
    rps = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    for alpha in list(ALPHAS) + [None]:
        pi = alpharank(rps, alpha)
        if np.max(np.abs(pi - 1.0 / 3.0)) > 1e-9:
            raise RuntimeError(f"RPS validation failed at alpha={alpha}: {pi}")
    # 2. strictly dominant strategy absorbs in the large-alpha regime
    dom = np.array([[0.5, 1.0, 1.0], [0.0, 0.5, 0.7], [0.0, 0.3, 0.5]])
    for alpha in (10.0, 100.0, None):
        pi = alpharank(dom, alpha)
        if pi[0] < 0.999:
            raise RuntimeError(f"dominant-strategy validation failed "
                               f"at alpha={alpha}: {pi}")
    if not sink_states(dom)[0] or sink_states(dom)[1:].any():
        raise RuntimeError("sink detection failed on dominant-strategy game")
    # 3. epsilon-RPS (biased) still sums to 1 and stays a proper distribution
    pi = alpharank(rps + 0.01 * np.eye(3), 1.0)
    if abs(pi.sum() - 1.0) > 1e-12 or (pi < 0).any():
        raise RuntimeError("stationary distribution not a distribution")
    print("self-validation passed (RPS uniform, dominant absorbs)")


# ---------------------------------------------------------------- stability
# (identical machinery to analyze_nash_averaging.py, tie-aware convention)

def snap(x: np.ndarray) -> np.ndarray:
    return np.round(x / TIE_TOL) * TIE_TOL


def spearman_rows(sample: np.ndarray, full: np.ndarray) -> np.ndarray:
    r_full = rankdata(full)
    r_samp = rankdata(sample, axis=1)
    r_full = (r_full - r_full.mean()) / r_full.std()
    with np.errstate(invalid="ignore", divide="ignore"):
        r_samp = (r_samp - r_samp.mean(axis=1, keepdims=True)) / r_samp.std(
            axis=1, keepdims=True
        )
        return (r_samp * r_full).mean(axis=1)


def stability_stats(sample: np.ndarray, full: np.ndarray) -> dict:
    """E19/E28-style stability numbers, tie-aware (Nash convention)."""
    sample = snap(sample)
    full = snap(full)
    b, p = sample.shape
    rho = spearman_rows(sample, full)
    top = full.max() - full <= TIE_TOL
    top_ix = int(np.argmax(full))
    rep_max = sample.max(axis=1)
    top1 = (rep_max - sample[:, top_ix]) <= TIE_TOL
    iu = np.triu_indices(p, k=1)
    full_sign = np.sign(full[iu[0]] - full[iu[1]])
    diff_sign = np.sign(sample[:, iu[0]] - sample[:, iu[1]])
    prob = (diff_sign != full_sign).mean(axis=0)
    order = np.argsort(-full, kind="stable")
    pair_index = {(min(a, c), max(a, c)): k for k, (a, c) in enumerate(zip(*iu))}
    adj = [float(prob[pair_index[(min(order[k], order[k + 1]),
                                 max(order[k], order[k + 1]))]])
           for k in range(p - 1)]
    return {
        "mean_spearman_vs_full": float(np.nanmean(rho)),
        "p5_spearman_vs_full": float(np.nanpercentile(rho, 5)),
        "top1_stability": float(np.mean(top1)),
        "n_tied_at_top_point": int(top.sum()),
        "mean_pair_flip_prob": float(prob.mean()),
        "max_pair_flip_prob": float(prob.max()),
        "n_pairs": int(prob.size),
        "n_pairs_flip_ge_5pct": int((prob >= 0.05).sum()),
        "n_pairs_flip_ge_25pct": int((prob >= 0.25).sum()),
        "mean_adjacent_flip_prob": float(np.mean(adj)),
    }


# ---------------------------------------------------------------- main

def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    self_validate()

    V = build_tensor()
    n_games = V.shape[2]
    print(f"tensor {V.shape} loaded in {time.time()-t0:.1f}s")
    healthy_ix = np.array([LABELS.index(p) for p in HEALTHY])
    group_ix = {"all10": np.arange(len(LABELS)), "healthy8": healthy_ix}
    group_labels = {"all10": list(LABELS), "healthy8": list(HEALTHY)}
    M_full = V.mean(axis=2)  # (10, 10, Q)

    # ---- macro-mean scores (E19 definition) + cross-check vs E28's CSV
    offdiag = ~np.eye(len(LABELS), dtype=bool)
    macro_full = {}
    for metric, q in METRICS.items():
        Mq = M_full[:, :, q]
        macro_full[metric] = np.array(
            [Mq[i, offdiag[i]].mean() for i in range(len(LABELS))])
    nash_pt = {}   # (metric, group) -> {policy: (nash_rating, nash_rank)}
    with open(NASH_DIR / "nash_ratings.csv") as fh:
        for row in csv.DictReader(fh):
            key = (row["metric"], row["group"])
            nash_pt.setdefault(key, {})[row["policy"]] = (
                float(row["rating"]), float(row["rank"]))
            i = LABELS.index(row["policy"])
            d = abs(float(row["macro_mean_score"]) - macro_full[row["metric"]][i])
            if d > 1e-6:
                raise RuntimeError(f"macro-mean mismatch vs nash_ratings.csv: "
                                   f"{row['policy']} {row['metric']} d={d}")
    print("macro-mean cross-check vs nash_averaging/nash_ratings.csv passed")

    # ---- point estimates: alpha sweep + inf limit, all 4 combos
    alpha_names = [str(a) for a in ALPHAS] + ["inf"]
    alpha_vals: list[float | None] = list(ALPHAS) + [None]
    point = {}       # (metric, group) -> dict
    sweep_rows = []
    for metric, q in METRICS.items():
        for group in GROUPS:
            ix = group_ix[group]
            labs = group_labels[group]
            M = M_full[np.ix_(ix, ix)][:, :, q]
            pis = {name: alpharank(M, a)
                   for name, a in zip(alpha_names, alpha_vals)}
            ranks = {name: rankdata(-snap(pi), method="average")
                     for name, pi in pis.items()}
            rho_vs_inf = {
                name: float(spearmanr(ranks[name], ranks["inf"]).statistic)
                for name in alpha_names}
            stab_alpha = None
            for name in alpha_names[:-1]:
                if np.array_equal(ranks[name], ranks["inf"]):
                    stab_alpha = name
                    break
            sinks = sink_states(M)
            point[(metric, group)] = {
                "labels": labs, "pis": pis, "ranks": ranks,
                "rho_vs_inf": rho_vs_inf, "stabilization_alpha": stab_alpha,
                "sinks": sinks,
            }
            top_by_alpha = {name: labs[int(np.argmax(pis[name]))]
                            for name in alpha_names}
            print(f"[{metric}/{group}] top-1 by alpha: {top_by_alpha}  "
                  f"stabilizes at alpha={stab_alpha}  "
                  f"sinks={[l for l, s in zip(labs, sinks) if s]}")
            for name in alpha_names:
                for k, l in enumerate(labs):
                    sweep_rows.append({
                        "metric": metric, "group": group, "alpha": name,
                        "policy": l, "stationary_mass": pis[name][k],
                        "rank": ranks[name][k],
                        "spearman_vs_inf": rho_vs_inf[name],
                    })

    with open(OUT_DIR / "alpha_sweep.csv", "w") as fh:
        fh.write("metric,group,alpha,policy,stationary_mass,rank,"
                 "ranking_spearman_vs_inf\n")
        for r in sweep_rows:
            fh.write(f"{r['metric']},{r['group']},{r['alpha']},{r['policy']},"
                     f"{r['stationary_mass']:.6g},{r['rank']:.1f},"
                     f"{r['spearman_vs_inf']:.6f}\n")

    # ---- alpha_rank_ratings.csv: three concepts side by side (point est.)
    with open(OUT_DIR / "alpha_rank_ratings.csv", "w") as fh:
        fh.write("metric,group,policy,alpharank_mass_inf,alpharank_rank_inf,"
                 "alpharank_mass_a10,alpharank_rank_a10,is_sink,"
                 "nash_rating,nash_rank,macro_mean_score,macro_mean_rank\n")
        for metric in METRICS:
            for group in GROUPS:
                ix = group_ix[group]
                labs = group_labels[group]
                pt = point[(metric, group)]
                mm = macro_full[metric][ix]
                mm_rank = rankdata(-mm, method="average")
                for k, l in enumerate(labs):
                    nr, nrank = nash_pt[(metric, group)][l]
                    fh.write(
                        f"{metric},{group},{l},"
                        f"{pt['pis']['inf'][k]:.6g},{pt['ranks']['inf'][k]:.1f},"
                        f"{pt['pis']['10.0'][k]:.6g},{pt['ranks']['10.0'][k]:.1f},"
                        f"{int(pt['sinks'][k])},{nr:.6f},{nrank:.1f},"
                        f"{mm[k]:.6f},{mm_rank[k]:.1f}\n")

    # ---- concept agreement (Spearman between the three rankings, #1 each)
    agreement = {}
    for metric in METRICS:
        for group in GROUPS:
            ix = group_ix[group]
            labs = group_labels[group]
            pt = point[(metric, group)]
            mm = macro_full[metric][ix]
            nash_r = np.array([nash_pt[(metric, group)][l][0] for l in labs])
            ar = pt["pis"]["inf"]
            rk_mm = rankdata(-snap(mm))
            rk_na = rankdata(-snap(nash_r))
            rk_ar = rankdata(-snap(ar))
            agreement[f"{metric}/{group}"] = {
                "top1_macro_mean": labs[int(np.argmax(mm))],
                "top1_nash": labs[int(np.argmax(snap(nash_r)))],
                "top1_alpharank_inf": labs[int(np.argmax(ar))],
                "spearman_mean_vs_nash":
                    float(spearmanr(rk_mm, rk_na).statistic),
                "spearman_mean_vs_alpharank":
                    float(spearmanr(rk_mm, rk_ar).statistic),
                "spearman_nash_vs_alpharank":
                    float(spearmanr(rk_na, rk_ar).statistic),
                "alpharank_mass_top": float(ar.max()),
                "stabilization_alpha": pt["stabilization_alpha"],
            }
            a = agreement[f"{metric}/{group}"]
            print(f"[{metric}/{group}] #1 mean={a['top1_macro_mean']} "
                  f"nash={a['top1_nash']} alpharank={a['top1_alpharank_inf']} "
                  f"| rho(mean,ar)={a['spearman_mean_vs_alpharank']:.3f} "
                  f"rho(nash,ar)={a['spearman_nash_vs_alpharank']:.3f} "
                  f"rho(mean,nash)={a['spearman_mean_vs_nash']:.3f}")

    # ---- bootstrap (paired, n=1000, B=2000, rng 20260902)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n_games, size=(B, n_games))
    t1 = time.time()
    Mboot = bootstrap_cell_means(V, idx)
    print(f"bootstrap meta-games in {time.time()-t1:.1f}s")

    stab_rows = []
    top_freq = {}
    for metric, q in METRICS.items():
        for group in GROUPS:
            ix = group_ix[group]
            labs = group_labels[group]
            npol = len(ix)
            for est, alpha in (("alpharank_inf", None),
                               ("alpharank_a10", BOOT_ALPHA)):
                boot = np.empty((B, npol))
                t2 = time.time()
                for b in range(B):
                    M = Mboot[b][np.ix_(ix, ix)][:, :, q]
                    boot[b] = alpharank(M, alpha)
                full = point[(metric, group)]["pis"][
                    "inf" if alpha is None else "10.0"]
                stab_rows.append({"metric": metric, "group": group,
                                  "estimator": est,
                                  **stability_stats(boot, full)})
                tops = np.argmax(snap(boot), axis=1)
                top_freq[f"{metric}/{group}/{est}"] = {
                    labs[k]: float((tops == k).mean())
                    for k in range(npol) if (tops == k).any()}
                print(f"[{metric}/{group}/{est}] {B} solves in "
                      f"{time.time()-t2:.1f}s  top1="
                      f"{stab_rows[-1]['top1_stability']:.3f}  "
                      f"rho={stab_rows[-1]['mean_spearman_vs_full']:.3f}")

    # ---- three_concepts_stability.csv: alpha-rank rows + E28 reference rows
    cols = ["mean_spearman_vs_full", "p5_spearman_vs_full", "top1_stability",
            "n_tied_at_top_point", "mean_pair_flip_prob", "max_pair_flip_prob",
            "n_pairs", "n_pairs_flip_ge_5pct", "n_pairs_flip_ge_25pct",
            "mean_adjacent_flip_prob"]
    with open(NASH_DIR / "nash_vs_mean_stability.csv") as fh:
        ref_lines = fh.read().strip().splitlines()
    with open(OUT_DIR / "three_concepts_stability.csv", "w") as fh:
        fh.write("metric,group,estimator," + ",".join(cols)
                 + ",bootstrap_source\n")
        for row in stab_rows:
            fh.write(f"{row['metric']},{row['group']},{row['estimator']},"
                     + ",".join(f"{row[c]:.6g}" for c in cols)
                     + ",B2000_rng20260902\n")
        for line in ref_lines[1:]:
            fh.write(line + ",copied_from_E28_nash_vs_mean_stability\n")

    # ---- full numeric record
    summary = {
        "config": {"B": B, "rng_seed": RNG_SEED, "n_games_per_cell": n_games,
                   "scheme": "paired", "m_population": M_POP,
                   "alphas": list(ALPHAS) + ["inf"], "inf_mix": INF_MIX,
                   "boot_alpha_graded": BOOT_ALPHA, "tie_tol": TIE_TOL,
                   "labels": list(LABELS), "healthy": list(HEALTHY)},
        "eig_fallbacks_total": _eig_fallbacks,
        "point": {
            f"{metric}/{group}": {
                "labels": point[(metric, group)]["labels"],
                "stationary_mass": {
                    name: [float(x) for x in point[(metric, group)]["pis"][name]]
                    for name in alpha_names},
                "ranking_spearman_vs_inf":
                    point[(metric, group)]["rho_vs_inf"],
                "stabilization_alpha":
                    point[(metric, group)]["stabilization_alpha"],
                "sink_states": [l for l, s in zip(
                    point[(metric, group)]["labels"],
                    point[(metric, group)]["sinks"]) if s],
            }
            for metric in METRICS for group in GROUPS
        },
        "concept_agreement": agreement,
        "stability": stab_rows,
        "bootstrap_top1_frequency": top_freq,
    }
    with open(OUT_DIR / "alpha_rank_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    print(f"eig fallbacks: {_eig_fallbacks}")
    print(f"total {time.time()-t0:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
