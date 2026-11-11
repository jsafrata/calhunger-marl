"""E28: Nash averaging vs the unstable leaderboard (EconML theme 1).

Question: does an economic solution concept (equilibrium rating, Balduzzi et
al. 2018 "Re-evaluating evaluation") beat the macro-mean leaderboard's
instability (E19)?

Data: merged 1000-game peer cross-play matrix (E19 seed extension), loaded via
``workshops/eval/crossplay_raw_common.load_peer_1v3(extended=True)`` plus the
self-play diagonal (``load_peer_selfplay``). Meta-game M[i, j] = mean focal-i
outcome (survival, or win) when playing 1-vs-3 copies of j; M[i, i] = the
self-play mixed-cell seat mean. All 100 cells share the same paired seed range
(game index i has the same game_seed everywhere), which the script asserts.

Nash averaging, exactly as implemented here
-------------------------------------------
1. Antisymmetrize: A = (M - M^T) / 2. CAVEAT (prominent, honest): the 1v3
   meta-game is NOT a symmetric 2-player game — M[i, j] and M[j, i] come from
   *different lineups* (i vs 3xj, and j vs 3xi), and payoffs are not
   constant-sum (both focals can survive). A measures the *relative advantage
   in mirrored 1v3 matchups*, which is the natural analogue of Balduzzi's
   antisymmetric construction for this data, not a literal zero-sum game.
2. A is antisymmetric, so the symmetric zero-sum game (A, -A) has value 0 and
   the optimal-strategy set of either player is the polytope
   P* = {p in simplex : (A p)_j <= 0 for all j}
   (p guarantees value >= 0 against every pure column since (A^T p) = -(A p)).
3. One Nash via LP (scipy.optimize.linprog, HiGHS): variables (p, v),
   maximize v s.t. v <= (A^T p)_j for all j, sum p = 1, p >= 0.
4. Maxent refinement: the maximum-entropy point of the convex polytope P* is
   unique; found with SLSQP (scipy.optimize.minimize) minimizing
   sum p log(p + 1e-12) with analytic gradients, subject to sum p = 1 and
   (A p)_j <= 0, bounds [0, 1], initialized at 0.95*LP-Nash + 0.05*uniform.
   The result is clipped to >= 0, renormalized, and VERIFIED: equilibrium
   residual max_j (A p*)_j must be <= 1e-6 (reported). If SLSQP fails the LP
   vertex Nash is used and counted (bootstrap fallback counter reported).
5. Rating of policy i = (A p*)_i. Support members rate exactly 0 (a structural
   property: Nash averaging declares a co-optimal *set*), non-support rate < 0.

Stability comparison (the actual E28 question)
----------------------------------------------
The Nash-averaging estimator is pushed through the SAME bootstrap as the E19
macro-mean leaderboard: game-level resampling at the full n=1000 budget,
paired scheme (ONE index vector shared by all 100 cells per replicate,
matching the campaign's paired-seed design), B = 2000, rng seed 20260831.
Per replicate: recompute M, A, maxent Nash, ratings, ranking — and, from the
same replicate index vector, the E19 macro-mean scores (mean over the 9
off-diagonal focal cells; identical definition to
``eval/analyze_bootstrap_leaderboard_stability.py``) for a like-for-like
comparison. Reference values in ``results/leaderboard_stability_1000/`` are
used as a sanity check (same estimator, different rng seed/B there).

Tie handling (Nash ratings are tied at 0 on the support by construction):
ratings are snapped to a 1e-7 grid before ranking/sign tests; Spearman uses
average ranks; a pairwise verdict "flips" when the snapped sign of the rating
difference differs from the point estimate's (ties count as disagreement with
a strict point-estimate verdict, matching E19's "ties count as flips"); top-1
is stable when the point-estimate top-rated policy is in the replicate's
maximal-rating set (within 1e-7). Support-level stability (per-policy support
frequency, exact-support match rate, mean Jaccard) is reported as well.

Replicator dynamics
-------------------
dt-Euler replicator (dp_i = p_i((Mp)_i - p'Mp), dt = 0.1) from the uniform
start on (a) the raw healthy-8 survival meta-game M and (b) its
antisymmetrization A; convergence vs cycling detected from the tail of the
trajectory (gradient norm, tail amplitude, autocorrelation period estimate).
Connects to the documented non-transitivity (E2 scripted rock-paper-scissors;
E10a mirrored win-channel dominance pairs, e.g. ippo_s2<->ippo_s4 = 30.3<->4.9
wins at ~equal survival).

Outputs (workshops/econml/results/nash_averaging/):
  meta_game_matrix.csv, nash_ratings.csv, nash_vs_mean_stability.csv,
  replicator_cycles.json, FINDINGS.md

Usage:  python3 workshops/econml/analyze_nash_averaging.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
from crossplay_raw_common import (  # noqa: E402
    HEALTHY,
    LABELS,
    QUANTITIES,
    RESULTS_ROOT,
    load_peer_1v3,
    load_peer_selfplay,
)

OUT_DIR = HERE / "results" / "nash_averaging"
REF_JSON = RESULTS_ROOT / "leaderboard_stability_1000" / "bootstrap_stability.json"

RNG_SEED = 20260831
B = 2000
TIE_TOL = 1e-7
EQ_TOL = 1e-6
METRICS = {"survival": QUANTITIES.index("survived"), "win": QUANTITIES.index("won")}
GROUPS = ("all10", "healthy8")

REPLICATOR_DT = 0.1
REPLICATOR_STEPS = 500_000
REPLICATOR_RECORD_EVERY = 500


# ---------------------------------------------------------------- data

def build_tensor() -> np.ndarray:
    """V[i, j, g, q]: per-game outcome of focal LABELS[i] vs 3x LABELS[j].

    Off-diagonal from the 1v3 cells; diagonal from the self-play mixed cells
    (per-game seat mean). Asserts the paired-seed alignment across all cells.
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


# ---------------------------------------------------------------- nash

def lp_nash(A: np.ndarray) -> np.ndarray:
    """One Nash of the symmetric zero-sum game on antisymmetric A (value 0)."""
    n = A.shape[0]
    # variables x = (p_0..p_{n-1}, v); maximize v  <=>  minimize -v
    c = np.zeros(n + 1)
    c[-1] = -1.0
    # v - (A^T p)_j <= 0  for each column j
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    A_eq = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")
    p = np.clip(res.x[:n], 0.0, None)
    return p / p.sum()


def maxent_nash(A: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Maximum-entropy Nash of antisymmetric A.

    Returns (p, equilibrium residual max_j (A p)_j, used_lp_fallback).
    """
    n = A.shape[0]
    p0 = lp_nash(A)
    eps = 1e-12

    def f(p):
        return float(np.sum(p * np.log(p + eps)))

    def grad(p):
        return np.log(p + eps) + p / (p + eps)

    cons = [
        {"type": "eq", "fun": lambda p: p.sum() - 1.0,
         "jac": lambda p: np.ones(n)},
        {"type": "ineq", "fun": lambda p: -(A @ p),  # (A p)_j <= 0
         "jac": lambda p: -A},
    ]
    x0 = 0.95 * p0 + 0.05 / n
    res = minimize(f, x0, jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    p = np.clip(res.x, 0.0, None)
    p = p / p.sum()
    resid = float(np.max(A @ p))
    if (not res.success) or resid > EQ_TOL:
        # fall back to the LP vertex Nash (a valid equilibrium, lower entropy)
        p_lp = p0 / p0.sum()
        resid_lp = float(np.max(A @ p_lp))
        if resid_lp <= max(resid, EQ_TOL):
            return p_lp, resid_lp, True
    return p, resid, False


def nash_ratings(M: np.ndarray) -> dict:
    A = (M - M.T) / 2.0
    p, resid, fallback = maxent_nash(A)
    r = A @ p
    support = p > 1e-6
    return {
        "A": A,
        "p": p,
        "ratings": r,
        "residual": resid,
        "fallback": fallback,
        "support": support,
        "entropy": float(-np.sum(p[p > 0] * np.log(p[p > 0]))),
        "value": float(p @ A @ p),  # 0 by antisymmetry (numerical check)
    }


# ---------------------------------------------------------------- stability

def snap(x: np.ndarray) -> np.ndarray:
    return np.round(x / TIE_TOL) * TIE_TOL


def spearman_rows(sample: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Spearman rho of each row of (B, P) vs full (P,), average ranks."""
    r_full = rankdata(full)
    r_samp = rankdata(sample, axis=1)
    r_full = (r_full - r_full.mean()) / r_full.std()
    with np.errstate(invalid="ignore", divide="ignore"):
        r_samp = (r_samp - r_samp.mean(axis=1, keepdims=True)) / r_samp.std(
            axis=1, keepdims=True
        )
        return (r_samp * r_full).mean(axis=1)


def stability_stats(sample: np.ndarray, full: np.ndarray, tie_aware: bool) -> dict:
    """E19-style stability numbers of replicate scores vs the point estimate.

    ``tie_aware`` (Nash): scores snapped to the 1e-7 grid first; top-1 stable
    = point argmax within the replicate's maximal set; pair flip = snapped
    sign differs. Macro-mean (tie_aware=False) reproduces E19's convention
    exactly (strict sign, ties count as flips, plain argmax).
    """
    if tie_aware:
        sample = snap(sample)
        full = snap(full)
    b, p = sample.shape
    rho = spearman_rows(sample, full)
    if tie_aware:
        top = full.max() - full <= TIE_TOL  # point-estimate maximal set
        top_ix = int(np.argmax(full))
        rep_max = sample.max(axis=1)
        top1 = (rep_max - sample[:, top_ix]) <= TIE_TOL
        n_top_point = int(top.sum())
    else:
        top1 = sample.argmax(axis=1) == int(np.argmax(full))
        n_top_point = 1
    iu = np.triu_indices(p, k=1)
    full_sign = np.sign(full[iu[0]] - full[iu[1]])
    diff_sign = np.sign(sample[:, iu[0]] - sample[:, iu[1]])
    flipped = diff_sign != full_sign
    prob = flipped.mean(axis=0)
    order = np.argsort(-full, kind="stable")
    pair_index = {(min(a, c), max(a, c)): k for k, (a, c) in enumerate(zip(*iu))}
    adj = [float(prob[pair_index[(min(order[k], order[k + 1]),
                                 max(order[k], order[k + 1]))]])
           for k in range(p - 1)]
    return {
        "mean_spearman_vs_full": float(np.nanmean(rho)),
        "p5_spearman_vs_full": float(np.nanpercentile(rho, 5)),
        "top1_stability": float(np.mean(top1)),
        "n_tied_at_top_point": n_top_point,
        "mean_pair_flip_prob": float(prob.mean()),
        "max_pair_flip_prob": float(prob.max()),
        "n_pairs": int(prob.size),
        "n_pairs_flip_ge_5pct": int((prob >= 0.05).sum()),
        "n_pairs_flip_ge_25pct": int((prob >= 0.25).sum()),
        "mean_adjacent_flip_prob": float(np.mean(adj)),
    }


def bootstrap_cell_means(V: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(B, n, n, Q) resampled meta-games; idx (B, n_games) shared (paired)."""
    n = V.shape[0]
    b = idx.shape[0]
    out = np.empty((b, n, n, V.shape[3]))
    for i in range(n):
        for j in range(n):
            out[:, i, j, :] = V[i, j][idx, :].mean(axis=1)
    return out


# ---------------------------------------------------------------- replicator

def run_replicator(M: np.ndarray, labels: list[str]) -> dict:
    n = M.shape[0]
    p = np.full(n, 1.0 / n)
    traj, times = [], []
    dt, steps, rec = REPLICATOR_DT, REPLICATOR_STEPS, REPLICATOR_RECORD_EVERY
    gnorm = np.nan
    for t in range(steps):
        f = M @ p
        dp = p * (f - p @ f)
        if t % rec == 0:
            traj.append(p.copy())
            times.append(t * dt)
        p = np.clip(p + dt * dp, 0.0, None)
        p /= p.sum()
        gnorm = float(np.linalg.norm(dp))
    traj = np.array(traj)
    tail = traj[int(0.6 * len(traj)):]
    amplitude = float(np.max(tail.max(axis=0) - tail.min(axis=0)))
    converged = bool(gnorm < 1e-10 or amplitude < 1e-6)
    period = None
    if not converged:
        # autocorrelation of the most active coordinate in the tail
        k = int(np.argmax(tail.std(axis=0)))
        x = tail[:, k] - tail[:, k].mean()
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        ac /= ac[0]
        peaks = [i for i in range(2, len(ac) - 1)
                 if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0.2]
        if peaks:
            period = float(peaks[0] * rec * dt)
    return {
        "labels": labels,
        "dt": dt,
        "steps": steps,
        "record_every": rec,
        "start": "uniform",
        "final_gradient_norm": gnorm,
        "tail_amplitude_max_coord": amplitude,
        "converged": converged,
        "cycle_period_estimate_time_units": period,
        "final_point": {l: float(v) for l, v in zip(labels, traj[-1])},
        "times": [float(t) for t in times[:: max(1, len(times) // 400)]],
        "trajectory": [[float(v) for v in row]
                       for row in traj[:: max(1, len(traj) // 400)]],
    }


# ---------------------------------------------------------------- main

def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    V = build_tensor()
    n_games = V.shape[2]
    print(f"tensor {V.shape} loaded in {time.time()-t0:.1f}s")
    healthy_ix = np.array([LABELS.index(p) for p in HEALTHY])
    group_ix = {"all10": np.arange(len(LABELS)), "healthy8": healthy_ix}
    group_labels = {"all10": LABELS, "healthy8": HEALTHY}

    M_full = V.mean(axis=2)  # (10, 10, Q)

    # ---- sanity check 1: macro-mean scores/ranking vs E19 1000-game reference
    with open(REF_JSON) as fh:
        ref = json.load(fh)
    macro_full = {}  # metric -> (10,) macro-mean over the 9 off-diagonal cells
    offdiag = ~np.eye(len(LABELS), dtype=bool)
    sanity = {"macro_mean_matches_reference": True, "max_abs_diff": 0.0}
    for metric, q in METRICS.items():
        Mq = M_full[:, :, q]
        scores = np.array([Mq[i, offdiag[i]].mean() for i in range(len(LABELS))])
        macro_full[metric] = scores
        ref_scores = np.array([ref["full_scores"][metric][p] for p in LABELS])
        d = float(np.max(np.abs(scores - ref_scores)))
        sanity["max_abs_diff"] = max(sanity["max_abs_diff"], d)
        if not np.allclose(scores, ref_scores, atol=1e-9):
            sanity["macro_mean_matches_reference"] = False
        same_rank = bool(
            (rankdata(-scores) == rankdata(-ref_scores)).all()
        )
        print(f"sanity macro-mean [{metric}]: max|diff|={d:.2e}, "
              f"ranking match={same_rank}")
        if not same_rank:
            sanity["macro_mean_matches_reference"] = False
    if not sanity["macro_mean_matches_reference"]:
        raise RuntimeError("macro-mean sanity check vs "
                           "leaderboard_stability_1000 FAILED")

    # ---- point-estimate Nash averaging (4 combos)
    point = {}
    for metric, q in METRICS.items():
        for group in GROUPS:
            ix = group_ix[group]
            M = M_full[np.ix_(ix, ix)][:, :, q]
            res = nash_ratings(M)
            point[(metric, group)] = res
            labs = group_labels[group]
            supp = [l for l, s in zip(labs, res["support"]) if s]
            print(f"[{metric}/{group}] residual={res['residual']:.2e} "
                  f"value={res['value']:.2e} support={supp} "
                  f"fallback={res['fallback']}")
            if res["residual"] > EQ_TOL:
                raise RuntimeError(f"equilibrium residual {res['residual']} "
                                   f"> {EQ_TOL} for {metric}/{group}")

    # ---- meta_game_matrix.csv (long form, both metrics, all 10)
    with open(OUT_DIR / "meta_game_matrix.csv", "w") as fh:
        fh.write("metric,focal,opponent,mean,se,n_games,cell_type\n")
        for metric, q in METRICS.items():
            for i, fo in enumerate(LABELS):
                for j, op in enumerate(LABELS):
                    x = V[i, j, :, q]
                    fh.write(
                        f"{metric},{fo},{op},{x.mean():.6f},"
                        f"{x.std(ddof=1)/np.sqrt(n_games):.6f},{n_games},"
                        f"{'selfplay' if i == j else '1v3'}\n"
                    )

    # ---- nash_ratings.csv
    with open(OUT_DIR / "nash_ratings.csv", "w") as fh:
        fh.write("metric,group,policy,nash_weight,rating,rank,in_support,"
                 "macro_mean_score,macro_mean_rank\n")
        for metric, q in METRICS.items():
            for group in GROUPS:
                ix = group_ix[group]
                labs = group_labels[group]
                res = point[(metric, group)]
                ranks = rankdata(-snap(res["ratings"]), method="average")
                mm = macro_full[metric][ix]
                mm_rank = rankdata(-mm, method="average")
                for k, l in enumerate(labs):
                    fh.write(
                        f"{metric},{group},{l},{res['p'][k]:.6f},"
                        f"{res['ratings'][k]:.6f},{ranks[k]:.1f},"
                        f"{int(res['support'][k])},{mm[k]:.6f},"
                        f"{mm_rank[k]:.1f}\n"
                    )

    # ---- bootstrap (paired, n=1000, B=2000, rng 20260831)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n_games, size=(B, n_games))
    t1 = time.time()
    Mboot = bootstrap_cell_means(V, idx)  # (B, 10, 10, Q)
    print(f"bootstrap meta-games in {time.time()-t1:.1f}s")

    stab_rows = []
    support_stats = {}
    fallback_counts = {}
    for metric, q in METRICS.items():
        for group in GROUPS:
            ix = group_ix[group]
            labs = group_labels[group]
            res_pt = point[(metric, group)]
            npol = len(ix)
            # macro-mean replicate scores (E19 definition: 9 off-diag cells,
            # computed over ALL 10 opponents, then restricted to the group —
            # identical to E19's stack over the 9 focal cells)
            od = ~np.eye(len(LABELS), dtype=bool)
            mm_boot_all = np.stack(
                [Mboot[:, i, od[i], q].mean(axis=1) for i in range(len(LABELS))],
                axis=1,
            )  # (B, 10)
            mm_boot = mm_boot_all[:, ix]
            mm_full = macro_full[metric][ix]

            nash_boot = np.empty((B, npol))
            supp_boot = np.zeros((B, npol), dtype=bool)
            n_fallback = 0
            t2 = time.time()
            for b in range(B):
                M = Mboot[b][np.ix_(ix, ix)][:, :, q]
                A = (M - M.T) / 2.0
                p, resid, fb = maxent_nash(A)
                n_fallback += int(fb)
                nash_boot[b] = A @ p
                supp_boot[b] = p > 1e-6
            fallback_counts[f"{metric}/{group}"] = n_fallback
            print(f"[{metric}/{group}] {B} nash solves in "
                  f"{time.time()-t2:.1f}s, fallbacks={n_fallback}")

            for est, samp, full, tie_aware in (
                ("nash", nash_boot, res_pt["ratings"], True),
                ("macro_mean", mm_boot, mm_full, False),
            ):
                row = {"metric": metric, "group": group, "estimator": est,
                       **stability_stats(samp, full, tie_aware)}
                stab_rows.append(row)

            pt_supp = res_pt["support"]
            inter = (supp_boot & pt_supp).sum(axis=1)
            union = (supp_boot | pt_supp).sum(axis=1)
            support_stats[f"{metric}/{group}"] = {
                "point_support": [l for l, s in zip(labs, pt_supp) if s],
                "point_support_size": int(pt_supp.sum()),
                "mean_replicate_support_size": float(supp_boot.sum(axis=1).mean()),
                "support_exact_match_rate": float(
                    (supp_boot == pt_supp).all(axis=1).mean()),
                "mean_jaccard_vs_point": float((inter / union).mean()),
                "per_policy_support_freq": {
                    l: float(supp_boot[:, k].mean()) for k, l in enumerate(labs)},
                "n_lp_fallbacks": n_fallback,
            }

    # reference rows (E19 1000-game run, rng 20260827) for the sanity column
    ref_rows = {}
    for e in ref["stability"]:
        if (e["n_games_per_cell"] == 1000 and e["scheme"] == "paired"
                and e["metric"] in METRICS):
            ref_rows[(e["metric"], e["group"])] = e

    cols = ["mean_spearman_vs_full", "p5_spearman_vs_full", "top1_stability",
            "n_tied_at_top_point", "mean_pair_flip_prob", "max_pair_flip_prob",
            "n_pairs", "n_pairs_flip_ge_5pct", "n_pairs_flip_ge_25pct",
            "mean_adjacent_flip_prob"]
    with open(OUT_DIR / "nash_vs_mean_stability.csv", "w") as fh:
        fh.write("metric,group,estimator," + ",".join(cols) + "\n")
        for row in stab_rows:
            fh.write(f"{row['metric']},{row['group']},{row['estimator']},"
                     + ",".join(f"{row[c]:.6g}" for c in cols) + "\n")
        for (metric, group), e in sorted(ref_rows.items()):
            vals = [e.get(c, "") for c in cols]
            vals[cols.index("n_tied_at_top_point")] = 1
            fh.write(f"{metric},{group},macro_mean_E19_reference,"
                     + ",".join(f"{v:.6g}" if v != "" else "" for v in vals)
                     + "\n")

    # ---- replicator dynamics (healthy-8 survival)
    q = METRICS["survival"]
    M_h = M_full[np.ix_(healthy_ix, healthy_ix)][:, :, q]
    A_h = (M_h - M_h.T) / 2.0
    rep = {
        "config": {"dt": REPLICATOR_DT, "steps": REPLICATOR_STEPS,
                   "record_every": REPLICATOR_RECORD_EVERY,
                   "note": ("dt-Euler replicator from uniform start; "
                            "raw_M = healthy-8 survival meta-game (incl. "
                            "self-play diagonal); antisym_A = (M - M^T)/2")},
        "raw_M": run_replicator(M_h, HEALTHY),
        "antisym_A": run_replicator(A_h, HEALTHY),
    }
    with open(OUT_DIR / "replicator_cycles.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    for k in ("raw_M", "antisym_A"):
        r = rep[k]
        print(f"replicator[{k}]: converged={r['converged']} "
              f"|grad|={r['final_gradient_norm']:.2e} "
              f"tail_amp={r['tail_amplitude_max_coord']:.2e} "
              f"period={r['cycle_period_estimate_time_units']}")

    # ---- summary json (full numeric record)
    summary = {
        "config": {"B": B, "rng_seed": RNG_SEED, "n_games_per_cell": n_games,
                   "scheme": "paired", "tie_tol": TIE_TOL, "eq_tol": EQ_TOL,
                   "labels": LABELS, "healthy": HEALTHY},
        "sanity": sanity,
        "point": {
            f"{metric}/{group}": {
                "labels": group_labels[group],
                "nash_weights": [float(x) for x in point[(metric, group)]["p"]],
                "ratings": [float(x) for x in point[(metric, group)]["ratings"]],
                "support": [l for l, s in zip(group_labels[group],
                                              point[(metric, group)]["support"]) if s],
                "equilibrium_residual": point[(metric, group)]["residual"],
                "game_value_pAp": point[(metric, group)]["value"],
                "entropy": point[(metric, group)]["entropy"],
                "lp_fallback": point[(metric, group)]["fallback"],
            }
            for metric in METRICS for group in GROUPS
        },
        "stability": stab_rows,
        "support_stability": support_stats,
        "reference_rows_E19_1000_paired": {
            f"{m}/{g}": e for (m, g), e in ref_rows.items()},
    }
    with open(OUT_DIR / "nash_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    print(f"total {time.time()-t0:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
