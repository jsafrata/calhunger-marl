"""Camera-ready held-out analyses (HPC campaign hpc/camera_ready_heldout).

Block A = the original longitudinal evaluation (seed base 900000; the data of
Figure 1 / Table 2).  Block B = the same cells replayed on the held-out seed
block (base 960000).  Peers = every checkpoint of every run scored against
held-out episode-matched populations (never used in any audit or selection).

(1) REPLICATION: Table 2 verdicts and the v4 trajectory (mean, 95% t-CI over
    runs) recomputed on block B; cell-level agreement A vs B.
(2) OUT-OF-SAMPLE REGRET: every selector (incl. the oracle) picks a checkpoint
    on one block and is scored on the other; two-fold cross-validated regret
    relative to an oracle picked the same way.  The in-sample oracle of the
    submitted paper is reported alongside for reference.
(3) HELD-OUT LEARNED POPULATION: trajectory, 700 -> 1200 verdict, peak-based
    decline and correlation with the v4 trajectory, for each held-out
    population.
(4) Appendix figure: v4 trajectory on both blocks with CI bands, and the
    held-out-population trajectories.

Usage:
    python eval/analyze_camera_ready_heldout.py --results-root results \
        --heldout-root results/camera_ready_wv --out-dir results/camera_ready_wv/analysis \
        --fig-dir figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

HEALTHY = [("ippo", 0), ("ippo", 1), ("ippo", 2), ("ippo", 3), ("ippo", 4),
           ("mappo", 0), ("mappo", 2), ("mappo", 3)]
ALL10 = HEALTHY + [("mappo", 1), ("mappo", 4)]
CKPTS = list(range(100, 1201, 100))
OPPS = ("v2", "v3", "v4", "v5")
FLAT_BAND = 2.0
T975 = {3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622}
LABEL_RE = re.compile(r"^(ippo|mappo)_s(\d)_c(\d{4})$")


# ---------------------------------------------------------------- loading --

def read_rows(paths):
    rows = []
    for p in paths:
        if p.is_file():
            rows.extend(csv.DictReader(open(p)))
    return rows


def series(rows, field, runs):
    out = {}
    for r in rows:
        if r.get("mode", "stochastic") != "stochastic":
            continue
        key = (r["algorithm"], int(r["seed"]))
        if key not in runs:
            continue
        try:
            val = float(r[field])
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault(key, {})[int(r["checkpoint"])] = val
    return out


def load_block_a(root: Path, runs):
    sc = {}
    for opp in OPPS:
        rows = read_rows([
            root / f"best_config_scripted_eval_100/default_180_mlp/{opp}/checkpoint_diagnostics.csv",
            root / f"seeds234/bestcfg/_scripted_eval_100/default_180_mlp/{opp}/checkpoint_diagnostics.csv",
        ])
        sc[opp] = {"surv": series(rows, "survival_percentage", runs),
                   "win": series(rows, "win_percentage", runs),
                   "cash": series(rows, "mean_terminal_cash", runs)}
    sp_rows = read_rows([
        root / "best_config_selfplay_survival_50/default_180_mlp/selfplay_survival_diagnostics.csv",
        root / "seeds234/bestcfg/_selfplay_survival_50/default_180_mlp/selfplay_survival_diagnostics.csv",
    ])
    sp = {"surv": series(sp_rows, "survival_percentage", runs),
          "price_entropy": series(sp_rows, "price_given_post_entropy", runs)}
    return sc, sp


def load_block_b(heldout_root: Path, runs):
    sc = {}
    for opp in OPPS:
        paths = sorted((heldout_root / "_scripted_eval_100_heldout").glob(f"*/{opp}/*/checkpoint_diagnostics.csv"))
        rows = read_rows(paths)
        sc[opp] = {"surv": series(rows, "survival_percentage", runs),
                   "win": series(rows, "win_percentage", runs),
                   "cash": series(rows, "mean_terminal_cash", runs)}
    return sc


def load_peers(heldout_root: Path, runs):
    """population name -> {(algo, seed) -> {ckpt -> survival%}} (+ win, cash)."""
    pops = {}
    base = heldout_root / "_heldout_peers_100"
    if not base.is_dir():
        return pops
    for pop_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        surv, win, cash, opp_desc = {}, {}, {}, set()
        for raw in sorted((pop_dir / "raw").glob("*.json")):
            report = json.loads(raw.read_text())
            for row in report["summary_rows"]:
                m = LABEL_RE.match(row["focal"])
                if not m:
                    continue
                key = (m.group(1), int(m.group(2)))
                if key not in runs:
                    continue
                c = int(m.group(3))
                surv.setdefault(key, {})[c] = float(row["survival_percentage"])
                win.setdefault(key, {})[c] = float(row["win_percentage"])
                cash.setdefault(key, {})[c] = float(row["mean_terminal_cash"])
                opp_desc.add(row["opponent"])
        pops[pop_dir.name] = {"surv": surv, "win": win, "cash": cash,
                              "opponent": sorted(opp_desc)}
    return pops


# ------------------------------------------------------------- statistics --

def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else float("nan")


def t_ci(vals):
    vals = list(vals)
    n = len(vals)
    m = mean(vals)
    if n < 2:
        return m, float("nan"), float("nan")
    half = T975[n - 1] * statistics.stdev(vals) / math.sqrt(n)
    return m, m - half, m + half


def pearson(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    return pearson(rank(x), rank(y))


def argmax_ckpt(s):
    best = max(s[c] for c in CKPTS)
    return max(c for c in CKPTS if s[c] == best)


def complete(sig, runs):
    return [k for k in runs if k in sig and all(c in sig[k] for c in CKPTS)]


def verdict_row(name, sig, runs, kind="good"):
    keys = complete(sig, runs)
    deltas = {k: sig[k][1200] - sig[k][700] for k in keys}
    band = 0.05 if kind == "entropy" else FLAT_BAND
    n_up = sum(1 for d in deltas.values() if d > band)
    n_down = sum(1 for d in deltas.values() if d < -band)
    if kind == "entropy":
        verdict = "certifies (sharpening)" if n_down > len(keys) / 2 else (
            "flags" if n_up > len(keys) / 2 else "silent")
    else:
        verdict = ("certifies improvement" if n_up > len(keys) / 2 and n_up > n_down
                   else "FLAGS DECLINE" if n_down > len(keys) / 2 and n_down > n_up
                   else "silent / mixed")
    m, lo, hi = t_ci(deltas.values())
    return {"signal": name, "n_runs": len(keys),
            "mean_at_700": round(mean(sig[k][700] for k in keys), 2),
            "mean_at_1200": round(mean(sig[k][1200] for k in keys), 2),
            "mean_delta": round(m, 2), "delta_ci95_lo": round(lo, 2), "delta_ci95_hi": round(hi, 2),
            "runs_up": n_up, "runs_down": n_down, "verdict": verdict}


def trajectory(sig, runs, label):
    rows = []
    keys = complete(sig, runs)
    for c in CKPTS:
        vals = [sig[k][c] for k in keys]
        m, lo, hi = t_ci(vals)
        rows.append({"set": label, "checkpoint": c, "n_runs": len(vals), "mean": round(m, 2),
                     "ci95_lo": round(lo, 2), "ci95_hi": round(hi, 2),
                     "min": round(min(vals), 1), "max": round(max(vals), 1)})
    return rows


def portfolio(sc, runs, opps):
    return {k: {c: mean(sc[o]["surv"][k][c] for o in opps) for c in CKPTS}
            for k in runs if all(k in sc[o]["surv"] and all(c in sc[o]["surv"][k] for c in CKPTS) for o in opps)}


def ablation(sc, sp, runs, label):
    rows = [
        verdict_row("self-play survival", sp["surv"], runs),
        verdict_row("self-play price|POST entropy", sp["price_entropy"], runs, "entropy"),
        verdict_row("cross-play cash vs v4", sc["v4"]["cash"], runs),
        verdict_row("cross-play survival vs v3", sc["v3"]["surv"], runs),
        verdict_row("cross-play survival vs v2", sc["v2"]["surv"], runs),
        verdict_row("cross-play survival vs v5", sc["v5"]["surv"], runs),
        verdict_row("cross-play survival vs v4", sc["v4"]["surv"], runs),
        verdict_row("cross-play win vs v4", sc["v4"]["win"], runs),
        verdict_row("portfolio mean survival (v2..v5)", portfolio(sc, runs, OPPS), runs),
        verdict_row("portfolio mean without v4 (v2,v3,v5)", portfolio(sc, runs, ("v2", "v3", "v5")), runs),
    ]
    for r in rows:
        r["block"] = label
    return rows


def selectors(sc, sp):
    """Selection rules as functions run -> checkpoint, defined on a block."""
    ho = portfolio(sc, list(sc["v4"]["surv"]), ("v2", "v3", "v5"))
    return {
        "final checkpoint": lambda k: 1200,
        "self-play survival": lambda k: argmax_ckpt(sp["surv"][k]),
        "cross-play cash vs v4": lambda k: argmax_ckpt(sc["v4"]["cash"][k]),
        "portfolio without v4 (v2,v3,v5)": lambda k: argmax_ckpt(ho[k]),
        "oracle (v4 survival)": lambda k: argmax_ckpt(sc["v4"]["surv"][k]),
    }


def regret_cv(sc_a, sp, sc_b, runs):
    """Two-fold cross-validated regret. Selection on one block, scoring on the
    other; oracle picked the same way (so oracle and selector are treated
    symmetrically). Self-play selection uses the single self-play series in
    both folds (its games are disjoint from both scripted blocks)."""
    runs = [k for k in runs if k in complete(sc_a["v4"]["surv"], runs) and k in complete(sc_b["v4"]["surv"], runs)
            and k in complete(sp["surv"], runs)]
    folds = [("pick A -> score B", selectors(sc_a, sp), sc_b["v4"]["surv"]),
             ("pick B -> score A", selectors(sc_b, sp), sc_a["v4"]["surv"])]
    rows = []
    per_sel = {}
    for fold_name, sel, score in folds:
        oracle_pick = {k: sel["oracle (v4 survival)"](k) for k in runs}
        oracle_val = {k: score[k][oracle_pick[k]] for k in runs}
        for name, pick in sel.items():
            dep = {k: score[k][pick(k)] for k in runs}
            reg = {k: oracle_val[k] - dep[k] for k in runs}
            m, lo, hi = t_ci(reg.values())
            rows.append({"fold": fold_name, "selector": name, "n_runs": len(runs),
                         "mean_deployed_v4": round(mean(dep.values()), 2),
                         "mean_regret": round(m, 2), "regret_ci95_lo": round(lo, 2),
                         "regret_ci95_hi": round(hi, 2), "worst_run_regret": round(max(reg.values()), 1)})
            per_sel.setdefault(name, []).append((dep, reg))
    # average the two folds per run, then CI over runs
    for name, (fa, fb) in {n: v for n, v in per_sel.items()}.items():
        dep = {k: (fa[0][k] + fb[0][k]) / 2 for k in runs}
        reg = {k: (fa[1][k] + fb[1][k]) / 2 for k in runs}
        m, lo, hi = t_ci(reg.values())
        rows.append({"fold": "two-fold mean", "selector": name, "n_runs": len(runs),
                     "mean_deployed_v4": round(mean(dep.values()), 2),
                     "mean_regret": round(m, 2), "regret_ci95_lo": round(lo, 2),
                     "regret_ci95_hi": round(hi, 2), "worst_run_regret": round(max(reg.values()), 1)})
    # in-sample reference (the submitted paper's construction) on each block
    for blk, sc in (("A in-sample", sc_a), ("B in-sample", sc_b)):
        sel = selectors(sc, sp)
        v4 = sc["v4"]["surv"]
        oracle_val = {k: v4[k][argmax_ckpt(v4[k])] for k in runs}
        for name, pick in sel.items():
            dep = {k: v4[k][pick(k)] for k in runs}
            reg = {k: oracle_val[k] - dep[k] for k in runs}
            m, lo, hi = t_ci(reg.values())
            rows.append({"fold": blk, "selector": name, "n_runs": len(runs),
                         "mean_deployed_v4": round(mean(dep.values()), 2),
                         "mean_regret": round(m, 2), "regret_ci95_lo": round(lo, 2),
                         "regret_ci95_hi": round(hi, 2), "worst_run_regret": round(max(reg.values()), 1)})
    return rows


def cell_agreement(sc_a, sc_b, runs):
    out = []
    for opp in OPPS:
        a, b = sc_a[opp]["surv"], sc_b[opp]["surv"]
        keys = [k for k in runs if k in complete(a, runs) and k in complete(b, runs)]
        xa = [a[k][c] for k in keys for c in CKPTS]
        xb = [b[k][c] for k in keys for c in CKPTS]
        if not xa:
            continue
        diffs = [p - q for p, q in zip(xb, xa)]
        out.append({"opponent": opp, "n_cells": len(xa), "pearson_r": round(pearson(xa, xb), 3),
                    "mean_B_minus_A": round(mean(diffs), 2),
                    "mean_abs_diff": round(mean(abs(d) for d in diffs), 2),
                    "sd_diff": round(statistics.stdev(diffs), 2) if len(diffs) > 1 else float("nan"),
                    "mean_A": round(mean(xa), 2), "mean_B": round(mean(xb), 2)})
    return out


def peer_analysis(pops, sc_a, sc_b, runs, label):
    rows, traj = [], []
    for pop, d in pops.items():
        sig = d["surv"]
        keys = complete(sig, runs)
        if not keys:
            continue
        v = verdict_row(f"survival vs held-out {pop}", sig, keys)
        v["block"] = label
        v["opponent"] = ";".join(d["opponent"])
        # peak-based decline
        peak_drop = {k: round(sig[k][1200] - sig[k][argmax_ckpt(sig[k])], 1) for k in keys}
        v["runs_declining_from_peak_gt_band"] = sum(1 for x in peak_drop.values() if x < -FLAT_BAND)
        v["mean_peak"] = round(mean(sig[k][argmax_ckpt(sig[k])] for k in keys), 2)
        v["median_peak_ckpt"] = sorted(argmax_ckpt(sig[k]) for k in keys)[len(keys) // 2]
        # correlation of the run-mean trajectory with the v4 trajectory (both blocks)
        mp = [mean(sig[k][c] for k in keys) for c in CKPTS]
        for blk, sc in (("A", sc_a), ("B", sc_b)):
            k4 = [k for k in keys if k in complete(sc["v4"]["surv"], keys)]
            if k4:
                m4 = [mean(sc["v4"]["surv"][k][c] for k in k4) for c in CKPTS]
                v[f"spearman_mean_traj_vs_v4_block{blk}"] = round(spearman(mp, m4), 3)
                # per-run series correlation, averaged
                rs = [spearman([sig[k][c] for c in CKPTS], [sc["v4"]["surv"][k][c] for c in CKPTS]) for k in k4]
                v[f"mean_per_run_spearman_vs_v4_block{blk}"] = round(mean(r for r in rs if not math.isnan(r)), 3)
        v["win_700"] = round(mean(d["win"][k][700] for k in keys), 2)
        v["win_1200"] = round(mean(d["win"][k][1200] for k in keys), 2)
        v["cash_700"] = round(mean(d["cash"][k][700] for k in keys), 2)
        v["cash_1200"] = round(mean(d["cash"][k][1200] for k in keys), 2)
        v["per_run_delta_700_1200"] = json.dumps({f"{k[0]}_s{k[1]}": round(sig[k][1200] - sig[k][700], 1) for k in keys})
        rows.append(v)
        traj += trajectory(sig, keys, f"{label}: held-out {pop}")
    return rows, traj


# ------------------------------------------------------------------ figure --

def make_figure(traj_rows, fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("matplotlib unavailable, skipping figure:", exc)
        return
    sets = {}
    for r in traj_rows:
        sets.setdefault(r["set"], []).append(r)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
    INK, INK2 = "#0b0b0b", "#52514e"
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e7e6e2", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xticks(CKPTS[1::2])
        ax.set_xlabel("training checkpoint (PPO iteration)")
        ax.set_ylim(0, 100)
    ax1.set_ylabel("focal-seat survival (%)")

    def band(ax, rows, color, label, ls="-"):
        m = [r["mean"] for r in rows]
        lo = [r["ci95_lo"] for r in rows]
        hi = [r["ci95_hi"] for r in rows]
        ax.fill_between(CKPTS, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(CKPTS, m, color=color, lw=1.9, marker="o", ms=3, ls=ls, label=label, zorder=3)

    pal = ["#eb6834", "#b8431a", "#2a78d6", "#1f9d8a", "#7b5cd6", "#c59a00"]
    i = 0
    for name, rows in sets.items():
        if "3xv4" in name and "healthy8" in name:
            blk = "original seeds" if "block A" in name else "held-out seeds"
            band(ax1, rows, pal[i % 2], f"vs 3x v4, {blk}", "-" if "block A" in name else "--")
            i += 1
    ax1.set_title("reference population, two disjoint game-seed blocks", fontsize=9, color=INK2)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    j = 2
    for name, rows in sets.items():
        if "held-out" in name and "healthy8" in name:
            raw = name.split("held-out ")[-1]
            pop = {"3x_ep_ippo_s2": "3 copies of a held-out IPPO peer",
                   "3x_ep_mappo_s2": "3 copies of a held-out MAPPO peer",
                   "mixed_ep3": "mixed table of 3 held-out peers"}.get(raw, raw)
            band(ax2, rows, pal[j % len(pal)], pop)
            j += 1
    ax2.set_title("held-out learned populations (never used in any audit)", fontsize=9, color=INK2)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"f6_wv_heldout_trajectories.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", fig_dir / "f6_wv_heldout_trajectories.{png,pdf}")


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--heldout-root", type=Path, default=Path("results/camera_ready_wv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/camera_ready_wv/analysis"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    abl_rows, traj_rows, reg_rows, agree_rows, peer_rows = [], [], [], [], []
    for label, runs in (("healthy8", HEALTHY), ("all10", ALL10)):
        sc_a, sp = load_block_a(args.results_root, runs)
        sc_b = load_block_b(args.heldout_root, runs)
        pops = load_peers(args.heldout_root, runs)
        have_b = bool(complete(sc_b["v4"]["surv"], runs))
        traj_rows += trajectory(sc_a["v4"]["surv"], runs, f"{label}: survival vs 3xv4, block A")
        abl_rows += ablation(sc_a, sp, runs, f"{label} block A")
        if have_b:
            traj_rows += trajectory(sc_b["v4"]["surv"], runs, f"{label}: survival vs 3xv4, block B")
            abl_rows += ablation(sc_b, sp, runs, f"{label} block B")
            for r in regret_cv(sc_a, sp, sc_b, runs):
                r["set"] = label
                reg_rows.append(r)
            for r in cell_agreement(sc_a, sc_b, runs):
                r["set"] = label
                agree_rows.append(r)
        pr, pt = peer_analysis(pops, sc_a, sc_b if have_b else sc_a, runs, label)
        peer_rows += pr
        traj_rows += pt

    write_csv(args.out_dir / "ablation_blocks.csv", abl_rows)
    write_csv(args.out_dir / "trajectories_ci.csv", traj_rows)
    write_csv(args.out_dir / "regret_crossvalidated.csv", reg_rows)
    write_csv(args.out_dir / "cell_agreement_A_vs_B.csv", agree_rows)
    write_csv(args.out_dir / "heldout_peer_populations.csv", peer_rows)
    make_figure(traj_rows, args.fig_dir)

    print("\n== (1) Table 2 verdicts per block ==")
    for r in abl_rows:
        print(f"  [{r['block']:<16}] {r['signal']:<40} {r['mean_at_700']:>7} -> {r['mean_at_1200']:>7} "
              f"(up {r['runs_up']}, dn {r['runs_down']}/{r['n_runs']}) d={r['mean_delta']} "
              f"[{r['delta_ci95_lo']},{r['delta_ci95_hi']}] {r['verdict']}")
    print("\n== cell agreement A vs B ==")
    for r in agree_rows:
        print(f"  [{r['set']}] {r['opponent']}: r={r['pearson_r']} B-A={r['mean_B_minus_A']} "
              f"|d|={r['mean_abs_diff']} sd={r['sd_diff']} (A {r['mean_A']}, B {r['mean_B']})")
    print("\n== (2) regret ==")
    for r in reg_rows:
        print(f"  [{r['set']:<8}] {r['fold']:<16} {r['selector']:<34} deploy {r['mean_deployed_v4']:>6} "
              f"regret {r['mean_regret']:>6} [{r['regret_ci95_lo']}, {r['regret_ci95_hi']}] worst {r['worst_run_regret']}")
    print("\n== (3) held-out learned populations ==")
    for r in peer_rows:
        print(f"  [{r['block']}] {r['signal']:<42} {r['mean_at_700']:>6} -> {r['mean_at_1200']:>6} "
              f"(up {r['runs_up']}, dn {r['runs_down']}/{r['n_runs']}) {r['verdict']}; peak {r['mean_peak']} "
              f"@~{r['median_peak_ckpt']}, {r['runs_declining_from_peak_gt_band']} decline from peak; "
              f"rho(mean traj, v4 A)={r.get('spearman_mean_traj_vs_v4_blockA')}; "
              f"win {r['win_700']}->{r['win_1200']}; cash {r['cash_700']}->{r['cash_1200']}")
    print("\n== trajectories (healthy8) ==")
    for r in traj_rows:
        if r["set"].startswith("healthy8"):
            print(f"  {r['set']:<50} {r['checkpoint']:>4}: {r['mean']:>6} [{r['ci95_lo']}, {r['ci95_hi']}]")
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
