"""E25: the verifier portfolio, longitudinally — four WV analyses on E11 data.

All four run on the existing best-config longitudinal diagnostics (the E11
rich-and-dead data: 8 healthy default_180_mlp runs, checkpoints 100..1200,
100 scripted-eval games/cell vs each of v2..v5, 50 self-play games/ckpt):

(a) PORTFOLIO ABLATION — for every verification signal, the verdict it
    renders on the ckpt-700 -> 1200 transition (the F1 collapse): certifies
    improvement / flags decline / silent, with per-run deltas and the count
    of runs each signal flags. Operationalizes "composing heterogeneous
    signals detects what each single signal certifies".
(b) CHECKPOINT-SELECTION REGRET — deploy the checkpoint each verifier
    selects (argmax over ckpts 100..1200, last on ties); regret = deployment
    survival (vs v4, the hardest archetype) given up relative to the oracle.
    The held-out portfolio selector uses only {v2,v3,v5} — it never sees v4.
(c) HELD-OUT VERIFIER — does the reduced portfolio mean({v2,v3,v5} survival)
    flag the transition that the held-out v4 exhibits?
(d) DETECTION BUDGET — per run, the games/cell needed to detect its
    peak -> 1200 v4-survival drop (one-sided two-proportion z, alpha=.05,
    power=.80, independent-binomial approximation — conservative vs the
    paired-seed design actually used).

Usage:
    python eval/analyze_wv_portfolio_longitudinal.py \
        --results-root results --out-dir results/wv_portfolio_longitudinal
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HEALTHY = [("ippo", 0), ("ippo", 1), ("ippo", 2), ("ippo", 3), ("ippo", 4),
           ("mappo", 0), ("mappo", 2), ("mappo", 3)]
CKPTS = list(range(100, 1201, 100))
OPPS = ("v2", "v3", "v4", "v5")
FLAT_BAND = 2.0  # pts; |delta| <= band reads as "silent/flat"
Z_A = 1.6449  # one-sided alpha = 0.05
Z_B = 0.8416  # power = 0.80


def read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if p.is_file():
            rows.extend(csv.DictReader(open(p)))
    if not rows:
        raise SystemExit(f"no data found in any of: {paths}")
    return rows


def series(rows: list[dict], field: str) -> dict[tuple[str, int], dict[int, float]]:
    """(algo, seed) -> {checkpoint: value}."""
    out: dict[tuple[str, int], dict[int, float]] = {}
    for r in rows:
        key = (r["algorithm"], int(r["seed"]))
        if key not in dict.fromkeys(HEALTHY):
            continue
        try:
            val = float(r[field])
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault(key, {})[int(r["checkpoint"])] = val
    return out


def load_all(root: Path):
    sc = {}
    for opp in OPPS:
        rows = read_rows([
            root / f"best_config_scripted_eval_100/default_180_mlp/{opp}/checkpoint_diagnostics.csv",
            root / f"seeds234/bestcfg/_scripted_eval_100/default_180_mlp/{opp}/checkpoint_diagnostics.csv",
        ])
        sc[opp] = {
            "surv": series(rows, "survival_percentage"),
            "win": series(rows, "win_percentage"),
            "cash": series(rows, "mean_terminal_cash"),
        }
    sp_rows = read_rows([
        root / "best_config_selfplay_survival_50/default_180_mlp/selfplay_survival_diagnostics.csv",
        root / "seeds234/bestcfg/_selfplay_survival_50/default_180_mlp/selfplay_survival_diagnostics.csv",
    ])
    sp = {
        "surv": series(sp_rows, "survival_percentage"),
        "price_entropy": series(sp_rows, "price_given_post_entropy"),
    }
    return sc, sp


def runs_with_full_series(sig: dict[tuple[str, int], dict[int, float]]):
    return [k for k in HEALTHY if k in sig and all(c in sig[k] for c in CKPTS)]


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# (a) portfolio ablation on the 700 -> 1200 transition
# ---------------------------------------------------------------------------


def ablation(sc, sp) -> list[dict]:
    signals: list[tuple[str, dict, str]] = [
        ("self-play survival", sp["surv"], "good"),
        ("self-play price|POST entropy (fingerprint)", sp["price_entropy"], "entropy"),
        ("cross-play cash vs v4 (raw score)", sc["v4"]["cash"], "good"),
        ("cross-play survival vs v4", sc["v4"]["surv"], "good"),
        ("cross-play survival vs v2", sc["v2"]["surv"], "good"),
        ("cross-play survival vs v3", sc["v3"]["surv"], "good"),
        ("cross-play survival vs v5", sc["v5"]["surv"], "good"),
        ("cross-play win vs v4", sc["v4"]["win"], "good"),
        ("portfolio mean survival (v2..v5)", None, "good"),  # built below
        ("held-out portfolio mean survival (v2,v3,v5)", None, "good"),
    ]
    built: dict[str, dict] = {}
    for name, opps in [("portfolio mean survival (v2..v5)", OPPS),
                       ("held-out portfolio mean survival (v2,v3,v5)", ("v2", "v3", "v5"))]:
        agg: dict[tuple[str, int], dict[int, float]] = {}
        for k in HEALTHY:
            if all(k in sc[o]["surv"] for o in opps):
                agg[k] = {c: mean(sc[o]["surv"][k][c] for o in opps)
                          for c in CKPTS if all(c in sc[o]["surv"][k] for o in opps)}
        built[name] = agg

    table = []
    for name, sig, kind in signals:
        sig = built.get(name, sig)
        keys = runs_with_full_series(sig)
        deltas = {k: sig[k][1200] - sig[k][700] for k in keys}
        band = 0.05 if kind == "entropy" else FLAT_BAND  # nats vs pct-points
        n_up = sum(1 for d in deltas.values() if d > band)
        n_down = sum(1 for d in deltas.values() if d < -band)
        if kind == "entropy":
            # sharpening (falling entropy) is the "learning is healthy" read
            verdict = "certifies (sharpening)" if n_down > len(keys) / 2 else (
                "flags" if n_up > len(keys) / 2 else "silent")
        else:
            verdict = ("certifies improvement" if n_up > len(keys) / 2 and n_up > n_down
                       else "FLAGS DECLINE" if n_down > len(keys) / 2 and n_down > n_up
                       else "silent / mixed")
        table.append({
            "signal": name,
            "n_runs": len(keys),
            "mean_at_700": round(mean(sig[k][700] for k in keys), 2),
            "mean_at_1200": round(mean(sig[k][1200] for k in keys), 2),
            "mean_delta": round(mean(deltas.values()), 2),
            "runs_up": n_up,
            "runs_down": n_down,
            "verdict_700_to_1200": verdict,
        })
    return table


# ---------------------------------------------------------------------------
# (b) checkpoint-selection regret (deployment = survival vs v4)
# ---------------------------------------------------------------------------


def argmax_ckpt(series_for_run: dict[int, float]) -> int:
    best = max(series_for_run[c] for c in CKPTS)
    return max(c for c in CKPTS if series_for_run[c] == best)  # last on ties


def selection_regret(sc, sp) -> list[dict]:
    v4 = sc["v4"]["surv"]
    heldout = {}
    for k in HEALTHY:
        heldout[k] = {c: mean(sc[o]["surv"][k][c] for o in ("v2", "v3", "v5"))
                      for c in CKPTS}
    selectors = {
        "final checkpoint (default practice)": lambda k: 1200,
        "self-play survival": lambda k: argmax_ckpt(sp["surv"][k]),
        "cross-play cash vs v4 (raw score)": lambda k: argmax_ckpt(sc["v4"]["cash"][k]),
        "held-out portfolio survival (v2,v3,v5)": lambda k: argmax_ckpt(heldout[k]),
        "oracle (v4 survival)": lambda k: argmax_ckpt(v4[k]),
    }
    keys = [k for k in HEALTHY
            if k in v4 and k in sp["surv"]
            and all(c in v4[k] and c in sp["surv"][k] for c in CKPTS)]
    rows = []
    oracle_dep = {k: v4[k][argmax_ckpt(v4[k])] for k in keys}
    for name, pick in selectors.items():
        picks = {k: pick(k) for k in keys}
        dep = {k: v4[k][picks[k]] for k in keys}
        rows.append({
            "selector": name,
            "median_ckpt_picked": sorted(picks.values())[len(picks) // 2],
            "mean_deployment_surv_v4": round(mean(dep.values()), 2),
            "mean_regret_vs_oracle": round(
                mean(oracle_dep[k] - dep[k] for k in keys), 2),
            "worst_run_regret": round(
                max(oracle_dep[k] - dep[k] for k in keys), 2),
            "per_run_ckpt": json.dumps(
                {f"{a}_s{s}": picks[(a, s)] for a, s in keys}),
        })
    return rows


# ---------------------------------------------------------------------------
# (c) held-out verifier detection + (d) detection budget
# ---------------------------------------------------------------------------


def two_prop_n(p1: float, p2: float) -> float:
    """Games per cell (per checkpoint) to detect p1 -> p2, one-sided
    alpha=.05, power=.80, independent binomial."""
    if p1 == p2:
        return float("inf")
    pbar = (p1 + p2) / 2
    num = (Z_A * math.sqrt(2 * pbar * (1 - pbar))
           + Z_B * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
    return (num / abs(p1 - p2)) ** 2


def detection(sc) -> tuple[list[dict], dict]:
    v4 = sc["v4"]["surv"]
    rows = []
    budgets_v4, budgets_ho = [], []
    n_flag_ho = 0
    keys = [k for k in HEALTHY if k in v4 and all(c in v4[k] for c in CKPTS)]
    for k in keys:
        peak_c = argmax_ckpt(v4[k])
        p_peak, p_end = v4[k][peak_c] / 100, v4[k][1200] / 100
        n_v4 = two_prop_n(p_peak, p_end) if p_peak > p_end else float("inf")
        ho = {c: mean(sc[o]["surv"][k][c] for o in ("v2", "v3", "v5"))
              for c in CKPTS}
        ho_peak_c = argmax_ckpt(ho)
        q_peak, q_end = ho[ho_peak_c] / 100, ho[1200] / 100
        ho_flags = (ho[ho_peak_c] - ho[1200]) > FLAT_BAND
        n_flag_ho += int(ho_flags)
        n_ho = two_prop_n(q_peak, q_end) if q_peak > q_end else float("inf")
        # held-out portfolio pools 3 archetype cells per checkpoint
        rows.append({
            "run": f"{k[0]}_s{k[1]}",
            "v4_peak_ckpt": peak_c, "v4_peak": round(p_peak * 100, 1),
            "v4_at_1200": round(p_end * 100, 1),
            "games_per_cell_to_detect_v4_drop": (
                round(n_v4) if math.isfinite(n_v4) else "n/a (no drop)"),
            "heldout_flags_decline": ho_flags,
            "heldout_peak_ckpt": ho_peak_c,
            "heldout_peak": round(q_peak * 100, 1),
            "heldout_at_1200": round(q_end * 100, 1),
            "games_per_cell_to_detect_heldout_drop": (
                round(n_ho / 3) if math.isfinite(n_ho) else "n/a (no drop)"),
        })
        if math.isfinite(n_v4):
            budgets_v4.append(n_v4)
        if math.isfinite(n_ho):
            budgets_ho.append(n_ho / 3)
    summary = {
        "runs_where_heldout_portfolio_flags": f"{n_flag_ho}/{len(keys)}",
        "median_games_per_cell_v4": round(sorted(budgets_v4)[len(budgets_v4) // 2]) if budgets_v4 else None,
        "max_games_per_cell_v4": round(max(budgets_v4)) if budgets_v4 else None,
        "median_games_per_cell_heldout": round(sorted(budgets_ho)[len(budgets_ho) // 2]) if budgets_ho else None,
        "note": ("independent-binomial approximation; the actual protocol "
                 "pairs seeds across checkpoints, so these are upper bounds. "
                 "held-out budget divides by its 3 pooled archetype cells."),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/wv_portfolio_longitudinal"))
    args = ap.parse_args()
    sc, sp = load_all(args.results_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    abl = ablation(sc, sp)
    write_csv(args.out_dir / "portfolio_ablation.csv", abl)
    reg = selection_regret(sc, sp)
    write_csv(args.out_dir / "selection_regret.csv", reg)
    det_rows, det_summary = detection(sc)
    write_csv(args.out_dir / "detection_budget.csv", det_rows)
    (args.out_dir / "detection_budget_summary.json").write_text(
        json.dumps(det_summary, indent=2))

    print("== (a) portfolio ablation: verdict on ckpt 700 -> 1200 ==")
    for r in abl:
        print(f"  {r['signal']:<46} {r['mean_at_700']:>7} -> {r['mean_at_1200']:>7} "
              f"(dn {r['runs_down']}/{r['n_runs']}, up {r['runs_up']}) {r['verdict_700_to_1200']}")
    print("\n== (b) checkpoint-selection regret (deployment = v4 survival) ==")
    for r in reg:
        print(f"  {r['selector']:<42} picks ~{r['median_ckpt_picked']}: "
              f"deploy {r['mean_deployment_surv_v4']}%, regret {r['mean_regret_vs_oracle']} "
              f"(worst {r['worst_run_regret']})")
    print("\n== (c)+(d) held-out verifier + detection budget ==")
    for r in det_rows:
        print(f"  {r['run']:<9} v4 {r['v4_peak']}@{r['v4_peak_ckpt']} -> {r['v4_at_1200']} "
              f"(n/cell {r['games_per_cell_to_detect_v4_drop']}); held-out flags: "
              f"{r['heldout_flags_decline']} ({r['heldout_peak']}@{r['heldout_peak_ckpt']} -> {r['heldout_at_1200']}, "
              f"n/cell {r['games_per_cell_to_detect_heldout_drop']})")
    print("\nsummary:", json.dumps(det_summary, indent=2))


if __name__ == "__main__":
    main()
