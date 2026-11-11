"""Camera-ready robustness of the Figure 1 headline (offline, existing E11 data).

Camera-ready additions (Who Verifies the Agents? @ NeurIPS 2026, paper 302):

  * CI reporting on the n=8 headline (camera-ready request): per-checkpoint mean
    survival vs 3x v4 with 95% t-intervals over runs, and the paired
    700 -> 1200 difference with its interval.
  * 36.8 vs 43.6 reconciliation (sqFy): 36.8 is the fixed-checkpoint mean
    at ckpt 700 over the 8 runs; 43.6 is the mean over runs of each run's own
    peak (per-run argmax). Same for "best self-play 21.9" vs "last 17.0".
  * Selection concern (camera-ready request): recompute the headline, the Table 2
    verdicts and the selection regret with ALL TEN runs (the marginal
    mappo_s1 and the failed mappo_s4 included) next to the eight healthy.

Reads the same campaign CSVs as eval/plot_rich_and_dead_bestconfig.py and
eval/analyze_wv_portfolio_longitudinal.py; nothing hand-entered.

Usage:
    python eval/analyze_camera_ready_headline_ci.py \
        --results-root results --out-dir results/camera_ready_wv/headline_ci
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

HEALTHY = [("ippo", 0), ("ippo", 1), ("ippo", 2), ("ippo", 3), ("ippo", 4),
           ("mappo", 0), ("mappo", 2), ("mappo", 3)]
ALL10 = HEALTHY + [("mappo", 1), ("mappo", 4)]
CKPTS = list(range(100, 1201, 100))
OPPS = ("v2", "v3", "v4", "v5")
FLAT_BAND = 2.0
# two-sided 97.5% t quantiles, df = n-1
T975 = {7: 2.3646, 9: 2.2622, 6: 2.4469, 8: 2.3060, 5: 2.5706, 4: 2.7764}


def read_rows(paths):
    rows = []
    for p in paths:
        if p.is_file():
            rows.extend(csv.DictReader(open(p)))
    if not rows:
        raise SystemExit(f"no data in {paths}")
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


def load(root: Path, runs):
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


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else float("nan")


def t_ci(vals):
    vals = list(vals)
    n = len(vals)
    m = mean(vals)
    if n < 2:
        return m, float("nan"), float("nan"), float("nan")
    sd = statistics.stdev(vals)
    half = T975[n - 1] * sd / math.sqrt(n)
    return m, m - half, m + half, sd


def argmax_ckpt(s):
    best = max(s[c] for c in CKPTS)
    return max(c for c in CKPTS if s[c] == best)


def trajectory_table(sig, runs, label):
    rows = []
    for c in CKPTS:
        vals = [sig[k][c] for k in runs if k in sig and c in sig[k]]
        m, lo, hi, sd = t_ci(vals)
        rows.append({"set": label, "checkpoint": c, "n_runs": len(vals), "mean": round(m, 2),
                     "ci95_lo": round(lo, 2), "ci95_hi": round(hi, 2), "sd": round(sd, 2),
                     "min": round(min(vals), 1), "max": round(max(vals), 1)})
    return rows


def paired_delta(sig, runs, a=700, b=1200):
    d = [sig[k][b] - sig[k][a] for k in runs if k in sig]
    m, lo, hi, sd = t_ci(d)
    n_down = sum(1 for x in d if x < -FLAT_BAND)
    n_up = sum(1 for x in d if x > FLAT_BAND)
    return {"from": a, "to": b, "n_runs": len(d), "mean_delta": round(m, 2),
            "ci95_lo": round(lo, 2), "ci95_hi": round(hi, 2), "runs_down": n_down,
            "runs_up": n_up, "per_run": {f"{k[0]}_s{k[1]}": round(sig[k][b] - sig[k][a], 1)
                                        for k in runs if k in sig}}


def reconcile(sc, sp, runs):
    v4 = sc["v4"]["surv"]
    out = {}
    out["fixed_ckpt_mean_700_v4"] = round(mean(v4[k][700] for k in runs), 2)
    out["fixed_ckpt_mean_1200_v4"] = round(mean(v4[k][1200] for k in runs), 2)
    out["best_fixed_ckpt_mean_v4"] = max((round(mean(v4[k][c] for k in runs), 2), c) for c in CKPTS)
    peaks = {k: argmax_ckpt(v4[k]) for k in runs}
    out["per_run_peak_ckpt_v4"] = {f"{k[0]}_s{k[1]}": peaks[k] for k in runs}
    out["mean_of_per_run_peaks_v4"] = round(mean(v4[k][peaks[k]] for k in runs), 2)
    sp_pick = {k: argmax_ckpt(sp["surv"][k]) for k in runs}
    out["selfplay_selected_ckpt"] = {f"{k[0]}_s{k[1]}": sp_pick[k] for k in runs}
    out["v4_at_selfplay_selected_mean"] = round(mean(v4[k][sp_pick[k]] for k in runs), 2)
    out["note"] = ("36.8 = mean over runs of survival at the single checkpoint 700; 43.6 = mean "
                   "over runs of each run's own best checkpoint (per-run argmax, so it exceeds "
                   "every fixed-checkpoint mean); 21.9 = mean v4 survival at each run's "
                   "self-play-selected checkpoint; 17.0 = mean at checkpoint 1200.")
    return out


def ablation(sc, sp, runs):
    built = {}
    for name, opps in [("portfolio mean survival (v2..v5)", OPPS),
                       ("held-out portfolio mean survival (v2,v3,v5)", ("v2", "v3", "v5"))]:
        built[name] = {k: {c: mean(sc[o]["surv"][k][c] for o in opps) for c in CKPTS}
                       for k in runs if all(k in sc[o]["surv"] for o in opps)}
    signals = [
        ("self-play survival", sp["surv"], "good"),
        ("self-play price|POST entropy (fingerprint)", sp["price_entropy"], "entropy"),
        ("cross-play cash vs v4 (raw score)", sc["v4"]["cash"], "good"),
        ("cross-play survival vs v3", sc["v3"]["surv"], "good"),
        ("cross-play survival vs v2", sc["v2"]["surv"], "good"),
        ("cross-play survival vs v5", sc["v5"]["surv"], "good"),
        ("cross-play survival vs v4", sc["v4"]["surv"], "good"),
        ("cross-play win vs v4", sc["v4"]["win"], "good"),
        ("portfolio mean survival (v2..v5)", built["portfolio mean survival (v2..v5)"], "good"),
        ("held-out portfolio mean survival (v2,v3,v5)",
         built["held-out portfolio mean survival (v2,v3,v5)"], "good"),
    ]
    table = []
    for name, sig, kind in signals:
        keys = [k for k in runs if k in sig and all(c in sig[k] for c in CKPTS)]
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
        table.append({"signal": name, "n_runs": len(keys),
                      "mean_at_700": round(mean(sig[k][700] for k in keys), 2),
                      "mean_at_1200": round(mean(sig[k][1200] for k in keys), 2),
                      "runs_up": n_up, "runs_down": n_down, "verdict": verdict})
    return table


def regret(sc, sp, runs):
    v4 = sc["v4"]["surv"]
    heldout = {k: {c: mean(sc[o]["surv"][k][c] for o in ("v2", "v3", "v5")) for c in CKPTS}
               for k in runs}
    selectors = {
        "final checkpoint": lambda k: 1200,
        "self-play survival": lambda k: argmax_ckpt(sp["surv"][k]),
        "cross-play cash vs v4": lambda k: argmax_ckpt(sc["v4"]["cash"][k]),
        "held-out portfolio (v2,v3,v5)": lambda k: argmax_ckpt(heldout[k]),
        "oracle (v4 survival, in-sample)": lambda k: argmax_ckpt(v4[k]),
    }
    oracle = {k: v4[k][argmax_ckpt(v4[k])] for k in runs}
    rows = []
    for name, pick in selectors.items():
        dep = {k: v4[k][pick(k)] for k in runs}
        reg = [oracle[k] - dep[k] for k in runs]
        m, lo, hi, _ = t_ci(reg)
        rows.append({"selector": name, "n_runs": len(runs),
                     "mean_deployed_v4": round(mean(dep.values()), 2),
                     "mean_regret": round(m, 2), "regret_ci95_lo": round(lo, 2),
                     "regret_ci95_hi": round(hi, 2), "worst_run_regret": round(max(reg), 1)})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/camera_ready_wv/headline_ci"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    traj_rows = []
    for label, runs in (("healthy8", HEALTHY), ("all10", ALL10)):
        sc, sp = load(args.results_root, runs)
        runs = [k for k in runs if k in sc["v4"]["surv"] and all(c in sc["v4"]["surv"][k] for c in CKPTS)]
        traj_rows += trajectory_table(sc["v4"]["surv"], runs, f"{label}: survival vs 3xv4")
        traj_rows += trajectory_table(sp["surv"], runs, f"{label}: self-play survival")
        traj_rows += trajectory_table(sc["v4"]["cash"], runs, f"{label}: cross-play cash vs v4")
        report[label] = {
            "runs": [f"{k[0]}_s{k[1]}" for k in runs],
            "v4_delta_700_1200": paired_delta(sc["v4"]["surv"], runs),
            "v4_delta_peak_1200_per_run": {
                f"{k[0]}_s{k[1]}": round(sc["v4"]["surv"][k][1200]
                                        - sc["v4"]["surv"][k][argmax_ckpt(sc["v4"]["surv"][k])], 1)
                for k in runs},
            "selfplay_delta_700_1200": paired_delta(sp["surv"], runs),
            "reconciliation": reconcile(sc, sp, runs),
            "ablation": ablation(sc, sp, runs),
            "regret": regret(sc, sp, runs),
        }
    write_csv(args.out_dir / "trajectory_ci.csv", traj_rows)
    (args.out_dir / "headline_robustness.json").write_text(json.dumps(report, indent=2))

    for label in ("healthy8", "all10"):
        r = report[label]
        print(f"\n===== {label} ({len(r['runs'])} runs) =====")
        print("v4 survival 700->1200:", json.dumps(r["v4_delta_700_1200"]))
        print("self-play 700->1200:", {k: v for k, v in r["selfplay_delta_700_1200"].items() if k != "per_run"})
        print("reconcile:", json.dumps({k: v for k, v in r["reconciliation"].items() if k != "note"}))
        print("-- ablation --")
        for a in r["ablation"]:
            print(f"  {a['signal']:<46} {a['mean_at_700']:>7} -> {a['mean_at_1200']:>7} "
                  f"(up {a['runs_up']}, dn {a['runs_down']} / {a['n_runs']}) {a['verdict']}")
        print("-- regret --")
        for g in r["regret"]:
            print(f"  {g['selector']:<34} deploy {g['mean_deployed_v4']:>6}  regret {g['mean_regret']:>6} "
                  f"[{g['regret_ci95_lo']}, {g['regret_ci95_hi']}] worst {g['worst_run_regret']}")
    print("\n-- v4 trajectory with 95% t-CI --")
    for row in traj_rows:
        if "3xv4" in row["set"]:
            print(f"  {row['set']:<28} ckpt {row['checkpoint']:>4}: {row['mean']:>6} "
                  f"[{row['ci95_lo']}, {row['ci95_hi']}]  min {row['min']} max {row['max']}")
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
