"""E36: checkpoint-selection regret with a separate evaluation stage.

Camera-ready addition (TAE): Section 7 selected the best checkpoint
and measured its advantage on the same longitudinal games; maximizing over
noisy checkpoint estimates inflates the oracle benchmark. Clarify the seed
split and evaluate the selected checkpoints on independent games.

Design
------
SELECTION stage = the original longitudinal data (E11/E25): every checkpoint
100..1200 of the eight healthy best-config runs scored on the SAME 100 paired
games per archetype (seeds 900000-900099) and 50 self-play games. Every
selector -- including the oracle -- picks one checkpoint per run from these.

EVALUATION stage = fresh games: every checkpoint of every run re-scored
against 3x v4 on 500 games with seeds 950000-950499 (HPC campaign
hpc/best_config_eval/run_array_scripted_fresh_v4.sbatch, job 6552675),
disjoint from the selection seeds.

Reported, deployment target = survival vs v4:
  * in-sample regret (as published): A[oracle_A pick] - A[rule pick];
  * out-of-sample value of every rule's pick: B[rule pick];
  * nested regret: B[oracle_A pick] - B[rule pick] -- the honest regret
    relative to a selector that saw the deployment metric on 100 games;
  * optimism of the in-sample oracle: mean A[oracle_A pick] - mean B[oracle_A pick];
  * reference only: B-oracle (argmax over B, in-sample on B at n=500).
Also: the fresh-seed replication of the longitudinal v4 trajectory itself
(peak checkpoint, 700 -> 1200 delta, runs up/down).

Usage:
    python workshops/eval/analyze_nested_selection_regret.py \
        --results-root workshops/results --out-dir workshops/results/nested_selection_regret
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
FLAT_BAND = 2.0


def read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if p.is_file():
            rows.extend(csv.DictReader(open(p)))
    if not rows:
        raise SystemExit(f"no data found in any of: {paths}")
    return rows


def series(rows, field):
    out = {}
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


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else float("nan")


def se(vals):
    vals = list(vals)
    if len(vals) < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1) / len(vals))


def argmax_ckpt(s: dict[int, float]) -> int:
    best = max(s[c] for c in CKPTS)
    return max(c for c in CKPTS if s[c] == best)  # last on ties, as in E25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("workshops/results"))
    ap.add_argument("--out-dir", type=Path, default=Path("workshops/results/nested_selection_regret"))
    ap.add_argument("--fresh-dir", type=Path, default=None,
                    help="dir holding default_180_mlp_seeds01/v4 and default_180_mlp_seeds234/v4")
    args = ap.parse_args()
    root = args.results_root
    fresh = args.fresh_dir or (root / "best_config_fresh_eval_v4_500")

    # selection stage (A)
    A = {}
    for opp in ("v2", "v3", "v4", "v5"):
        rows = read_rows([
            root / f"best_config_scripted_eval_100/default_180_mlp/{opp}/checkpoint_diagnostics.csv",
            root / f"seeds234/bestcfg/_scripted_eval_100/default_180_mlp/{opp}/checkpoint_diagnostics.csv",
        ])
        A[opp] = {"surv": series(rows, "survival_percentage"),
                  "win": series(rows, "win_percentage"),
                  "cash": series(rows, "mean_terminal_cash")}
    sp_rows = read_rows([
        root / "best_config_selfplay_survival_50/default_180_mlp/selfplay_survival_diagnostics.csv",
        root / "seeds234/bestcfg/_selfplay_survival_50/default_180_mlp/selfplay_survival_diagnostics.csv",
    ])
    sp = series(sp_rows, "survival_percentage")
    heldout = {k: {c: mean(A[o]["surv"][k][c] for o in ("v2", "v3", "v5")) for c in CKPTS}
               for k in HEALTHY}

    # evaluation stage (B)
    b_rows = read_rows([
        fresh / "default_180_mlp_seeds01/v4/checkpoint_diagnostics.csv",
        fresh / "default_180_mlp_seeds234/v4/checkpoint_diagnostics.csv",
    ])
    for r in b_rows:
        if int(r["n_games"]) != 500 or int(r["game_seed_base"]) != 950000 or r["opponent"] != "v4":
            raise SystemExit(f"unexpected evaluation-stage row: {r}")
    B = {"surv": series(b_rows, "survival_percentage"),
         "win": series(b_rows, "win_percentage"),
         "cash": series(b_rows, "mean_terminal_cash")}

    keys = [k for k in HEALTHY if all(c in B["surv"].get(k, {}) for c in CKPTS)
            and all(c in A["v4"]["surv"].get(k, {}) and c in sp.get(k, {}) for c in CKPTS)]
    if len(keys) != 8:
        print(f"WARNING: only {len(keys)} runs have complete A and B series: {keys}")

    a_v4 = A["v4"]["surv"]
    b_v4 = B["surv"]
    selectors = {
        "final checkpoint (default practice)": lambda k: 1200,
        "self-play survival": lambda k: argmax_ckpt(sp[k]),
        "cross-play cash vs v4 (raw score)": lambda k: argmax_ckpt(A["v4"]["cash"][k]),
        "held-out portfolio survival (v2,v3,v5)": lambda k: argmax_ckpt(heldout[k]),
        "oracle on selection games (v4 survival, n=100)": lambda k: argmax_ckpt(a_v4[k]),
    }
    picks = {name: {k: f(k) for k in keys} for name, f in selectors.items()}
    oracle_name = "oracle on selection games (v4 survival, n=100)"
    oracle_A_pick = picks[oracle_name]
    b_oracle_pick = {k: argmax_ckpt(b_v4[k]) for k in keys}  # in-sample on B, reference only

    rows_out = []
    for name in selectors:
        p = picks[name]
        a_val = {k: a_v4[k][p[k]] for k in keys}
        b_val = {k: b_v4[k][p[k]] for k in keys}
        in_regret = [a_v4[k][oracle_A_pick[k]] - a_val[k] for k in keys]
        nested_regret = [b_v4[k][oracle_A_pick[k]] - b_val[k] for k in keys]
        ref_regret = [b_v4[k][b_oracle_pick[k]] - b_val[k] for k in keys]
        rows_out.append({
            "selector": name,
            "median_ckpt_picked": sorted(p.values())[len(p) // 2],
            "in_sample_value_A": round(mean(a_val.values()), 2),
            "in_sample_regret_A": round(mean(in_regret), 2),
            "out_of_sample_value_B": round(mean(b_val.values()), 2),
            "out_of_sample_value_B_se": round(se(b_val.values()), 2),
            "nested_regret_B_vs_oracleA_pick": round(mean(nested_regret), 2),
            "nested_regret_se": round(se(nested_regret), 2),
            "nested_regret_runs_positive": sum(1 for v in nested_regret if v > 0),
            "regret_vs_B_oracle_reference": round(mean(ref_regret), 2),
            "per_run_ckpt": json.dumps({f"{a}_s{s}": p[(a, s)] for a, s in keys}),
            "per_run_value_B": json.dumps({f"{a}_s{s}": b_val[(a, s)] for a, s in keys}),
        })

    optimism = mean(a_v4[k][oracle_A_pick[k]] for k in keys) - mean(b_v4[k][oracle_A_pick[k]] for k in keys)
    optimism_B_oracle = mean(b_v4[k][b_oracle_pick[k]] for k in keys) - mean(b_v4[k][oracle_A_pick[k]] for k in keys)

    # fresh-seed replication of the trajectory
    trajA = [mean(a_v4[k][c] for k in keys) for c in CKPTS]
    trajB = [mean(b_v4[k][c] for k in keys) for c in CKPTS]
    dB = {k: b_v4[k][1200] - b_v4[k][700] for k in keys}
    peakdropB = {k: b_v4[k][argmax_ckpt(b_v4[k])] - b_v4[k][1200] for k in keys}
    corr_num = sum((a_v4[k][c] - mean(trajA)) * (b_v4[k][c] - mean(trajB)) for k in keys for c in CKPTS)
    # simple Pearson over all (run, ckpt) cells
    xa = [a_v4[k][c] for k in keys for c in CKPTS]
    xb = [b_v4[k][c] for k in keys for c in CKPTS]
    ma, mb = mean(xa), mean(xb)
    pear = (sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
            / math.sqrt(sum((x - ma) ** 2 for x in xa) * sum((y - mb) ** 2 for y in xb)))
    replication = {
        "n_runs": len(keys),
        "trajectory_A_n100": [round(v, 1) for v in trajA],
        "trajectory_B_n500": [round(v, 1) for v in trajB],
        "peak_ckpt_A": CKPTS[trajA.index(max(trajA))], "peak_ckpt_B": CKPTS[trajB.index(max(trajB))],
        "mean_700_B": round(trajB[CKPTS.index(700)], 1), "mean_1200_B": round(trajB[-1], 1),
        "runs_down_700_to_1200_B": sum(1 for v in dB.values() if v < -FLAT_BAND),
        "runs_up_700_to_1200_B": sum(1 for v in dB.values() if v > FLAT_BAND),
        "runs_declining_gt5_from_B_peak": sum(1 for v in peakdropB.values() if v > 5),
        "pearson_A_vs_B_over_96_cells": round(pear, 3),
        "mean_abs_diff_A_minus_B_pts": round(mean(abs(a_v4[k][c] - b_v4[k][c]) for k in keys for c in CKPTS), 2),
        "mean_A_minus_B_pts": round(mean(a_v4[k][c] - b_v4[k][c] for k in keys for c in CKPTS), 2),
        "in_sample_oracle_optimism_pts": round(optimism, 2),
        "B_oracle_minus_oracleA_pick_on_B_pts": round(optimism_B_oracle, 2),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "nested_selection_regret.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    (args.out_dir / "replication_summary.json").write_text(json.dumps(replication, indent=2))
    lines = ["# E36: selection regret with a separate evaluation stage (deployment = v4 survival)", "",
             "Selection on the original 100 games/ckpt (seeds 900000-900099); evaluation of the picked",
             "checkpoint on 500 fresh games (seeds 950000-950499). Means over 8 healthy runs; SE over runs.", "",
             "| selector | median ckpt | in-sample value (A) | in-sample regret (A, published) | out-of-sample value (B) | nested regret (B, vs oracle-A pick) | runs with regret>0 | regret vs B-oracle (ref.) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows_out:
        lines.append(f"| {r['selector']} | {r['median_ckpt_picked']} | {r['in_sample_value_A']} | {r['in_sample_regret_A']} | "
                     f"{r['out_of_sample_value_B']} +- {r['out_of_sample_value_B_se']} | {r['nested_regret_B_vs_oracleA_pick']} +- {r['nested_regret_se']} | "
                     f"{r['nested_regret_runs_positive']}/8 | {r['regret_vs_B_oracle_reference']} |")
    lines += ["", f"- in-sample oracle optimism (A value of oracle-A pick minus its B value): **{replication['in_sample_oracle_optimism_pts']} pts**",
              f"- B-oracle (argmax on B, in-sample at n=500) exceeds the oracle-A pick on B by {replication['B_oracle_minus_oracleA_pick_on_B_pts']} pts",
              f"- fresh-seed replication of the v4 trajectory: peak ckpt {replication['peak_ckpt_B']} (A: {replication['peak_ckpt_A']}); "
              f"700 -> 1200: {replication['mean_700_B']} -> {replication['mean_1200_B']} (runs down {replication['runs_down_700_to_1200_B']}/8, up {replication['runs_up_700_to_1200_B']}/8); "
              f"{replication['runs_declining_gt5_from_B_peak']}/8 runs decline >5 pts from their B peak",
              f"- A vs B agreement over 96 (run, ckpt) cells: Pearson {replication['pearson_A_vs_B_over_96_cells']}, mean |A-B| {replication['mean_abs_diff_A_minus_B_pts']} pts, mean A-B {replication['mean_A_minus_B_pts']} pts",
              f"- trajectories A: {replication['trajectory_A_n100']}", f"- trajectories B: {replication['trajectory_B_n500']}", ""]
    (args.out_dir / "nested_selection_regret_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
