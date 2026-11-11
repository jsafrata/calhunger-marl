"""E32: declared-target audit of the longitudinal (rich-and-dead) data.

Camera-ready addition (TAE): "survival against one archetype, average
survival across a population mixture, and winning the game are different
objectives; under the stated reward the expected return is
P(win) + P(survival) - 1, so survival alone is not the training objective."

This script scores the SAME eight healthy best-config runs (E11 data:
checkpoints 100..1200, 100 paired games per archetype per checkpoint, 50
self-play games per checkpoint) under every target a careful evaluator could
declare, and reports the verdict each target renders on the checkpoint-700
-> 1200 transition. Targets:

  survival / win / expected return, each vs the storage trader v4
      (the deployment archetype of the regret analysis),
  survival / expected return under the uniform v2..v5 mixture
      (the "average over a population" target),
  self-play survival and self-play expected return (the symmetric
      diagonal has P(win) = 1/4 exactly, so E[r] = S_selfplay - 3/4).

Expected return per cell is computed from the summary rates:
  E[r] = P(win) + P(survive) - 1  (r = +1 win, 0 survive-not-win, -1 die).

Usage:
    python eval/analyze_objective_targets.py \
        --results-root results --out-dir results/objective_targets
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HEALTHY = [("ippo", 0), ("ippo", 1), ("ippo", 2), ("ippo", 3), ("ippo", 4),
           ("mappo", 0), ("mappo", 2), ("mappo", 3)]
CKPTS = list(range(100, 1201, 100))
OPPS = ("v2", "v3", "v4", "v5")
BAND_PCT = 2.0     # percentage points: |delta| <= band reads as flat
BAND_RET = 0.02    # expected-return units (same 2% of range)


def read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if p.is_file():
            rows.extend(csv.DictReader(open(p)))
    if not rows:
        raise SystemExit(f"no data found in any of: {paths}")
    return rows


def series(rows: list[dict], field: str) -> dict[tuple[str, int], dict[int, float]]:
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
    sp = {"surv": series(sp_rows, "survival_percentage")}
    return sc, sp


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else float("nan")


def build_targets(sc, sp) -> dict[str, tuple[dict, str]]:
    """name -> (series {run: {ckpt: value}}, unit) ; unit in {'pct','ret'}."""
    def ret(opp):
        return {k: {c: (sc[opp]["win"][k][c] + sc[opp]["surv"][k][c] - 100.0) / 100.0
                    for c in CKPTS} for k in HEALTHY}

    def mix(field_fn):
        return {k: {c: mean(field_fn(o)[k][c] for o in OPPS) for c in CKPTS}
                for k in HEALTHY}

    rets = {o: ret(o) for o in OPPS}
    targets = {
        "survival vs v4 (deployment target of Sec. 7)": (sc["v4"]["surv"], "pct"),
        "win rate vs v4": (sc["v4"]["win"], "pct"),
        "expected return vs v4": (rets["v4"], "ret"),
        "survival vs v2": (sc["v2"]["surv"], "pct"),
        "survival vs v3": (sc["v3"]["surv"], "pct"),
        "survival vs v5": (sc["v5"]["surv"], "pct"),
        "expected return vs v2": (rets["v2"], "ret"),
        "expected return vs v3": (rets["v3"], "ret"),
        "expected return vs v5": (rets["v5"], "ret"),
        "mixture survival (uniform v2..v5)": (mix(lambda o: sc[o]["surv"]), "pct"),
        "mixture expected return (uniform v2..v5)": (mix(lambda o: rets[o]), "ret"),
        "self-play survival": (sp["surv"], "pct"),
        "self-play expected return (P(win)=1/4 by symmetry)": (
            {k: {c: sp["surv"][k][c] / 100.0 - 0.75 for c in CKPTS} for k in HEALTHY}, "ret"),
    }
    return targets


def audit(targets) -> list[dict]:
    rows = []
    for name, (sig, unit) in targets.items():
        keys = [k for k in HEALTHY if k in sig and all(c in sig[k] for c in CKPTS)]
        band = BAND_PCT if unit == "pct" else BAND_RET
        deltas = {k: sig[k][1200] - sig[k][700] for k in keys}
        n_up = sum(1 for d in deltas.values() if d > band)
        n_down = sum(1 for d in deltas.values() if d < -band)
        traj = [mean(sig[k][c] for k in keys) for c in CKPTS]
        peak_c = CKPTS[max(range(len(CKPTS)), key=lambda i: traj[i])]
        per_run_peak = {f"{a}_s{s}": max(CKPTS, key=lambda c: (sig[(a, s)][c], c))
                        for a, s in keys}
        if n_up > len(keys) / 2 and n_up > n_down:
            verdict = "improvement"
        elif n_down > len(keys) / 2 and n_down > n_up:
            verdict = "DECLINE"
        else:
            verdict = "mixed / flat"
        fmt = (lambda v: round(v, 1)) if unit == "pct" else (lambda v: round(v, 3))
        rows.append({
            "target": name,
            "unit": "percentage points" if unit == "pct" else "expected return",
            "n_runs": len(keys),
            "mean_at_700": fmt(traj[CKPTS.index(700)]),
            "mean_at_1200": fmt(traj[-1]),
            "mean_delta_700_to_1200": fmt(traj[-1] - traj[CKPTS.index(700)]),
            "runs_up": n_up,
            "runs_down": n_down,
            "verdict_700_to_1200": verdict,
            "peak_ckpt_of_mean_trajectory": peak_c,
            "mean_at_peak": fmt(max(traj)),
            "mean_trajectory": json.dumps([fmt(v) for v in traj]),
            "per_run_peak_ckpt": json.dumps(per_run_peak),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md(path: Path, rows: list[dict]) -> None:
    lines = [
        "# E32: declared-target audit of the ckpt 700 -> 1200 transition (8 healthy runs)",
        "",
        "E[r] = P(win) + P(survive) - 1 per cell. A run counts as up/down if its delta",
        f"exceeds +-{BAND_PCT} pts (rates) or +-{BAND_RET} (returns). Verdict needs a majority.",
        "",
        "| target | 700 | 1200 | delta | up/down (of 8) | verdict | peak ckpt (mean) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['mean_at_700']} | {r['mean_at_1200']} | "
            f"{r['mean_delta_700_to_1200']} | {r['runs_up']}/{r['runs_down']} | "
            f"{r['verdict_700_to_1200']} | {r['peak_ckpt_of_mean_trajectory']} |"
        )
    lines += ["", "## Mean trajectories (ckpt 100..1200)", ""]
    for r in rows:
        lines.append(f"- {r['target']}: {r['mean_trajectory']}")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/objective_targets"))
    args = ap.parse_args()
    sc, sp = load_all(args.results_root)
    rows = audit(build_targets(sc, sp))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "objective_targets.csv", rows)
    write_md(args.out_dir / "objective_targets_summary.md", rows)
    for r in rows:
        print(f"{r['target']:<52} {r['mean_at_700']:>8} -> {r['mean_at_1200']:>8}  "
              f"up {r['runs_up']} / down {r['runs_down']}  {r['verdict_700_to_1200']}  "
              f"(peak {r['peak_ckpt_of_mean_trajectory']})")


if __name__ == "__main__":
    main()
