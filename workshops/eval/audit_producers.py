"""Audit — does the producer-pair assignment materially affect outcomes?

We stratify per-seat outcomes by (cadence, produced-food-pair) and per-config,
then report mean ± 95 % CI (game-clustered bootstrap where relevant).

Question we want to answer: does any produced-pair give players a
systematically better or worse chance of surviving / trading, once
cadence and config are controlled for?

Output CSV columns:
    config, cadence, produced_pair (e.g. AB), n_seats,
    survival_rate, survival_rate_ci,
    death_tick_median, death_tick_ci,
    units_bought_mean, units_bought_ci,
    units_sold_mean, units_sold_ci,
    final_cash_mean, final_cash_ci
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


N_BOOT = 2000
ROOT = Path(__file__).resolve().parent / "results" / "main"


def _pair_key(produces: list[str]) -> str:
    return "".join(sorted(produces))


def _bootstrap_ci(xs: list, stat=np.mean, n_boot=N_BOOT, seed=0) -> tuple[float, float]:
    if len(xs) < 2:
        v = float(stat(xs)) if xs else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    a = np.asarray(xs, dtype=float)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        boots[i] = stat(a[idx])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    # (config, cadence, pair) -> lists of per-seat outcomes.
    per_stratum: dict[tuple[str, str, str], dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    # (cadence, pair) collapsed across configs for the "controlled" view.
    collapsed: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )

    for f in sorted(ROOT.glob("*.jsonl")):
        cfg = f.stem
        with open(f) as fh:
            for line in fh:
                g = json.loads(line)
                if "error" in g:
                    continue
                final = g["final"]
                elapsed = final["elapsed_seconds"]
                for pb, pf in zip(g["players_behavior"], final["players"]):
                    cad = pb["cadence"]
                    pair = _pair_key(pf["produces"])
                    survived = pf["survived"]
                    died_at = pf["died_at"] if pf["died_at"] is not None else elapsed
                    for k, v in (
                        ("survived", survived),
                        ("death_tick", died_at),
                        ("units_bought", pb["units_bought"]),
                        ("units_sold", pb["units_sold"]),
                        ("final_cash", pf["final_cash"]),
                    ):
                        per_stratum[(cfg, cad, pair)][k].append(v)
                        collapsed[(cad, pair)][k].append(v)

    # Per-(config, cadence, pair) table
    out_path = ROOT / "producer_pair_effects.csv"
    fields = [
        "config", "cadence", "produced_pair", "n_seats",
        "survival_rate", "survival_rate_ci95",
        "death_tick_median", "death_tick_ci95",
        "units_bought_mean", "units_bought_ci95",
        "units_sold_mean", "units_sold_ci95",
        "final_cash_mean", "final_cash_ci95",
    ]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (cfg, cad, pair), buckets in sorted(per_stratum.items()):
            n = len(buckets["survived"])
            if n == 0:
                continue
            sr_lo, sr_hi = _bootstrap_ci(buckets["survived"], np.mean, seed=hash(("s", cfg, cad, pair)) & 0xFFFF)
            dt_lo, dt_hi = _bootstrap_ci(buckets["death_tick"], np.median, seed=hash(("d", cfg, cad, pair)) & 0xFFFF)
            ub_lo, ub_hi = _bootstrap_ci(buckets["units_bought"], np.mean, seed=hash(("b", cfg, cad, pair)) & 0xFFFF)
            us_lo, us_hi = _bootstrap_ci(buckets["units_sold"], np.mean, seed=hash(("u", cfg, cad, pair)) & 0xFFFF)
            fc_lo, fc_hi = _bootstrap_ci(buckets["final_cash"], np.mean, seed=hash(("c", cfg, cad, pair)) & 0xFFFF)
            w.writerow({
                "config": cfg, "cadence": cad, "produced_pair": pair, "n_seats": n,
                "survival_rate": f"{np.mean(buckets['survived']):.3f}",
                "survival_rate_ci95": f"[{sr_lo:.3f}, {sr_hi:.3f}]",
                "death_tick_median": f"{np.median(buckets['death_tick']):.0f}",
                "death_tick_ci95": f"[{dt_lo:.0f}, {dt_hi:.0f}]",
                "units_bought_mean": f"{np.mean(buckets['units_bought']):.2f}",
                "units_bought_ci95": f"[{ub_lo:.2f}, {ub_hi:.2f}]",
                "units_sold_mean": f"{np.mean(buckets['units_sold']):.2f}",
                "units_sold_ci95": f"[{us_lo:.2f}, {us_hi:.2f}]",
                "final_cash_mean": f"{np.mean(buckets['final_cash']):.2f}",
                "final_cash_ci95": f"[{fc_lo:.2f}, {fc_hi:.2f}]",
            })
    print(f"[audit-producers] wrote per-stratum table → {out_path}")

    # Collapsed (cadence, pair) — the "controlled for config" view. Each row's
    # CI uses stratified bootstrap (per-config resampling) so the interval is
    # honest about the fact that different configs contribute different
    # numbers of seats.
    out2 = ROOT / "producer_pair_by_cadence.csv"
    with open(out2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cadence", "produced_pair", "n_seats", "survival_rate", "survival_rate_ci95",
                    "death_tick_median", "units_bought_mean", "final_cash_mean"])
        for (cad, pair), buckets in sorted(collapsed.items()):
            n = len(buckets["survived"])
            sr_lo, sr_hi = _bootstrap_ci(buckets["survived"], np.mean, seed=hash(("sc", cad, pair)) & 0xFFFF)
            w.writerow([
                cad, pair, n,
                f"{np.mean(buckets['survived']):.3f}",
                f"[{sr_lo:.3f}, {sr_hi:.3f}]",
                f"{np.median(buckets['death_tick']):.0f}",
                f"{np.mean(buckets['units_bought']):.2f}",
                f"{np.mean(buckets['final_cash']):.2f}",
            ])
    print(f"[audit-producers] wrote collapsed table → {out2}")

    # Report the effect size — do any pairs stand out per cadence?
    print("\nCadence-controlled per-pair survival rates:")
    for cad in ("fast", "medium", "slow"):
        rows = [
            (pair, np.mean(collapsed[(cad, pair)]["survived"]),
             len(collapsed[(cad, pair)]["survived"]))
            for pair in ["AB", "AC", "AD", "BC", "BD", "CD"]
            if (cad, pair) in collapsed
        ]
        if not rows:
            continue
        rates = [r[1] for r in rows]
        spread = max(rates) - min(rates)
        print(f"  {cad:6s} span={spread:.3f}  " +
              " ".join(f"{p}={r:.2f}(n={n})" for p, r, n in rows))


if __name__ == "__main__":
    main()
