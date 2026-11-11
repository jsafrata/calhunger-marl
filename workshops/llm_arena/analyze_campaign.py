"""Zero-token analyses of the cached 90 s LLM campaign logs.

Produces, next to llm_matrix.csv:
1. ``death_times.csv``  — when LLM seats die, per model x population:
   count per feeding tick (10..90), plus early/mid/late shares.
2. ``llm_matrix_ci.csv`` — 95% bootstrap CIs (10k resamples over games)
   for survival% and win% of every (model, population) cell.
3. ``fingerprints.csv`` — action mix (wait/post/take/cancel %) per
   model x population, with a per-model stability summary (range of
   each action share across populations).

Rerunnable; late-arriving games upgrade all three.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CAMPAIGN = Path(__file__).resolve().parents[1] / "results" / "llm_arena" / "campaign_90s"
MODELS = ("claude", "gpt", "deepseek", "gemini")
OPPONENTS = ("selfplay", "3llm_mix", "v2", "v3", "v4", "v5",
             "ippo_s0", "ippo_s2", "mappo_s1", "mappo_s3")
RNG = np.random.default_rng(0)
N_BOOT = 10_000


def load(cell: str) -> list[dict]:
    d = CAMPAIGN / cell / "games"
    if not d.is_dir():
        return []
    return [json.loads(p.read_text())["game"] for p in sorted(d.glob("game_*.json"))]


def cells_for(model: str, opp: str) -> list[dict]:
    if opp == "selfplay":
        return load(f"{model}_selfplay")
    if opp == "3llm_mix":
        return load("mixed_4llm")
    return load(f"{model}_vs_{opp}")


def per_game_seat_stats(games: list[dict], spec: str):
    """Per game: (n_seats, survivors, winners) for this spec."""
    out = []
    for g in games:
        seats = [s for s in g["seats"] if s["spec"] == spec]
        if seats:
            out.append((
                len(seats),
                sum(s["survived"] for s in seats),
                sum(s["won"] for s in seats),
            ))
    return out


def bootstrap_ci(stats, index: int) -> tuple[float, float]:
    """CI for a per-seat rate; resample GAMES (the independent unit)."""
    arr = np.array(stats, dtype=float)
    n = len(arr)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    totals = arr[idx].sum(axis=1)  # (N_BOOT, 3)
    rates = 100.0 * totals[:, index] / totals[:, 0]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def main() -> None:
    death_rows, ci_rows, fp_rows = [], [], []
    fp_by_model = defaultdict(dict)

    for model in MODELS:
        spec = f"llm_{model}"
        for opp in OPPONENTS:
            games = cells_for(model, opp)
            if not games:
                continue

            # --- deaths ---------------------------------------------------
            deaths = Counter()
            n_seats = n_dead = 0
            for g in games:
                for s in g["seats"]:
                    if s["spec"] != spec:
                        continue
                    n_seats += 1
                    if not s["survived"]:
                        n_dead += 1
                        deaths[int(s["survival_time_ticks"])] += 1
            row = {"model": model, "opponent": opp, "n_seats": n_seats,
                   "n_deaths": n_dead}
            for t in range(10, 100, 10):
                row[f"t{t}"] = deaths.get(t, 0)
            if n_dead:
                row["early_share_t10_30"] = round(
                    100 * sum(deaths.get(t, 0) for t in (10, 20, 30)) / n_dead, 1)
                row["late_share_t60_90"] = round(
                    100 * sum(deaths.get(t, 0) for t in (60, 70, 80, 90)) / n_dead, 1)
            else:
                row["early_share_t10_30"] = row["late_share_t60_90"] = ""
            death_rows.append(row)

            # --- bootstrap CIs -------------------------------------------
            stats = per_game_seat_stats(games, spec)
            tot = np.array(stats).sum(axis=0)
            surv_lo, surv_hi = bootstrap_ci(stats, 1)
            win_lo, win_hi = bootstrap_ci(stats, 2)
            ci_rows.append({
                "model": model, "opponent": opp, "n_games": len(stats),
                "survival_pct": round(100 * tot[1] / tot[0], 1),
                "survival_ci_lo": round(surv_lo, 1),
                "survival_ci_hi": round(surv_hi, 1),
                "win_pct": round(100 * tot[2] / tot[0], 1),
                "win_ci_lo": round(win_lo, 1),
                "win_ci_hi": round(win_hi, 1),
            })

            # --- fingerprints --------------------------------------------
            acts = Counter()
            for g in games:
                for d in g["llm_decisions"]:
                    if d["spec"] != spec:
                        continue
                    a = d["decision"]["action"] if d["decision"] else "wait"
                    acts[a] += 1
            n_dec = sum(acts.values())
            mix = {a: round(100 * acts[a] / n_dec, 1)
                   for a in ("wait", "post", "take", "cancel")}
            fp_rows.append({"model": model, "opponent": opp,
                            "n_decisions": n_dec, **mix})
            fp_by_model[model][opp] = mix

    with (CAMPAIGN / "death_times.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(death_rows[0]))
        w.writeheader(); w.writerows(death_rows)
    with (CAMPAIGN / "llm_matrix_ci.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ci_rows[0]))
        w.writeheader(); w.writerows(ci_rows)
    with (CAMPAIGN / "fingerprints.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fp_rows[0]))
        w.writeheader(); w.writerows(fp_rows)

    # --- console digest --------------------------------------------------
    print("=== death timing (pooled per model) ===")
    for model in MODELS:
        pooled = Counter(); n_dead = 0
        for r in death_rows:
            if r["model"] == model:
                for t in range(10, 100, 10):
                    pooled[t] += r[f"t{t}"]
                n_dead += r["n_deaths"]
        if not n_dead:
            continue
        early = 100 * sum(pooled[t] for t in (10, 20, 30)) / n_dead
        late = 100 * sum(pooled[t] for t in (60, 70, 80, 90)) / n_dead
        top = ", ".join(f"t={t}:{c}" for t, c in pooled.most_common(3))
        print(f"{model:9s} deaths={n_dead:4d}  early(10-30)={early:4.1f}%  "
              f"late(60-90)={late:4.1f}%  top: {top}")

    print("\n=== fingerprint stability (min-max of action share across populations) ===")
    for model in MODELS:
        mixes = fp_by_model[model]
        if not mixes:
            continue
        for act in ("wait", "post", "take"):
            vals = [m[act] for m in mixes.values()]
            print(f"{model:9s} {act:6s} {min(vals):5.1f}–{max(vals):5.1f}%"
                  f"  (mean {sum(vals)/len(vals):5.1f})")

    print("\n=== tightest / widest CIs ===")
    for r in sorted(ci_rows, key=lambda r: r["survival_ci_hi"] - r["survival_ci_lo"])[:2]:
        print(f"tight: {r['model']} vs {r['opponent']}: surv {r['survival_pct']}"
              f" [{r['survival_ci_lo']}–{r['survival_ci_hi']}]")
    for r in sorted(ci_rows, key=lambda r: r["survival_ci_lo"] - r["survival_ci_hi"])[:2]:
        print(f"wide:  {r['model']} vs {r['opponent']}: surv {r['survival_pct']}"
              f" [{r['survival_ci_lo']}–{r['survival_ci_hi']}] (n={r['n_games']})")
    print(f"\nwrote death_times.csv, llm_matrix_ci.csv, fingerprints.csv in {CAMPAIGN}")


if __name__ == "__main__":
    main()
