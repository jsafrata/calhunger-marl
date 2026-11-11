"""Analysis for the ladder_validity_dose_response campaign (E17/E18).

Experiment A (ladder validity): does survival against the scripted ladder
predict survival against independently trained peers?  Spearman rank
correlation across the 10 best-config policies between vs-ladder survival
(``_ladder_validity_100``) and mean off-diagonal 1v3 peer survival
(``_peer_crossplay_100``).

Experiment B (population dose-response): survival of focal F as a function
of k = number of stranger copies among its 3 co-players, per ordered pair
(F, S):

- k=0: ``F vs self`` rows of the peer matrix (self-play diagonal)
- k=1: ``F vs S`` secondary rows of the 1v3 runs (F held 3 seats)
- k=2: ``F vs 2xS`` rows of the new 2+2 matrix
- k=3: ``F vs 3xS`` 1v3 rows

Usage (repo root):
    python3 workshops/eval/analyze_ladder_validity_dose_response.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1] / "results" / "seeds234"
PEER = ROOT / "_peer_crossplay_100" / "crossplay_matrix.csv"
LADDER = ROOT / "_ladder_validity_100" / "crossplay_matrix.csv"
MIX22 = ROOT / "_peer_mix22_100" / "crossplay_matrix.csv"
OUT = ROOT / "ladder_validity_dose_response_summary.md"

LABELS = [f"{a}_s{s}" for a in ("ippo", "mappo") for s in range(5)]
HEALTHY = {f"ippo_s{s}" for s in range(5)} | {"mappo_s0", "mappo_s2", "mappo_s3"}
BOTS = ["v1", "v2", "v3", "v4", "v5"]


def load(path: Path) -> list[dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    peer = load(PEER)
    ladder = load(LADDER)
    mix22 = load(MIX22)

    # ---- Experiment A: ladder validity -----------------------------------
    # vs-ladder survival/wins per policy per bot
    vs_bot: dict[str, dict[str, tuple[float, float]]] = {p: {} for p in LABELS}
    for row in ladder:
        if row["focal"] in vs_bot and row["opponent"].startswith("3xscripted_"):
            bot = row["opponent"].removeprefix("3xscripted_")
            vs_bot[row["focal"]][bot] = (
                float(row["survival_percentage"]),
                float(row["win_percentage"]),
            )

    # mean off-diagonal peer survival/wins per policy (focal side of 1v3)
    peer_out: dict[str, list[tuple[float, float]]] = {p: [] for p in LABELS}
    for row in peer:
        if (
            row["lineup_mode"] == "1v3"
            and row["focal"] in peer_out
            and row["opponent"].startswith("3x")
        ):
            peer_out[row["focal"]].append(
                (float(row["survival_percentage"]), float(row["win_percentage"]))
            )
    peer_surv = {p: float(np.mean([s for s, _ in v])) for p, v in peer_out.items()}
    peer_win = {p: float(np.mean([w for _, w in v])) for p, v in peer_out.items()}

    lines = ["# Ladder validity + population dose-response (campaign 1724783)", ""]
    lines += ["## Experiment A: does the scripted ladder proxy real peers?", ""]
    header = (
        "| policy | " + " | ".join(f"vs {b}" for b in BOTS)
        + " | mean v2-v5 | peer off-diag surv | peer off-diag win |"
    )
    lines += [header, "|" + "---|" * (len(BOTS) + 4)]
    lad_mean = {}
    for p in LABELS:
        cells = [f"{vs_bot[p][b][0]:.0f}" for b in BOTS]
        lad_mean[p] = float(np.mean([vs_bot[p][b][0] for b in BOTS[1:]]))
        lines.append(
            f"| {p} | " + " | ".join(cells)
            + f" | {lad_mean[p]:.1f} | {peer_surv[p]:.1f} | {peer_win[p]:.1f} |"
        )
    lines.append("")

    def corr(xs: list[float], ys: list[float]) -> str:
        rho, pval = spearmanr(xs, ys)
        return f"rho={rho:+.2f} (p={pval:.3f})"

    for subset_name, subset in (("all 10", LABELS), ("healthy 8", [p for p in LABELS if p in HEALTHY])):
        y = [peer_surv[p] for p in subset]
        lines.append(f"Spearman ({subset_name} policies), vs peer off-diagonal survival:")
        lines.append("")
        for b in BOTS:
            x = [vs_bot[p][b][0] for p in subset]
            lines.append(f"- vs {b} survival: {corr(x, y)}")
        x = [lad_mean[p] for p in subset]
        lines.append(f"- mean v2-v5 survival: {corr(x, y)}")
        xw = [float(np.mean([vs_bot[p][b][1] for b in BOTS[1:]])) for p in subset]
        yw = [peer_win[p] for p in subset]
        lines.append(f"- mean v2-v5 WIN vs peer off-diag WIN: {corr(xw, yw)}")
        lines.append("")

    # ---- Experiment B: dose-response -------------------------------------
    # survival[(F, S)][k]
    dose: dict[tuple[str, str], dict[int, float]] = {}
    selfplay: dict[str, float] = {}
    for row in peer:
        f, o = row["focal"], row["opponent"]
        if f not in LABELS:
            continue
        if o == "self":
            selfplay[f] = float(row["survival_percentage"])
        elif row["lineup_mode"] == "1v3" and o.startswith("3x") and o[2:] in LABELS:
            dose.setdefault((f, o[2:]), {})[3] = float(row["survival_percentage"])
        elif row["lineup_mode"] == "1v3" and o in LABELS:
            dose.setdefault((f, o), {})[1] = float(row["survival_percentage"])
    for row in mix22:
        f, o = row["focal"], row["opponent"]
        if f in LABELS and o.startswith("2x") and o[2:] in LABELS:
            dose.setdefault((f, o[2:]), {})[2] = float(row["survival_percentage"])
    for (f, s), ks in dose.items():
        ks[0] = selfplay[f]

    pairs = sorted(dose)
    complete = [p for p in pairs if all(k in dose[p] for k in range(4))]
    lines += ["## Experiment B: survival vs number of strangers k at the table", ""]
    lines.append(f"Ordered pairs with complete k=0..3 curves: {len(complete)} / {len(pairs)}")
    lines.append("")
    for name, keep in (
        ("all ordered pairs", complete),
        ("healthy focal AND healthy stranger", [(f, s) for f, s in complete if f in HEALTHY and s in HEALTHY]),
    ):
        lines.append(f"Mean survival, {name} (n={len(keep)}):")
        lines.append("")
        for k in range(4):
            vals = [dose[p][k] for p in keep]
            lines.append(
                f"- k={k}: {float(np.mean(vals)):.1f}% "
                f"(median {float(np.median(vals)):.1f}, min {min(vals):.1f}, max {max(vals):.1f})"
            )
        lines.append("")
    lines.append("Per-pair curves (survival % at k=0,1,2,3):")
    lines.append("")
    lines.append("| focal | stranger | k=0 | k=1 | k=2 | k=3 |")
    lines.append("|---|---|---|---|---|---|")
    for f, s in complete:
        ks = dose[(f, s)]
        lines.append(
            f"| {f} | {s} | {ks[0]:.1f} | {ks[1]:.1f} | {ks[2]:.1f} | {ks[3]:.1f} |"
        )

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: lines.index("Per-pair curves (survival % at k=0,1,2,3):")]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
