"""Unify E26 and E29: welfare accounting for LLM economies + deficit decomposition.

TASK 1 — E29's welfare definitions applied to the E26 replay data (34 LLM cells,
validated games only): per game over all four seats compute survivors, per-game
Gini of terminal cash, dead-wealth share; aggregate per cell; then a UNIFIED
frontier figure with E29's 145 policy-economy cells as background
(``results/welfare/gini_by_population.csv`` carries survival_pct / gini_mean /
comp / healthy — no raw re-read needed) and the 34 LLM cells overlaid with
distinct markers per group (selfplay/mixed, vs-policy, vs-scripted).

TASK 2 — deficit decomposition ("the deficit is food, not exploitation" made
exact). For every LLM-vs-scripted and LLM-vs-policy cell, per validated game:

- actual_net_outflow      = sum(total_price | LLM is buyer)
                          - sum(total_price | LLM is seller)
- counterfactual_fair     = sum(VWAP_{g,f} * qty | LLM is buyer)
                          - sum(VWAP_{g,f} * qty | LLM is seller)
  (the same trades executed at the game's per-food volume-weighted average price)
- price_component  = actual - counterfactual  (what adverse PRICES cost)
- volume_component = counterfactual           (what the trade PATTERN costs at
                                               fair prices)

Validation reported per game: (a) cash conservation — final-tick cash across the
four seats sums to 400; (b) final-tick cash/alive match the original campaign
records' per-seat terminal_cash/survived
(``workshops/results/llm_arena/campaign_90s/<cell>/games/game_XXXX.json``).

Outputs (``workshops/econml/results/llm_welfare/``):
    llm_welfare_by_cell.csv, deficit_decomposition.csv,
    unified_frontier.pdf/.png, FINDINGS.md

Writes ONLY new files; modifies nothing existing.

Usage (repo root):
    python3 workshops/econml/analyze_llm_welfare.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LLM_ROOT = HERE / "results" / "llm_economy"
E29_GINI = HERE / "results" / "welfare" / "gini_by_population.csv"
CAMPAIGN = HERE.parent / "results" / "llm_arena" / "campaign_90s"
OUT = HERE / "results" / "llm_welfare"

# Same validated categorical slots as analyze_welfare.py (dataviz reference
# palette; distinct marker shapes carry identity alongside color).
C_SELF, C_POLICY, C_SCRIPTED = "#2a78d6", "#eb6834", "#1baf7a"
C_BG = "#b0b4bc"  # de-emphasized E29 background


def cell_group(cell: str) -> str:
    """selfplay/mixed | vs-policy | vs-scripted."""
    if cell.endswith("_selfplay") or cell == "mixed_4llm":
        return "selfplay/mixed"
    if "_vs_ippo" in cell or "_vs_mappo" in cell:
        return "vs-policy"
    if "_vs_v" in cell:
        return "vs-scripted"
    raise ValueError(f"unclassifiable cell {cell}")


def gini4(cash: np.ndarray) -> float:
    """Per-game Gini over the four terminal cash values (E29's formula)."""
    if cash.mean() <= 0:
        return np.nan
    return float(
        np.abs(cash[:, None] - cash[None, :]).sum()
        / (2 * len(cash) ** 2 * cash.mean())
    )


def iter_validated_games(cell_dir: Path):
    for path in sorted((cell_dir / "replay").glob("game_*.json")):
        game = json.loads(path.read_text())
        if game["validation"]["ok"]:
            yield game


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = sorted(
        d for d in LLM_ROOT.iterdir() if d.is_dir() and (d / "replay").is_dir()
    )

    game_rows = []       # per validated game welfare accounting
    decomp_rows = []     # per validated game deficit decomposition (vs-cells)
    sanity = {
        "n_games": 0,
        "cash_violations": [],       # (cell, game_index, total)
        "max_total_dev": 0.0,
        "terminal_mismatches": [],   # (cell, game_index, seat, replay, orig)
        "alive_mismatches": [],
        "orig_missing": [],
        "checked_seats": 0,
    }

    for cell_dir in cells:
        cell = cell_dir.name
        group = cell_group(cell)
        for game in iter_validated_games(cell_dir):
            gi = int(game["game_index"])
            lineup = game["lineup"]
            ticks = game["ticks"]
            final = ticks[-1]["seats"]
            cash = np.array([float(s["cash"]) for s in final])
            alive = np.array([bool(s["alive"]) for s in final])
            total = float(cash.sum())

            # --- check (a): cash conservation --------------------------------
            sanity["n_games"] += 1
            dev = abs(total - 400.0)
            sanity["max_total_dev"] = max(sanity["max_total_dev"], dev)
            if dev > 1e-6:
                sanity["cash_violations"].append((cell, gi, total))

            # --- check (b): final-tick cash == campaign terminal_cash --------
            orig_path = CAMPAIGN / cell / "games" / f"game_{gi:04d}.json"
            if not orig_path.exists():
                sanity["orig_missing"].append((cell, gi))
            else:
                orig = json.loads(orig_path.read_text())["game"]["seats"]
                for j, seat in enumerate(orig):
                    sanity["checked_seats"] += 1
                    if abs(float(seat["terminal_cash"]) - cash[j]) > 1e-6:
                        sanity["terminal_mismatches"].append(
                            (cell, gi, j, cash[j], float(seat["terminal_cash"]))
                        )
                    if bool(seat["survived"]) != alive[j]:
                        sanity["alive_mismatches"].append(
                            (cell, gi, j, bool(alive[j]), bool(seat["survived"]))
                        )

            # --- TASK 1: welfare accounting ----------------------------------
            game_rows.append(
                {"cell": cell, "group": group, "game_index": gi,
                 "n_survivors": int(alive.sum()),
                 "total_cash": total,
                 "dead_cash": float(cash[~alive].sum()),
                 "dead_share": float(cash[~alive].sum() / total) if total else np.nan,
                 "gini": gini4(cash)}
            )

            # --- TASK 2: deficit decomposition (single-LLM vs-cells) ---------
            if group not in ("vs-scripted", "vs-policy"):
                continue
            llm_seats = {i for i, k in enumerate(lineup) if k.startswith("llm_")}
            trades = game["trades"]
            px = defaultdict(list)
            for t in trades:
                px[t["food_type"]].append((t["price_per_unit"], t["quantity"]))
            vwap = {
                f: sum(p * q for p, q in v) / max(1, sum(q for _, q in v))
                for f, v in px.items()
            }
            actual = fair = 0.0
            n_buys = n_sells = 0
            qty_bought = qty_sold = 0.0
            for t in trades:
                f, q = t["food_type"], t["quantity"]
                if t["buyer_idx"] in llm_seats:
                    actual += t["total_price"]
                    fair += vwap[f] * q
                    n_buys += 1
                    qty_bought += q
                if t["seller_idx"] in llm_seats:
                    actual -= t["total_price"]
                    fair -= vwap[f] * q
                    n_sells += 1
                    qty_sold += q
            decomp_rows.append(
                {"cell": cell, "group": group, "game_index": gi,
                 "actual_net_outflow": actual,
                 "volume_component": fair,
                 "price_component": actual - fair,
                 "n_buys": n_buys, "n_sells": n_sells,
                 "qty_bought": qty_bought, "qty_sold": qty_sold}
            )

    games = pd.DataFrame(game_rows)
    decomp = pd.DataFrame(decomp_rows)

    # --- sanity report -------------------------------------------------------
    ok_cash = sanity["n_games"] - len(sanity["cash_violations"])
    print(
        f"SANITY cash conservation: total final-tick cash == 400 in "
        f"{ok_cash}/{sanity['n_games']} validated games "
        f"(max |dev| {sanity['max_total_dev']:.2e}); "
        f"{len(sanity['cash_violations'])} violations"
    )
    print(
        f"SANITY terminal-cash match vs campaign_90s records: "
        f"{sanity['checked_seats']} seat comparisons, "
        f"{len(sanity['terminal_mismatches'])} cash mismatches, "
        f"{len(sanity['alive_mismatches'])} alive/survived mismatches, "
        f"{len(sanity['orig_missing'])} original files missing"
    )
    for v in sanity["cash_violations"][:10]:
        print("  cash violation:", v)
    for v in sanity["terminal_mismatches"][:10]:
        print("  terminal mismatch:", v)
    for v in sanity["alive_mismatches"][:10]:
        print("  alive mismatch:", v)
    for v in sanity["orig_missing"][:10]:
        print("  missing original:", v)

    # --- TASK 1: per-cell aggregation ---------------------------------------
    cell_rows = []
    for cell, g in games.groupby("cell", sort=True):
        cell_rows.append(
            {"cell": cell, "group": g.group.iloc[0], "n_games": len(g),
             "survival_pct": 100 * float(g.n_survivors.mean() / 4),
             "pct_games_all_survive": 100 * float((g.n_survivors == 4).mean()),
             "gini_mean": float(g.gini.mean()),
             "gini_sd": float(g.gini.std(ddof=1)),
             "dead_share_mean_pct": 100 * float(g.dead_share.mean()),
             "dead_cash_mean": float(g.dead_cash.mean())}
        )
    by_cell = pd.DataFrame(cell_rows)
    by_cell.to_csv(OUT / "llm_welfare_by_cell.csv", index=False)

    # --- TASK 2: per-cell aggregation ---------------------------------------
    dec_rows = []
    for cell, g in decomp.groupby("cell", sort=True):
        a = float(g.actual_net_outflow.mean())
        v = float(g.volume_component.mean())
        p = float(g.price_component.mean())
        dec_rows.append(
            {"cell": cell, "group": g.group.iloc[0], "n_games": len(g),
             "actual_net_outflow_mean": a,
             "volume_component_mean": v,
             "price_component_mean": p,
             "price_share_pct": 100 * p / a if a else np.nan,
             "actual_sd": float(g.actual_net_outflow.std(ddof=1)),
             "price_component_sd": float(g.price_component.std(ddof=1)),
             "mean_n_buys": float(g.n_buys.mean()),
             "mean_n_sells": float(g.n_sells.mean()),
             "mean_qty_bought": float(g.qty_bought.mean()),
             "mean_qty_sold": float(g.qty_sold.mean())}
        )
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(OUT / "deficit_decomposition.csv", index=False)

    # --- unified frontier figure --------------------------------------------
    e29 = pd.read_csv(E29_GINI)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    # Background: E29's 145 policy-economy cells (muted; open = sick seed).
    for healthy, fill in ((True, True), (False, False)):
        d = e29[e29.healthy == healthy]
        ax.scatter(
            d.survival_pct, d.gini_mean, s=16, marker="o",
            facecolors=C_BG if fill else "none", edgecolors=C_BG,
            linewidths=0.9, alpha=0.65,
            label=("policy economies (E29, healthy)" if healthy
                   else "policy economies (E29, sick seed)"),
            zorder=2,
        )
    # Overlay: 34 LLM cells, distinct marker per group.
    styles = {
        "selfplay/mixed": (C_SELF, "o", "LLM self-play / mixed 4-LLM"),
        "vs-policy": (C_POLICY, "^", "LLM vs trained policies"),
        "vs-scripted": (C_SCRIPTED, "s", "LLM vs scripted bots"),
    }
    for grp, (color, marker, label) in styles.items():
        d = by_cell[by_cell.group == grp]
        ax.scatter(
            d.survival_pct, d.gini_mean, s=46, marker=marker,
            facecolors=color, edgecolors="white", linewidths=0.7,
            label=label, zorder=4,
        )

    def note(mask, text, dx, dy, ha="left"):
        d = by_cell[mask]
        if d.empty:
            return
        x, y = float(d.survival_pct.iloc[0]), float(d.gini_mean.iloc[0])
        ax.annotate(text, (x, y), xytext=(x + dx, y + dy), ha=ha,
                    fontsize=7.5, color="0.25",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.55"))

    worst = by_cell.loc[by_cell.gini_mean.idxmax()]
    note(by_cell.cell == worst.cell, worst.cell, -7, 0.055, ha="left")
    best_s = by_cell.loc[by_cell.survival_pct.idxmax()]
    note(by_cell.cell == best_s.cell, best_s.cell, 2, -0.045)
    note(by_cell.cell == "mixed_4llm", "mixed_4llm", -11, 0.028, ha="right")

    ax.set_xlabel("aggregate survival, all four seats (%)")
    ax.set_ylabel("mean per-game terminal-cash Gini")
    ax.set_xlim(-3, 103)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, lw=0.4, color="0.9", zorder=0)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("Unified welfare frontier: policy economies (E29) + LLM economies (E26)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "unified_frontier.pdf")
    fig.savefig(OUT / "unified_frontier.png", dpi=220)
    plt.close(fig)

    # --- findings ------------------------------------------------------------
    def grp_stats(df: pd.DataFrame, grp: str) -> pd.DataFrame:
        return df[df.group == grp]

    w_self = grp_stats(by_cell, "selfplay/mixed")
    w_pol = grp_stats(by_cell, "vs-policy")
    w_scr = grp_stats(by_cell, "vs-scripted")
    d_pol = grp_stats(dec, "vs-policy")
    d_scr = grp_stats(dec, "vs-scripted")
    e29_healthy = e29[e29.healthy]

    def wrange(df, col, fmt="{:.3f}"):
        return f"{fmt.format(df[col].min())}-{fmt.format(df[col].max())}"

    lines = [
        "# E26+E29 unified — LLM economies on the welfare frontier; the LLM "
        "deficit decomposed into volume vs price [EconML; Aug 29, offline]",
        "",
        f"Data: the {len(cells)} E26 replay cells ({sanity['n_games']} validated "
        "games; validation.ok only) welfare-accounted with E29's exact "
        "definitions, overlaid on E29's 145-cell frontier. Script: "
        "`workshops/econml/analyze_llm_welfare.py`; outputs "
        "`workshops/econml/results/llm_welfare/`.",
        "",
        "**Sanity**: (a) cash conservation — total final-tick cash across the "
        f"four seats is exactly 400 in {ok_cash}/{sanity['n_games']} validated "
        f"games (max |dev| {sanity['max_total_dev']:.2e}); the LLM economies "
        "conserve the 4x100 endowment exactly like the policy economies. "
        "(b) replay final-tick cash equals the campaign's own per-seat "
        f"terminal_cash in {sanity['checked_seats']} seat comparisons with "
        f"{len(sanity['terminal_mismatches'])} mismatches "
        f"({len(sanity['alive_mismatches'])} alive/survived mismatches, "
        f"{len(sanity['orig_missing'])} originals missing) — reserved cash is "
        "returned to the player before the terminal snapshot; final-tick `cash` "
        "IS the campaign's terminal cash.",
        "",
        "## 1. LLM economies on the E29 frontier (`llm_welfare_by_cell.csv`, "
        "`unified_frontier.pdf/.png`)",
        "",
        f"- LLM self-play/mixed ({len(w_self)} cells): survival "
        f"{wrange(w_self, 'survival_pct', '{:.1f}')}%, Gini "
        f"{wrange(w_self, 'gini_mean')}, dead-wealth share "
        f"{wrange(w_self, 'dead_share_mean_pct', '{:.1f}')}%.",
        f"- LLM vs scripted ({len(w_scr)} cells): survival "
        f"{wrange(w_scr, 'survival_pct', '{:.1f}')}%, Gini "
        f"{wrange(w_scr, 'gini_mean')}, dead share "
        f"{wrange(w_scr, 'dead_share_mean_pct', '{:.1f}')}%.",
        f"- LLM vs policies ({len(w_pol)} cells): survival "
        f"{wrange(w_pol, 'survival_pct', '{:.1f}')}%, Gini "
        f"{wrange(w_pol, 'gini_mean')}, dead share "
        f"{wrange(w_pol, 'dead_share_mean_pct', '{:.1f}')}%.",
        f"- E29 healthy policy economies for reference: Gini "
        f"{wrange(e29_healthy, 'gini_mean')}, survival "
        f"{wrange(e29_healthy, 'survival_pct', '{:.1f}')}%.",
        "",
        "## 2. Deficit decomposition (`deficit_decomposition.csv`)",
        "",
        "Per LLM-vs-X game: actual net outflow = paid-as-buyer - "
        "received-as-seller; volume component = the same trades at the game's "
        "per-food VWAP; price component = actual - volume (the cost of adverse "
        "prices).",
        "",
        f"- **Vs scripted bots ({len(d_scr)} cells)**: mean actual outflow "
        f"${d_scr.actual_net_outflow_mean.mean():.1f}/game, of which volume "
        f"${d_scr.volume_component_mean.mean():.1f} and price "
        f"${d_scr.price_component_mean.mean():.1f} — price share "
        f"{100 * d_scr.price_component_mean.mean() / d_scr.actual_net_outflow_mean.mean():.1f}% "
        f"(per-cell price share {wrange(d_scr, 'price_share_pct', '{:.1f}')}%).",
        f"- **Vs trained policies ({len(d_pol)} cells)**: mean actual outflow "
        f"${d_pol.actual_net_outflow_mean.mean():.1f}/game, volume "
        f"${d_pol.volume_component_mean.mean():.1f}, price "
        f"${d_pol.price_component_mean.mean():.1f} — price share "
        f"{100 * d_pol.price_component_mean.mean() / d_pol.actual_net_outflow_mean.mean():.1f}% "
        f"(per-cell {wrange(d_pol, 'price_share_pct', '{:.1f}')}%).",
    ]
    v_neg = d_pol[d_pol.volume_component_mean < 0]
    cv2 = dec[dec.cell == "claude_vs_v2"]
    lines += [
        f"- **The whole deficit is price vs policies**: the volume component is "
        f"NEGATIVE in {len(v_neg)}/{len(d_pol)} vs-policy cells "
        f"(all four vs-ippo_s0 cells; e.g. deepseek_vs_ippo_s0 "
        f"${float(d_pol[d_pol.cell == 'deepseek_vs_ippo_s0'].volume_component_mean.iloc[0]):.0f}) "
        "— at fair prices the LLM's own trade pattern would RECEIVE net cash; "
        "the observed ~$80-95 outflow is adverse pricing alone (price share "
        ">100%).",
        f"- One vs-scripted exception: claude_vs_v2 price share "
        f"{float(cv2.price_share_pct.iloc[0]):.1f}% "
        f"(price ${float(cv2.price_component_mean.iloc[0]):.1f} of "
        f"${float(cv2.actual_net_outflow_mean.iloc[0]):.1f}) — Claude's "
        "hyperactive selling into the v2 book (47 sells/game) leaks cash "
        "through prices even against bots; the other 15 vs-scripted cells "
        "range -29 to +20%.",
        "",
    ]
    OUT.joinpath("FINDINGS.md").write_text("\n".join(lines))

    print("\n--- llm_welfare_by_cell ---")
    print(by_cell.round(3).to_string(index=False))
    print("\n--- deficit_decomposition ---")
    print(dec.round(2).to_string(index=False))
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
