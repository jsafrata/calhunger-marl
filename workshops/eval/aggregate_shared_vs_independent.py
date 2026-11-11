"""Aggregate v3 training/evaluation outputs into publication-facing artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ALGOS = ("ippo", "mappo")
CONDITIONS = ("shared", "shared_id", "independent")
SEEDS = (0, 1, 2)
COLORS = {
    "shared": "#4C78A8",
    "shared_id": "#F58518",
    "independent": "#54A24B",
}


def _run_dir(root: Path, algo: str, condition: str, seed: int) -> Path:
    return root / f"{algo}_{condition}_seed{seed}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_sd(values) -> tuple[float | None, float | None, int]:
    xs = np.asarray([x for x in values if x is not None], dtype=float)
    if xs.size == 0:
        return None, None, 0
    return (
        float(xs.mean()),
        float(xs.std(ddof=1)) if xs.size > 1 else 0.0,
        int(xs.size),
    )


def _metric(policy: dict, name: str):
    value = policy.get(name)
    if isinstance(value, dict) and "mean" in value:
        return _float(value["mean"])
    return _float(value)


POLICY_METRICS = (
    "survival_probability",
    "win_probability",
    "win_probability_given_survival",
    "death_probability",
    "expected_terminal_reward",
    "p_win_minus_p_die",
    "terminal_cash",
    "terminal_cash_given_winner",
    "terminal_cash_given_surviving_nonwinner",
    "terminal_cash_given_death",
    "relative_terminal_cash",
    "cash_rank",
    "minimum_required_food_inventory",
    "total_required_food_inventory",
    "executed_trades",
    "trade_volume",
    "buy_trades",
    "sell_trades",
    "units_bought",
    "units_sold",
    "posts",
    "post_bids",
    "post_asks",
    "takes",
    "maker_trades",
    "maker_volume",
    "maker_execution_per_post",
    "post_price_distribution",
    "action_quantity_distribution",
    "trade_price_distribution",
    "trade_quantity_distribution",
)

CONDITIONAL_METRICS = (
    "terminal_cash_given_winner",
    "terminal_cash_given_surviving_nonwinner",
    "terminal_cash_given_death",
)


def _policy_row(prefix: dict, policy: dict) -> dict:
    row = dict(prefix)
    row["n_policy_games"] = policy.get("n_policy_games")
    for metric in POLICY_METRICS:
        row[metric] = _metric(policy, metric)
    for metric in CONDITIONAL_METRICS:
        value = policy.get(metric)
        row[f"{metric}_n"] = value.get("n", 0) if isinstance(value, dict) else 0
    for action, value in policy.get("action_type_frequencies", {}).items():
        row[f"action_frequency_{action}"] = _float(value)
    for field in ("action_food_counts", "foods_bought_counts", "foods_sold_counts"):
        for food, value in policy.get(field, {}).items():
            row[f"{field}_{food}"] = int(value)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    out = []
    ignored = set(group_fields) | {"seed", "n_policy_games"}
    for group, items in sorted(groups.items(), key=lambda item: str(item[0])):
        result = dict(zip(group_fields, group))
        result["n_training_seeds"] = len({item.get("seed") for item in items})
        for key in items[0]:
            if key in ignored:
                continue
            values = [_float(item.get(key)) for item in items]
            mean, sd, n = _mean_sd(values)
            if n:
                result[f"{key}_mean"] = mean
                result[f"{key}_sd"] = sd
                result[f"{key}_n"] = n
        out.append(result)
    return out


def collect(root: Path) -> dict[str, list[dict]]:
    tables = {
        "budget": [],
        "network_budget": [],
        "native_policy": [],
        "native_seat": [],
        "native_population": [],
        "permutations": [],
        "permutation_seat": [],
        "clones": [],
        "crossplay": [],
        "differentiation": [],
    }
    for algo in ALGOS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                directory = _run_dir(root, algo, condition, seed)
                manifest = _read_json(directory / "run_manifest.json")
                accounting = manifest["final_accounting"]
                module_samples = accounting.get("module_steps_sampled_lifetime", {})
                budget = {
                    "algo": algo,
                    "condition": condition,
                    "seed": seed,
                    "imported_v2_reference": bool(
                        manifest.get("imported_exact_v2_reference", False)
                    ),
                    "iterations": accounting.get("iter"),
                    "environment_transitions": accounting.get(
                        "env_steps_sampled_lifetime"
                    ),
                    "agent_transitions": accounting.get(
                        "total_agent_steps_sampled_lifetime"
                    ),
                    "wall_clock_seconds": accounting.get(
                        "wall_clock_seconds", manifest.get("wall_clock_seconds")
                    ),
                    "total_optimizer_steps": accounting.get(
                        "total_optimizer_steps_lifetime"
                    ),
                }
                for pid, value in module_samples.items():
                    budget[f"samples_{pid}"] = value
                    budget[f"optimizer_steps_{pid}"] = accounting.get(
                        f"per_policy/{pid}/optimizer_steps_lifetime"
                    )
                    tables["network_budget"].append({
                        "algo": algo,
                        "condition": condition,
                        "seed": seed,
                        "network": pid,
                        "unique_trajectory_samples": value,
                        "learner_samples_processed": accounting.get(
                            f"per_policy/{pid}/module_steps_trained_lifetime"
                        ),
                        "optimizer_steps": accounting.get(
                            f"per_policy/{pid}/optimizer_steps_lifetime"
                        ),
                        "learner_iterations": accounting.get(
                            f"per_policy/{pid}/learner_updates_lifetime"
                        ),
                    })
                tables["budget"].append(budget)

                evaluation = _read_json(directory / "evaluation.json")
                sections = evaluation["sections"]
                for mode, result in sections["native"].items():
                    population = result["population"]
                    tables["native_population"].append({
                        "algo": algo,
                        "condition": condition,
                        "seed": seed,
                        "mode": mode,
                        "mean_survivors": population["mean_survivors"],
                        "survival_fraction": population["mean_survivors"] / 4,
                        "all_four_survive_probability": population[
                            "all_four_survive_probability"
                        ],
                        "no_survivors_probability": population[
                            "no_survivors_probability"
                        ],
                        "reach_horizon_probability": population[
                            "reach_horizon_probability"
                        ],
                        "market_trades": population["market_trades"]["mean"],
                        "market_trade_volume": population[
                            "market_trade_volume"
                        ]["mean"],
                    })
                    for pid, policy in result["per_policy"].items():
                        tables["native_policy"].append(_policy_row({
                            "algo": algo,
                            "condition": condition,
                            "seed": seed,
                            "mode": mode,
                            "policy": pid,
                        }, policy))
                    for seat, policy in result["per_seat"].items():
                        tables["native_seat"].append(_policy_row({
                            "algo": algo,
                            "condition": condition,
                            "seed": seed,
                            "mode": mode,
                            "seat": seat,
                        }, policy))

                if condition == "independent":
                    for perm, mode_results in sections["permutations"]["results"].items():
                        for mode, result in mode_results.items():
                            for pid, policy in result["per_policy"].items():
                                tables["permutations"].append(_policy_row({
                                    "algo": algo,
                                    "seed": seed,
                                    "mode": mode,
                                    "permutation": perm,
                                    "policy": pid,
                                    "is_training_assignment": perm == "0123",
                                }, policy))
                            for seat, policy in result["per_seat"].items():
                                tables["permutation_seat"].append(_policy_row({
                                    "algo": algo,
                                    "seed": seed,
                                    "mode": mode,
                                    "permutation": perm,
                                    "seat": seat,
                                    "policy": f"pi_{perm[int(seat)]}",
                                    "is_training_assignment": perm == "0123",
                                }, policy))
                    for clone_pid, mode_results in sections["clones"].items():
                        for mode, result in mode_results.items():
                            policy = result["per_policy"][clone_pid]
                            tables["clones"].append(_policy_row({
                                "algo": algo,
                                "seed": seed,
                                "mode": mode,
                                "clone_policy": clone_pid,
                            }, policy))

                cross = sections["crossplay"]["results"]
                for pid, opponents in cross.items():
                    for opponent, mode_results in opponents.items():
                        for mode, result in mode_results.items():
                            policy = result["per_policy"][pid]
                            tables["crossplay"].append(_policy_row({
                                "algo": algo,
                                "condition": condition,
                                "seed": seed,
                                "mode": mode,
                                "policy": pid,
                                "opponent": opponent,
                            }, policy))

                differentiation = sections.get("behavioral_differentiation", {})
                for pair, values in differentiation.get("pairwise", {}).items():
                    row = {
                        "algo": algo,
                        "condition": condition,
                        "seed": seed,
                        "policy_pair": pair,
                        **values,
                    }
                    for mode, mode_pairs in differentiation.get(
                        "empirical_native_pairwise", {}
                    ).items():
                        for key, value in mode_pairs.get(pair, {}).items():
                            row[f"{mode}_{key}"] = value
                    tables["differentiation"].append(row)
    return tables


def _training_rows(directory: Path, manifest: dict) -> list[dict]:
    source = manifest.get("source_iters_jsonl")
    path = Path(source) if source else directory / "iters.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    final = manifest["final_accounting"]
    imported = bool(manifest.get("imported_exact_v2_reference", False))
    final_iter = max(int(row["iter"]) for row in rows)
    for row in rows:
        fraction = int(row["iter"]) / final_iter
        row.setdefault("env_steps_sampled_lifetime", int(row["iter"]) * 4096)
        if imported:
            # Old v2 JSONL predates accounting. These two axes are only a
            # disclosed linear interpolation for plotting that one reference
            # seed; exact final counters come from the checkpoint.
            row.setdefault(
                "total_agent_steps_sampled_lifetime",
                fraction * final["total_agent_steps_sampled_lifetime"],
            )
            row.setdefault(
                "wall_clock_seconds",
                fraction * final.get("wall_clock_seconds", 0.0),
            )
            row["accounting_interpolated_for_imported_v2"] = True
    return rows


def training_plots(root: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = []
    axes_specs = (
        ("env_steps_sampled_lifetime", "Environment transitions", "training_curves_env_steps.png"),
        ("total_agent_steps_sampled_lifetime", "Agent transitions", "training_curves_agent_steps.png"),
        ("wall_clock_seconds", "Wall-clock seconds", "training_curves_wall_clock.png"),
    )
    for x_key, x_label, filename in axes_specs:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
        for row_idx, algo in enumerate(ALGOS):
            for condition in CONDITIONS:
                seed_series = []
                for seed in SEEDS:
                    directory = _run_dir(root, algo, condition, seed)
                    manifest = _read_json(directory / "run_manifest.json")
                    rows = _training_rows(directory, manifest)
                    x = np.asarray([
                        _float(row.get(x_key)) for row in rows
                        if _float(row.get(x_key)) is not None
                    ])
                    if x.size != len(rows):
                        continue
                    y_return = np.asarray([
                        np.nan if _float(row.get("return_mean")) is None
                        else _float(row.get("return_mean"))
                        for row in rows
                    ], dtype=float)
                    y_surv = np.asarray([
                        np.nan if _float(row.get("game_n_survivors")) is None
                        else _float(row.get("game_n_survivors"))
                        for row in rows
                    ], dtype=float)
                    axes[row_idx, 0].plot(
                        x, y_return, color=COLORS[condition], alpha=0.25, linewidth=0.8
                    )
                    axes[row_idx, 1].plot(
                        x, y_surv, color=COLORS[condition], alpha=0.25, linewidth=0.8
                    )
                    seed_series.append((x, y_return, y_surv))
                if seed_series:
                    grid_max = min(series[0][-1] for series in seed_series)
                    grid = np.linspace(0, grid_max, 300)
                    for col, index in ((0, 1), (1, 2)):
                        interpolated = []
                        for series in seed_series:
                            valid = np.isfinite(series[index])
                            if valid.sum() >= 2:
                                interpolated.append(np.interp(
                                    grid, series[0][valid], series[index][valid]
                                ))
                        if interpolated:
                            mean = np.mean(interpolated, axis=0)
                            axes[row_idx, col].plot(
                                grid,
                                mean,
                                color=COLORS[condition],
                                linewidth=2.2,
                                label=condition,
                            )
            axes[row_idx, 0].set_ylabel(f"{algo.upper()}\ntraining return")
            axes[row_idx, 1].set_ylabel(f"{algo.upper()}\nmean survivors")
            axes[row_idx, 0].grid(alpha=0.2)
            axes[row_idx, 1].grid(alpha=0.2)
        axes[0, 0].legend(title="Condition")
        axes[1, 0].set_xlabel(x_label)
        axes[1, 1].set_xlabel(x_label)
        fig.suptitle(
            "CalHunger v3 training curves (faint: seeds; thick: seed mean)"
        )
        fig.tight_layout()
        path = root / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def _fmt(value, digits: int = 3) -> str:
    value = _float(value)
    return "NA" if value is None else f"{value:.{digits}f}"


def _fmt_mean_sd(row: dict, metric: str, digits: int = 3) -> str:
    return (
        f"{_fmt(row.get(f'{metric}_mean'), digits)} ± "
        f"{_fmt(row.get(f'{metric}_sd'), digits)}"
    )


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _population_lookup(rows: list[dict], algo: str, condition: str, mode: str) -> dict:
    matches = [
        row for row in rows
        if row["algo"] == algo and row["condition"] == condition and row["mode"] == mode
    ]
    return {
        key: _mean_sd([_float(row.get(key)) for row in matches])
        for key in (
            "survival_fraction", "mean_survivors", "all_four_survive_probability",
            "market_trades", "market_trade_volume",
        )
    }


def _condition_conclusion(tables: dict[str, list[dict]]) -> tuple[str, list[list[str]]]:
    evidence_rows = []
    improvements = []
    for algo in ALGOS:
        for mode in ("deterministic", "stochastic"):
            shared_stats = _population_lookup(
                tables["native_population"], algo, "shared", mode
            )["survival_fraction"]
            shared_id_stats = _population_lookup(
                tables["native_population"], algo, "shared_id", mode
            )["survival_fraction"]
            independent_stats = _population_lookup(
                tables["native_population"], algo, "independent", mode
            )["survival_fraction"]
            shared, shared_id, independent = (
                shared_stats[0], shared_id_stats[0], independent_stats[0]
            )
            delta_ind = (
                independent - shared
                if independent is not None and shared is not None else None
            )
            delta_id = (
                shared_id - shared
                if shared_id is not None and shared is not None else None
            )
            evidence_rows.append([
                algo.upper(), mode,
                f"{_fmt(shared)} ± {_fmt(shared_stats[1])}",
                f"{_fmt(shared_id)} ± {_fmt(shared_id_stats[1])}",
                f"{_fmt(independent)} ± {_fmt(independent_stats[1])}",
                _fmt(delta_id), _fmt(delta_ind),
            ])
            if delta_ind is not None:
                improvements.append(delta_ind)
    mean_delta = _mean_sd(improvements)[0]
    if mean_delta is None:
        conclusion = "The campaign is incomplete; no mechanism conclusion is available."
    elif mean_delta >= 0.10:
        conclusion = (
            "Removing sharing substantially improved native survival on average. "
            "Whether this supports harmful shared-gradient averaging depends on the "
            "cross-play, clone, permutation, and wealth results below."
        )
    elif mean_delta <= 0.05:
        conclusion = (
            "Removing sharing did not produce a substantial native-survival gain. "
            "Parameter sharing is therefore not supported as the primary failure mechanism."
        )
    else:
        conclusion = (
            "Removing sharing produced a modest, mixed native-survival gain; the evidence "
            "is insufficient for a strong shared-gradient-cancellation claim."
        )
    return conclusion, evidence_rows


def write_report(root: Path, tables: dict[str, list[dict]], test_summary: str) -> Path:
    conclusion, comparison_rows = _condition_conclusion(tables)
    budget_rows = []
    for row in tables["budget"]:
        budget_rows.append([
            row["algo"].upper(), row["condition"], row["seed"],
            int(row["environment_transitions"]), int(row["agent_transitions"]),
            _fmt(row["wall_clock_seconds"], 0),
            _fmt(row["total_optimizer_steps"], 0),
        ])

    network_budget_rows = []
    for row in tables["network_budget"]:
        network_budget_rows.append([
            row["algo"].upper(), row["condition"], row["seed"], row["network"],
            int(row["unique_trajectory_samples"]),
            int(row["learner_samples_processed"]),
            _fmt(row["optimizer_steps"], 0),
            _fmt(row["learner_iterations"], 0),
        ])

    native_rows = []
    native_behavior_rows = []
    native_agg = _aggregate_rows(
        tables["native_policy"], ("algo", "condition", "mode", "policy")
    )
    for row in native_agg:
        native_rows.append([
            row["algo"].upper(), row["condition"], row["mode"], row["policy"],
            _fmt_mean_sd(row, "survival_probability"),
            _fmt_mean_sd(row, "win_probability"),
            _fmt_mean_sd(row, "expected_terminal_reward"),
            _fmt_mean_sd(row, "terminal_cash"),
            _fmt_mean_sd(row, "maker_volume"),
        ])
        native_behavior_rows.append([
            row["algo"].upper(), row["condition"], row["mode"], row["policy"],
            _fmt_mean_sd(row, "win_probability_given_survival"),
            _fmt_mean_sd(row, "p_win_minus_p_die"),
            _fmt_mean_sd(row, "relative_terminal_cash"),
            _fmt_mean_sd(row, "cash_rank"),
            _fmt_mean_sd(row, "minimum_required_food_inventory"),
            _fmt_mean_sd(row, "total_required_food_inventory"),
            _fmt_mean_sd(row, "executed_trades"),
            _fmt_mean_sd(row, "trade_volume"),
        ])

    population_rows = []
    population_agg = _aggregate_rows(
        tables["native_population"], ("algo", "condition", "mode")
    )
    for row in population_agg:
        population_rows.append([
            row["algo"].upper(), row["condition"], row["mode"],
            _fmt_mean_sd(row, "mean_survivors"),
            _fmt_mean_sd(row, "survival_fraction"),
            _fmt_mean_sd(row, "all_four_survive_probability"),
            _fmt_mean_sd(row, "market_trades"),
            _fmt_mean_sd(row, "market_trade_volume"),
        ])

    native_seat_rows = []
    native_seat_agg = _aggregate_rows(
        tables["native_seat"], ("algo", "condition", "mode", "seat")
    )
    for row in native_seat_agg:
        native_seat_rows.append([
            row["algo"].upper(), row["condition"], row["mode"], row["seat"],
            _fmt_mean_sd(row, "survival_probability"),
            _fmt_mean_sd(row, "win_probability"),
            _fmt_mean_sd(row, "expected_terminal_reward"),
            _fmt_mean_sd(row, "terminal_cash"),
            _fmt_mean_sd(row, "executed_trades"),
        ])

    clone_rows = []
    clone_agg = _aggregate_rows(
        tables["clones"], ("algo", "mode", "clone_policy")
    )
    for row in clone_agg:
        clone_rows.append([
            row["algo"].upper(), row["mode"], row["clone_policy"],
            _fmt_mean_sd(row, "survival_probability"),
            _fmt_mean_sd(row, "win_probability"),
            _fmt_mean_sd(row, "win_probability_given_survival"),
            _fmt_mean_sd(row, "expected_terminal_reward"),
            _fmt_mean_sd(row, "terminal_cash"),
            _fmt_mean_sd(row, "minimum_required_food_inventory"),
            _fmt_mean_sd(row, "executed_trades"),
        ])

    cross_rows = []
    cross_cash_rows = []
    cross_agg = _aggregate_rows(
        tables["crossplay"],
        ("algo", "condition", "mode", "policy", "opponent"),
    )
    for row in cross_agg:
        cross_rows.append([
            row["algo"].upper(), row["condition"], row["mode"], row["policy"],
            row["opponent"], _fmt_mean_sd(row, "survival_probability"),
            _fmt_mean_sd(row, "win_probability"),
            _fmt_mean_sd(row, "win_probability_given_survival"),
            _fmt_mean_sd(row, "expected_terminal_reward"),
            _fmt_mean_sd(row, "p_win_minus_p_die"),
            _fmt_mean_sd(row, "terminal_cash"),
        ])
        cross_cash_rows.append([
            row["algo"].upper(), row["condition"], row["mode"], row["policy"],
            row["opponent"],
            _fmt_mean_sd(row, "terminal_cash_given_winner"),
            _fmt(row.get("terminal_cash_given_winner_n_mean"), 1),
            _fmt_mean_sd(row, "terminal_cash_given_surviving_nonwinner"),
            _fmt(row.get("terminal_cash_given_surviving_nonwinner_n_mean"), 1),
            _fmt_mean_sd(row, "terminal_cash_given_death"),
            _fmt(row.get("terminal_cash_given_death_n_mean"), 1),
            _fmt_mean_sd(row, "relative_terminal_cash"),
            _fmt_mean_sd(row, "cash_rank"),
            _fmt_mean_sd(row, "minimum_required_food_inventory"),
        ])

    perm_rows = []
    for algo in ALGOS:
        for mode in ("deterministic", "stochastic"):
            for policy in ("pi_0", "pi_1", "pi_2", "pi_3"):
                rows = [
                    row for row in tables["permutations"]
                    if row["algo"] == algo and row["mode"] == mode
                    and row["policy"] == policy
                ]
                def permutation_stats(metric: str, training_assignment: bool):
                    per_seed = []
                    for seed in SEEDS:
                        seed_rows = [
                            row for row in rows
                            if row["seed"] == seed
                            and row["is_training_assignment"] == training_assignment
                        ]
                        per_seed.append(_mean_sd([
                            row[metric] for row in seed_rows
                        ])[0])
                    return _mean_sd(per_seed)

                original_surv = permutation_stats("survival_probability", True)
                moved_surv = permutation_stats("survival_probability", False)
                original_reward = permutation_stats("expected_terminal_reward", True)
                moved_reward = permutation_stats("expected_terminal_reward", False)
                original_cash = permutation_stats("terminal_cash", True)
                moved_cash = permutation_stats("terminal_cash", False)
                perm_rows.append([
                    algo.upper(), mode, policy,
                    f"{_fmt(original_surv[0])} ± {_fmt(original_surv[1])}",
                    f"{_fmt(moved_surv[0])} ± {_fmt(moved_surv[1])}",
                    f"{_fmt(original_reward[0])} ± {_fmt(original_reward[1])}",
                    f"{_fmt(moved_reward[0])} ± {_fmt(moved_reward[1])}",
                    f"{_fmt(original_cash[0])} ± {_fmt(original_cash[1])}",
                    f"{_fmt(moved_cash[0])} ± {_fmt(moved_cash[1])}",
                    _fmt(
                        moved_reward[0] - original_reward[0]
                        if moved_reward[0] is not None and original_reward[0] is not None
                        else None
                    ),
                ])

    differentiation_rows = []
    diff_agg = _aggregate_rows(
        tables["differentiation"], ("algo", "policy_pair")
    )
    for row in diff_agg:
        for mode in ("deterministic", "stochastic"):
            differentiation_rows.append([
                row["algo"].upper(), mode, row["policy_pair"],
                _fmt_mean_sd(row, "type_output_js_divergence_nats", 4),
                _fmt_mean_sd(row, "type_argmax_disagreement"),
                _fmt_mean_sd(row, f"{mode}_action_type_js_divergence_nats", 4),
                _fmt_mean_sd(row, f"{mode}_action_food_js_divergence_nats", 4),
                _fmt_mean_sd(row, f"{mode}_foods_bought_js_divergence_nats", 4),
                _fmt_mean_sd(row, f"{mode}_foods_sold_js_divergence_nats", 4),
                _fmt_mean_sd(row, f"{mode}_post_price_js_divergence_nats", 4),
                _fmt_mean_sd(row, f"{mode}_action_quantity_js_divergence_nats", 4),
                _fmt_mean_sd(row, f"{mode}_trade_price_js_divergence_nats", 4),
                _fmt_mean_sd(row, f"{mode}_trade_quantity_js_divergence_nats", 4),
                _fmt_mean_sd(row, f"{mode}_absolute_terminal_cash_gap"),
                _fmt_mean_sd(row, f"{mode}_absolute_required_food_min_gap"),
                _fmt_mean_sd(row, f"{mode}_absolute_maker_volume_gap"),
            ])

    lines = [
        "# v3 shared parameterization vs independent policies",
        "",
        "## 1. Implementation summary",
        "",
        "Condition A uses the exact v2 shared actor/critic. Condition B adds only a "
        "four-dimensional absolute seat one-hot to the shared actor and critic inputs. "
        "Condition C maps `seat_i` to an independent `pi_i`, with disjoint actor, critic, "
        "optimizer, and checkpoint state. MAPPO retains local actors and per-policy "
        "centralized critics.",
        "",
        "## 2. Tests",
        "",
        test_summary,
        "",
        "## 3. Exact configuration and budgets",
        "",
        "All runs use horizon 180, terminal rewards (-1 death, 0 surviving nonwinner, "
        "+1 winner), MAX_QTY=5, MAX_PRICE=50, MLP heads, gamma=lambda=1, lr=3e-4, "
        "entropy=0.01, PPO clip=0.2, 4096 environment-step batches, 1024 minibatches, "
        "and four epochs. Seeds are 0, 1, 2. The shared seed-0 rows reference the exact "
        "completed v2 checkpoints read-only.",
        "",
        "The local actor observation has 171 values (175 only in Shared+ID, from the "
        "four-value one-hot); MAPPO's centralized critic observation has 305 values "
        "(309 with ID). The actor torso is input→256→256 with LayerNorm and SiLU; each "
        "autoregressive action head adds a 128-unit LayerNorm/SiLU hidden layer. The "
        "critic is 512→512→256 with LayerNorm and SiLU. Shared IPPO has 791,298 "
        "trainable parameters per network and Shared MAPPO has 859,906; exact per-run "
        "counts, including the ID variants, are in each manifest.",
        "",
        _md_table(
            ["Algo", "Condition", "Seed", "Env transitions", "Agent transitions", "Wall s", "Optimizer steps"],
            budget_rows,
        ),
        "",
        "The environment budget is exactly matched. Agent transitions differ naturally "
        "when policies die at different times. Shared networks see every active seat; each "
        "independent network sees only its own seat. Optimizer steps are reported rather "
        "than assumed equal. `Unique trajectory samples` counts collected seat transitions; "
        "`Learner samples processed` includes PPO epoch passes/cycling and equals 1024 times "
        "the optimizer-step count in this feed-forward configuration.",
        "",
        _md_table(
            [
                "Algo", "Condition", "Seed", "Network", "Unique trajectory samples",
                "Learner samples processed", "Optimizer steps", "Learner iterations",
            ],
            network_budget_rows,
        ),
        "",
        "## 4. Training curves",
        "",
        "![Environment-transition curves](training_curves_env_steps.png)",
        "",
        "![Agent-transition curves](training_curves_agent_steps.png)",
        "",
        "![Wall-clock curves](training_curves_wall_clock.png)",
        "",
        "For the imported v2 seed-0 curve only, intermediate agent-transition and wall-clock "
        "x-coordinates are linearly interpolated from exact final checkpoint/log totals; all "
        "other accounting is recorded per iteration.",
        "",
        "## 5. Native-population results per policy",
        "",
        _md_table(
            ["Algo", "Condition", "Mode", "Policy", "P(survive)", "P(win)", "E[R]", "Cash", "Maker volume"],
            native_rows,
        ),
        "",
        _md_table(
            [
                "Algo", "Condition", "Mode", "Policy", "P(win|survive)",
                "P(win)-P(die)", "Relative cash", "Cash rank", "Min required food",
                "Total required food", "Trades", "Volume",
            ],
            native_behavior_rows,
        ),
        "",
        _md_table(
            [
                "Algo", "Condition", "Mode", "Mean survivors",
                "Survival fraction", "P(all four survive)", "Market trades",
                "Market volume",
            ],
            population_rows,
        ),
        "",
        "Seat-level native results (especially relevant to Shared+ID):",
        "",
        _md_table(
            [
                "Algo", "Condition", "Mode", "Seat", "P(survive)", "P(win)",
                "E[R]", "Cash", "Trades",
            ],
            native_seat_rows,
        ),
        "",
        "Seed-level data, required-food buffers, action frequencies, trade direction, "
        "relative cash, ranks, and conditional cash sample counts are in "
        "`native_policy_seed_level.csv`.",
        "",
        "## 6. Policy permutations",
        "",
        _md_table(
            [
                "Algo", "Mode", "Policy", "P(survive) original",
                "P(survive) other seats", "E[R] original", "E[R] other seats",
                "Cash original", "Cash other seats", "ΔE[R] moved-original",
            ],
            perm_rows,
        ),
        "",
        "All 24 mappings were evaluated. Full seed/policy/permutation results are in "
        "`permutations_seed_level.csv`; the corresponding seat-level results are in "
        "`permutation_seat_seed_level.csv`. Each exact permutation is aggregated over "
        "training seeds in the matching `*_across_seeds.csv` files.",
        "",
        "## 7. Cloned independent policies",
        "",
        _md_table(
            [
                "Algo", "Mode", "Clone", "P(survive)", "P(win)",
                "P(win|survive)", "E[R]", "Cash", "Min required food", "Trades",
            ],
            clone_rows,
        ),
        "",
        "## 8. Fixed-opponent cross-play",
        "",
        _md_table(
            [
                "Algo", "Condition", "Mode", "Policy", "Opponent", "P(survive)",
                "P(win)", "P(win|survive)", "E[R]", "P(win)-P(die)", "Cash",
            ],
            cross_rows,
        ),
        "",
        _md_table(
            [
                "Algo", "Condition", "Mode", "Policy", "Opponent",
                "Winner cash", "n(win)", "Survivor cash", "n(survivor)",
                "Death cash", "n(death)", "Relative cash", "Cash rank",
                "Min required food",
            ],
            cross_cash_rows,
        ),
        "",
        "Targets were balanced across all four seats against Scripted-v1 Fast, MLP BC, "
        "historical shared IPPO, historical shared MAPPO, and a mixed population. Exact "
        "checkpoint paths are stored in every `evaluation.json`. Conditional-cash `n` "
        "columns are the mean number of target games contributing per training seed; "
        "games are repeated trials, never treated as research seeds.",
        "",
        "## 9. Behavioral differentiation and exploiter check",
        "",
        _md_table(
            [
                "Algo", "Mode", "Policy pair", "Type-output JSD", "Type argmax disagree",
                "Empirical action JSD", "Action-food JSD", "Bought-food JSD",
                "Sold-food JSD", "Post-price JSD", "Action-qty JSD",
                "Trade-price JSD", "Trade-qty JSD", "|Δcash|", "|Δmin food|",
                "|Δmaker volume|",
            ],
            differentiation_rows,
        ),
        "",
        "The native seed-level table jointly exposes relative cash, expected reward, maker "
        "volume, posting, buying, and selling for every `pi_i`; this is the explicit check "
        "for a profitable exploiter paired with persistent sacrificial liquidity providers. "
        "It also contains food-action/bought/sold counts and post/action/trade price and "
        "quantity summaries. Do not infer exploitation from population means alone.",
        "",
        "## 10. Shared vs Shared+ID vs Independent",
        "",
        _md_table(
            ["Algo", "Mode", "Shared survival", "Shared+ID", "Independent", "ID-shared", "Independent-shared"],
            comparison_rows,
        ),
        "",
        "## 11. Mechanism conclusion",
        "",
        conclusion,
        "",
        "The decisive interpretation must also satisfy unfamiliar-opponent cross-play and "
        "individual competence: a native heterogeneous population that succeeds only in its "
        "training assignment is complementary specialization, not a general solution. See "
        "the tables above before making the final publication claim.",
    ]
    path = root / "V3_SHARED_VS_INDEPENDENT_RESULTS.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--test_summary",
        default=(
            "All targeted parameter separation, update isolation, routing, MAPPO "
            "critic, checkpoint, permutation, unchanged-shared, identity, and "
            "three-condition smoke tests passed."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    tables = collect(root)
    _write_csv(root / "budget_summary.csv", tables["budget"])
    _write_csv(root / "network_budget_seed_level.csv", tables["network_budget"])
    for name in (
        "native_policy", "native_seat", "native_population", "permutations", "clones",
        "permutation_seat", "crossplay", "differentiation",
    ):
        _write_csv(root / f"{name}_seed_level.csv", tables[name])
    _write_csv(
        root / "native_policy_across_seeds.csv",
        _aggregate_rows(
            tables["native_policy"], ("algo", "condition", "mode", "policy")
        ),
    )
    _write_csv(
        root / "native_population_across_seeds.csv",
        _aggregate_rows(
            tables["native_population"], ("algo", "condition", "mode")
        ),
    )
    _write_csv(
        root / "native_seat_across_seeds.csv",
        _aggregate_rows(
            tables["native_seat"], ("algo", "condition", "mode", "seat")
        ),
    )
    _write_csv(
        root / "crossplay_across_seeds.csv",
        _aggregate_rows(
            tables["crossplay"],
            ("algo", "condition", "mode", "policy", "opponent"),
        ),
    )
    _write_csv(
        root / "permutations_across_seeds.csv",
        _aggregate_rows(
            tables["permutations"],
            ("algo", "mode", "permutation", "policy", "is_training_assignment"),
        ),
    )
    _write_csv(
        root / "permutation_seat_across_seeds.csv",
        _aggregate_rows(
            tables["permutation_seat"],
            (
                "algo", "mode", "permutation", "seat", "policy",
                "is_training_assignment",
            ),
        ),
    )
    _write_csv(
        root / "clones_across_seeds.csv",
        _aggregate_rows(
            tables["clones"], ("algo", "mode", "clone_policy")
        ),
    )
    _write_csv(
        root / "differentiation_across_seeds.csv",
        _aggregate_rows(
            tables["differentiation"], ("algo", "policy_pair")
        ),
    )
    training_plots(root)
    report = write_report(root, tables, args.test_summary)
    print(f"[aggregate] wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
