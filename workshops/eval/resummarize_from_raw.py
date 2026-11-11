"""Regenerate extended scripted-diagnostics summaries from cached raw JSONs.

Walks a results root, finds every directory whose ``raw/`` cache was written by
``eval/checkpoint_scripted_diagnostics.py`` (files named
``<algo>_seed<N>_iter<NNNN>_1learned_3scripted[_<opp>]_<mode>_<games>.json``),
and re-runs ``summarize_result`` on the stored ``raw_result`` (plus stored
``head_diagnostics`` where present) to produce the extended metric columns.

No games are played and Ray is never initialised — RLlib/torch are imported
only because the diagnostics module imports them at module scope.

Outputs go ONLY under ``workshops/results/`` mirroring the relative path of
each source directory, as ``checkpoint_diagnostics_extended.csv`` (+ ``.json``).
A validation report ``workshops/results/RESUMMARIZE_REPORT.md`` compares the
columns shared with each source directory's committed
``checkpoint_diagnostics.csv`` row-for-row (float tolerance 1e-9).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

WORKSHOPS_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = WORKSHOPS_ROOT.parent / "research"

# The diagnostics module resolves `agents.*`, `env.*` and (via its own
# sys.path insert of the workshops root) `eval._warnings` — make sure both
# roots are importable before loading it by file path.
for path in (str(WORKSHOPS_ROOT / "eval"), str(RESEARCH_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_diagnostics_module():
    module_path = WORKSHOPS_ROOT / "eval" / "checkpoint_scripted_diagnostics.py"
    spec = importlib.util.spec_from_file_location(
        "workshops_checkpoint_scripted_diagnostics", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CSD = _load_diagnostics_module()

RAW_NAME_RE = re.compile(
    r"^(?P<algorithm>.+)_seed(?P<seed>\d+)_iter(?P<checkpoint>\d+)_"
    r"1learned_3scripted(?:_(?P<opponent>v\d+))?_"
    r"(?P<mode>deterministic|stochastic)_(?P<n_games>\d+)\.json$"
)

# Extended metrics derived purely from per-game rollout statistics stored in
# every generation of raw cache.
GAME_DERIVED_EXTENDED = (
    "win_given_survival_percentage",
    "mean_trades_as_buyer",
    "mean_trades_as_seller",
    "mean_posted_bids",
    "mean_posted_asks",
    "mean_takes",
    "frac_wait",
    "frac_post",
    "survivor_frac_min_required_0_to_1",
)
# Extended metrics that need the policy network's head distributions on real
# observations; computable from cache only when the raw JSON stored a
# ``head_diagnostics`` block at evaluation time.
HEAD_DERIVED_EXTENDED = (
    "type_prob_post_mean",
    "type_entropy_mean",
    "price_given_post_entropy",
)

# Fields whose availability in ``raw_result`` we track per file so the report
# can state which extended metrics are genuinely cache-derived (vs. silently
# defaulted by ``summarize_result``'s .get fallbacks).
RAW_REQUIRED_KEYS = {
    "win_given_survival_percentage": ("terminal_metrics",),
    "mean_trades_as_buyer": ("learned_seat_completed_trades_as_buyer_mean",),
    "mean_trades_as_seller": ("learned_seat_completed_trades_as_seller_mean",),
    "mean_posted_bids": ("learned_seat_posted_bids_mean",),
    "mean_posted_asks": ("learned_seat_posted_asks_mean",),
    "mean_takes": ("learned_seat_takes_mean",),
    "frac_wait": ("learned_seat_type_freq_mean",),
    "frac_post": ("learned_seat_type_freq_mean",),
}

FLOAT_TOLERANCE = 1e-9


def find_scripted_raw_dirs(results_root: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Return (dirs containing scripted-diagnostics raw caches, skipped raw dirs)."""
    processed: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for raw_dir in sorted(results_root.rglob("raw")):
        if not raw_dir.is_dir():
            continue
        files = sorted(raw_dir.glob("*.json"))
        scripted = [f for f in files if RAW_NAME_RE.match(f.name)]
        if scripted:
            processed.append(raw_dir.parent)
        else:
            reason = (
                "raw cache is from the self-play survival diagnostics "
                "(no *_1learned_3scripted_* files)"
                if files
                else "raw directory contains no JSON files"
            )
            skipped.append((raw_dir.parent, reason))
    # Top-level result sets with no raw/ cache anywhere (e.g. campaigns that
    # only stored deep_eval reports or plots) — listed so the report is
    # explicit about why they produce no extended summary.
    for top in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if not any(d.is_dir() for d in top.rglob("raw")):
            skipped.append(
                (top, "no scripted-diagnostics raw/ cache anywhere under it")
            )
    return processed, skipped


def resummarize_dir(source_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild extended summary rows for one diagnostics output directory."""
    rows: list[dict[str, Any]] = []
    missing_raw_keys: dict[str, set[str]] = {}
    n_with_head = 0
    for raw_path in sorted((source_dir / "raw").glob("*.json")):
        match = RAW_NAME_RE.match(raw_path.name)
        if not match:
            continue
        report = json.loads(raw_path.read_text())
        raw_result = report["raw_result"]
        cached_summary = report.get("summary", {})
        head_diagnostics = report.get("head_diagnostics")
        if head_diagnostics:
            n_with_head += 1
        mode = report.get("mode", match.group("mode"))
        summary = CSD.summarize_result(
            raw_result,
            algorithm=str(
                cached_summary.get("algorithm", match.group("algorithm"))
            ),
            seed=int(cached_summary.get("seed", match.group("seed"))),
            checkpoint=int(
                cached_summary.get("checkpoint", match.group("checkpoint"))
            ),
            n_games=int(report.get("n_games", match.group("n_games"))),
            deterministic=(mode == "deterministic"),
            game_seed_base=int(report["game_seed_base"]),
            opponent=str(
                report.get("opponent", match.group("opponent") or "v1")
            ),
            head_diagnostics=head_diagnostics,
        )
        rows.append(summary)
        # Track which game-derived metrics were genuinely present in the raw
        # record (summarize_result silently defaults missing ones to 0.0).
        for metric, keys in RAW_REQUIRED_KEYS.items():
            if any(key not in raw_result for key in keys):
                missing_raw_keys.setdefault(metric, set()).add(raw_path.name)
        buckets = raw_result.get("terminal_metrics", {}).get(
            "survivor_min_required_buckets"
        )
        if buckets is None or "frac_0_to_1" not in buckets:
            missing_raw_keys.setdefault(
                "survivor_frac_min_required_0_to_1", set()
            ).add(raw_path.name)

    rows.sort(key=lambda row: (row["algorithm"], row["seed"], row["checkpoint"]))
    info = {
        "n_rows": len(rows),
        "n_with_head_diagnostics": n_with_head,
        "missing_raw_keys": {
            metric: sorted(files) for metric, files in missing_raw_keys.items()
        },
    }
    return rows, info


def _parse_cell(value: str) -> Any:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _cells_equal(committed: Any, regenerated: Any) -> bool:
    if isinstance(committed, float) and isinstance(regenerated, (int, float)):
        return abs(committed - float(regenerated)) <= FLOAT_TOLERANCE
    if committed is None and regenerated is None:
        return True
    return str(committed) == str(regenerated)


def validate_against_committed(
    source_dir: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    committed_path = source_dir / "checkpoint_diagnostics.csv"
    if not committed_path.is_file():
        return {"status": "no committed CSV", "shared_columns": [], "mismatches": []}
    with committed_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        committed_rows = list(reader)
        committed_columns = list(reader.fieldnames or [])
    shared = [c for c in committed_columns if c in CSD.SUMMARY_FIELDS]
    sort_key = lambda r: (str(r["algorithm"]), int(r["seed"]), int(r["checkpoint"]))
    committed_rows.sort(key=sort_key)
    regenerated_rows = sorted(rows, key=sort_key)
    result: dict[str, Any] = {
        "shared_columns": shared,
        "committed_rows": len(committed_rows),
        "regenerated_rows": len(regenerated_rows),
        "mismatches": [],
    }
    if len(committed_rows) != len(regenerated_rows):
        result["status"] = "FAIL (row count mismatch)"
        return result
    for i, (old, new) in enumerate(zip(committed_rows, regenerated_rows)):
        for column in shared:
            committed_value = _parse_cell(old[column])
            regenerated_value = new[column]
            if isinstance(committed_value, float) and regenerated_value is not None:
                regenerated_value = float(regenerated_value)
            if not _cells_equal(committed_value, regenerated_value):
                result["mismatches"].append(
                    {
                        "row": i,
                        "id": (old["algorithm"], old["seed"], old["checkpoint"]),
                        "column": column,
                        "committed": old[column],
                        "regenerated": new[column],
                    }
                )
    result["status"] = "PASS" if not result["mismatches"] else "FAIL"
    return result


def write_extended_outputs(
    output_dir: Path, rows: list[dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "checkpoint_diagnostics_extended.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSD.SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "checkpoint_diagnostics_extended.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


def write_report(
    report_path: Path,
    results_root: Path,
    entries: list[dict[str, Any]],
    skipped: list[tuple[Path, str]],
) -> None:
    lines = [
        "# Resummarize-from-raw validation report",
        "",
        f"Results root: `{results_root}`",
        "",
        "Extended summaries regenerated from cached raw JSONs only — no games "
        "were re-run and Ray was never initialised.",
        "",
        "## Metric computability from cache",
        "",
        "Game-derived extended metrics, computable from every raw cache "
        "generation (all needed per-game fields are present in `raw_result`):",
        "",
    ]
    lines += [f"- `{m}`" for m in GAME_DERIVED_EXTENDED]
    lines += [
        "",
        "Head-derived extended metrics, computable from cache ONLY where the "
        "raw JSON stored a `head_diagnostics` block (episode-matched and "
        "best-config campaigns). Everywhere else these columns are empty and "
        "would need re-rollouts (the observation sample `_head_diag_sample` "
        "is never persisted, so recomputing them requires restoring the "
        "checkpoint and replaying games):",
        "",
    ]
    lines += [f"- `{m}`" for m in HEAD_DERIVED_EXTENDED]
    lines += ["", "## Directories processed", ""]
    header = (
        "| Source dir (relative) | Rows | Head diag rows | Validation | "
        "Shared cols checked |"
    )
    lines += [header, "|---|---|---|---|---|"]
    for entry in entries:
        validation = entry["validation"]
        lines.append(
            "| `{rel}` | {rows} | {head} | {status} | {ncols} |".format(
                rel=entry["relative"],
                rows=entry["info"]["n_rows"],
                head=entry["info"]["n_with_head_diagnostics"],
                status=validation["status"],
                ncols=len(validation["shared_columns"]),
            )
        )
    mismatch_entries = [
        e for e in entries if e["validation"].get("mismatches")
    ]
    if mismatch_entries:
        lines += ["", "## Validation mismatches", ""]
        for entry in mismatch_entries:
            lines.append(f"### `{entry['relative']}`")
            for mm in entry["validation"]["mismatches"][:20]:
                lines.append(
                    f"- row {mm['row']} {mm['id']} `{mm['column']}`: "
                    f"committed={mm['committed']} regenerated={mm['regenerated']}"
                )
    missing_entries = [
        e for e in entries if e["info"]["missing_raw_keys"]
    ]
    lines += ["", "## Raw-cache field gaps", ""]
    if missing_entries:
        for entry in missing_entries:
            lines.append(f"### `{entry['relative']}`")
            for metric, files in entry["info"]["missing_raw_keys"].items():
                lines.append(
                    f"- `{metric}` missing source fields in {len(files)} "
                    f"raw file(s), e.g. `{files[0]}`"
                )
    else:
        lines.append(
            "None — every raw file in every processed directory contained all "
            "fields needed for the game-derived extended metrics."
        )
    lines += ["", "## Directories skipped", ""]
    if skipped:
        for path, reason in skipped:
            rel = path.relative_to(results_root)
            lines.append(f"- `{rel}`: {reason}")
    else:
        lines.append("None.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate extended scripted-diagnostics summaries from raw caches"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESEARCH_ROOT / "eval" / "results",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSHOPS_ROOT / "results",
    )
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    output_root = args.output_root.resolve()

    processed_dirs, skipped = find_scripted_raw_dirs(results_root)
    entries: list[dict[str, Any]] = []
    for source_dir in processed_dirs:
        relative = source_dir.relative_to(results_root)
        rows, info = resummarize_dir(source_dir)
        validation = validate_against_committed(source_dir, rows)
        write_extended_outputs(output_root / relative, rows)
        entries.append(
            {"relative": str(relative), "info": info, "validation": validation}
        )
        print(
            f"{relative}: {info['n_rows']} rows, "
            f"validation {validation['status']}"
        )

    write_report(output_root / "RESUMMARIZE_REPORT.md", results_root, entries, skipped)
    n_fail = sum(
        1 for e in entries if not e["validation"]["status"].startswith("PASS")
    )
    print(
        f"processed {len(entries)} directories "
        f"({sum(e['info']['n_rows'] for e in entries)} rows), "
        f"{n_fail} validation failure(s); skipped {len(skipped)} raw dirs"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
