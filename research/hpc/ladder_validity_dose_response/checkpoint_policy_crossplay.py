"""Learned-policy vs learned-policy cross-play harness.

The missing experiment for the paper: self-play-trained checkpoints
evaluated against *independently trained* peer policies (other seeds, or
IPPO vs MAPPO), not just against scripted bots.

Lineup modes
------------
- ``1v3``: one focal policy seat vs three seats of a single opponent
  policy (or scripted bot).  The focal seat rotates cyclically so each
  policy occupies each physical seat equally often.
- ``mixed``: four arbitrary per-seat specs (learned and/or scripted),
  rotated cyclically for exact per-seat exposure balance.

Conventions mirrored from ``eval/checkpoint_scripted_diagnostics.py`` and
``eval/deep_eval.py`` (research repo):

- Fixed ``--game-seed-base``: game ``g`` always plays with seed
  ``base + g``, so every pair/checkpoint sees identical initial states.
- ``seed_everything(game_seed_base)`` is called before each job so the
  stochastic action-sampling stream is identical across jobs.
- Stochastic sampling uses ``module._forward_exploration`` (training
  temperature); ``--deterministic`` switches to argmax via
  ``module._forward_inference``.
- Observations are encoded with the rollout env's own config and the
  module's declared observation contract (``include_survival_features``,
  ``include_trade_history``).
- Raw per-job JSON cache under ``<output>/raw/`` with metadata
  validation; re-running resumes from cache, ``--overwrite`` replaces.
- ``crossplay_matrix.csv`` is rebuilt from every raw file in the output
  dir at the end of each run, so a matrix can be accumulated across
  invocations.

Spec grammar
------------
- ``policy:LABEL=/path/to/checkpoint`` — RLlib checkpoint, shared policy.
- ``policy:LABEL=/path/to/checkpoint@MODULE_ID`` — explicit RLlib module
  id (e.g. ``pi_0`` for the v3 independent-policy runs).
- ``scripted:v1`` .. ``scripted:v5`` — scripted bot behind the standard
  "fast" cadence wrapper.

Example
-------
PYTHONPATH=research python workshops/eval/checkpoint_policy_crossplay.py \
    --mode 1v3 \
    --focal "policy:ippo_s0=research/eval/results/small_2seed_partial/ippo_seed0_headmlp/checkpoints/iter_1000" \
    --opponent "policy:ippo_s1=research/eval/results/small_2seed_partial/ippo_seed1_headmlp/checkpoints/iter_1000" \
    --environment-variant small --n-games 8 \
    --output workshops/results/policy_crossplay/smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_RESEARCH_ROOT = Path(__file__).resolve().parents[2] / "research"
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from agents.cadenced_scripted import CadencedScripted  # noqa: E402
from agents.scripted_v1 import ScriptedV1Agent  # noqa: E402
from agents.scripted_v2 import ScriptedV2Agent  # noqa: E402
from agents.scripted_v3 import ScriptedV3Agent  # noqa: E402
from agents.scripted_v4 import ScriptedV4Agent  # noqa: E402
from agents.scripted_v5 import ScriptedV5Agent  # noqa: E402
from env.env import FoodTradingEnv  # noqa: E402
from env.small_env import SCALED_ENVIRONMENTS  # noqa: E402

SCRIPTED_CLASSES = {
    "v1": ScriptedV1Agent,
    "v2": ScriptedV2Agent,
    "v3": ScriptedV3Agent,
    "v4": ScriptedV4Agent,
    "v5": ScriptedV5Agent,
}

# Same env registry as the research eval scripts: the default 180-second
# env plus the 30/60/90/120/150-second scaled variants.
ENVIRONMENT_SPECS: dict[str, tuple[type[FoodTradingEnv], int]] = {
    "default": (FoodTradingEnv, 180),
    **{
        name: (env_class, int(config.game_duration_seconds))
        for name, (config, env_class) in SCALED_ENVIRONMENTS.items()
    },
}

SUMMARY_FIELDS = (
    "focal",
    "opponent",
    "lineup_mode",
    "mode",
    "n_games",
    "n_focal_seat_games",
    "game_seed_base",
    "environment_variant",
    "environment_class",
    "game_duration_seconds",
    "survival_percentage",
    "win_percentage",
    "win_given_survival_percentage",
    "mean_terminal_cash",
    "mean_terminal_cash_given_survival",
    "mean_survival_time_ticks",
    "mean_trades_as_buyer",
    "mean_trades_as_seller",
)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeatSpec:
    """One seat's policy: a checkpoint-backed module or a scripted bot."""

    kind: str  # "policy" | "scripted"
    label: str
    checkpoint_path: Path | None = None
    module_id: str | None = None  # None → shared policy

    @property
    def key(self) -> str:
        """Identity used for caching and CSV rows."""
        if self.kind == "scripted":
            return f"scripted_{self.label}"
        return self.label

    def identity(self) -> dict[str, Any]:
        """Cache-metadata identity (resolved path so moves invalidate)."""
        if self.kind == "scripted":
            return {"kind": "scripted", "label": self.label}
        return {
            "kind": "policy",
            "label": self.label,
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "module_id": self.module_id,
        }


def parse_spec(text: str) -> SeatSpec:
    """Parse ``policy:LABEL=PATH[@MODULE]`` or ``scripted:vN``."""
    if text.startswith("scripted:"):
        version = text.removeprefix("scripted:")
        if version not in SCRIPTED_CLASSES:
            raise ValueError(
                f"unknown scripted version {version!r}; "
                f"choose from {sorted(SCRIPTED_CLASSES)}"
            )
        return SeatSpec(kind="scripted", label=version)
    if text.startswith("policy:"):
        body = text.removeprefix("policy:")
        label, sep, path_text = body.partition("=")
        if not sep or not label or not path_text:
            raise ValueError(
                f"policy spec must look like policy:LABEL=PATH, got {text!r}"
            )
        module_id = None
        if "@" in path_text:
            path_text, module_id = path_text.rsplit("@", 1)
        path = Path(path_text).expanduser()
        if not (path / "rllib_checkpoint.json").is_file():
            raise FileNotFoundError(
                f"not an RLlib checkpoint directory: {path}"
            )
        return SeatSpec(
            kind="policy",
            label=label,
            checkpoint_path=path,
            module_id=module_id,
        )
    raise ValueError(
        f"spec must start with 'policy:' or 'scripted:', got {text!r}"
    )


# ---------------------------------------------------------------------------
# Seat-balanced lineups
# ---------------------------------------------------------------------------


def rotated_lineups(n_games: int) -> list[tuple[int, int, int, int]]:
    """Cyclic seat rotations: game ``g`` gives seat ``s`` the spec at
    slot ``(s - g) % 4``.

    With four slots this places every slot in every physical seat exactly
    ``n_games / 4`` times, so both the 1v3 focal seat and the four specs
    of a mixed lineup are exactly seat-balanced.  The rotation order is
    deterministic on purpose: game ``g`` must see seed ``base + g`` with
    the same lineup in every job so pairs are compared on identical games.
    """
    if n_games <= 0 or n_games % 4:
        raise ValueError("n_games must be a positive multiple of 4")
    return [
        tuple((seat - game) % 4 for seat in range(4))  # type: ignore[misc]
        for game in range(n_games)
    ]


def lineup_slots(
    lineup_mode: str,
    focal: SeatSpec | None,
    opponent: SeatSpec | None,
    seats: Sequence[SeatSpec] | None,
) -> list[SeatSpec]:
    """The four rotation slots. Slot 0 is the focal seat in 1v3 mode."""
    if lineup_mode == "1v3":
        if focal is None or opponent is None:
            raise ValueError("1v3 mode requires --focal and --opponent")
        return [focal, opponent, opponent, opponent]
    if lineup_mode == "mixed":
        if seats is None or len(seats) != 4:
            raise ValueError("mixed mode requires exactly four --seat specs")
        return list(seats)
    raise ValueError(f"unknown lineup mode {lineup_mode!r}")


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class ScriptedSeat:
    """Scripted bot behind the standard fast cadence wrapper."""

    def __init__(self, spec: SeatSpec) -> None:
        self._factory = SCRIPTED_CLASSES[spec.label]
        self._agents: dict[int, CadencedScripted] = {}

    def begin_game(self, game_seed: int, seats: Sequence[int]) -> None:
        # Same per-seat seeding as scripted_version_tournament / deep_eval.
        self._agents = {}
        for seat in seats:
            agent = CadencedScripted(
                "fast",
                seed=game_seed * 17 + seat,
                inner_factory=self._factory,
            )
            agent.reset()
            self._agents[seat] = agent

    def act(self, env: FoodTradingEnv, obs_raw: dict) -> dict[int, int]:
        return {
            seat: agent.act(obs_raw[seat], env.legal_actions(seat))
            for seat, agent in self._agents.items()
        }


class PolicySeat:
    """Checkpoint-backed module driving one or more seats, batched.

    Encoding, recurrent-state handling and sampling mirror
    ``eval/deep_eval.py:play_games`` exactly.
    """

    def __init__(self, module, deterministic: bool) -> None:
        import torch  # deferred: scripted-only runs never need torch

        self._torch = torch
        self.module = module
        self.deterministic = deterministic
        self.include_survival = bool(
            getattr(module, "include_survival_features", False)
        )
        self.include_trades = bool(
            getattr(module, "include_trade_history", False)
        )
        # v3 Shared+ID condition: obs carry an agent one-hot (175-D).
        self.include_agent_id = bool(
            getattr(module, "include_agent_id", False)
        )
        self._init_state = module.get_initial_state()
        self._seats: list[int] = []
        self._states: dict[int, dict] = {}

    def begin_game(self, game_seed: int, seats: Sequence[int]) -> None:
        del game_seed  # policy RNG is the torch stream seeded per job
        torch = self._torch
        self._seats = list(seats)
        self._states = {
            seat: {
                k: torch.from_numpy(np.asarray(v)).float()
                for k, v in self._init_state.items()
            }
            for seat in self._seats
            if self._init_state
        }

    def act(self, env: FoodTradingEnv, obs_raw: dict) -> dict[int, int]:
        from env.rl.global_observation_encoder import (
            encode_global_observation,
        )
        from env.rl.observation_encoder import encode_observation
        from ray.rllib.core.columns import Columns

        torch = self._torch
        cfg = env.config
        enc = [
            {
                "observations": encode_observation(
                    obs_raw[seat],
                    cfg,
                    include_agent_id=self.include_agent_id,
                    include_survival_features=self.include_survival,
                ),
                "action_mask": env.legal_actions(seat).astype(np.float32),
                "critic_observations": encode_global_observation(
                    env.state,
                    seat,
                    cfg,
                    include_trade_history=self.include_trades,
                ),
            }
            for seat in self._seats
        ]
        batch_obs = {
            "observations": torch.tensor(
                np.stack([e["observations"] for e in enc]),
                dtype=torch.float32,
            ),
            "action_mask": torch.tensor(
                np.stack([e["action_mask"] for e in enc])
            ),
            "critic_observations": torch.tensor(
                np.stack([e["critic_observations"] for e in enc]),
                dtype=torch.float32,
            ),
        }
        batch = {Columns.OBS: batch_obs}
        if self._init_state:
            batch[Columns.STATE_IN] = {
                k: torch.stack([self._states[s][k] for s in self._seats])
                for k in self._init_state
            }
        with torch.no_grad():
            if self.deterministic:
                out = self.module._forward_inference(batch)
            else:
                out = self.module._forward_exploration(batch)
        if self._init_state and Columns.STATE_OUT in out:
            for pos, seat in enumerate(self._seats):
                self._states[seat] = {
                    k: v[pos].detach()
                    for k, v in out[Columns.STATE_OUT].items()
                }
        return {
            seat: int(out[Columns.ACTIONS][pos].item())
            for pos, seat in enumerate(self._seats)
        }


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors checkpoint_scripted_diagnostics)
# ---------------------------------------------------------------------------


_RAY_STARTED = False


def _ensure_ray() -> None:
    global _RAY_STARTED
    if _RAY_STARTED:
        return
    # Ray worker actors deserialize the checkpointed env creator, so they
    # must be able to import the research packages.  sys.path edits do not
    # propagate to workers — PYTHONPATH does.
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(_RESEARCH_ROOT)] + ([existing] if existing else [])
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(parts))

    from eval._warnings import silence_ray_deprecations

    silence_ray_deprecations()
    import ray
    from env.rl.rllib_config import register_nash_hunger_env

    ray.init(
        ignore_reinit_error=True,
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
    )
    register_nash_hunger_env()
    _RAY_STARTED = True


def _shutdown_ray() -> None:
    global _RAY_STARTED
    if _RAY_STARTED:
        import ray

        ray.shutdown()
        _RAY_STARTED = False


def _restore_algorithm(
    checkpoint_path: Path,
    *,
    attempts: int = 3,
    retry_delay_seconds: float = 0.5,
):
    """Restore an RLlib checkpoint, retrying interrupted filesystem calls
    (same policy as eval/checkpoint_scripted_diagnostics.py)."""
    from ray.rllib.algorithms.algorithm import Algorithm

    for attempt in range(1, attempts + 1):
        try:
            # Resolve: pyarrow rejects relative paths as scheme-less URIs.
            return Algorithm.from_checkpoint(
                str(checkpoint_path.expanduser().resolve())
            )
        except InterruptedError:
            if attempt == attempts:
                raise
            delay = retry_delay_seconds * attempt
            print(
                f"checkpoint listing interrupted; retrying "
                f"({attempt}/{attempts}) in {delay:.1f}s: {checkpoint_path}",
                flush=True,
            )
            time.sleep(delay)


class ModulePool:
    """Loads each distinct (checkpoint, module_id) once.

    Checkpoints are restored strictly one at a time and the Algorithm is
    stopped as soon as its local RLModule has been extracted — keeping
    two Algorithms alive deadlocks: each checkpoint's env-runner actors
    claim the small ``ray.init(num_cpus=2)`` budget, so a second
    ``Algorithm.from_checkpoint`` waits forever for actors that can never
    schedule.  The extracted module is a plain local torch module and
    stays fully usable after ``stop()`` (verified: identical logits).
    """

    def __init__(self) -> None:
        self._modules: dict[tuple[str, str | None], Any] = {}

    def module_for(self, spec: SeatSpec):
        from env.rl.rllib_config import SHARED_POLICY_ID

        path_key = str(spec.checkpoint_path.resolve())
        module_key = (path_key, spec.module_id)
        if module_key not in self._modules:
            _ensure_ray()
            algorithm = _restore_algorithm(spec.checkpoint_path)
            try:
                module_id = spec.module_id or SHARED_POLICY_ID
                module = algorithm.get_module(module_id)
                module.eval()
            finally:
                algorithm.stop()
            self._modules[module_key] = module
        return self._modules[module_key]

    def close(self) -> None:
        self._modules.clear()


# ---------------------------------------------------------------------------
# Game playing
# ---------------------------------------------------------------------------


def play_crossplay_games(
    slots: Sequence[SeatSpec],
    n_games: int,
    game_seed_base: int,
    *,
    deterministic: bool,
    environment_class: type[FoodTradingEnv] = FoodTradingEnv,
    module_pool: ModulePool | None = None,
) -> list[dict[str, Any]]:
    """Play seat-balanced games; return one record per game.

    ``slots`` is the four-entry rotation base from ``lineup_slots``.
    Every game ``g`` plays with seed ``game_seed_base + g`` and lineup
    ``rotated_lineups(n_games)[g]``, so results are a pure function of
    (slots, n_games, game_seed_base, mode, environment) — the property
    the raw-cache identity relies on.
    """
    lineups = rotated_lineups(n_games)

    # One runner per distinct spec key; identical specs share a runner
    # (and, for policies, a single loaded module).
    runners: dict[str, ScriptedSeat | PolicySeat] = {}
    for spec in slots:
        if spec.key in runners:
            continue
        if spec.kind == "scripted":
            runners[spec.key] = ScriptedSeat(spec)
        else:
            if module_pool is None:
                raise ValueError(
                    "policy specs require a ModulePool; scripted-only "
                    "lineups may pass module_pool=None"
                )
            runners[spec.key] = PolicySeat(
                module_pool.module_for(spec), deterministic
            )

    if any(spec.kind == "policy" for spec in slots):
        # Reset the stochastic-sampling RNG so every job sees the same
        # game seeds AND the same action-sampling stream (convention from
        # checkpoint_scripted_diagnostics).
        from env.rl.rllib_config import seed_everything

        seed_everything(game_seed_base)

    records: list[dict[str, Any]] = []
    for game_index, slot_of_seat in enumerate(lineups):
        game_seed = game_seed_base + game_index
        lineup = [slots[slot] for slot in slot_of_seat]
        env = environment_class(seed=game_seed)
        obs_raw = env.reset()

        seats_by_key: dict[str, list[int]] = {}
        for seat, spec in enumerate(lineup):
            seats_by_key.setdefault(spec.key, []).append(seat)
        # Deterministic runner order: by first seat occupied this game.
        ordered_keys = sorted(
            seats_by_key, key=lambda key: seats_by_key[key][0]
        )
        for key in ordered_keys:
            runners[key].begin_game(game_seed, seats_by_key[key])

        trades_seen = 0
        buyer_trades = [0, 0, 0, 0]
        seller_trades = [0, 0, 0, 0]
        while not env.is_done():
            actions: dict[int, int] = {}
            for key in ordered_keys:
                actions.update(runners[key].act(env, obs_raw))
            # Per-step mask-legality invariant (tests/scripted_harness).
            for seat, action in actions.items():
                if not env.legal_actions(seat)[action]:
                    raise RuntimeError(
                        f"{lineup[seat].key} returned illegal action "
                        f"{action} in game {game_index}, seat {seat}"
                    )
            obs_raw, _, _, _ = env.step(actions)
            for trade in env.state.trades[trades_seen:]:
                buyer_trades[trade.buyer_idx] += 1
                seller_trades[trade.seller_idx] += 1
            trades_seen = len(env.state.trades)

        result = env.get_result()
        winners = set(result["winner_idxs"])
        seat_records = []
        for seat, spec in enumerate(lineup):
            player = result["players"][seat]
            alive = player["status"] == "alive"
            survival_time = (
                player["died_at"]
                if player["died_at"] is not None
                else result["elapsed_seconds"]
            )
            seat_records.append(
                {
                    "seat": seat,
                    "spec": spec.key,
                    "survived": bool(alive),
                    "won": seat in winners,
                    "terminal_cash": float(player["final_cash"]),
                    "survival_time_ticks": float(survival_time),
                    "trades_as_buyer": buyer_trades[seat],
                    "trades_as_seller": seller_trades[seat],
                }
            )
        records.append(
            {
                "game_index": game_index,
                "game_seed": game_seed,
                "lineup": [spec.key for spec in lineup],
                "end_reason": result["end_reason"],
                "n_trades": result["n_trades"],
                "seats": seat_records,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _opponent_description(
    slots: Sequence[SeatSpec], focal_key: str
) -> str:
    """Compact description of everyone who is not the focal spec."""
    counts: dict[str, int] = {}
    for spec in slots:
        if spec.key != focal_key:
            counts[spec.key] = counts.get(spec.key, 0) + 1
    if not counts:
        return "self"  # uniform lineup: policy self-play baseline
    return "+".join(
        (f"{count}x{key}" if count > 1 else key)
        for key, count in sorted(counts.items())
    )


def summarize_records(
    records: Sequence[dict[str, Any]],
    slots: Sequence[SeatSpec],
    *,
    lineup_mode: str,
    mode: str,
    game_seed_base: int,
    environment_variant: str,
    environment_class: str,
    game_duration_seconds: int,
) -> list[dict[str, Any]]:
    """One summary row per distinct spec in the lineup.

    Metric names follow eval/checkpoint_scripted_diagnostics.py's summary
    so downstream tables can join the scripted-baseline results directly.
    """
    per_key: dict[str, dict[str, list[float]]] = {}
    for record in records:
        for seat_record in record["seats"]:
            values = per_key.setdefault(
                seat_record["spec"],
                {
                    "survived": [],
                    "won": [],
                    "terminal_cash": [],
                    "survival_time_ticks": [],
                    "trades_as_buyer": [],
                    "trades_as_seller": [],
                },
            )
            values["survived"].append(float(seat_record["survived"]))
            values["won"].append(float(seat_record["won"]))
            values["terminal_cash"].append(seat_record["terminal_cash"])
            values["survival_time_ticks"].append(
                seat_record["survival_time_ticks"]
            )
            values["trades_as_buyer"].append(
                float(seat_record["trades_as_buyer"])
            )
            values["trades_as_seller"].append(
                float(seat_record["trades_as_seller"])
            )

    ordered_keys = []
    for spec in slots:  # slot order: focal first in 1v3 mode
        if spec.key not in ordered_keys:
            ordered_keys.append(spec.key)

    rows = []
    for key in ordered_keys:
        values = per_key[key]
        n_seat_games = len(values["survived"])
        survivors = sum(values["survived"])
        winners = sum(values["won"])
        cash_given_survival = [
            cash
            for cash, survived in zip(
                values["terminal_cash"], values["survived"]
            )
            if survived
        ]
        rows.append(
            {
                "focal": key,
                "opponent": _opponent_description(slots, key),
                "lineup_mode": lineup_mode,
                "mode": mode,
                "n_games": len(records),
                "n_focal_seat_games": n_seat_games,
                "game_seed_base": game_seed_base,
                "environment_variant": environment_variant,
                "environment_class": environment_class,
                "game_duration_seconds": game_duration_seconds,
                "survival_percentage": 100.0 * survivors / n_seat_games,
                "win_percentage": 100.0 * winners / n_seat_games,
                "win_given_survival_percentage": (
                    100.0 * winners / survivors if survivors else 0.0
                ),
                "mean_terminal_cash": float(
                    np.mean(values["terminal_cash"])
                ),
                "mean_terminal_cash_given_survival": (
                    float(np.mean(cash_given_survival))
                    if cash_given_survival
                    else float("nan")
                ),
                "mean_survival_time_ticks": float(
                    np.mean(values["survival_time_ticks"])
                ),
                "mean_trades_as_buyer": float(
                    np.mean(values["trades_as_buyer"])
                ),
                "mean_trades_as_seller": float(
                    np.mean(values["trades_as_seller"])
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Raw cache + CSV (pattern from checkpoint_scripted_diagnostics)
# ---------------------------------------------------------------------------


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def raw_result_path(
    output_dir: Path,
    slots: Sequence[SeatSpec],
    lineup_mode: str,
    mode: str,
    n_games: int,
) -> Path:
    if lineup_mode == "1v3":
        stem = f"{slots[0].key}_vs_{slots[1].key}_1v3"
    else:
        stem = "mixed_" + "_".join(spec.key for spec in slots)
    name = _sanitize(f"{stem}_{mode}_{n_games}") + ".json"
    return output_dir / "raw" / name


def job_identity(
    slots: Sequence[SeatSpec],
    *,
    lineup_mode: str,
    mode: str,
    n_games: int,
    game_seed_base: int,
    environment_variant: str,
    environment_class: str,
    game_duration_seconds: int,
) -> dict[str, Any]:
    return {
        "lineup_mode": lineup_mode,
        "slots": [spec.identity() for spec in slots],
        "mode": mode,
        "n_games": n_games,
        "game_seed_base": game_seed_base,
        "environment_variant": environment_variant,
        "environment_class": environment_class,
        "game_duration_seconds": game_duration_seconds,
    }


def load_cached_report(
    path: Path, identity: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the cached report if its identity matches; raise on clash."""
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if report.get("identity") != identity:
        raise ValueError(
            f"cached result metadata does not match this run: {path}; "
            "use --overwrite to replace it"
        )
    return report


def rebuild_matrix(output_dir: Path) -> list[dict[str, Any]]:
    """Rebuild crossplay_matrix.csv from every raw file in the dir.

    This is what lets the full matrix accumulate across many single-pair
    invocations while staying resumable.
    """
    rows: list[dict[str, Any]] = []
    raw_dir = output_dir / "raw"
    for raw_path in sorted(raw_dir.glob("*.json")):
        report = json.loads(raw_path.read_text())
        rows.extend(report["summary_rows"])
    rows.sort(
        key=lambda row: (
            row["lineup_mode"],
            row["mode"],
            row["focal"],
            row["opponent"],
        )
    )
    csv_path = output_dir / "crossplay_matrix.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "crossplay_matrix.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoint-vs-checkpoint cross-play harness"
    )
    parser.add_argument("--mode", dest="lineup_mode", default="1v3",
                        choices=("1v3", "mixed"))
    parser.add_argument(
        "--focal",
        help="1v3 focal seat spec (policy:LABEL=PATH[@MODULE])",
    )
    parser.add_argument(
        "--opponent",
        help="1v3 opponent spec (policy:... or scripted:vN), 3 seats",
    )
    parser.add_argument(
        "--seat",
        action="append",
        dest="seats",
        help="mixed-mode per-seat spec; pass exactly four times",
    )
    parser.add_argument("--n-games", type=int, default=8,
                        help="multiple of 4 (seat balance)")
    parser.add_argument("--game-seed-base", type=int, default=900_000)
    parser.add_argument(
        "--environment-variant",
        default="default",
        choices=sorted(ENVIRONMENT_SPECS),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="argmax actions instead of stochastic training sampling",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a matching raw JSON instead of resuming from it",
    )
    parser.add_argument("--name", default="run")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "output dir (default "
            "workshops/results/policy_crossplay/<name>)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    focal = parse_spec(args.focal) if args.focal else None
    opponent = parse_spec(args.opponent) if args.opponent else None
    seats = (
        [parse_spec(text) for text in args.seats] if args.seats else None
    )
    slots = lineup_slots(args.lineup_mode, focal, opponent, seats)
    if args.lineup_mode == "1v3" and focal.kind != "policy":
        raise ValueError("the 1v3 focal seat must be a policy spec")

    environment_class, expected_duration = ENVIRONMENT_SPECS[
        args.environment_variant
    ]
    probe = environment_class(seed=0)
    game_duration_seconds = int(probe.config.game_duration_seconds)
    if game_duration_seconds != expected_duration:
        raise ValueError(
            f"{environment_class.__name__} advertises "
            f"{game_duration_seconds}s; registry expected "
            f"{expected_duration}s"
        )

    output_dir = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parents[1]
        / "results"
        / "policy_crossplay"
        / args.name
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)

    mode = "deterministic" if args.deterministic else "stochastic"
    identity = job_identity(
        slots,
        lineup_mode=args.lineup_mode,
        mode=mode,
        n_games=args.n_games,
        game_seed_base=args.game_seed_base,
        environment_variant=args.environment_variant,
        environment_class=environment_class.__name__,
        game_duration_seconds=game_duration_seconds,
    )
    raw_path = raw_result_path(
        output_dir, slots, args.lineup_mode, mode, args.n_games
    )

    cached = None if args.overwrite else load_cached_report(
        raw_path, identity
    )
    if cached is not None:
        print(f"cached: {raw_path.name}")
        rows = rebuild_matrix(output_dir)
        print(
            f"matrix: {len(rows)} rows in "
            f"{output_dir / 'crossplay_matrix.csv'}"
        )
        return 0

    lineup_text = (
        f"{slots[0].key} vs 3x {slots[1].key}"
        if args.lineup_mode == "1v3"
        else " | ".join(spec.key for spec in slots)
    )
    print(
        f"playing {args.n_games} games ({mode}, "
        f"{args.environment_variant}, seed base "
        f"{args.game_seed_base}): {lineup_text}",
        flush=True,
    )

    pool = ModulePool()
    started = time.time()
    try:
        records = play_crossplay_games(
            slots,
            args.n_games,
            args.game_seed_base,
            deterministic=args.deterministic,
            environment_class=environment_class,
            module_pool=pool,
        )
    finally:
        pool.close()
        _shutdown_ray()
    elapsed = time.time() - started

    summary_rows = summarize_records(
        records,
        slots,
        lineup_mode=args.lineup_mode,
        mode=mode,
        game_seed_base=args.game_seed_base,
        environment_variant=args.environment_variant,
        environment_class=environment_class.__name__,
        game_duration_seconds=game_duration_seconds,
    )
    report = {
        "identity": identity,
        "elapsed_seconds": elapsed,
        "seconds_per_game": elapsed / args.n_games,
        "summary_rows": summary_rows,
        "games": records,
    }
    raw_path.write_text(json.dumps(report, indent=2) + "\n")

    rows = rebuild_matrix(output_dir)
    for row in summary_rows:
        print(
            f"  {row['focal']:>24} vs {row['opponent']:<24} "
            f"survival={row['survival_percentage']:5.1f}%  "
            f"win={row['win_percentage']:5.1f}%  "
            f"cash={row['mean_terminal_cash']:8.2f}"
        )
    print(
        f"{args.n_games} games in {elapsed:.1f}s "
        f"({elapsed / args.n_games:.2f}s/game); matrix now {len(rows)} "
        f"rows in {output_dir / 'crossplay_matrix.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
