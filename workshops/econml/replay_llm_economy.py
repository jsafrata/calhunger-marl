"""E26: deterministic replay of the LLM-arena campaign → full market data.

Replays every cached game of ``workshops/results/llm_arena/campaign_90s``
with the recorded LLM actions fed back verbatim (zero API calls), scripted
seats re-run from their per-game seeds, and policy seats re-run from the
original local checkpoints with the torch stream re-seeded per cell
(``seed_everything(game_seed_base)``) and games replayed in ascending
order — the exact discipline of ``llm_arena.arena.play_arena_games``.

The point: the arena kept only trade *counts*; the env keeps a complete
attributed ledger (``env.state.trades``) plus live order books and player
balances.  Replay recovers, per game:

- every executed trade (price, qty, buyer, seller, maker/taker ids, tick),
- per-tick per-seat cash / reserved cash / inventory / status,
- per-tick best bid/ask + open depth per food,
- an order registry (order id -> owner seat, side, price, food) built by
  scanning the books each tick (orders that fill within their posting tick
  are attributed via that tick's applied action when possible).

Validation is mandatory and per game: the replay must reproduce every
recorded LLM ``env_applied`` verdict at its tick, plus terminal
``n_trades``, per-seat ``terminal_cash``, ``survived``,
``survival_time_ticks`` and buyer/seller trade counts.  Games failing any
check are written with ``validation.ok = false`` and excluded downstream;
per-cell rates go to ``validation.csv``.  Policy cells are the at-risk
group (the sampling stream restarts at the job seed only on a clean
sequential pass — resumes/aborts in the original campaign break alignment
from that game on).

Usage:
  python3 replay_llm_economy.py [--cells claude_vs_v2 ...] [--limit N]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WORKSHOPS = _HERE.parent
_RESEARCH = _WORKSHOPS.parent / "research"
for p in (str(_RESEARCH), str(_WORKSHOPS / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

import checkpoint_policy_crossplay as crossplay  # noqa: E402
from env.actions import ACTION_WAIT  # noqa: E402

CAMPAIGN = _WORKSHOPS / "results" / "llm_arena" / "campaign_90s"
OUT_ROOT = _HERE / "results" / "llm_economy"


# ---------------------------------------------------------------------------
# Seat reconstruction from cached identity
# ---------------------------------------------------------------------------


def spec_from_identity(slot: dict) -> object:
    if slot["kind"] == "scripted":
        return crossplay.SeatSpec(kind="scripted", label=slot["label"])
    if slot["kind"] == "policy":
        return crossplay.SeatSpec(
            kind="policy",
            label=slot["label"],
            checkpoint_path=Path(slot["checkpoint_path"]),
            module_id=slot.get("module_id"),
        )
    if slot["kind"] == "llm":
        # Stand-in spec: replay never queries a provider; only .key matters
        # (it must reproduce the recorded lineup keys / seat mapping).
        return _LLMStub(label=slot["label"])
    raise ValueError(f"unknown slot kind: {slot['kind']!r}")


@dataclasses.dataclass(frozen=True)
class _LLMStub:
    label: str
    kind: str = "llm"

    @property
    def key(self) -> str:
        return f"llm_{self.label}"


class RecordedSeat:
    """Feeds the recorded LLM actions back at their ticks."""

    def __init__(self) -> None:
        self._seats: list[int] = []
        self._actions: dict[tuple[int, int], dict] = {}

    def load_game(self, decisions: list[dict]) -> None:
        self._actions = {
            (d["tick"], d["seat"]): d for d in decisions
        }

    def begin_game(self, game_seed: int, seats) -> None:
        del game_seed
        self._seats = list(seats)

    def act(self, env, obs_raw) -> dict[int, int]:
        del obs_raw
        tick = env.state.elapsed_seconds
        out: dict[int, int] = {}
        for seat in self._seats:
            if env.state.players[seat].status != "alive":
                out[seat] = ACTION_WAIT  # arena: dead seats wait, no record
                continue
            rec = self._actions.get((tick, seat))
            # A missing record for an alive seat = divergence (or an
            # aborted original game); WAIT keeps the replay alive and the
            # validation layer flags the game.
            out[seat] = ACTION_WAIT if rec is None else int(rec["action_index"])
        return out


# ---------------------------------------------------------------------------
# Book / balance snapshots
# ---------------------------------------------------------------------------


def _book_top(book) -> dict:
    def top(orders):
        for o in orders:  # sorted best-first by construction
            if o.status == "open" and o.remaining_quantity > 0:
                return o
        return None

    bid, ask = top(book.bids), top(book.asks)
    depth_bid = sum(
        o.remaining_quantity for o in book.bids
        if o.status == "open" and o.remaining_quantity > 0
    )
    depth_ask = sum(
        o.remaining_quantity for o in book.asks
        if o.status == "open" and o.remaining_quantity > 0
    )
    return {
        "best_bid": bid.price_per_unit if bid else None,
        "best_ask": ask.price_per_unit if ask else None,
        "depth_bid": depth_bid,
        "depth_ask": depth_ask,
    }


def _scan_orders(state, registry: dict[int, dict]) -> None:
    for food, book in state.order_books.items():
        for side, orders in (("bid", book.bids), ("ask", book.asks)):
            for o in orders:
                if o.id not in registry:
                    registry[o.id] = {
                        "owner": o.player_idx,
                        "side": side,
                        "food": food,
                        "price": o.price_per_unit,
                        "tick": o.created_at_tick,
                    }


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------


def replay_game(
    game: dict,
    runners: dict[str, object],
    slots_by_key: dict[str, object],
    environment_class,
) -> dict:
    lineup_keys: list[str] = game["lineup"]
    env = environment_class(seed=game["game_seed"])
    obs_raw = env.reset()

    seats_by_key: dict[str, list[int]] = {}
    for seat, key in enumerate(lineup_keys):
        seats_by_key.setdefault(key, []).append(seat)
    ordered_keys = sorted(seats_by_key, key=lambda k: seats_by_key[k][0])
    for key in ordered_keys:
        runner = runners[key]
        if isinstance(runner, RecordedSeat):
            runner.load_game(game["llm_decisions"])
        runner.begin_game(game["game_seed"], seats_by_key[key])
    # The arena runs non-LLM ("sync") seats first, in first-seat order —
    # this ordering is what the torch sampling stream saw; reproduce it.
    sync_keys = [
        k for k in ordered_keys if not isinstance(runners[k], RecordedSeat)
    ]
    llm_keys = [k for k in ordered_keys if isinstance(runners[k], RecordedSeat)]

    recorded = {
        (d["tick"], d["seat"]): d for d in game["llm_decisions"]
    }
    mismatches: list[dict] = []
    trades_seen = 0
    buyer_trades = [0, 0, 0, 0]
    seller_trades = [0, 0, 0, 0]
    ticks: list[dict] = []
    order_registry: dict[int, dict] = {}
    take_actions: dict[int, list] = {}

    crashed = None
    while not env.is_done():
        tick = env.state.elapsed_seconds
        actions: dict[int, int] = {}
        for key in sync_keys:
            actions.update(runners[key].act(env, obs_raw))
        for key in llm_keys:
            actions.update(runners[key].act(env, obs_raw))
        try:
            obs_raw, _, _, info = env.step(actions)
        except Exception as exc:  # divergence can feed the env an
            crashed = f"tick {tick}: {type(exc).__name__}: {exc}"  # illegal action
            break

        # --- validation: recorded env verdicts must reproduce ----------
        for seat in range(4):
            rec = recorded.get((tick, seat))
            if rec is None:
                continue
            applied = info.actions_applied.get(seat)
            if applied != rec["env_applied"]:
                mismatches.append(
                    {
                        "tick": tick,
                        "seat": seat,
                        "recorded": rec["env_applied"],
                        "replayed": applied,
                    }
                )
        for seat, applied in info.actions_applied.items():
            if isinstance(applied, dict) and applied.get("kind") == "take":
                take_actions.setdefault(tick, []).append(seat)

        for trade in env.state.trades[trades_seen:]:
            buyer_trades[trade.buyer_idx] += 1
            seller_trades[trade.seller_idx] += 1
        trades_seen = len(env.state.trades)

        _scan_orders(env.state, order_registry)
        snap = {
            "tick": tick,
            "seats": [
                {
                    "cash": p.cash,
                    "reserved_cash": p.reserved_cash,
                    "inventory": dict(p.inventory),
                    "alive": p.status == "alive",
                }
                for p in env.state.players
            ],
            "books": {
                food: _book_top(book)
                for food, book in env.state.order_books.items()
            },
        }
        ticks.append(snap)

    result = env.get_result() if crashed is None else None

    checks: dict[str, bool] = {"no_crash": crashed is None}
    if result is not None:
        checks["n_trades"] = result["n_trades"] == game["n_trades"]
        checks["end_reason"] = result["end_reason"] == game["end_reason"]
        seat_ok = True
        for rec in game["seats"]:
            seat = rec["seat"]
            player = result["players"][seat]
            alive = player["status"] == "alive"
            survival_time = (
                player["died_at"]
                if player["died_at"] is not None
                else result["elapsed_seconds"]
            )
            seat_ok &= (
                alive == rec["survived"]
                and float(player["final_cash"]) == rec["terminal_cash"]
                and float(survival_time) == rec["survival_time_ticks"]
                and buyer_trades[seat] == rec["trades_as_buyer"]
                and seller_trades[seat] == rec["trades_as_seller"]
            )
        checks["seats"] = seat_ok
    checks["env_applied"] = not mismatches

    ok = all(checks.values())
    return {
        "game_index": game["game_index"],
        "game_seed": game["game_seed"],
        "lineup": lineup_keys,
        "validation": {
            "ok": ok,
            "checks": checks,
            "n_env_applied_mismatches": len(mismatches),
            "first_mismatches": mismatches[:3],
            "crashed": crashed,
        },
        "producers": (
            [list(p.produces) for p in env.state.players]
        ),
        "trades": [
            dataclasses.asdict(t) for t in env.state.trades
        ],
        "orders": {str(k): v for k, v in order_registry.items()},
        "takes_by_tick": {str(k): v for k, v in take_actions.items()},
        "ticks": ticks,
    }


# ---------------------------------------------------------------------------
# Campaign loop
# ---------------------------------------------------------------------------


def replay_cell(cell_dir: Path, out_dir: Path, limit: int | None) -> dict:
    game_files = sorted(cell_dir.glob("games/game_*.json"))
    if limit is not None:
        game_files = game_files[:limit]
    if not game_files:
        return {"cell": cell_dir.name, "n_games": 0}
    identity = json.loads(game_files[0].read_text())["identity"]
    variant = identity["environment_variant"]
    environment_class, _ = crossplay.ENVIRONMENT_SPECS[variant]

    slots = [spec_from_identity(s) for s in identity["slots"]]
    slots_by_key = {s.key: s for s in slots}

    has_policy = any(getattr(s, "kind", "") == "policy" for s in slots)
    module_pool = crossplay.ModulePool() if has_policy else None
    runners: dict[str, object] = {}
    for spec in slots:
        if spec.key in runners:
            continue
        if spec.kind == "scripted":
            runners[spec.key] = crossplay.ScriptedSeat(spec)
        elif spec.kind == "policy":
            runners[spec.key] = crossplay.PolicySeat(
                module_pool.module_for(spec),
                deterministic=(identity["mode"] == "deterministic"),
            )
        else:
            runners[spec.key] = RecordedSeat()

    if has_policy:
        from env.rl.rllib_config import seed_everything

        seed_everything(identity["game_seed_base"])

    out_games = out_dir / "replay"
    out_games.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    expected = identity["n_games"]
    present = {int(p.stem.split("_")[1]) for p in game_files}
    missing = [i for i in range(expected) if i not in present]
    for path in game_files:
        game = json.loads(path.read_text())["game"]
        replayed = replay_game(
            game, runners, slots_by_key, environment_class
        )
        n_ok += bool(replayed["validation"]["ok"])
        (out_games / path.name).write_text(
            json.dumps(replayed, separators=(",", ":")) + "\n"
        )
    row = {
        "cell": cell_dir.name,
        "n_games": len(game_files),
        "n_missing_originals": len(missing),
        "n_validated": n_ok,
        "validation_rate": round(100.0 * n_ok / len(game_files), 2),
        "has_policy_seats": has_policy,
    }
    print(
        f"{cell_dir.name}: {n_ok}/{len(game_files)} validated "
        f"({row['validation_rate']}%)"
        + (f", {len(missing)} originals missing" if missing else ""),
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="skip cells whose replay dir already has as many files as games",
    )
    args = parser.parse_args()

    cells = sorted(
        d for d in CAMPAIGN.iterdir()
        if (d / "games").is_dir()
        and (args.cells is None or d.name in args.cells)
    )
    rows = []
    for cell in cells:
        out_dir = OUT_ROOT / cell.name
        if args.skip_existing:
            n_src = len(list((cell / "games").glob("game_*.json")))
            n_done = len(list((out_dir / "replay").glob("game_*.json")))
            if n_src and n_done >= n_src:
                print(f"{cell.name}: already replayed, skipping", flush=True)
                continue
        rows.append(replay_cell(cell, out_dir, args.limit))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    val_path = OUT_ROOT / "validation.csv"
    header = [
        "cell", "n_games", "n_missing_originals", "n_validated",
        "validation_rate", "has_policy_seats",
    ]
    exists = val_path.is_file()
    with val_path.open("a") as fh:
        if not exists:
            fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(k, "")) for k in header) + "\n")
    print(f"validation table appended: {val_path}", flush=True)


if __name__ == "__main__":
    main()
