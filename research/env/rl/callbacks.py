"""RLlib callbacks for true game-result metrics + per-episode reward-channel
and invalid-action bookkeeping. All metrics are logged as episode means so
they show up in the standard `env_runners/…` and `game/…` panels."""
from __future__ import annotations

from typing import Any

from ray.rllib.callbacks.callbacks import RLlibCallback


class NashHungerCallbacks(RLlibCallback):
    """Log per-seat game outcomes AND separate dense vs. terminal reward
    channels + invalid-action count per episode."""

    _AGENT_IDS = ("seat_0", "seat_1", "seat_2", "seat_3")

    @staticmethod
    def _all_infos_for_agent(episode: Any, agent_id: str) -> list[dict]:
        """Return every info dict this agent produced this episode, in order.

        In Ray 2.53 new-API-stack, `episode.get_infos()` returns a *dict*
        keyed by agent id, where each value is a list of step-info dicts
        for that agent. Older APIs returned a list-of-dicts across all
        steps. We handle both, plus the very legacy `last_info_for`
        fallback that only exposed the terminal info.
        """
        try:
            infos = episode.get_infos()
        except (AttributeError, TypeError):
            infos = None
        # New API: dict[agent_id -> list[info]]
        if isinstance(infos, dict) and agent_id in infos:
            per_agent = infos[agent_id]
            if isinstance(per_agent, list):
                # Drop the initial empty {} that get_infos() emits before
                # the first step so aggregation doesn't blow up on it.
                return [i for i in per_agent if isinstance(i, dict) and i]
            if isinstance(per_agent, dict):
                return [per_agent] if per_agent else []
        # Old API: list[dict[agent_id -> info]]
        if isinstance(infos, list):
            return [
                s[agent_id] for s in infos
                if isinstance(s, dict) and agent_id in s and s[agent_id]
            ]
        # Legacy last_info_for fallback.
        try:
            info = episode.last_info_for(agent_id) or {}
            return [info] if info else []
        except AttributeError:
            return []

    @classmethod
    def _terminal_metrics(cls, episode: Any) -> dict | None:
        """The env attaches an `episode_metrics` payload on the terminal
        step of any active agent. Search a few agents in case one seat
        died earlier and doesn't get a terminal info dict."""
        for agent_id in cls._AGENT_IDS:
            infos = cls._all_infos_for_agent(episode, agent_id)
            if not infos:
                continue
            terminal = infos[-1]
            metrics = terminal.get("episode_metrics") if isinstance(terminal, dict) else None
            if metrics:
                return metrics
        return None

    def on_episode_end(
        self,
        *,
        episode,
        metrics_logger=None,
        **kwargs,
    ) -> None:
        # ---- reward-channel + invariant aggregation (always emit) ----
        # These are the training-invariants the checklist needs — they come
        # from per-step infos and do NOT depend on whether the terminal
        # `episode_metrics` payload was attached (which isn't guaranteed on
        # multi-agent episodes ending by death of the last active seat).
        dense_by_seat: list[float] = []
        terminal_by_seat: list[float] = []
        rejected_by_seat: list[int] = []
        rejections_by_code: dict[str, int] = {}
        per_seat_step_counts: list[int] = []
        for agent_id in self._AGENT_IDS:
            infos = self._all_infos_for_agent(episode, agent_id)
            per_seat_step_counts.append(len(infos))
            dense_by_seat.append(sum(float(i.get("reward_dense", 0.0)) for i in infos))
            terminal_by_seat.append(
                sum(float(i.get("reward_terminal", 0.0)) for i in infos)
            )
            rejected_by_seat.append(
                sum(1 for i in infos if i.get("rejected_action"))
            )
            for i in infos:
                code = i.get("rejection_code")
                if code:
                    rejections_by_code[code] = rejections_by_code.get(code, 0) + 1

        n = max(1, sum(1 for c in per_seat_step_counts if c > 0))
        values: dict[str, float] = {
            "reward/dense_per_seat": sum(dense_by_seat) / n,
            "reward/terminal_per_seat": sum(terminal_by_seat) / n,
            "reward/total_per_seat": (
                (sum(dense_by_seat) + sum(terminal_by_seat)) / n
            ),
            "reward/dense_share": (
                (sum(dense_by_seat) / (sum(dense_by_seat) + sum(terminal_by_seat)))
                if (sum(dense_by_seat) + sum(terminal_by_seat)) > 0
                else 0.0
            ),
            # Renamed from "invalid_actions_per_seat" for accuracy — these
            # are actions the env couldn't apply, but many are legal-under-
            # the-mask rejections at the simultaneous-clearing stage rather
            # than mask violations. The per-code breakdown below tells
            # them apart. Kept as an aggregate metric so old dashboards
            # keep reading a value.
            "action_rejections/per_seat_total": sum(rejected_by_seat) / n,
            "action_rejections/no_order_to_cancel_per_seat": (
                rejections_by_code.get("no_order_to_cancel", 0) / n
            ),
            "action_rejections/insufficient_cash_per_seat": (
                rejections_by_code.get("insufficient_cash", 0) / n
            ),
            "action_rejections/insufficient_food_per_seat": (
                rejections_by_code.get("insufficient_food", 0) / n
            ),
            "action_rejections/player_dead_per_seat": (
                rejections_by_code.get("player_dead", 0) / n
            ),
            "action_rejections/other_per_seat": (
                (sum(rejected_by_seat) - sum(rejections_by_code.get(k, 0)
                    for k in ("no_order_to_cancel", "insufficient_cash",
                              "insufficient_food", "player_dead"))) / n
            ),
        }

        # ---- game / seat outcome metrics (only if the payload is present) ----
        # `episode_metrics` gets attached in the multi-agent env on the
        # `game_done` step for active seats. On multi-agent episodes ending
        # via last-survivor-death, the RLlib callback may not see it — in
        # that case we skip these fields rather than fabricate zeros.
        metrics = self._terminal_metrics(episode)
        if metrics:
            values["game/episode_length"] = float(metrics["elapsed_seconds"])
            values["game/trades"] = float(metrics["n_trades"])
            values["game/reached_horizon"] = float(
                metrics.get("reached_horizon", False)
            )
            seat_survival = []
            seat_survived_to_180 = []
            seat_died_before_180 = []
            terminal_rewards = []
            for idx, player in enumerate(metrics["players"]):
                values[f"seat_{idx}/win"] = float(player["win"])
                values[f"seat_{idx}/survived"] = float(player["survived"])
                # v2 explicit fields — never confused with the archive
                # metric of "alive whenever the game ended".
                values[f"seat_{idx}/survived_to_180"] = float(
                    player.get("survived_to_180", player["survived"])
                )
                values[f"seat_{idx}/died_before_180"] = float(
                    player.get("died_before_180", not player["survived"])
                )
                values[f"seat_{idx}/final_cash"] = float(player["final_cash"])
                values[f"seat_{idx}/final_total_inventory"] = float(
                    player.get("final_total_inventory", 0)
                )
                values[f"seat_{idx}/death_time"] = float(player["death_time"])
                values[f"seat_{idx}/terminal_rank"] = float(
                    player.get("terminal_rank", 0)
                )
                values[f"seat_{idx}/terminal_reward"] = float(
                    player.get("terminal_reward", 0.0)
                )
                seat_survival.append(float(player["survived"]))
                seat_survived_to_180.append(
                    float(player.get("survived_to_180", player["survived"]))
                )
                seat_died_before_180.append(
                    float(player.get("died_before_180", not player["survived"]))
                )
                terminal_rewards.append(
                    float(player.get("terminal_reward", 0.0))
                )
            values["game/n_survivors"] = float(sum(seat_survival))
            values["game/n_survived_to_180"] = float(sum(seat_survived_to_180))
            values["game/n_died_before_180"] = float(sum(seat_died_before_180))
            values["game/terminal_reward_per_seat"] = (
                sum(terminal_rewards) / max(1, len(terminal_rewards))
            )
            values["game/reached_time_limit"] = float(
                metrics.get("end_reason") == "time_limit"
            )
            values["game/no_survivors"] = float(sum(seat_survival) == 0)
        else:
            # Even without episode_metrics we can compute game/n_survivors
            # from the seat step-counts: a seat that took FEWER steps than
            # the max died earlier. Give the training-side monitor a signal
            # to use.
            max_steps = max(per_seat_step_counts) if per_seat_step_counts else 0
            n_survivors = sum(1 for c in per_seat_step_counts if c == max_steps)
            values["game/n_survivors"] = float(n_survivors)
            values["game/episode_length"] = float(max_steps)

        if metrics_logger is not None:
            for key, value in values.items():
                metrics_logger.log_value(key, value, reduce="mean")
        else:
            episode.custom_metrics.update(values)
