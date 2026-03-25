from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import config
from .models import PlayerCommand
from .simulation import GameSimulation


@dataclass
class BotEndpoint:
    player_id: int
    writer: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    connected: bool = True
    registered: bool = False
    hello_sent: bool = False

    def send(self, payload: dict[str, Any]) -> bool:
        raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with self.lock:
            try:
                self.writer.write(raw)
                self.writer.flush()
            except OSError:
                self.connected = False
                return False
        return True

    def close(self) -> None:
        with self.lock:
            try:
                self.writer.close()
            except OSError:
                pass
        self.connected = False


class MatchCoordinator:
    def __init__(
        self,
        simulation: GameSimulation,
        auto_restart_delay_seconds: float | None = None,
        simulation_factory: Callable[[], GameSimulation] | None = None,
        spawn_assignment_seed: int | None = None,
        series_total_rounds: int = 1,
    ) -> None:
        self.simulation = simulation
        self.auto_restart_delay_seconds = (
            None if auto_restart_delay_seconds is None else max(0.0, float(auto_restart_delay_seconds))
        )
        self.simulation_factory = simulation_factory
        self.series_total_rounds = max(1, int(series_total_rounds))

        self._lock = threading.RLock()
        self._commands: dict[int, PlayerCommand] = {1: PlayerCommand(), 2: PlayerCommand()}
        self._bots: dict[int, BotEndpoint] = {}
        self._bot_to_player: dict[int, int] = {}
        self._bot_names: dict[int, str] = {}
        self._rng = random.Random(spawn_assignment_seed)

        self._stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None

        self._round_started = False
        self._round_announced = False
        self._match_completed = False
        self._restart_at_monotonic: float | None = None
        self._use_initial_simulation_once = True

        self._series_rounds_completed = 0
        self._series_scores: dict[int, int] = {1: 0, 2: 0}
        self._series_final_result: dict[str, Any] | None = None
        self._match_characters: dict[int, str] = {}

    @staticmethod
    def _int_or(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _float_or(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _floor_layer_payload(cls, level_payload: dict[str, Any]) -> dict[str, Any]:
        raw_tiles = level_payload.get("floor_tiles") or []
        width = max(0.0, cls._float_or(level_payload.get("width"), 0.0))
        height = max(0.0, cls._float_or(level_payload.get("height"), 0.0))
        if not raw_tiles:
            return {"grid_size": 64, "cells": []}

        first_size = max(1, cls._int_or(raw_tiles[0].get("size"), 64))
        cells: set[tuple[int, int]] = set()
        for tile in raw_tiles:
            tile_x = cls._int_or(tile.get("x"), 0)
            tile_y = cls._int_or(tile.get("y"), 0)
            if tile_x < 0 or tile_y < 0 or tile_x >= width or tile_y >= height:
                continue
            cells.add((tile_x // first_size, tile_y // first_size))

        return {
            "grid_size": first_size,
            "cells": [[cell_x, cell_y] for cell_x, cell_y in sorted(cells, key=lambda item: (item[1], item[0]))],
        }

    @classmethod
    def _build_bot_level_payload(cls, snapshot: dict[str, Any]) -> dict[str, Any]:
        level_payload = snapshot.get("level") or {}
        static_obstacles = []
        for obstacle in snapshot.get("obstacles", []):
            if not isinstance(obstacle, dict):
                continue
            static_obstacles.append(
                {
                    "id": cls._int_or(obstacle.get("id"), 0),
                    "kind": str(obstacle.get("kind") or ""),
                    "center": list(obstacle.get("center") or [0.0, 0.0]),
                    "half_size": list(obstacle.get("half_size") or [0.0, 0.0]),
                    "solid": bool(obstacle.get("solid", True)),
                }
            )

        return {
            "identifier": str(level_payload.get("identifier") or ""),
            "width": cls._float_or(level_payload.get("width"), 0.0),
            "height": cls._float_or(level_payload.get("height"), 0.0),
            "floor": cls._floor_layer_payload(level_payload),
            "player_spawns": [list(item) for item in (level_payload.get("player_spawns") or [])],
            "static_obstacles": static_obstacles,
        }

    @classmethod
    def _build_bot_snapshot(cls, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": str(snapshot.get("status") or ""),
            "tick": cls._int_or(snapshot.get("tick"), 0),
            "time_seconds": cls._float_or(snapshot.get("time_seconds"), 0.0),
            "time_limit_seconds": cls._float_or(snapshot.get("time_limit_seconds"), 0.0),
            "result": None if snapshot.get("result") is None else dict(snapshot.get("result") or {}),
            "players": [dict(player) for player in (snapshot.get("players") or []) if isinstance(player, dict)],
            "pickups": [dict(item) for item in (snapshot.get("pickups") or []) if isinstance(item, dict)],
            "projectiles": [dict(item) for item in (snapshot.get("projectiles") or []) if isinstance(item, dict)],
            "obstacles": [dict(item) for item in (snapshot.get("obstacles") or []) if isinstance(item, dict)],
            "breakables": [dict(item) for item in (snapshot.get("breakables") or []) if isinstance(item, dict)],
            "letterboxes": [dict(item) for item in (snapshot.get("letterboxes") or []) if isinstance(item, dict)],
        }

    def _bot_series_payload(self) -> dict[str, Any]:
        current_round = self._series_rounds_completed + (1 if self._round_started else 0)
        if self.series_total_rounds <= 1:
            current_round = 1 if (self._round_started or self._series_rounds_completed > 0) else 0
        current_round = max(0, min(self.series_total_rounds, current_round))
        if self.series_total_rounds <= 1:
            current_round = max(1, current_round)
        return {
            "enabled": self.series_total_rounds > 1,
            "round_index": current_round,
            "total_rounds": self.series_total_rounds,
            "completed_rounds": self._series_rounds_completed,
            "score": {
                "1": self._series_scores.get(1, 0),
                "2": self._series_scores.get(2, 0),
            },
        }

    def _prepare_match_characters(self) -> None:
        self._match_characters = self.simulation.sample_match_characters(self._rng)

    def _start_round(self) -> None:
        if self._match_completed:
            self._series_rounds_completed = 0
            self._series_scores = {1: 0, 2: 0}
            self._series_final_result = None

        if self._match_completed or not self._match_characters:
            self._prepare_match_characters()

        if self._use_initial_simulation_once:
            self._use_initial_simulation_once = False
            self.simulation.reset_round()
        elif self.simulation_factory is not None:
            self.simulation = self.simulation_factory()
        else:
            self.simulation.reset_round()

        self.simulation.set_match_characters(self._match_characters)

        self._commands = {1: PlayerCommand(), 2: PlayerCommand()}
        self._assign_random_spawn_mapping()
        self._round_started = True
        self._round_announced = False
        self._match_completed = False
        self._restart_at_monotonic = None

    def _series_snapshot(self) -> dict[str, Any]:
        current_round = self._series_rounds_completed + (1 if self._round_started else 0)
        if self.series_total_rounds <= 1:
            current_round = 1 if (self._round_started or self._series_rounds_completed > 0) else 0
        current_round = max(0, min(self.series_total_rounds, current_round))
        return {
            "enabled": self.series_total_rounds > 1,
            "round": current_round,
            "total_rounds": self.series_total_rounds,
            "completed_rounds": self._series_rounds_completed,
            "score": {
                "1": self._series_scores.get(1, 0),
                "2": self._series_scores.get(2, 0),
            },
            "final_result": self._series_final_result,
        }

    def _finalize_round_result(self, round_result: dict[str, Any] | None, level_identifier: str) -> dict[str, Any] | None:
        if round_result is None:
            return None

        result_payload = dict(round_result)
        winner_id = result_payload.get("winner_id")
        if winner_id in {1, 2}:
            self._series_scores[int(winner_id)] = self._series_scores.get(int(winner_id), 0) + 1

        self._series_rounds_completed += 1

        if self.series_total_rounds <= 1:
            return result_payload

        result_payload["series_round"] = self._series_rounds_completed
        result_payload["series_total_rounds"] = self.series_total_rounds
        result_payload["series_score"] = {
            "1": self._series_scores.get(1, 0),
            "2": self._series_scores.get(2, 0),
        }
        result_payload["series_finished"] = self._series_rounds_completed >= self.series_total_rounds
        result_payload["level_identifier"] = level_identifier

        if result_payload["series_finished"]:
            self._series_final_result = dict(result_payload)
        return result_payload

    def _assign_random_spawn_mapping(self) -> None:
        bot_ids = list(self._bots.keys())
        if len(bot_ids) != 2:
            self._bot_to_player = {}
            return

        self._rng.shuffle(bot_ids)
        self._bot_to_player = {
            bot_ids[0]: 1,
            bot_ids[1]: 2,
        }

    def start(self) -> None:
        if self._loop_thread is not None:
            return

        self._loop_thread = threading.Thread(target=self._run_loop, name="game-loop", daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)

        with self._lock:
            for bot in self._bots.values():
                bot.close()
            self._bots.clear()
            self._bot_names.clear()
            self._bot_to_player.clear()

    def connect_bot(self, writer: Any) -> int | None:
        with self._lock:
            if 1 not in self._bots:
                bot_id = 1
            elif 2 not in self._bots:
                bot_id = 2
            else:
                return None

            endpoint = BotEndpoint(player_id=bot_id, writer=writer)
            self._bots[bot_id] = endpoint
            self._bot_names.setdefault(bot_id, f"bot_{bot_id}")
        return bot_id

    def disconnect_bot(self, bot_id: int) -> None:
        with self._lock:
            endpoint = self._bots.pop(bot_id, None)
            if endpoint is not None:
                endpoint.connected = False
            self._restart_at_monotonic = None

            if self._round_started and not self.simulation.is_finished():
                player_id = self._bot_to_player.get(bot_id)
                player = self.simulation.players.get(player_id) if player_id is not None else None
                if player is not None:
                    player.alive = False
            self._bot_to_player.pop(bot_id, None)
            self._bot_names.pop(bot_id, None)

    def register_bot(self, bot_id: int, payload: dict[str, Any]) -> None:
        with self._lock:
            endpoint = self._bots.get(bot_id)
            if endpoint is None:
                return

            raw_name = payload.get("name")
            if isinstance(raw_name, str):
                name = raw_name.strip()
                if name:
                    self._bot_names[bot_id] = name[:40]

            endpoint.registered = True
            if endpoint.hello_sent:
                return

            ok = endpoint.send(
                {
                    "type": "hello",
                    "player_id": bot_id,
                    "tick_rate": config.TICK_RATE,
                }
            )
            endpoint.hello_sent = ok
            if not ok:
                self.disconnect_bot(bot_id)

    def update_command(self, bot_id: int, payload: dict[str, Any]) -> None:
        cmd = PlayerCommand.from_payload(payload)
        with self._lock:
            player_id = self._bot_to_player.get(bot_id)
            if player_id is None:
                return
            self._commands[player_id] = cmd

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self.simulation.get_snapshot()
            if self._series_final_result is not None and not self._round_started:
                snapshot["result"] = self._series_final_result
            all_registered = len(self._bots) == 2 and all(endpoint.registered for endpoint in self._bots.values())
            if not self._match_completed and (len(self._bots) < 2 or not all_registered) and not self._round_started:
                snapshot["status"] = "waiting_for_bots"
            snapshot["bots_connected"] = sorted(self._bots.keys())
            snapshot["bot_player_map"] = [
                {
                    "bot_id": bot_id,
                    "bot_name": self._bot_names.get(bot_id, f"bot_{bot_id}"),
                    "player_id": player_id,
                }
                for bot_id, player_id in sorted(self._bot_to_player.items())
            ]
            player_to_bot = {player_id: bot_id for bot_id, player_id in self._bot_to_player.items()}
            snapshot["color_roles"] = [
                {
                    "color": "red",
                    "player_id": 1,
                    "bot_id": player_to_bot.get(1),
                    "bot_name": self._bot_names.get(player_to_bot.get(1, -1), None),
                },
                {
                    "color": "green",
                    "player_id": 2,
                    "bot_id": player_to_bot.get(2),
                    "bot_name": self._bot_names.get(player_to_bot.get(2, -1), None),
                },
            ]
            snapshot["series"] = self._series_snapshot()
            return snapshot

    def _send_round_start(self) -> None:
        snapshot = self.simulation.get_snapshot()
        level = self._build_bot_level_payload(snapshot)
        series = self._bot_series_payload()
        disconnected: list[int] = []
        for bot_id, endpoint in self._bots.items():
            player_id = self._bot_to_player.get(bot_id)
            if player_id is None:
                disconnected.append(bot_id)
                continue
            enemy_id = 1 if player_id == 2 else 2
            ok = endpoint.send(
                {
                    "type": "round_start",
                    "player_id": player_id,
                    "enemy_id": enemy_id,
                    "tick_rate": config.TICK_RATE,
                    "level": level,
                    "series": series,
                }
            )
            if not ok:
                disconnected.append(bot_id)

        for bot_id in disconnected:
            self.disconnect_bot(bot_id)

    def _send_tick(self) -> None:
        snapshot = self.simulation.get_snapshot()
        bot_snapshot = self._build_bot_snapshot(snapshot)
        players_by_id = {p["id"]: p for p in bot_snapshot["players"]}
        disconnected: list[int] = []
        for bot_id, endpoint in self._bots.items():
            player_id = self._bot_to_player.get(bot_id)
            if player_id is None:
                disconnected.append(bot_id)
                continue
            enemy_id = 1 if player_id == 2 else 2
            ok = endpoint.send(
                {
                    "type": "tick",
                    "tick": snapshot["tick"],
                    "time_seconds": snapshot["time_seconds"],
                    "you": players_by_id.get(player_id),
                    "enemy": players_by_id.get(enemy_id),
                    "snapshot": bot_snapshot,
                }
            )
            if not ok:
                disconnected.append(bot_id)

        for bot_id in disconnected:
            self.disconnect_bot(bot_id)

    def _send_round_end(self, result_payload: dict[str, Any] | None = None) -> None:
        snapshot = self.simulation.get_snapshot()
        result = result_payload if result_payload is not None else snapshot.get("result")
        disconnected: list[int] = []
        for bot_id, endpoint in self._bots.items():
            ok = endpoint.send({"type": "round_end", "result": result})
            if not ok:
                disconnected.append(bot_id)

        for bot_id in disconnected:
            self.disconnect_bot(bot_id)

    def _run_loop(self) -> None:
        tick_interval = 1.0 / config.TICK_RATE

        while not self._stop_event.is_set():
            started_at = time.perf_counter()

            with self._lock:
                connected = len(self._bots) == 2
                registered = connected and all(endpoint.registered for endpoint in self._bots.values())

                if registered and not self._round_started:
                    now = time.perf_counter()
                    can_start = True
                    if self._restart_at_monotonic is not None and now < self._restart_at_monotonic:
                        can_start = False
                    elif self._match_completed and self.auto_restart_delay_seconds is None:
                        can_start = False

                    if can_start:
                        self._start_round()

                if registered and self._round_started:
                    if not self._round_announced:
                        self._send_round_start()
                        self._round_announced = True

                    self.simulation.step(dict(self._commands), dt=config.TICK_DT)
                    self._send_tick()

                    if self.simulation.is_finished():
                        round_snapshot = self.simulation.get_snapshot()
                        round_result = self._finalize_round_result(
                            round_snapshot.get("result"),
                            (round_snapshot.get("level") or {}).get("identifier", ""),
                        )
                        self._send_round_end(round_result)
                        self._round_started = False

                        # Continue next level within the same multi-level series.
                        has_more_series_rounds = self._series_rounds_completed < self.series_total_rounds
                        if has_more_series_rounds:
                            self._match_completed = False
                            if self.auto_restart_delay_seconds is not None:
                                self._restart_at_monotonic = time.perf_counter() + self.auto_restart_delay_seconds
                            else:
                                self._restart_at_monotonic = time.perf_counter()
                        else:
                            self._match_completed = True
                            if self.auto_restart_delay_seconds is not None:
                                self._restart_at_monotonic = time.perf_counter() + self.auto_restart_delay_seconds
                            else:
                                # Keep final state and stop stepping until manual restart/reconnect.
                                self._restart_at_monotonic = None

            elapsed = time.perf_counter() - started_at
            sleep_for = tick_interval - elapsed
            if sleep_for > 0.0:
                time.sleep(sleep_for)
