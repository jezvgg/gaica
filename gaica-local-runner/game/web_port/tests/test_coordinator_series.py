from __future__ import annotations

import json
import threading
import time
import types
import unittest

from game.coordinator import MatchCoordinator


class _MemoryWriter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: list[dict] = []

    def write(self, raw: bytes) -> None:
        text = raw.decode("utf-8")
        with self._lock:
            for line in text.splitlines():
                if line.strip():
                    self._messages.append(json.loads(line))

    def flush(self) -> None:
        return

    def close(self) -> None:
        return

    def messages(self) -> list[dict]:
        with self._lock:
            return list(self._messages)


class _FakeRoundSimulation:
    def __init__(self, level_identifier: str, winner_id: int | None) -> None:
        self.level = types.SimpleNamespace(identifier=level_identifier)
        self.players = {
            1: types.SimpleNamespace(alive=True),
            2: types.SimpleNamespace(alive=True),
        }
        self._winner_id = winner_id
        self.match_characters = {1: "grapefruit", 2: "lime"}
        self.reset_round()

    def reset_round(self) -> None:
        self.status = "running"
        self.tick = 0
        self.time_seconds = 0.0
        self.result = None
        self.players[1].alive = True
        self.players[2].alive = True

    def step(self, _commands: dict, dt: float) -> None:
        if self.result is not None:
            return
        self.tick += 1
        self.time_seconds += dt
        if self.tick >= 2:
            self.status = "finished"
            self.result = types.SimpleNamespace(
                winner_id=self._winner_id,
                reason="elimination",
                duration_seconds=self.time_seconds,
            )

    def is_finished(self) -> bool:
        return self.result is not None

    @staticmethod
    def sample_match_characters(_rng) -> dict[int, str]:
        return {1: "orange", 2: "lemon"}

    def set_match_characters(self, characters: dict[int, str]) -> None:
        self.match_characters = {
            1: str(characters.get(1, "grapefruit")),
            2: str(characters.get(2, "lime")),
        }

    def get_snapshot(self) -> dict:
        return {
            "status": self.status,
            "tick": self.tick,
            "time_seconds": self.time_seconds,
            "time_limit_seconds": 99.0,
            "result": (
                {
                    "winner_id": self.result.winner_id,
                    "reason": self.result.reason,
                    "duration_seconds": self.result.duration_seconds,
                }
                if self.result is not None
                else None
            ),
            "level": {
                "identifier": self.level.identifier,
                "width": 320,
                "height": 192,
                "floor_tiles": [],
                "top_tiles": [],
                "small_tiles": [],
                "player_spawns": [],
            },
            "players": [
                {
                    "id": 1,
                    "position": [80.0, 96.0],
                    "facing": [1.0, 0.0],
                    "alive": self.players[1].alive,
                    "weapon": None,
                    "shoot_cooldown": 0.0,
                    "kick_cooldown": 0.0,
                    "stun_remaining": 0.0,
                    "color": "#ef4444",
                    "character": self.match_characters[1],
                },
                {
                    "id": 2,
                    "position": [160.0, 96.0],
                    "facing": [-1.0, 0.0],
                    "alive": self.players[2].alive,
                    "weapon": None,
                    "shoot_cooldown": 0.0,
                    "kick_cooldown": 0.0,
                    "stun_remaining": 0.0,
                    "color": "#22c55e",
                    "character": self.match_characters[2],
                },
            ],
            "pickups": [],
            "projectiles": [],
            "obstacles": [],
            "breakables": [],
            "effects": [],
            "debris": [],
            "letterboxes": [],
        }


class TestCoordinatorSeries(unittest.TestCase):
    def test_connect_does_not_send_hello_until_register(self) -> None:
        coordinator = MatchCoordinator(_FakeRoundSimulation("L0", winner_id=1))
        writer = _MemoryWriter()
        try:
            bot_id = coordinator.connect_bot(writer)
            self.assertEqual(bot_id, 1)
            self.assertEqual(writer.messages(), [])

            coordinator.register_bot(bot_id, {"type": "register", "name": "alpha", "protocol": "standalone"})
            messages = writer.messages()
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["type"], "hello")
            self.assertEqual(messages[0]["player_id"], 1)
        finally:
            coordinator.stop()

    def test_match_characters_stay_stable_across_series_rounds(self) -> None:
        rounds = [
            _FakeRoundSimulation("L0", winner_id=1),
            _FakeRoundSimulation("L1", winner_id=2),
        ]
        counter = {"idx": 1}

        def factory() -> _FakeRoundSimulation:
            idx = counter["idx"]
            counter["idx"] = idx + 1
            return rounds[idx]

        coordinator = MatchCoordinator(
            rounds[0],
            simulation_factory=factory,
            series_total_rounds=2,
        )

        coordinator._start_round()
        first_chars = dict(coordinator.simulation.match_characters)
        coordinator._series_rounds_completed = 1
        coordinator._start_round()
        second_chars = dict(coordinator.simulation.match_characters)

        self.assertEqual(first_chars, second_chars)

    def test_series_winner_is_decided_by_aggregate_score(self) -> None:
        rounds = [
            _FakeRoundSimulation("L0", winner_id=1),
            _FakeRoundSimulation("L1", winner_id=2),
            _FakeRoundSimulation("L2", winner_id=1),
        ]
        counter = {"idx": 1}

        def factory() -> _FakeRoundSimulation:
            idx = counter["idx"]
            counter["idx"] = idx + 1
            return rounds[idx]

        coordinator = MatchCoordinator(
            rounds[0],
            simulation_factory=factory,
            series_total_rounds=3,
        )
        writer_a = _MemoryWriter()
        writer_b = _MemoryWriter()

        try:
            bot_a = coordinator.connect_bot(writer_a)
            bot_b = coordinator.connect_bot(writer_b)
            self.assertEqual(bot_a, 1)
            self.assertEqual(bot_b, 2)
            coordinator.register_bot(bot_a, {"type": "register", "name": "alpha", "protocol": "standalone"})
            coordinator.register_bot(bot_b, {"type": "register", "name": "beta", "protocol": "standalone"})

            coordinator.start()

            deadline = time.time() + 3.0
            final_result = None
            while time.time() < deadline:
                messages = writer_a.messages()
                round_ends = [m for m in messages if m.get("type") == "round_end"]
                if round_ends and (round_ends[-1].get("result") or {}).get("series_finished"):
                    final_result = round_ends[-1].get("result")
                    break
                time.sleep(0.02)

            self.assertIsNotNone(final_result, "series final result was not produced")
            assert final_result is not None
            self.assertEqual(final_result.get("winner_id"), 1)
            self.assertEqual(final_result.get("reason"), "elimination")
            self.assertEqual((final_result.get("series_score") or {}).get("1"), 2)
            self.assertEqual((final_result.get("series_score") or {}).get("2"), 1)
            self.assertEqual(final_result.get("series_round"), 3)
            self.assertEqual(final_result.get("series_total_rounds"), 3)
            self.assertTrue(final_result.get("series_finished"))
            self.assertIn(final_result.get("level_identifier"), {"L0", "L1", "L2"})
            self.assertNotIn("last_level_result", final_result)
            self.assertNotIn("series_rounds_played", final_result)

            all_round_ends = [m.get("result") for m in writer_a.messages() if m.get("type") == "round_end"]
            self.assertGreaterEqual(len(all_round_ends), 3)
            self.assertEqual((all_round_ends[0] or {}).get("reason"), "elimination")
            self.assertEqual((all_round_ends[-1] or {}).get("reason"), "elimination")

            snapshot = coordinator.get_snapshot()
            self.assertEqual((snapshot.get("result") or {}).get("reason"), "elimination")
            self.assertTrue((snapshot.get("result") or {}).get("series_finished"))
            self.assertEqual(((snapshot.get("series") or {}).get("score") or {}).get("1"), 2)
            self.assertEqual(((snapshot.get("series") or {}).get("score") or {}).get("2"), 1)
        finally:
            coordinator.stop()


if __name__ == "__main__":
    unittest.main()
