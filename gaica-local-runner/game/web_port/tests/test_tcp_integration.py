from __future__ import annotations

import json
import socket
import threading
import time
import unittest

from game.bot_tcp_server import BotTCPServer
from game.coordinator import MatchCoordinator
from game.models import LevelData, TileDraw, Vec2, WeaponType
from game.simulation import GameSimulation


class TestTCPIntegration(unittest.TestCase):
    def test_two_bots_complete_single_round(self) -> None:
        floor_tiles = []
        for y in range(0, 192, 64):
            for x in range(0, 320, 64):
                floor_tiles.append(
                    TileDraw(
                        x=x,
                        y=y,
                        tile_id=7,
                        src_x=0,
                        src_y=0,
                        layer="Floor",
                        size=64,
                    )
                )

        level = LevelData(
            identifier="TcpTest",
            width=320,
            height=192,
            floor_tiles=floor_tiles,
            top_tiles=[],
            small_tiles=[],
            player_spawns=[Vec2(80.0, 96.0), Vec2(160.0, 96.0)],
            weapon_spawns=[(Vec2(80.0, 96.0), WeaponType.UZI)],
            box_spawns=[],
            obstacles=[],
            breakables=[],
            letterboxes=[],
        )

        simulation = GameSimulation(level, seed=11)
        coordinator = MatchCoordinator(simulation)
        tcp_server = BotTCPServer("127.0.0.1", 0, coordinator)

        try:
            coordinator.start()
            tcp_server.start()

            round_end_result: dict | None = None
            observed_round_start: dict | None = None

            def bot_logic(bot_name: str, aggressive: bool) -> None:
                nonlocal round_end_result, observed_round_start
                sock = socket.create_connection((tcp_server.actual_host, tcp_server.actual_port), timeout=5.0)
                reader = sock.makefile("rb")
                writer = sock.makefile("wb")
                writer.write((json.dumps({"type": "register", "name": bot_name, "protocol": "standalone"}) + "\n").encode("utf-8"))
                writer.flush()

                seq = 0
                try:
                    start = time.time()
                    while time.time() - start < 8.0:
                        raw = reader.readline()
                        if not raw:
                            break
                        msg = json.loads(raw.decode("utf-8"))

                        if msg.get("type") == "round_start":
                            if observed_round_start is None:
                                observed_round_start = msg
                            continue

                        if msg.get("type") == "round_end":
                            round_end_result = msg.get("result")
                            break

                        if msg.get("type") != "tick":
                            continue

                        you = msg.get("you")
                        enemy = msg.get("enemy")
                        if not you:
                            continue

                        cmd = {
                            "type": "command",
                            "seq": seq,
                            "move": [0.0, 0.0],
                            "aim": [1.0, 0.0],
                            "shoot": False,
                            "kick": False,
                        }

                        if aggressive:
                            if enemy:
                                dx = enemy["position"][0] - you["position"][0]
                                dy = enemy["position"][1] - you["position"][1]
                                mag = (dx * dx + dy * dy) ** 0.5
                                if mag > 1e-8:
                                    cmd["aim"] = [dx / mag, dy / mag]

                        if not you.get("weapon"):
                            cmd["pickup"] = seq >= 20
                            cmd["move"] = [1.0, 0.0]
                        else:
                            cmd["shoot"] = True
                            cmd["move"] = [0.4, 0.0]

                        writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
                        writer.flush()
                        seq += 1
                finally:
                    try:
                        writer.close()
                        reader.close()
                        sock.close()
                    except OSError:
                        pass

            t1 = threading.Thread(target=bot_logic, args=("bot1", True), daemon=True)
            t2 = threading.Thread(target=bot_logic, args=("bot2", False), daemon=True)

            t1.start()
            t2.start()
            t1.join(timeout=10.0)
            t2.join(timeout=10.0)

            self.assertIsNotNone(round_end_result)
            self.assertIsNotNone(observed_round_start)
            self.assertIn(round_end_result["winner_id"], {1, 2})
            self.assertEqual(round_end_result["reason"], "elimination")
            assert observed_round_start is not None
            self.assertIn("floor", observed_round_start["level"])
            self.assertNotIn("floor_tiles", observed_round_start["level"])
            self.assertIn("round_index", observed_round_start["series"])
        finally:
            tcp_server.stop()
            coordinator.stop()

    def test_spawn_assignment_is_random_per_round_not_connection_order(self) -> None:
        floor_tiles = []
        for y in range(0, 192, 64):
            for x in range(0, 320, 64):
                floor_tiles.append(
                    TileDraw(
                        x=x,
                        y=y,
                        tile_id=7,
                        src_x=0,
                        src_y=0,
                        layer="Floor",
                        size=64,
                    )
                )

        level = LevelData(
            identifier="SpawnRandomnessTest",
            width=320,
            height=192,
            floor_tiles=floor_tiles,
            top_tiles=[],
            small_tiles=[],
            player_spawns=[Vec2(80.0, 96.0), Vec2(240.0, 96.0)],
            weapon_spawns=[],
            box_spawns=[],
            obstacles=[],
            breakables=[],
            letterboxes=[],
        )

        simulation = GameSimulation(level, seed=42, round_time_limit_seconds=0.22)
        coordinator = MatchCoordinator(
            simulation,
            auto_restart_delay_seconds=0.05,
            spawn_assignment_seed=1,
        )
        tcp_server = BotTCPServer("127.0.0.1", 0, coordinator)

        done = threading.Event()
        lock = threading.Lock()
        round_starts: dict[str, list[int]] = {"alpha": [], "beta": []}

        try:
            coordinator.start()
            tcp_server.start()

            def bot_logic(bot_name: str) -> None:
                sock = socket.create_connection((tcp_server.actual_host, tcp_server.actual_port), timeout=5.0)
                sock.settimeout(0.5)
                reader = sock.makefile("rb")
                writer = sock.makefile("wb")
                writer.write((json.dumps({"type": "register", "name": bot_name, "protocol": "standalone"}) + "\n").encode("utf-8"))
                writer.flush()

                seq = 0
                try:
                    start = time.time()
                    while time.time() - start < 8.0 and not done.is_set():
                        try:
                            raw = reader.readline()
                        except TimeoutError:
                            continue
                        if not raw:
                            break
                        msg = json.loads(raw.decode("utf-8"))
                        msg_type = msg.get("type")

                        if msg_type == "round_start":
                            pid = int(msg.get("player_id", 0))
                            with lock:
                                round_starts[bot_name].append(pid)
                                if len(round_starts["alpha"]) >= 6 and len(round_starts["beta"]) >= 6:
                                    done.set()
                            continue

                        if msg_type != "tick":
                            continue

                        cmd = {
                            "type": "command",
                            "seq": seq,
                            "move": [0.0, 0.0],
                            "aim": [1.0, 0.0],
                            "shoot": False,
                            "kick": False,
                        }
                        writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
                        writer.flush()
                        seq += 1
                finally:
                    try:
                        writer.close()
                        reader.close()
                        sock.close()
                    except OSError:
                        pass

            t1 = threading.Thread(target=bot_logic, args=("alpha",), daemon=True)
            t2 = threading.Thread(target=bot_logic, args=("beta",), daemon=True)
            t1.start()
            t2.start()

            success = done.wait(timeout=10.0)
            self.assertTrue(success, f"insufficient rounds observed: {round_starts}")

            # Each connection should receive both spawn slots across rounds.
            self.assertEqual(set(round_starts["alpha"]), {1, 2})
            self.assertEqual(set(round_starts["beta"]), {1, 2})

            snapshot = coordinator.get_snapshot()
            roles = snapshot.get("color_roles", [])
            self.assertEqual(len(roles), 2)
            self.assertEqual({r.get("color") for r in roles}, {"red", "green"})
        finally:
            tcp_server.stop()
            coordinator.stop()

    def test_auto_restart_sends_next_round_start_without_reconnect(self) -> None:
        floor_tiles = []
        for y in range(0, 192, 64):
            for x in range(0, 320, 64):
                floor_tiles.append(
                    TileDraw(
                        x=x,
                        y=y,
                        tile_id=7,
                        src_x=0,
                        src_y=0,
                        layer="Floor",
                        size=64,
                    )
                )

        level = LevelData(
            identifier="TcpRestartTest",
            width=320,
            height=192,
            floor_tiles=floor_tiles,
            top_tiles=[],
            small_tiles=[],
            player_spawns=[Vec2(80.0, 96.0), Vec2(160.0, 96.0)],
            weapon_spawns=[(Vec2(80.0, 96.0), WeaponType.UZI)],
            box_spawns=[],
            obstacles=[],
            breakables=[],
            letterboxes=[],
        )

        simulation = GameSimulation(level, seed=21, round_time_limit_seconds=8.0)
        coordinator = MatchCoordinator(simulation, auto_restart_delay_seconds=0.2)
        tcp_server = BotTCPServer("127.0.0.1", 0, coordinator)

        success_event = threading.Event()
        bot_stats: dict[str, dict[str, int]] = {
            "bot1": {"round_start": 0, "round_end": 0},
            "bot2": {"round_start": 0, "round_end": 0},
        }
        stats_lock = threading.Lock()

        try:
            coordinator.start()
            tcp_server.start()

            def bot_logic(bot_name: str) -> None:
                sock = socket.create_connection((tcp_server.actual_host, tcp_server.actual_port), timeout=5.0)
                reader = sock.makefile("rb")
                writer = sock.makefile("wb")
                writer.write((json.dumps({"type": "register", "name": bot_name, "protocol": "standalone"}) + "\n").encode("utf-8"))
                writer.flush()

                seq = 0
                try:
                    start = time.time()
                    while time.time() - start < 10.0 and not success_event.is_set():
                        raw = reader.readline()
                        if not raw:
                            break
                        msg = json.loads(raw.decode("utf-8"))
                        msg_type = msg.get("type")

                        if msg_type == "round_start":
                            with stats_lock:
                                bot_stats[bot_name]["round_start"] += 1
                            continue

                        if msg_type == "round_end":
                            with stats_lock:
                                bot_stats[bot_name]["round_end"] += 1
                                if (
                                    bot_stats[bot_name]["round_end"] >= 1
                                    and bot_stats[bot_name]["round_start"] >= 2
                                ):
                                    success_event.set()
                            continue

                        if msg_type != "tick":
                            continue

                        you = msg.get("you")
                        enemy = msg.get("enemy")
                        if not you:
                            continue

                        cmd = {
                            "type": "command",
                            "seq": seq,
                            "move": [0.0, 0.0],
                            "aim": [1.0, 0.0],
                            "shoot": False,
                            "kick": False,
                        }

                        if enemy:
                            dx = enemy["position"][0] - you["position"][0]
                            dy = enemy["position"][1] - you["position"][1]
                            mag = (dx * dx + dy * dy) ** 0.5
                            if mag > 1e-8:
                                cmd["aim"] = [dx / mag, dy / mag]

                        if not you.get("weapon"):
                            cmd["pickup"] = True
                            cmd["move"] = [1.0, 0.0]
                        else:
                            cmd["shoot"] = True

                        writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
                        writer.flush()
                        seq += 1
                finally:
                    try:
                        writer.close()
                        reader.close()
                        sock.close()
                    except OSError:
                        pass

            t1 = threading.Thread(target=bot_logic, args=("bot1",), daemon=True)
            t2 = threading.Thread(target=bot_logic, args=("bot2",), daemon=True)
            t1.start()
            t2.start()

            success = success_event.wait(timeout=12.0)
            self.assertTrue(success, f"auto restart not observed, stats={bot_stats}")
        finally:
            tcp_server.stop()
            coordinator.stop()

    def test_auto_restart_rotates_levels_with_factory(self) -> None:
        floor_tiles = []
        for y in range(0, 192, 64):
            for x in range(0, 320, 64):
                floor_tiles.append(
                    TileDraw(
                        x=x,
                        y=y,
                        tile_id=7,
                        src_x=0,
                        src_y=0,
                        layer="Floor",
                        size=64,
                    )
                )

        level_a = LevelData(
            identifier="RotateA",
            width=320,
            height=192,
            floor_tiles=floor_tiles,
            top_tiles=[],
            small_tiles=[],
            player_spawns=[Vec2(80.0, 96.0), Vec2(160.0, 96.0)],
            weapon_spawns=[(Vec2(80.0, 96.0), WeaponType.UZI)],
            box_spawns=[],
            obstacles=[],
            breakables=[],
            letterboxes=[],
        )
        level_b = LevelData(
            identifier="RotateB",
            width=320,
            height=192,
            floor_tiles=floor_tiles,
            top_tiles=[],
            small_tiles=[],
            player_spawns=[Vec2(80.0, 96.0), Vec2(160.0, 96.0)],
            weapon_spawns=[(Vec2(80.0, 96.0), WeaponType.UZI)],
            box_spawns=[],
            obstacles=[],
            breakables=[],
            letterboxes=[],
        )

        factory_counter = {"value": 0}

        def simulation_factory() -> GameSimulation:
            idx = factory_counter["value"]
            factory_counter["value"] = idx + 1
            level = level_a if (idx % 2 == 0) else level_b
            return GameSimulation(level, seed=100 + idx, round_time_limit_seconds=6.0)

        simulation = simulation_factory()
        coordinator = MatchCoordinator(
            simulation,
            auto_restart_delay_seconds=0.2,
            simulation_factory=simulation_factory,
        )
        tcp_server = BotTCPServer("127.0.0.1", 0, coordinator)

        done = threading.Event()
        bot1_round_levels: list[str] = []
        lock = threading.Lock()

        try:
            coordinator.start()
            tcp_server.start()

            def bot_logic(track_levels: bool) -> None:
                sock = socket.create_connection((tcp_server.actual_host, tcp_server.actual_port), timeout=5.0)
                reader = sock.makefile("rb")
                writer = sock.makefile("wb")
                writer.write((json.dumps({"type": "register", "name": "tracked" if track_levels else "observer", "protocol": "standalone"}) + "\n").encode("utf-8"))
                writer.flush()
                seq = 0
                try:
                    start = time.time()
                    while time.time() - start < 12.0 and not done.is_set():
                        raw = reader.readline()
                        if not raw:
                            break
                        msg = json.loads(raw.decode("utf-8"))
                        msg_type = msg.get("type")

                        if msg_type == "round_start":
                            if track_levels:
                                level = (msg.get("level") or {}).get("identifier")
                                if level:
                                    with lock:
                                        bot1_round_levels.append(level)
                                        if len(set(bot1_round_levels)) >= 2:
                                            done.set()
                            continue

                        if msg_type != "tick":
                            continue

                        you = msg.get("you")
                        enemy = msg.get("enemy")
                        if not you:
                            continue

                        cmd = {
                            "type": "command",
                            "seq": seq,
                            "move": [0.0, 0.0],
                            "aim": [1.0, 0.0],
                            "shoot": False,
                            "kick": False,
                        }

                        if enemy:
                            dx = enemy["position"][0] - you["position"][0]
                            dy = enemy["position"][1] - you["position"][1]
                            mag = (dx * dx + dy * dy) ** 0.5
                            if mag > 1e-8:
                                cmd["aim"] = [dx / mag, dy / mag]

                        if not you.get("weapon"):
                            cmd["pickup"] = True
                            cmd["move"] = [1.0, 0.0]
                        else:
                            cmd["shoot"] = True

                        writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
                        writer.flush()
                        seq += 1
                finally:
                    try:
                        writer.close()
                        reader.close()
                        sock.close()
                    except OSError:
                        pass

            t1 = threading.Thread(target=bot_logic, args=(True,), daemon=True)
            t2 = threading.Thread(target=bot_logic, args=(False,), daemon=True)
            t1.start()
            t2.start()

            success = done.wait(timeout=12.0)
            self.assertTrue(success, f"round levels observed={bot1_round_levels}")
        finally:
            tcp_server.stop()
            coordinator.stop()


if __name__ == "__main__":
    unittest.main()
