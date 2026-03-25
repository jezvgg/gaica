from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import sys
import time
import unittest
from urllib.request import urlopen

from game import config
from game.bot_tcp_server import BotTCPServer
from game.coordinator import MatchCoordinator
from game.level_loader import load_level
from game.models import PlayerCommand, Vec2
from game.simulation import GameSimulation
from game.web_server import WebServer


def _norm(v: Vec2) -> Vec2:
    return v.normalize()


def _aggressive_command(sim: GameSimulation, player_id: int, seq: int) -> PlayerCommand:
    you = sim.players[player_id]
    enemy = sim.players[1 if player_id == 2 else 2]

    if not you.alive:
        return PlayerCommand(seq=seq)

    to_enemy = enemy.position - you.position
    aim = _norm(to_enemy)

    move = Vec2(0.0, 0.0)
    pickup = False
    shoot = False
    kick = False

    if you.current_weapon is None:
        nearest_pickup = None
        nearest_dist = 1e9
        for pickup in sim.pickups.values():
            if pickup.cooldown > 0.0:
                continue
            dist = pickup.position.distance_to(you.position)
            if dist < nearest_dist:
                nearest_pickup = pickup
                nearest_dist = dist

        if nearest_pickup is not None:
            to_pickup = nearest_pickup.position - you.position
            move = _norm(to_pickup)
            if move.length() > 0.0:
                aim = move
            pickup = nearest_dist <= 22.0
        else:
            move = _norm(to_enemy)
    else:
        dist = to_enemy.length()
        if dist > 150.0:
            move = _norm(to_enemy)
        else:
            move = Vec2(0.0, 0.0)

        shoot = enemy.alive
        kick = dist <= 28.0 and aim.length() > 0.0

    return PlayerCommand(
        seq=seq,
        move=move,
        aim=aim if aim.length() > 0.0 else Vec2(1.0, 0.0),
        shoot=shoot,
        kick=kick,
        pickup=pickup,
    )


def _randomish_command(sim: GameSimulation, player_id: int, seq: int, rnd: random.Random) -> PlayerCommand:
    you = sim.players[player_id]
    enemy = sim.players[1 if player_id == 2 else 2]

    if not you.alive:
        return PlayerCommand(seq=seq)

    if seq % 20 == 0:
        angle = rnd.uniform(0.0, 6.283185307179586)
        _randomish_command.last_move = Vec2(rnd.uniform(-1.0, 1.0), rnd.uniform(-1.0, 1.0)).normalize()
        _randomish_command.last_aim = Vec2(
            (enemy.position.x - you.position.x) + 0.2 * rnd.uniform(-1.0, 1.0),
            (enemy.position.y - you.position.y) + 0.2 * rnd.uniform(-1.0, 1.0),
        ).normalize()

    move = getattr(_randomish_command, "last_move", Vec2(1.0, 0.0))
    aim = getattr(_randomish_command, "last_aim", Vec2(1.0, 0.0))

    pickup = False
    if you.current_weapon is None and rnd.random() < 0.22:
        pickup = True

    return PlayerCommand(
        seq=seq,
        move=move,
        aim=aim if aim.length() > 0.0 else Vec2(1.0, 0.0),
        shoot=bool(you.current_weapon) and rnd.random() < 0.75,
        kick=rnd.random() < 0.06,
        pickup=pickup,
    )


class TestRealisticE2E(unittest.TestCase):
    def test_real_level_scripted_bots_finish_round(self) -> None:
        root = Path(__file__).resolve().parents[2]
        ldtk = root / "web_port" / "assets" / "levels" / "test_ldtk_project.ldtk"

        level = load_level(ldtk, level_index=0)
        sim = GameSimulation(level, seed=101, round_time_limit_seconds=35.0)

        rnd = random.Random(33)
        for seq in range(int(35 * config.TICK_RATE + 300)):
            cmd1 = _aggressive_command(sim, 1, seq)
            cmd2 = _randomish_command(sim, 2, seq, rnd)
            sim.step({1: cmd1, 2: cmd2})
            if sim.is_finished():
                break

        self.assertTrue(sim.is_finished(), "Round did not finish on real level")
        self.assertIsNotNone(sim.result)
        self.assertIn(sim.result.reason, {"elimination", "time_limit", "tick_limit"})
        self.assertGreater(sim.tick, 0)
        self.assertGreater(sim._next_projectile_id, 1)

    def test_full_stack_with_builtin_bot_processes(self) -> None:
        root = Path(__file__).resolve().parents[2]
        ldtk = root / "web_port" / "assets" / "levels" / "test_ldtk_project.ldtk"
        assets = root / "hotline-miami-like" / "assets"
        static = root / "web_port" / "static"
        aggressive_bot = root.parent / "examples" / "test_bots" / "aggressive" / "main.py"
        defensive_bot = root.parent / "examples" / "test_bots" / "defensive" / "main.py"

        level = load_level(ldtk, level_index=0)
        simulation = GameSimulation(level, seed=202, round_time_limit_seconds=18.0)
        coordinator = MatchCoordinator(simulation)
        tcp_server = BotTCPServer("127.0.0.1", 0, coordinator)
        web_server = WebServer("127.0.0.1", 0, coordinator, static_dir=static, assets_dir=assets)

        processes: list[subprocess.Popen[bytes]] = []

        try:
            coordinator.start()
            tcp_server.start()
            web_server.start()

            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(aggressive_bot),
                        "127.0.0.1",
                        str(tcp_server.actual_port),
                    ]
                )
            )
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(defensive_bot),
                        "127.0.0.1",
                        str(tcp_server.actual_port),
                    ]
                )
            )

            final_state: dict | None = None
            max_connected = 0
            deadline = time.time() + 25.0
            while time.time() < deadline:
                with urlopen(f"http://127.0.0.1:{web_server.actual_port}/api/state", timeout=1.0) as response:
                    state = json.loads(response.read().decode("utf-8"))

                max_connected = max(max_connected, len(state.get("bots_connected", [])))

                if state.get("result") is not None:
                    final_state = state
                    break

                time.sleep(0.1)

            self.assertIsNotNone(final_state, "full stack match did not finish in time")
            assert final_state is not None
            self.assertEqual(final_state.get("status"), "finished")
            self.assertIn(final_state["result"]["reason"], {"elimination", "time_limit", "tick_limit"})
            self.assertGreaterEqual(max_connected, 2)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()

            web_server.stop()
            tcp_server.stop()
            coordinator.stop()


if __name__ == "__main__":
    unittest.main()
