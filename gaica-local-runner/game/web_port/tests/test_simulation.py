from __future__ import annotations

import unittest

from game import config
from game.models import PlayerCommand, Vec2, WeaponInstance, WeaponType
from game.simulation import GameSimulation
from tests.test_helpers import (
    build_breakable_test_level,
    build_door_test_level,
    build_flat_test_level,
    build_glass_test_level,
    build_letterbox_test_level,
    build_multi_spawn_test_level,
    build_wall_pin_test_level,
)


class TestSimulation(unittest.TestCase):
    @staticmethod
    def _step_for_seconds(sim: GameSimulation, seconds: float, commands: dict[int, PlayerCommand] | None = None) -> None:
        ticks = max(1, int(seconds * config.TICK_RATE))
        for _ in range(ticks):
            sim.step(commands or {})

    def test_pickup_then_shoot_ends_round(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=1)
        sim.players[1].position = Vec2(64.0, 96.0)
        sim.players[2].position = Vec2(224.0, 96.0)

        # Wait until pickup cooldown is over.
        self._step_for_seconds(sim, config.WEAPON_PICKUP_COOLDOWN + 0.05)

        sim.step(
            {
                1: PlayerCommand(seq=1, move=Vec2(), aim=Vec2(1.0, 0.0), pickup=True),
                2: PlayerCommand(seq=1, move=Vec2(), aim=Vec2(-1.0, 0.0)),
            }
        )
        self.assertIsNotNone(sim.players[1].current_weapon)

        # Wait for initial shoot cooldown after pickup.
        self._step_for_seconds(
            sim,
            config.REVOLVER_SHOT_COOLDOWN + 0.05,
            {1: PlayerCommand(seq=2, move=Vec2(), aim=Vec2(1.0, 0.0))},
        )

        for tick in range(int(2.0 * config.TICK_RATE)):
            sim.step(
                {
                    1: PlayerCommand(seq=100 + tick, move=Vec2(), aim=Vec2(1.0, 0.0), shoot=True),
                    2: PlayerCommand(seq=100 + tick, move=Vec2(), aim=Vec2(-1.0, 0.0)),
                }
            )
            if sim.is_finished():
                break

        self.assertTrue(sim.is_finished())
        self.assertIsNotNone(sim.result)
        self.assertEqual(sim.result.winner_id, 1)

    def test_kick_causes_stun_and_drop(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=2)

        sim.players[1].position = Vec2(100.0, 96.0)
        sim.players[2].position = Vec2(120.0, 96.0)
        sim.players[1].facing = Vec2(1.0, 0.0)
        sim.players[1].kick_cooldown = 0.0

        sim.players[2].current_weapon = WeaponInstance(weapon_type=WeaponType.REVOLVER, ammo=4)

        before_pickups = len(sim.pickups)

        sim.step(
            {
                1: PlayerCommand(seq=10, move=Vec2(), aim=Vec2(1.0, 0.0), kick=True),
                2: PlayerCommand(seq=10, move=Vec2(), aim=Vec2(-1.0, 0.0)),
            }
        )

        self.assertGreater(sim.players[2].stun_remaining, 0.0)
        self.assertIsNone(sim.players[2].current_weapon)
        self.assertGreater(len(sim.pickups), before_pickups)
        effect_types = [effect.get("type") for effect in sim.effects]
        self.assertIn("kick_arc", effect_types)
        self.assertIn("kick_hit", effect_types)
        kick_hit = next(effect for effect in sim.effects if effect.get("type") == "kick_hit")
        self.assertAlmostEqual(float(kick_hit.get("travel_speed", 0.0)), config.STUN_SPEED)
        self.assertAlmostEqual(float(kick_hit.get("duration", 0.0)), 0.24)

    def test_breakable_destroyed_by_bullets(self) -> None:
        sim = GameSimulation(build_breakable_test_level(), seed=3)

        sim.players[1].position = Vec2(120.0, 96.0)
        sim.players[2].position = Vec2(280.0, 96.0)
        sim.players[1].current_weapon = WeaponInstance(weapon_type=WeaponType.REVOLVER, ammo=3)
        sim.players[1].shoot_cooldown = 0.0

        for tick in range(int(1.0 * config.TICK_RATE)):
            sim.step({1: PlayerCommand(seq=tick + 1, aim=Vec2(1.0, 0.0), shoot=True)})

        # The box should be broken by one revolver hit (breakability 1.0).
        alive_breakables = [b for b in sim.breakables.values() if b.alive]
        self.assertEqual(alive_breakables, [])
        self.assertFalse(sim.obstacles[10].solid)
        self.assertGreater(len(sim.debris), 0)
        self.assertTrue(all(d.get("type") == "box_debris" for d in sim.debris))

    def test_letterbox_has_cooldown(self) -> None:
        sim = GameSimulation(build_letterbox_test_level(), seed=4)

        sim.players[1].position = Vec2(108.0, 96.0)
        sim.players[1].facing = Vec2(1.0, 0.0)
        sim.players[1].kick_cooldown = 0.0

        before = len(sim.pickups)
        sim.step({1: PlayerCommand(seq=1, kick=True, aim=Vec2(1.0, 0.0))})
        after_first_kick = len(sim.pickups)

        # Kick again immediately; cooldown should prevent a second spawn.
        sim.players[1].kick_cooldown = 0.0
        sim.step({1: PlayerCommand(seq=2, kick=True, aim=Vec2(1.0, 0.0))})
        after_second_kick = len(sim.pickups)

        self.assertGreater(after_first_kick, before)
        self.assertEqual(after_second_kick, after_first_kick)

    def test_weapon_spawn_pickups_use_current_max_ammo(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=23)

        self.assertEqual(len(sim.pickups), 1)
        pickup = next(iter(sim.pickups.values()))
        self.assertEqual(pickup.weapon_type, WeaponType.REVOLVER)
        self.assertEqual(pickup.ammo, config.REVOLVER_MAX_AMMO)

    def test_letterbox_spawns_uzi_with_current_max_ammo(self) -> None:
        sim = GameSimulation(build_letterbox_test_level(), seed=24)

        sim.players[1].position = Vec2(108.0, 96.0)
        sim.players[1].facing = Vec2(1.0, 0.0)
        sim.players[1].kick_cooldown = 0.0

        sim.step({1: PlayerCommand(seq=1, kick=True, aim=Vec2(1.0, 0.0))})

        spawned_uzi_pickups = [pickup for pickup in sim.pickups.values() if pickup.weapon_type == WeaponType.UZI]
        self.assertEqual(len(spawned_uzi_pickups), 1)
        self.assertEqual(spawned_uzi_pickups[0].ammo, config.UZI_MAX_AMMO)

    def test_empty_weapon_pickup_is_removed(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=5)

        sim.players[1].position = Vec2(64.0, 96.0)
        sim.players[1].facing = Vec2(1.0, 0.0)
        sim.players[1].current_weapon = WeaponInstance(weapon_type=WeaponType.REVOLVER, ammo=0)

        sim.step({1: PlayerCommand(seq=1, throw=True, aim=Vec2(1.0, 0.0))})
        empty_pickup_ids = [pickup_id for pickup_id, pickup in sim.pickups.items() if pickup.ammo == 0]
        self.assertGreater(len(empty_pickup_ids), 0)

        # Empty pickup should despawn after timeout.
        self._step_for_seconds(sim, config.EMPTY_PICKUP_REMOVE_SECONDS + 0.6)

        remaining_empty = [pickup for pickup in sim.pickups.values() if pickup.ammo == 0]
        self.assertEqual(remaining_empty, [])

    def test_empty_weapon_pickup_exposes_despawn_remaining_in_snapshot(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=25)

        sim.players[1].position = Vec2(64.0, 96.0)
        sim.players[1].facing = Vec2(1.0, 0.0)
        sim.players[1].current_weapon = WeaponInstance(weapon_type=WeaponType.REVOLVER, ammo=0)

        sim.step({1: PlayerCommand(seq=1, throw=True, aim=Vec2(1.0, 0.0))})
        snapshot = sim.get_snapshot()
        empty_pickups = [pickup for pickup in snapshot["pickups"] if pickup["ammo"] == 0]

        self.assertEqual(len(empty_pickups), 1)
        self.assertIn("despawn_remaining", empty_pickups[0])
        self.assertGreater(float(empty_pickups[0]["despawn_remaining"]), 0.0)
        self.assertLessEqual(float(empty_pickups[0]["despawn_remaining"]), config.EMPTY_PICKUP_REMOVE_SECONDS)

    def test_stun_breaks_box_on_collision(self) -> None:
        sim = GameSimulation(build_breakable_test_level(), seed=6)

        sim.players[1].position = Vec2(130.0, 96.0)
        sim.players[2].position = Vec2(150.0, 96.0)
        sim.players[1].facing = Vec2(1.0, 0.0)
        sim.players[1].kick_cooldown = 0.0

        # Kick player 2 towards the box center at x=160.
        sim.step({1: PlayerCommand(seq=7, kick=True, aim=Vec2(1.0, 0.0))})

        for _ in range(int(0.35 * config.TICK_RATE)):
            sim.step({})
            if not sim.breakables[10].alive:
                break

        alive_breakables = [b for b in sim.breakables.values() if b.alive]
        self.assertEqual(alive_breakables, [])
        self.assertLessEqual(sim.players[2].stun_remaining, 1e-6)
        self.assertTrue(sim.players[2].alive)

        pos_after_hit = sim.players[2].position
        sim.step({})
        self.assertLess(sim.players[2].position.distance_to(pos_after_hit), 0.05)

    def test_shoot_creates_tracer_effect(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=9)
        sim.players[1].current_weapon = WeaponInstance(weapon_type=WeaponType.REVOLVER, ammo=3)
        sim.players[1].shoot_cooldown = 0.0
        sim.players[1].position = Vec2(64.0, 96.0)
        sim.players[2].position = Vec2(250.0, 96.0)

        sim.step({1: PlayerCommand(seq=1, aim=Vec2(1.0, 0.0), shoot=True)})
        effect_types = [effect.get("type") for effect in sim.effects]
        self.assertIn("tracer", effect_types)
        self.assertIn("muzzle", effect_types)

    def test_snapshot_exposes_player_velocity(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=9)
        sim.players[1].position = Vec2(64.0, 96.0)
        sim.players[2].position = Vec2(224.0, 96.0)

        sim.step({1: PlayerCommand(seq=1, move=Vec2(1.0, 0.0), aim=Vec2(1.0, 0.0))})
        snapshot = sim.get_snapshot()
        players_by_id = {player["id"]: player for player in snapshot["players"]}

        self.assertIn("velocity", players_by_id[1])
        self.assertEqual(players_by_id[2]["velocity"], [0.0, 0.0])
        self.assertGreater(players_by_id[1]["velocity"][0], 0.0)
        self.assertEqual(players_by_id[1]["velocity"][1], 0.0)

    def test_snapshot_level_exposes_floor_cells(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=26)

        snapshot = sim.get_snapshot()
        level_payload = snapshot["level"]

        self.assertIn("floor", level_payload)
        self.assertEqual(level_payload["floor"]["grid_size"], 64)
        self.assertGreater(len(level_payload["floor"]["cells"]), 0)
        self.assertIn([0, 0], level_payload["floor"]["cells"])

    def test_pickup_and_throw_create_effects(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=10)
        sim.players[1].position = Vec2(64.0, 96.0)
        sim.players[2].position = Vec2(224.0, 96.0)

        self._step_for_seconds(sim, config.WEAPON_PICKUP_COOLDOWN + 0.05)

        sim.step({1: PlayerCommand(seq=1, aim=Vec2(1.0, 0.0), pickup=True)})
        effect_types = [effect.get("type") for effect in sim.effects]
        self.assertIn("pickup", effect_types)

        sim.step({1: PlayerCommand(seq=2, aim=Vec2(1.0, 0.0), throw=True)})
        effect_types = [effect.get("type") for effect in sim.effects]
        self.assertIn("throw", effect_types)

    def test_explicit_drop_uses_lower_impulse_than_throw(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=19)
        sim.players[1].position = Vec2(64.0, 96.0)
        sim.players[2].position = Vec2(224.0, 96.0)
        self._step_for_seconds(sim, config.WEAPON_PICKUP_COOLDOWN + 0.05)

        sim.step({1: PlayerCommand(seq=1, aim=Vec2(1.0, 0.0), pickup=True)})
        self.assertIsNotNone(sim.players[1].current_weapon)

        sim.step({1: PlayerCommand(seq=2, aim=Vec2(1.0, 0.0), drop=True)})
        self.assertIsNone(sim.players[1].current_weapon)
        dropped_velocity = max((pickup.velocity.length() for pickup in sim.pickups.values()), default=0.0)
        self.assertGreater(dropped_velocity, 0.0)
        self.assertLess(dropped_velocity, config.WEAPON_THROW_IMPULSE / 160.0)

        self._step_for_seconds(sim, config.WEAPON_PICKUP_COOLDOWN + 0.05)
        sim.step({1: PlayerCommand(seq=3, aim=Vec2(1.0, 0.0), pickup=True)})
        self.assertIsNotNone(sim.players[1].current_weapon)
        sim.step({1: PlayerCommand(seq=4, aim=Vec2(1.0, 0.0), throw=True)})
        thrown_velocity = max((pickup.velocity.length() for pickup in sim.pickups.values()), default=0.0)
        self.assertGreater(thrown_velocity, dropped_velocity)

    def test_match_characters_are_reflected_in_snapshot(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=20)

        sim.set_match_characters({1: "orange", 2: "grapefruit"})
        snapshot = sim.get_snapshot()
        players_by_id = {player["id"]: player for player in snapshot["players"]}

        self.assertEqual(players_by_id[1]["character"], "orange")
        self.assertEqual(players_by_id[2]["character"], "grapefruit")
        self.assertEqual(players_by_id[1]["color"], "#fe930a")
        self.assertEqual(players_by_id[2]["color"], "#fe3232")

    def test_match_characters_are_forced_distinct(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=21)

        sim.set_match_characters({1: "lime", 2: "lime"})
        snapshot = sim.get_snapshot()
        players_by_id = {player["id"]: player for player in snapshot["players"]}

        self.assertEqual(players_by_id[1]["character"], "lime")
        self.assertNotEqual(players_by_id[1]["character"], players_by_id[2]["character"])

    def test_juice_stain_records_creation_time(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=22)
        sim.time_seconds = 1.25

        sim._spawn_juice_stain(Vec2(160.0, 96.0), "orange")

        stain = sim.debris[-1]
        self.assertEqual(stain["type"], "juice_stain")
        self.assertEqual(stain["character"], "orange")
        self.assertAlmostEqual(float(stain["created_at"]), 1.25)
        self.assertEqual(int(stain["created_tick"]), sim.tick)

    def test_kick_opens_door_and_creates_effect(self) -> None:
        sim = GameSimulation(build_door_test_level(), seed=11)
        sim.players[1].position = Vec2(114.0, 96.0)
        sim.players[1].facing = Vec2(1.0, 0.0)
        sim.players[1].kick_cooldown = 0.0

        sim.step({1: PlayerCommand(seq=1, kick=True, aim=Vec2(1.0, 0.0))})
        door_open = sim.door_open_timers.get(30, 0.0)
        effect_types = [effect.get("type") for effect in sim.effects]

        self.assertGreater(door_open, 0.0)
        self.assertIn("door_open", effect_types)

    def test_glass_break_turns_obstacle_non_solid(self) -> None:
        sim = GameSimulation(build_glass_test_level(), seed=12)
        sim.players[1].position = Vec2(120.0, 96.0)
        sim.players[2].position = Vec2(280.0, 96.0)
        sim.players[1].current_weapon = WeaponInstance(weapon_type=WeaponType.UZI, ammo=8)
        sim.players[1].shoot_cooldown = 0.0

        for tick in range(int(0.8 * config.TICK_RATE)):
            sim.step({1: PlayerCommand(seq=1 + tick, aim=Vec2(1.0, 0.0), shoot=True)})
        self.assertFalse(sim.obstacles[11].solid)

    def test_wall_pin_kicks_do_not_ring_out_victim(self) -> None:
        sim = GameSimulation(build_wall_pin_test_level(), seed=13)
        sim.players[1].position = Vec2(96.0, 96.0)
        sim.players[2].position = Vec2(78.0, 96.0)
        sim.players[1].facing = Vec2(-1.0, 0.0)
        sim.players[2].facing = Vec2(1.0, 0.0)
        sim.players[1].kick_cooldown = 0.0

        for tick in range(int(3.0 * config.TICK_RATE)):
            sim.step(
                {
                    1: PlayerCommand(
                        seq=tick,
                        move=Vec2(-1.0, 0.0),
                        aim=Vec2(-1.0, 0.0),
                        kick=(tick % 10 == 0),
                    ),
                    2: PlayerCommand(seq=tick, move=Vec2(0.0, 0.0), aim=Vec2(1.0, 0.0)),
                }
            )
            if tick % 10 == 0:
                sim.players[1].kick_cooldown = 0.0
            self.assertTrue(sim.players[2].alive)

        self.assertTrue(sim._is_player_grounded(sim.players[2].position))

    def test_player_movement_has_small_inertia(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=14)
        start = Vec2(sim.players[1].position.x, sim.players[1].position.y)

        sim.step({1: PlayerCommand(seq=1, move=Vec2(1.0, 0.0), aim=Vec2(1.0, 0.0))})
        first = sim.players[1].position
        moved_first = first.x - start.x
        full_step = config.PLAYER_SPEED * config.TICK_DT

        self.assertGreater(moved_first, 0.1)
        self.assertLess(moved_first, full_step * 0.95)

        sim.step({1: PlayerCommand(seq=2, move=Vec2(0.0, 0.0), aim=Vec2(1.0, 0.0))})
        second = sim.players[1].position
        moved_second = second.x - first.x

        self.assertGreater(moved_second, 0.05)
        self.assertLess(moved_second, moved_first)

    def test_move_input_magnitude_affects_speed(self) -> None:
        def run_for_mag(mag: float) -> float:
            sim = GameSimulation(build_flat_test_level(), seed=15)
            x0 = sim.players[1].position.x
            for tick in range(12):
                sim.step({1: PlayerCommand(seq=tick, move=Vec2(mag, 0.0), aim=Vec2(1.0, 0.0))})
            return sim.players[1].position.x - x0

        full = run_for_mag(1.0)
        slow = run_for_mag(0.45)

        self.assertGreater(full, 1.0)
        self.assertGreater(slow, 0.5)
        self.assertLess(slow, full * 0.72)

    def test_round_ends_in_draw_after_tick_limit(self) -> None:
        sim = GameSimulation(build_flat_test_level(), seed=18, round_time_limit_seconds=9999.0)
        for _ in range(config.ROUND_TICK_LIMIT + 5):
            sim.step({})
            if sim.is_finished():
                break

        self.assertTrue(sim.is_finished())
        self.assertIsNotNone(sim.result)
        assert sim.result is not None
        self.assertIsNone(sim.result.winner_id)
        self.assertEqual(sim.result.reason, "tick_limit")

    def test_round_spawn_points_randomize_across_available_spawns(self) -> None:
        level = build_multi_spawn_test_level()
        sim = GameSimulation(level, seed=16)

        p1_positions = set()
        p2_positions = set()
        for _ in range(14):
            sim.reset_round()
            p1_positions.add(tuple(round(v, 2) for v in sim.players[1].position.to_list()))
            p2_positions.add(tuple(round(v, 2) for v in sim.players[2].position.to_list()))

        self.assertGreaterEqual(len(p1_positions), 3)
        self.assertGreaterEqual(len(p2_positions), 3)
        self.assertFalse(p1_positions.isdisjoint(p2_positions))


if __name__ == "__main__":
    unittest.main()
