from __future__ import annotations

from dataclasses import asdict
import math
import random
from typing import Any

from . import config
from .models import (
    BreakableState,
    LevelData,
    ObstacleRect,
    PickupWeapon,
    PlayerCommand,
    PlayerState,
    Projectile,
    RoundResult,
    Vec2,
    WeaponInstance,
    WeaponType,
    WEAPON_STATS,
)
from .physics import (
    circle_intersects_rect,
    kick_target_in_front,
    ray_segment_aabb_intersection,
    ray_segment_circle_intersection,
    resolve_circle_world,
)


class GameSimulation:
    """Single-round 2-player simulation closely ported from the Rust game parameters."""

    MATCH_CHARACTERS = ("orange", "lime", "grapefruit", "lemon")
    CHARACTER_COLORS = {
        "orange": "#fe930a",
        "lime": "#28fe0b",
        "grapefruit": "#fe3232",
        "lemon": "#fcfe05",
    }

    def __init__(
        self,
        level: LevelData,
        seed: int | None = None,
        round_time_limit_seconds: float | None = None,
    ) -> None:
        self.level = level
        self.random = random.Random(seed)
        self.round_time_limit_seconds = (
            config.ROUND_TIME_LIMIT_SECONDS
            if round_time_limit_seconds is None
            else float(round_time_limit_seconds)
        )

        self.tick: int = 0
        self.time_seconds: float = 0.0

        self.players: dict[int, PlayerState] = {}
        self.pickups: dict[int, PickupWeapon] = {}
        self.projectiles: dict[int, Projectile] = {}
        self.obstacles: dict[int, ObstacleRect] = {}
        self.breakables: dict[int, BreakableState] = {}
        self.letterbox_cooldowns: dict[int, float] = {}
        self.door_open_timers: dict[int, float] = {}
        self.effects: list[dict[str, Any]] = []
        self.debris: list[dict[str, Any]] = []

        self.result: RoundResult | None = None
        self.status: str = "waiting"

        self._next_pickup_id = 1
        self._next_projectile_id = 1
        self._next_effect_id = 1
        self._floor_cells = {(tile.x // int(config.TILE_SIZE), tile.y // int(config.TILE_SIZE)) for tile in self.level.floor_tiles}
        self.match_characters: dict[int, str] = {
            1: "grapefruit",
            2: "lime",
        }

        self._static_level_payload = self._build_static_level_payload()
        self.reset_round()

    @classmethod
    def sample_match_characters(cls, rng: random.Random) -> dict[int, str]:
        variants = list(cls.MATCH_CHARACTERS)
        rng.shuffle(variants)
        return {
            1: variants[0],
            2: variants[1],
        }

    @classmethod
    def normalize_match_characters(cls, characters: dict[int, str] | None) -> dict[int, str]:
        requested = characters or {}
        assigned: dict[int, str] = {}
        used: set[str] = set()
        defaults = {
            1: "grapefruit",
            2: "lime",
        }

        for player_id in (1, 2):
            candidate = cls.normalize_character(requested.get(player_id), fallback=defaults[player_id])
            if candidate not in used:
                assigned[player_id] = candidate
                used.add(candidate)
                continue

            for fallback in cls.MATCH_CHARACTERS:
                if fallback in used:
                    continue
                assigned[player_id] = fallback
                used.add(fallback)
                break

        return assigned

    @classmethod
    def normalize_character(cls, raw: str | None, fallback: str = "lemon") -> str:
        candidate = str(raw or fallback).strip().lower()
        if candidate in cls.CHARACTER_COLORS:
            return candidate
        return fallback

    @classmethod
    def character_color(cls, character: str) -> str:
        normalized = cls.normalize_character(character)
        return cls.CHARACTER_COLORS[normalized]

    def _build_static_level_payload(self) -> dict[str, Any]:
        floor_tiles = [asdict(tile) for tile in self.level.floor_tiles]
        if floor_tiles:
            grid_size = max(1, int(floor_tiles[0].get("size", config.TILE_SIZE)))
            floor_cells = sorted(
                {
                    (int(tile.get("x", 0)) // grid_size, int(tile.get("y", 0)) // grid_size)
                    for tile in floor_tiles
                    if 0 <= int(tile.get("x", 0)) < self.level.width and 0 <= int(tile.get("y", 0)) < self.level.height
                },
                key=lambda item: (item[1], item[0]),
            )
        else:
            grid_size = int(config.TILE_SIZE)
            floor_cells = []

        return {
            "identifier": self.level.identifier,
            "width": self.level.width,
            "height": self.level.height,
            "floor": {
                "grid_size": grid_size,
                "cells": [[cell_x, cell_y] for cell_x, cell_y in floor_cells],
            },
            "floor_tiles": floor_tiles,
            "top_tiles": [asdict(tile) for tile in self.level.top_tiles],
            "small_tiles": [asdict(tile) for tile in self.level.small_tiles],
            "player_spawns": [spawn.to_list() for spawn in self.level.player_spawns],
        }

    def reset_round(self) -> None:
        self.tick = 0
        self.time_seconds = 0.0
        self.result = None
        self.status = "running"
        self._next_pickup_id = 1
        self._next_projectile_id = 1
        self._next_effect_id = 1

        spawn_indices = list(range(len(self.level.player_spawns)))
        self.random.shuffle(spawn_indices)
        spawn_1 = self.level.player_spawns[spawn_indices[0]]
        spawn_2 = self.level.player_spawns[spawn_indices[1]]

        self.players = {
            1: PlayerState(
                player_id=1,
                position=Vec2(spawn_1.x, spawn_1.y),
                facing=Vec2(1.0, 0.0),
                color="#ef4444",
                kick_cooldown=config.PLAYER_KICK_COOLDOWN,
            ),
            2: PlayerState(
                player_id=2,
                position=Vec2(spawn_2.x, spawn_2.y),
                facing=Vec2(-1.0, 0.0),
                color="#22c55e",
                kick_cooldown=config.PLAYER_KICK_COOLDOWN,
            ),
        }
        self._apply_match_characters()

        self.pickups = {}
        self.projectiles = {}
        self.effects = []
        self.debris = []

        self.obstacles = {
            obstacle.obstacle_id: ObstacleRect(
                obstacle_id=obstacle.obstacle_id,
                kind=obstacle.kind,
                center=Vec2(obstacle.center.x, obstacle.center.y),
                half_size=Vec2(obstacle.half_size.x, obstacle.half_size.y),
                solid=obstacle.solid,
            )
            for obstacle in self.level.obstacles
        }

        self.breakables = {
            b.breakable_id: BreakableState(
                breakable_id=b.breakable_id,
                obstacle_id=b.obstacle_id,
                variant=b.variant,
                threshold=b.threshold,
                current_value=0.0,
                rect_center=Vec2(b.rect_center.x, b.rect_center.y),
                rect_half_size=Vec2(b.rect_half_size.x, b.rect_half_size.y),
                alive=True,
            )
            for b in self.level.breakables
        }

        self.letterbox_cooldowns = {
            obstacle_id: 0.0
            for obstacle_id, obstacle in self.obstacles.items()
            if obstacle.kind == "letterbox"
        }
        self.door_open_timers = {
            obstacle_id: 0.0
            for obstacle_id, obstacle in self.obstacles.items()
            if obstacle.kind == "door"
        }

        for spawn_pos, weapon_type in self.level.weapon_spawns:
            stats = WEAPON_STATS[weapon_type]
            self._spawn_pickup(
                weapon_type=weapon_type,
                ammo=stats.max_ammo,
                position=Vec2(spawn_pos.x, spawn_pos.y),
                velocity=Vec2(0.0, 0.0),
                cooldown=config.WEAPON_PICKUP_COOLDOWN,
            )

    def _apply_match_characters(self) -> None:
        for player_id, player in self.players.items():
            character = self.normalize_character(self.match_characters.get(player_id), fallback="lemon")
            player.character = character
            player.color = self.character_color(character)

    def set_match_characters(self, characters: dict[int, str]) -> None:
        self.match_characters = self.normalize_match_characters(characters)
        self._apply_match_characters()

    def is_finished(self) -> bool:
        return self.result is not None

    def _add_effect(self, effect_type: str, ttl: float, **payload: Any) -> None:
        self.effects.append(
            {
                "id": self._next_effect_id,
                "type": effect_type,
                "ttl": ttl,
                **payload,
            }
        )
        self._next_effect_id += 1

    def _tick_effects(self, dt: float) -> None:
        next_effects: list[dict[str, Any]] = []
        for effect in self.effects:
            ttl = float(effect.get("ttl", 0.0)) - dt
            if ttl <= 0.0:
                continue
            effect["ttl"] = ttl
            next_effects.append(effect)
        self.effects = next_effects

    def _is_obstacle_solid(self, obstacle: ObstacleRect) -> bool:
        if not obstacle.solid:
            return False
        if obstacle.kind == "door" and self.door_open_timers.get(obstacle.obstacle_id, 0.0) > 0.0:
            return False
        return True

    def _active_solids(self) -> list[ObstacleRect]:
        return [obstacle for obstacle in self.obstacles.values() if self._is_obstacle_solid(obstacle)]

    def _open_door(self, obstacle_id: int, seconds: float = 1.0) -> None:
        if obstacle_id in self.door_open_timers:
            was_closed = self.door_open_timers[obstacle_id] <= 0.0
            self.door_open_timers[obstacle_id] = max(self.door_open_timers[obstacle_id], seconds)
            if was_closed:
                door = self.obstacles.get(obstacle_id)
                if door is not None:
                    self._add_effect(
                        "door_open",
                        ttl=0.30,
                        position=door.center.to_list(),
                    )

    def step(self, commands: dict[int, PlayerCommand], dt: float = config.TICK_DT) -> None:
        if self.result is not None:
            return

        self.tick += 1
        self.time_seconds += dt

        for player in self.players.values():
            if not player.alive:
                continue

            player.shoot_cooldown = max(0.0, player.shoot_cooldown - dt)
            player.kick_cooldown = max(0.0, player.kick_cooldown - dt)

            if player.stun_remaining > 0.0:
                player.stun_remaining = max(0.0, player.stun_remaining - dt)

        for pickup in self.pickups.values():
            pickup.cooldown = max(0.0, pickup.cooldown - dt)

        for obstacle_id, value in list(self.letterbox_cooldowns.items()):
            self.letterbox_cooldowns[obstacle_id] = max(0.0, value - dt)

        for obstacle_id, value in list(self.door_open_timers.items()):
            self.door_open_timers[obstacle_id] = max(0.0, value - dt)

        self._tick_effects(dt)

        self._process_actions(commands)
        self._move_players(commands, dt)
        self._resolve_player_collisions()
        self._handle_stun_breakable_collisions()
        self._shoot(commands)
        self._update_projectiles(dt)
        self._update_pickups(dt)
        self._resolve_breakables()
        self._check_round_end()

    def _process_actions(self, commands: dict[int, PlayerCommand]) -> None:
        for player_id, player in self.players.items():
            if not player.alive:
                continue

            cmd = commands.get(player_id, PlayerCommand())

            aim = cmd.aim.normalize()
            if aim.length() > 0.0:
                player.facing = aim

            if player.stun_remaining > 0.0:
                continue

            if cmd.kick and cmd.seq != player.last_kick_seq and player.kick_cooldown <= 0.0:
                player.last_kick_seq = cmd.seq
                self._apply_kick(player_id)

            if cmd.pickup and cmd.seq != player.last_pickup_seq:
                player.last_pickup_seq = cmd.seq
                self._pickup_nearest_weapon(player_id)
                continue

            if cmd.throw and cmd.seq != player.last_throw_seq:
                player.last_throw_seq = cmd.seq
                self._throw_current_weapon(player_id)
                continue

            if cmd.drop and cmd.seq != player.last_drop_seq:
                player.last_drop_seq = cmd.seq
                self._drop_current_weapon(player_id)
                continue

    def _move_players(self, commands: dict[int, PlayerCommand], dt: float) -> None:
        for player_id, player in self.players.items():
            if not player.alive:
                continue

            previous_position = player.position
            if player.stun_remaining > 0.0:
                direction = player.stun_direction.normalize()
                move_delta = direction * (config.STUN_SPEED * dt)
                # Keep stun knockback deterministic; inertia applies to regular movement only.
                player.velocity = Vec2()
            else:
                cmd = commands.get(player_id, PlayerCommand())
                move_input = cmd.move
                mag = move_input.length()
                if mag > 1.0:
                    move_input = move_input * (1.0 / mag)

                desired_velocity = move_input * config.PLAYER_SPEED
                velocity_delta = desired_velocity - player.velocity
                max_delta = (
                    config.PLAYER_MOVE_ACCELERATION
                    if desired_velocity.length() > 1e-6
                    else config.PLAYER_MOVE_DECELERATION
                ) * dt

                delta_mag = velocity_delta.length()
                if delta_mag > max_delta and delta_mag > 1e-8:
                    velocity_delta = velocity_delta * (max_delta / delta_mag)

                player.velocity = player.velocity + velocity_delta
                if desired_velocity.length() <= 1e-6 and player.velocity.length() < 1.0:
                    player.velocity = Vec2()
                move_delta = player.velocity * dt

            new_position = player.position + move_delta
            for obstacle_id, remaining in self.door_open_timers.items():
                if remaining > 0.0:
                    continue
                door = self.obstacles.get(obstacle_id)
                if door is None or not door.solid:
                    continue
                if circle_intersects_rect(new_position, config.PLAYER_RADIUS + 1.0, door):
                    self._open_door(obstacle_id, seconds=1.0)

            solids = self._active_solids()
            collided_with_solid = any(
                circle_intersects_rect(new_position, config.PLAYER_RADIUS, obstacle)
                for obstacle in solids
            )
            resolved_position = resolve_circle_world(new_position, config.PLAYER_RADIUS, solids)

            if self._is_player_grounded(resolved_position):
                player.position = resolved_position
                if player.stun_remaining <= 0.0:
                    actual_move = (resolved_position - previous_position).length()
                    expected_move = move_delta.length()
                    if expected_move > 1e-6 and actual_move < expected_move * 0.4:
                        player.velocity = player.velocity * 0.25
                continue

            # If a wall collision would eject player outside floor due penetration resolution,
            # keep previous grounded position instead of counting it as ring-out.
            if collided_with_solid and self._is_player_grounded(previous_position):
                fallback = resolve_circle_world(previous_position, config.PLAYER_RADIUS, solids)
                if self._is_player_grounded(fallback):
                    player.position = fallback
                    player.velocity = Vec2()
                    continue

            player.position = resolved_position
            player.velocity = Vec2()
            player.alive = False

    def _resolve_player_collisions(self) -> None:
        alive_players = [player for player in self.players.values() if player.alive]
        if len(alive_players) < 2:
            return

        p1, p2 = alive_players[0], alive_players[1]
        before_p1 = p1.position
        before_p2 = p2.position
        delta = p2.position - p1.position
        dist = delta.length()
        min_dist = config.PLAYER_RADIUS * 2.0

        if dist >= min_dist:
            return

        if dist <= 1e-8:
            normal = Vec2(1.0, 0.0)
        else:
            normal = delta * (1.0 / dist)

        overlap = (min_dist - dist) * 0.5
        p1.position = p1.position - normal * overlap
        p2.position = p2.position + normal * overlap

        solids = self._active_solids()
        for player, fallback in ((p1, before_p1), (p2, before_p2)):
            corrected = resolve_circle_world(player.position, config.PLAYER_RADIUS, solids)
            if self._is_player_grounded(corrected):
                player.position = corrected
                continue

            fallback_corrected = resolve_circle_world(fallback, config.PLAYER_RADIUS, solids)
            if self._is_player_grounded(fallback_corrected):
                player.position = fallback_corrected
                continue

            player.position = corrected
            player.velocity = Vec2()
            player.alive = False

    def _handle_stun_breakable_collisions(self) -> None:
        for player in self.players.values():
            if not player.alive or player.stun_remaining <= 0.0:
                continue

            hit_breakable = False
            for breakable in self.breakables.values():
                if not breakable.alive:
                    continue

                obstacle = self.obstacles.get(breakable.obstacle_id)
                if obstacle is None or not self._is_obstacle_solid(obstacle):
                    continue

                if circle_intersects_rect(player.position, config.PLAYER_RADIUS, obstacle):
                    # Ported from Rust STUN_BREAKABILITY = 100.
                    breakable.current_value += 100.0
                    hit_breakable = True

            # If knockback crashed into a breakable object, stop knockback immediately.
            if hit_breakable:
                player.stun_remaining = 0.0
                player.stun_direction = Vec2()
                player.velocity = Vec2()

    def _shoot(self, commands: dict[int, PlayerCommand]) -> None:
        for player_id, player in self.players.items():
            if not player.alive:
                continue

            if player.stun_remaining > 0.0:
                continue

            cmd = commands.get(player_id, PlayerCommand())
            if not cmd.shoot:
                continue

            weapon = player.current_weapon
            if weapon is None:
                continue

            if player.shoot_cooldown > 0.0:
                continue

            if weapon.ammo <= 0:
                continue

            stats = WEAPON_STATS[weapon.weapon_type]

            spread_x = self.random.uniform(stats.spread_x[0], stats.spread_x[1])
            spread_y = self.random.uniform(stats.spread_y[0], stats.spread_y[1])

            shot_direction = Vec2(player.facing.x + spread_x, player.facing.y + spread_y).normalize()
            if shot_direction.length() <= 0.0:
                shot_direction = Vec2(1.0, 0.0)

            weapon.ammo -= 1
            player.shoot_cooldown = stats.shot_cooldown
            muzzle_pos = player.position + shot_direction * 18.0
            self._add_effect(
                "muzzle",
                ttl=0.14,
                position=muzzle_pos.to_list(),
                direction=shot_direction.to_list(),
                weapon=weapon.weapon_type.value,
                owner=player_id,
            )

            projectile = Projectile(
                projectile_id=self._next_projectile_id,
                owner_id=player_id,
                weapon_type=weapon.weapon_type,
                position=Vec2(player.position.x, player.position.y),
                velocity=shot_direction * config.BULLET_SPEED,
                remaining_life=config.BULLET_LIFETIME,
            )
            self.projectiles[self._next_projectile_id] = projectile
            self._next_projectile_id += 1

    def _update_projectiles(self, dt: float) -> None:
        to_remove: set[int] = set()

        for projectile_id, projectile in list(self.projectiles.items()):
            old_pos = projectile.position
            new_pos = old_pos + projectile.velocity * dt
            segment = new_pos - old_pos

            nearest_t = None
            hit_player_id: int | None = None
            hit_obstacle_id: int | None = None

            # Check players first.
            for player in self.players.values():
                if not player.alive or player.player_id == projectile.owner_id:
                    continue

                t = ray_segment_circle_intersection(old_pos, new_pos, player.position, config.PLAYER_RADIUS)
                if t is None:
                    continue

                if nearest_t is None or t < nearest_t:
                    nearest_t = t
                    hit_player_id = player.player_id
                    hit_obstacle_id = None

            # Check obstacles.
            for obstacle in self.obstacles.values():
                if not self._is_obstacle_solid(obstacle):
                    continue

                t = ray_segment_aabb_intersection(old_pos, new_pos, obstacle)
                if t is None:
                    continue

                if nearest_t is None or t < nearest_t:
                    nearest_t = t
                    hit_obstacle_id = obstacle.obstacle_id
                    hit_player_id = None

            hit_position = None
            if nearest_t is not None:
                hit_position = old_pos + segment * nearest_t
                tracer_end = hit_position
            else:
                tracer_end = new_pos

            self._add_effect(
                "tracer",
                ttl=0.32,
                start=old_pos.to_list(),
                end=tracer_end.to_list(),
                weapon=projectile.weapon_type.value,
            )

            if hit_player_id is not None:
                victim = self.players[hit_player_id]
                victim.alive = False
                impact_pos = hit_position or new_pos
                self._add_effect(
                    "blood",
                    ttl=0.78,
                    duration=0.78,
                    position=impact_pos.to_list(),
                    character=victim.character,
                )
                self._spawn_juice_stain(impact_pos, victim.character)
                self._add_effect(
                    "impact",
                    ttl=0.30,
                    position=impact_pos.to_list(),
                    material="player",
                )
                to_remove.add(projectile_id)
                continue

            if hit_obstacle_id is not None:
                self._apply_bullet_breakability(hit_obstacle_id, projectile.weapon_type)
                obstacle = self.obstacles.get(hit_obstacle_id)
                impact_pos = hit_position or new_pos
                self._add_effect(
                    "impact",
                    ttl=0.30,
                    position=impact_pos.to_list(),
                    material=(obstacle.kind if obstacle else "wall"),
                )
                if obstacle and obstacle.kind == "door":
                    self._open_door(hit_obstacle_id, seconds=0.9)
                to_remove.add(projectile_id)
                continue

            projectile.position = new_pos
            projectile.remaining_life -= dt
            if projectile.remaining_life <= 0.0:
                to_remove.add(projectile_id)

        for projectile_id in to_remove:
            self.projectiles.pop(projectile_id, None)

    def _update_pickups(self, dt: float) -> None:
        to_remove: set[int] = set()
        for pickup_id, pickup in self.pickups.items():
            if pickup.velocity.length() <= 1e-4:
                pickup.velocity = Vec2(0.0, 0.0)
            else:
                pickup.position = pickup.position + pickup.velocity * dt
                damping_factor = max(0.0, 1.0 - config.WEAPON_PICKUP_LINEAR_DAMPING * dt)
                pickup.velocity = pickup.velocity * damping_factor

                pickup.position = resolve_circle_world(
                    pickup.position,
                    6.0,
                    self._active_solids(),
                )

            if pickup.ammo == 0:
                if pickup.empty_remove_remaining is None:
                    pickup.empty_remove_remaining = config.EMPTY_PICKUP_REMOVE_SECONDS
                else:
                    pickup.empty_remove_remaining -= dt
                    if pickup.empty_remove_remaining <= 0.0:
                        to_remove.add(pickup_id)

        for pickup_id in to_remove:
            self.pickups.pop(pickup_id, None)

    def _apply_kick(self, kicker_id: int) -> None:
        kicker = self.players[kicker_id]
        if not kicker.alive:
            return

        kicker.kick_cooldown = config.PLAYER_KICK_COOLDOWN
        forward = kicker.facing.normalize()
        kick_zone_center = kicker.position + forward * 15.0
        kick_visual_origin = kicker.position + forward * 6.0
        self._add_effect(
            "kick_arc",
            ttl=0.18,
            duration=0.18,
            position=kick_visual_origin.to_list(),
            direction=forward.to_list(),
            owner=kicker_id,
        )

        for target in self.players.values():
            if not target.alive or target.player_id == kicker_id:
                continue

            if not kick_target_in_front(
                player_pos=kicker.position,
                forward=forward,
                target_pos=target.position,
                max_distance=config.KICK_RANGE,
                min_dot=config.KICK_ARC_DOT_THRESHOLD,
            ):
                continue

            target.stun_remaining = config.STUN_DURATION
            target.stun_direction = Vec2(forward.x, forward.y)
            self._add_effect(
                "kick_hit",
                ttl=0.24,
                duration=0.24,
                position=target.position.to_list(),
                source=kick_visual_origin.to_list(),
                direction=forward.to_list(),
                travel_speed=config.STUN_SPEED,
                owner=kicker_id,
                target=target.player_id,
            )
            self._add_effect(
                "impact",
                ttl=0.18,
                position=target.position.to_list(),
                material="player",
            )

            if target.current_weapon is not None:
                self._drop_weapon(target, impulse=config.WEAPON_DROP_IMPULSE, direction=forward)

        for pickup in self.pickups.values():
            if pickup.position.distance_to(kick_zone_center) <= 14.0:
                pickup.velocity = pickup.velocity + forward * 120.0

        for obstacle in self.obstacles.values():
            if obstacle.kind not in {"letterbox", "box", "glass", "door"}:
                continue
            if not self._is_obstacle_solid(obstacle):
                continue

            if not circle_intersects_rect(kick_zone_center, 14.0, obstacle):
                continue

            if obstacle.kind == "letterbox":
                self._try_spawn_letterbox_weapon(obstacle.obstacle_id, obstacle.center, forward)
            elif obstacle.kind == "door":
                self._open_door(obstacle.obstacle_id, seconds=1.2)
                self._add_effect(
                    "impact",
                    ttl=0.20,
                    position=obstacle.center.to_list(),
                    material="door",
                )
            else:
                self._add_effect(
                    "impact",
                    ttl=0.16,
                    position=obstacle.center.to_list(),
                    material=obstacle.kind,
                )
                self._apply_kick_breakability(obstacle.obstacle_id)

    def _try_spawn_letterbox_weapon(self, letterbox_id: int, position: Vec2, forward: Vec2) -> None:
        pickups_on_arena = len(self.pickups)
        if pickups_on_arena > 8:
            return
        if self.letterbox_cooldowns.get(letterbox_id, 0.0) > 0.0:
            return

        self._spawn_pickup(
            weapon_type=WeaponType.UZI,
            ammo=WEAPON_STATS[WeaponType.UZI].max_ammo,
            position=Vec2(position.x, position.y),
            velocity=Vec2(-forward.x * 80.0, -forward.y * 80.0),
            cooldown=config.WEAPON_PICKUP_COOLDOWN,
        )
        self._add_effect(
            "spawn",
            ttl=0.35,
            position=position.to_list(),
            item="Uzi",
        )
        self.letterbox_cooldowns[letterbox_id] = config.LETTERBOX_COOLDOWN_SECONDS

    def _nearest_pickup_for_player(self, player: PlayerState) -> PickupWeapon | None:
        nearest_pickup: PickupWeapon | None = None
        nearest_dist = 10_000.0

        for pickup in self.pickups.values():
            if pickup.cooldown > 0.0:
                continue

            dist = pickup.position.distance_to(player.position)
            if dist <= config.WEAPON_PICKUP_DISTANCE and dist < nearest_dist:
                nearest_dist = dist
                nearest_pickup = pickup

        return nearest_pickup

    def _pickup_nearest_weapon(self, player_id: int) -> bool:
        player = self.players[player_id]
        if not player.alive:
            return False

        nearest_pickup = self._nearest_pickup_for_player(player)
        if nearest_pickup is not None:
            if player.current_weapon is not None:
                self._drop_weapon(player, impulse=config.WEAPON_DROP_IMPULSE, direction=player.facing)

            player.current_weapon = WeaponInstance(
                weapon_type=nearest_pickup.weapon_type,
                ammo=nearest_pickup.ammo,
            )

            # Ported behaviour: newly picked weapon cannot fire instantly.
            player.shoot_cooldown = WEAPON_STATS[player.current_weapon.weapon_type].shot_cooldown

            self._add_effect(
                "pickup",
                ttl=0.30,
                position=nearest_pickup.position.to_list(),
                weapon=nearest_pickup.weapon_type.value,
                owner=player_id,
            )
            self.pickups.pop(nearest_pickup.pickup_id, None)
            return True
        return False

    def _drop_current_weapon(self, player_id: int) -> bool:
        player = self.players[player_id]
        if not player.alive or player.current_weapon is None:
            return False
        self._drop_weapon(player, impulse=config.WEAPON_DROP_IMPULSE, direction=player.facing)
        return True

    def _throw_current_weapon(self, player_id: int) -> bool:
        player = self.players[player_id]
        if not player.alive or player.current_weapon is None:
            return False
        self._drop_weapon(player, impulse=config.WEAPON_THROW_IMPULSE, direction=player.facing)
        return True

    def _drop_weapon(self, player: PlayerState, impulse: float, direction: Vec2) -> None:
        if player.current_weapon is None:
            return

        direction_norm = direction.normalize()
        if direction_norm.length() <= 0.0:
            direction_norm = Vec2(1.0, 0.0)

        # Convert impulse to a manageable velocity for our simplified simulation.
        throw_speed = impulse / 160.0

        self._spawn_pickup(
            weapon_type=player.current_weapon.weapon_type,
            ammo=player.current_weapon.ammo,
            position=Vec2(player.position.x, player.position.y),
            velocity=direction_norm * throw_speed,
            cooldown=config.WEAPON_PICKUP_COOLDOWN,
            empty_remove_remaining=(
                config.EMPTY_PICKUP_REMOVE_SECONDS
                if player.current_weapon.ammo == 0
                else None
            ),
        )
        self._add_effect(
            "throw",
            ttl=0.30,
            position=player.position.to_list(),
            weapon=player.current_weapon.weapon_type.value,
            owner=player.player_id,
        )

        player.current_weapon = None

    def _spawn_pickup(
        self,
        weapon_type: WeaponType,
        ammo: int,
        position: Vec2,
        velocity: Vec2,
        cooldown: float,
        empty_remove_remaining: float | None = None,
    ) -> None:
        pickup_id = self._next_pickup_id
        self._next_pickup_id += 1

        self.pickups[pickup_id] = PickupWeapon(
            pickup_id=pickup_id,
            weapon_type=weapon_type,
            ammo=ammo,
            position=position,
            velocity=velocity,
            cooldown=cooldown,
            empty_remove_remaining=empty_remove_remaining,
        )

    def _apply_bullet_breakability(self, obstacle_id: int, weapon_type: WeaponType) -> None:
        stats = WEAPON_STATS[weapon_type]
        for breakable in self.breakables.values():
            if not breakable.alive or breakable.obstacle_id != obstacle_id:
                continue

            breakable.current_value += stats.breakability

    def _apply_kick_breakability(self, obstacle_id: int) -> None:
        for breakable in self.breakables.values():
            if not breakable.alive or breakable.obstacle_id != obstacle_id:
                continue

            breakable.current_value += 0.1

    def _resolve_breakables(self) -> None:
        for breakable in self.breakables.values():
            if not breakable.alive:
                continue

            if breakable.current_value < breakable.threshold:
                continue

            breakable.alive = False
            obstacle = self.obstacles.get(breakable.obstacle_id)
            if obstacle is not None:
                obstacle.solid = False
            if breakable.variant == "Box":
                self._spawn_box_debris(breakable.rect_center, breakable.rect_half_size)
            self._add_effect(
                "break",
                ttl=0.65,
                position=breakable.rect_center.to_list(),
                variant=breakable.variant,
            )

    def _spawn_box_debris(self, center: Vec2, half_size: Vec2) -> None:
        count = 18
        radius = max(half_size.x, half_size.y) + 9.0
        for i in range(count):
            angle = self.random.uniform(0.0, 6.283185307179586)
            dist = self.random.uniform(2.0, radius)
            px = center.x + math.cos(angle) * dist
            py = center.y + math.sin(angle) * dist
            size = self.random.randint(1, 3)
            tone = "light" if (i % 3) else "dark"
            self.debris.append(
                {
                    "type": "box_debris",
                    "position": [px, py],
                    "size": size,
                    "tone": tone,
                }
            )

    def _spawn_juice_stain(self, center: Vec2, character: str) -> None:
        self.debris.append(
            {
                "type": "juice_stain",
                "position": [center.x, center.y],
                "character": self.normalize_character(character, fallback="lemon"),
                "radius": self.random.uniform(16.0, 19.0),
                "seed": self.random.randint(1, 1_000_000),
                "created_at": self.time_seconds,
                "created_tick": self.tick,
            }
        )

    def _is_player_grounded(self, position: Vec2) -> bool:
        grid_x = int(position.x // config.TILE_SIZE)
        grid_y = int(position.y // config.TILE_SIZE)
        return (grid_x, grid_y) in self._floor_cells

    def _check_round_end(self) -> None:
        alive_players = [p.player_id for p in self.players.values() if p.alive]

        if len(alive_players) <= 1:
            winner = alive_players[0] if len(alive_players) == 1 else None
            self.result = RoundResult(
                winner_id=winner,
                reason="elimination",
                duration_seconds=self.time_seconds,
            )
            self.status = "finished"
            return

        if self.tick >= config.ROUND_TICK_LIMIT:
            self.result = RoundResult(
                winner_id=None,
                reason="tick_limit",
                duration_seconds=self.time_seconds,
            )
            self.status = "finished"
            return

        if self.time_seconds >= self.round_time_limit_seconds:
            self.result = RoundResult(
                winner_id=None,
                reason="time_limit",
                duration_seconds=self.time_seconds,
            )
            self.status = "finished"

    def get_snapshot(self) -> dict[str, Any]:
        players = []
        for player in self.players.values():
            players.append(
                {
                    "id": player.player_id,
                    "position": player.position.to_list(),
                    "facing": player.facing.to_list(),
                    "velocity": player.velocity.to_list(),
                    "alive": player.alive,
                    "color": player.color,
                    "character": player.character,
                    "weapon": (
                        {
                            "type": player.current_weapon.weapon_type.value,
                            "ammo": player.current_weapon.ammo,
                        }
                        if player.current_weapon
                        else None
                    ),
                    "shoot_cooldown": player.shoot_cooldown,
                    "kick_cooldown": player.kick_cooldown,
                    "stun_remaining": player.stun_remaining,
                }
            )

        pickups = []
        for pickup in self.pickups.values():
            pickup_payload = {
                "id": pickup.pickup_id,
                "type": pickup.weapon_type.value,
                "ammo": pickup.ammo,
                "position": pickup.position.to_list(),
                "cooldown": pickup.cooldown,
            }
            if pickup.empty_remove_remaining is not None:
                pickup_payload["despawn_remaining"] = max(0.0, pickup.empty_remove_remaining)
            pickups.append(pickup_payload)

        projectiles = []
        for projectile in self.projectiles.values():
            projectiles.append(
                {
                    "id": projectile.projectile_id,
                    "owner": projectile.owner_id,
                    "type": projectile.weapon_type.value,
                    "position": projectile.position.to_list(),
                    "velocity": projectile.velocity.to_list(),
                    "remaining_life": projectile.remaining_life,
                }
            )

        obstacles = []
        for obstacle in self.obstacles.values():
            obstacles.append(
                {
                    "id": obstacle.obstacle_id,
                    "kind": obstacle.kind,
                    "center": obstacle.center.to_list(),
                    "half_size": obstacle.half_size.to_list(),
                    "solid": self._is_obstacle_solid(obstacle),
                }
            )

        breakables = []
        for breakable in self.breakables.values():
            breakables.append(
                {
                    "id": breakable.breakable_id,
                    "obstacle_id": breakable.obstacle_id,
                    "variant": breakable.variant,
                    "current": breakable.current_value,
                    "threshold": breakable.threshold,
                    "alive": breakable.alive,
                    "center": breakable.rect_center.to_list(),
                    "half_size": breakable.rect_half_size.to_list(),
                }
            )

        effects = []
        for effect in self.effects:
            payload = {k: v for k, v in effect.items() if k != "ttl"}
            effects.append(payload)

        letterboxes = []
        for obstacle_id, cooldown in self.letterbox_cooldowns.items():
            obstacle = self.obstacles.get(obstacle_id)
            if obstacle is None:
                continue
            letterboxes.append(
                {
                    "id": obstacle_id,
                    "position": obstacle.center.to_list(),
                    "cooldown": cooldown,
                    "ready": cooldown <= 0.0,
                }
            )

        return {
            "status": self.status,
            "tick": self.tick,
            "time_seconds": self.time_seconds,
            "time_limit_seconds": self.round_time_limit_seconds,
            "result": (
                {
                    "winner_id": self.result.winner_id,
                    "reason": self.result.reason,
                    "duration_seconds": self.result.duration_seconds,
                }
                if self.result
                else None
            ),
            "level": self._static_level_payload,
            "players": players,
            "pickups": pickups,
            "projectiles": projectiles,
            "obstacles": obstacles,
            "breakables": breakables,
            "effects": effects,
            "debris": list(self.debris),
            "letterboxes": letterboxes,
        }
