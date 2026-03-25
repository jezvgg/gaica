from __future__ import annotations

from game import config
from game.models import BreakableState, LevelData, ObstacleRect, TileDraw, Vec2, WeaponType


def build_flat_test_level() -> LevelData:
    floor_tiles = []
    for y in range(0, 192, 64):
        for x in range(0, 320, 64):
            floor_tiles.append(
                TileDraw(
                    x=x,
                    y=y,
                    tile_id=7,
                    src_x=0,
                    src_y=128,
                    layer="Floor",
                    size=64,
                )
            )

    return LevelData(
        identifier="TestLevel",
        width=320,
        height=192,
        floor_tiles=floor_tiles,
        top_tiles=[],
        small_tiles=[],
        player_spawns=[Vec2(64.0, 96.0), Vec2(224.0, 96.0)],
        weapon_spawns=[(Vec2(64.0, 96.0), WeaponType.REVOLVER)],
        box_spawns=[],
        obstacles=[],
        breakables=[],
        letterboxes=[],
    )


def build_breakable_test_level() -> LevelData:
    level = build_flat_test_level()
    box_obstacle = ObstacleRect(
        obstacle_id=10,
        kind="box",
        center=Vec2(160.0, 96.0),
        half_size=Vec2(8.0, 8.0),
        solid=True,
    )
    box_breakable = BreakableState(
        breakable_id=10,
        obstacle_id=10,
        variant="Box",
        threshold=1.0,
        current_value=0.0,
        rect_center=Vec2(160.0, 96.0),
        rect_half_size=Vec2(8.0, 8.0),
        alive=True,
    )

    level.obstacles.append(box_obstacle)
    level.breakables.append(box_breakable)
    return level


def build_glass_test_level() -> LevelData:
    level = build_flat_test_level()
    glass_obstacle = ObstacleRect(
        obstacle_id=11,
        kind="glass",
        center=Vec2(160.0, 96.0),
        half_size=Vec2(config.WALL_COLLIDER_HALF_W, config.WALL_COLLIDER_HALF_H),
        solid=True,
    )
    glass_breakable = BreakableState(
        breakable_id=11,
        obstacle_id=11,
        variant="Glass",
        threshold=config.GLASS_BREAK_THRESHOLD,
        current_value=0.0,
        rect_center=Vec2(160.0, 96.0),
        rect_half_size=Vec2(config.WALL_COLLIDER_HALF_W, config.WALL_COLLIDER_HALF_H),
        alive=True,
    )

    level.obstacles.append(glass_obstacle)
    level.breakables.append(glass_breakable)
    return level


def build_letterbox_test_level() -> LevelData:
    level = build_flat_test_level()
    letterbox = ObstacleRect(
        obstacle_id=20,
        kind="letterbox",
        center=Vec2(128.0, 96.0),
        half_size=Vec2(config.LETTERBOX_HALF_W, config.LETTERBOX_HALF_H),
        solid=True,
    )
    level.obstacles.append(letterbox)
    level.letterboxes.append(letterbox)
    return level


def build_door_test_level() -> LevelData:
    level = build_flat_test_level()
    door = ObstacleRect(
        obstacle_id=30,
        kind="door",
        center=Vec2(128.0, 96.0),
        half_size=Vec2(config.DOOR_COLLIDER_HALF_W, config.DOOR_COLLIDER_HALF_H),
        solid=True,
    )
    level.obstacles.append(door)
    return level


def build_wall_pin_test_level() -> LevelData:
    floor_tiles = []
    for y in range(0, 192, 64):
        for x in (64, 128, 192, 256):
            floor_tiles.append(
                TileDraw(
                    x=x,
                    y=y,
                    tile_id=7,
                    src_x=0,
                    src_y=128,
                    layer="Floor",
                    size=64,
                )
            )

    level = LevelData(
        identifier="WallPinLevel",
        width=320,
        height=192,
        floor_tiles=floor_tiles,
        top_tiles=[],
        small_tiles=[],
        player_spawns=[Vec2(96.0, 96.0), Vec2(128.0, 96.0)],
        weapon_spawns=[],
        box_spawns=[],
        obstacles=[],
        breakables=[],
        letterboxes=[],
    )

    level.obstacles.append(
        ObstacleRect(
            obstacle_id=40,
            kind="wall",
            center=Vec2(64.0, 96.0),
            half_size=Vec2(config.WALL_COLLIDER_HALF_W, config.WALL_COLLIDER_HALF_H),
            solid=True,
        )
    )
    return level


def build_multi_spawn_test_level() -> LevelData:
    level = build_flat_test_level()
    level.player_spawns = [
        Vec2(64.0, 64.0),
        Vec2(256.0, 64.0),
        Vec2(64.0, 128.0),
        Vec2(256.0, 128.0),
    ]
    return level
