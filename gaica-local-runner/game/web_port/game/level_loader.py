from __future__ import annotations

import json
from pathlib import Path
import random

from . import config
from .models import (
    BreakableState,
    LevelData,
    ObstacleRect,
    TileDraw,
    Vec2,
    WeaponType,
)


def _load_tile_tag_map(project: dict) -> dict[int, list[str]]:
    tag_map: dict[int, list[str]] = {}
    for tileset in project["defs"]["tilesets"]:
        if tileset.get("identifier") != "MafiaTileset":
            continue

        for entry in tileset.get("enumTags", []):
            tag = entry["enumValueId"]
            for tile_id in entry.get("tileIds", []):
                tag_map.setdefault(tile_id, []).append(tag)
        break

    return tag_map


def _tile_center(tile: dict, grid_size: int) -> Vec2:
    return Vec2(tile["px"][0] + grid_size / 2.0, tile["px"][1] + grid_size / 2.0)


def _to_tile_draw(tile: dict, layer: str, grid_size: int) -> TileDraw:
    return TileDraw(
        x=int(tile["px"][0]),
        y=int(tile["px"][1]),
        tile_id=int(tile["t"]),
        src_x=int(tile["src"][0]),
        src_y=int(tile["src"][1]),
        layer=layer,
        size=grid_size,
    )


def get_levels_count(ldtk_path: str | Path) -> int:
    data = json.loads(Path(ldtk_path).read_text(encoding="utf-8"))
    return len(data.get("levels", []))


def load_level(
    ldtk_path: str | Path,
    level_index: int | None = None,
    seed: int | None = None,
) -> LevelData:
    data = json.loads(Path(ldtk_path).read_text(encoding="utf-8"))
    levels = data["levels"]
    if not levels:
        raise ValueError("No levels in LDtk file")

    if level_index is None:
        level_index = random.Random(seed).randrange(len(levels))

    if level_index < 0 or level_index >= len(levels):
        raise ValueError(f"level_index {level_index} is out of range")

    level = levels[level_index]
    tag_map = _load_tile_tag_map(data)

    floor_tiles: list[TileDraw] = []
    top_tiles: list[TileDraw] = []
    small_tiles: list[TileDraw] = []

    player_spawns: list[Vec2] = []
    weapon_spawns: list[tuple[Vec2, WeaponType]] = []
    box_spawns: list[Vec2] = []

    obstacles: list[ObstacleRect] = []
    breakables: list[BreakableState] = []
    letterboxes: list[ObstacleRect] = []

    next_obstacle_id = 1
    next_breakable_id = 1

    for layer in level.get("layerInstances", []):
        layer_name = layer["__identifier"]
        layer_type = layer["__type"]
        grid_size = int(layer["__gridSize"])

        if layer_type == "Entities":
            for ent in layer.get("entityInstances", []):
                identifier = ent["__identifier"]
                px = ent["px"]
                width = ent.get("width", grid_size)
                height = ent.get("height", grid_size)
                center = Vec2(px[0] + width / 2.0, px[1] + height / 2.0)

                if identifier == "PlayerSpawnPoint":
                    player_spawns.append(center)
                elif identifier == "WeaponSpawnPoint":
                    weapon_value = "Revolver"
                    for f in ent.get("fieldInstances", []):
                        if f.get("__identifier") == "Weapons":
                            weapon_value = f.get("__value") or "Revolver"
                            break
                    weapon_type = WeaponType.UZI if weapon_value == "Uzi" else WeaponType.REVOLVER
                    weapon_spawns.append((center, weapon_type))
                elif identifier == "BoxSpawnPoint":
                    # Map authoring offset fix: move all box spawns by -0.5 cell.
                    box_spawns.append(
                        Vec2(
                            center.x - config.TILE_SIZE * 0.5,
                            center.y - config.TILE_SIZE * 0.5,
                        )
                    )
            continue

        for tile in layer.get("gridTiles", []):
            tile_draw = _to_tile_draw(tile, layer_name, grid_size)
            if layer_name == "Floor":
                floor_tiles.append(tile_draw)
            elif layer_name == "TopTiles":
                top_tiles.append(tile_draw)
            elif layer_name == "Small_grid":
                small_tiles.append(tile_draw)

            tile_tags = tag_map.get(int(tile["t"]), [])
            if not tile_tags:
                continue

            center = _tile_center(tile, grid_size)

            for tag in tile_tags:
                # Wall colliders are ported from level/loading.rs get_wall_transform_and_collider.
                if tag == "WallLeft":
                    obstacles.append(
                        ObstacleRect(
                            obstacle_id=next_obstacle_id,
                            kind="wall",
                            center=Vec2(center.x - config.TILE_SIZE / 2.0 + config.WALL_COLLIDER_HALF_W, center.y),
                            half_size=Vec2(config.WALL_COLLIDER_HALF_W, config.WALL_COLLIDER_HALF_H),
                        )
                    )
                    next_obstacle_id += 1
                elif tag == "WallRight":
                    obstacles.append(
                        ObstacleRect(
                            obstacle_id=next_obstacle_id,
                            kind="wall",
                            center=Vec2(center.x + config.TILE_SIZE / 2.0 - config.WALL_COLLIDER_HALF_W, center.y),
                            half_size=Vec2(config.WALL_COLLIDER_HALF_W, config.WALL_COLLIDER_HALF_H),
                        )
                    )
                    next_obstacle_id += 1
                elif tag == "WallTop":
                    obstacles.append(
                        ObstacleRect(
                            obstacle_id=next_obstacle_id,
                            kind="wall",
                            center=Vec2(center.x, center.y - config.TILE_SIZE / 2.0 + config.WALL_COLLIDER_HALF_W),
                            half_size=Vec2(config.WALL_COLLIDER_HALF_H, config.WALL_COLLIDER_HALF_W),
                        )
                    )
                    next_obstacle_id += 1
                elif tag == "WallBottom":
                    obstacles.append(
                        ObstacleRect(
                            obstacle_id=next_obstacle_id,
                            kind="wall",
                            center=Vec2(center.x, center.y + config.TILE_SIZE / 2.0 - config.WALL_COLLIDER_HALF_W),
                            half_size=Vec2(config.WALL_COLLIDER_HALF_H, config.WALL_COLLIDER_HALF_W),
                        )
                    )
                    next_obstacle_id += 1
                elif tag.startswith("DoubleDoor"):
                    is_vertical = tag in {"DoubleDoorLeft", "DoubleDoorRight"}
                    door_half = (
                        Vec2(config.DOOR_COLLIDER_HALF_W, config.DOOR_COLLIDER_HALF_H)
                        if is_vertical
                        else Vec2(config.DOOR_COLLIDER_HALF_H, config.DOOR_COLLIDER_HALF_W)
                    )
                    if tag == "DoubleDoorLeft":
                        door_center = Vec2(center.x - config.TILE_SIZE / 2.0 + config.DOOR_COLLIDER_HALF_W, center.y)
                    elif tag == "DoubleDoorRight":
                        door_center = Vec2(center.x + config.TILE_SIZE / 2.0 - config.DOOR_COLLIDER_HALF_W, center.y)
                    elif tag == "DoubleDoorTop":
                        door_center = Vec2(center.x, center.y - config.TILE_SIZE / 2.0 + config.DOOR_COLLIDER_HALF_W)
                    else:  # DoubleDoorBottom
                        door_center = Vec2(center.x, center.y + config.TILE_SIZE / 2.0 - config.DOOR_COLLIDER_HALF_W)

                    obstacles.append(
                        ObstacleRect(
                            obstacle_id=next_obstacle_id,
                            kind="door",
                            center=door_center,
                            half_size=door_half,
                            solid=True,
                        )
                    )
                    next_obstacle_id += 1
                elif tag == "GlassRight":
                    glass_center = Vec2(center.x + config.TILE_SIZE / 2.0 - config.WALL_COLLIDER_HALF_W, center.y)
                    glass_half = Vec2(config.WALL_COLLIDER_HALF_W, config.WALL_COLLIDER_HALF_H)

                    obstacles.append(
                        ObstacleRect(
                            obstacle_id=next_obstacle_id,
                            kind="glass",
                            center=glass_center,
                            half_size=glass_half,
                            solid=True,
                        )
                    )
                    next_obstacle_id += 1

                    breakables.append(
                        BreakableState(
                            breakable_id=next_breakable_id,
                            obstacle_id=next_obstacle_id - 1,
                            variant="Glass",
                            threshold=config.GLASS_BREAK_THRESHOLD,
                            current_value=0.0,
                            rect_center=glass_center,
                            rect_half_size=glass_half,
                        )
                    )
                    next_breakable_id += 1
                elif tag == "Letterbox":
                    letterbox = ObstacleRect(
                        obstacle_id=next_obstacle_id,
                        kind="letterbox",
                        center=center,
                        half_size=Vec2(config.LETTERBOX_HALF_W, config.LETTERBOX_HALF_H),
                        solid=True,
                    )
                    obstacles.append(letterbox)
                    letterboxes.append(letterbox)
                    next_obstacle_id += 1

    for center in box_spawns:
        box_half = Vec2(config.TILE_SIZE / 8.0, config.TILE_SIZE / 8.0)
        obstacles.append(
            ObstacleRect(
                obstacle_id=next_obstacle_id,
                kind="box",
                center=center,
                half_size=box_half,
                solid=True,
            )
        )
        next_obstacle_id += 1

        breakables.append(
            BreakableState(
                breakable_id=next_breakable_id,
                obstacle_id=next_obstacle_id - 1,
                variant="Box",
                threshold=config.BOX_BREAK_THRESHOLD,
                current_value=0.0,
                rect_center=center,
                rect_half_size=box_half,
            )
        )
        next_breakable_id += 1

    if len(player_spawns) < 2:
        raise ValueError("Expected at least 2 player spawn points")

    return LevelData(
        identifier=level["identifier"],
        width=int(level["pxWid"]),
        height=int(level["pxHei"]),
        floor_tiles=floor_tiles,
        top_tiles=top_tiles,
        small_tiles=small_tiles,
        player_spawns=player_spawns,
        weapon_spawns=weapon_spawns,
        box_spawns=box_spawns,
        obstacles=obstacles,
        breakables=breakables,
        letterboxes=letterboxes,
    )
