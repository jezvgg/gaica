from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - local fallback for machines without numpy
    np = None

if TYPE_CHECKING:
    from gaica_bot.models import LevelInfo, ObstacleView, TickMessage, Vec2


GridMask = Any


def _new_mask(width: int, height: int) -> GridMask:
    if np is not None:
        return np.zeros((height, width), dtype=np.bool_)
    return [[False for _ in range(width)] for _ in range(height)]


def _copy_mask(mask: GridMask) -> GridMask:
    if np is not None:
        return mask.copy()
    return [row[:] for row in mask]


def _set_mask_cell(mask: GridMask, cell_x: int, cell_y: int, value: bool) -> None:
    if np is not None:
        mask[cell_y, cell_x] = value
        return
    mask[cell_y][cell_x] = value


def _get_mask_cell(mask: GridMask, cell_x: int, cell_y: int) -> bool:
    if np is not None:
        return bool(mask[cell_y, cell_x])
    return bool(mask[cell_y][cell_x])


def _and_not(left: GridMask, right: GridMask) -> GridMask:
    if np is not None:
        return left & ~right

    height = len(left)
    width = len(left[0]) if height else 0
    result = _new_mask(width, height)
    for cell_y in range(height):
        for cell_x in range(width):
            result[cell_y][cell_x] = bool(left[cell_y][cell_x] and not right[cell_y][cell_x])
    return result


@dataclass(slots=True)
class CellMapSnapshot:
    level_identifier: str
    cell_size: int
    width_cells: int
    height_cells: int
    floor_mask: GridMask
    solid_mask: GridMask
    walkable_mask: GridMask
    pickup_mask: GridMask
    player_mask: GridMask
    projectile_mask: GridMask
    breakable_mask: GridMask
    letterbox_mask: GridMask
    obstacle_cells: dict[int, tuple[tuple[int, int], ...]]
    pickup_cells: dict[int, tuple[int, int]]
    player_cells: dict[int, tuple[int, int]]
    projectile_cells: dict[int, tuple[int, int]]
    breakable_cells: dict[int, tuple[int, int]]
    letterbox_cells: dict[int, tuple[int, int]]

    def contains_cell(self, cell_x: int, cell_y: int) -> bool:
        return 0 <= cell_x < self.width_cells and 0 <= cell_y < self.height_cells

    def cell_for_position(self, position: "Vec2") -> tuple[int, int] | None:
        cell_x = int(position.x // self.cell_size)
        cell_y = int(position.y // self.cell_size)
        if not self.contains_cell(cell_x, cell_y):
            return None
        return (cell_x, cell_y)

    def cell_center(self, cell_x: int, cell_y: int) -> tuple[float, float]:
        return (
            (cell_x + 0.5) * self.cell_size,
            (cell_y + 0.5) * self.cell_size,
        )

    def is_walkable_cell(self, cell_x: int, cell_y: int) -> bool:
        if not self.contains_cell(cell_x, cell_y):
            return False
        return _get_mask_cell(self.walkable_mask, cell_x, cell_y)

    def is_floor_cell(self, cell_x: int, cell_y: int) -> bool:
        if not self.contains_cell(cell_x, cell_y):
            return False
        return _get_mask_cell(self.floor_mask, cell_x, cell_y)

    def is_world_walkable(self, position: "Vec2") -> bool:
        cell = self.cell_for_position(position)
        if cell is None:
            return False
        return self.is_walkable_cell(*cell)

    def copy_walkable_mask(self) -> GridMask:
        return _copy_mask(self.walkable_mask)


@dataclass(slots=True)
class CellMapBuilder:
    level_identifier: str
    cell_size: int
    width_cells: int
    height_cells: int
    floor_mask: GridMask
    _obstacle_cells_cache: dict[int, tuple[tuple[int, int], ...]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_level(cls, level: "LevelInfo") -> "CellMapBuilder":
        cell_size = max(1, level.floor.grid_size)
        max_floor_x = max((cell_x for cell_x, _ in level.floor.cells), default=-1)
        max_floor_y = max((cell_y for _, cell_y in level.floor.cells), default=-1)
        width_cells = max(1, math.ceil(level.width / cell_size), max_floor_x + 1)
        height_cells = max(1, math.ceil(level.height / cell_size), max_floor_y + 1)
        floor_mask = _new_mask(width_cells, height_cells)
        for cell_x, cell_y in level.floor.cells:
            if 0 <= cell_x < width_cells and 0 <= cell_y < height_cells:
                _set_mask_cell(floor_mask, cell_x, cell_y, True)

        builder = cls(
            level_identifier=level.identifier,
            cell_size=cell_size,
            width_cells=width_cells,
            height_cells=height_cells,
            floor_mask=floor_mask,
        )
        for obstacle in level.static_obstacles:
            builder._cache_cells_for_obstacle(obstacle)
        return builder

    def build_tick_map(self, tick: "TickMessage") -> CellMapSnapshot:
        solid_mask = _new_mask(self.width_cells, self.height_cells)
        pickup_mask = _new_mask(self.width_cells, self.height_cells)
        player_mask = _new_mask(self.width_cells, self.height_cells)
        projectile_mask = _new_mask(self.width_cells, self.height_cells)
        breakable_mask = _new_mask(self.width_cells, self.height_cells)
        letterbox_mask = _new_mask(self.width_cells, self.height_cells)

        obstacle_cells: dict[int, tuple[tuple[int, int], ...]] = {}
        pickup_cells: dict[int, tuple[int, int]] = {}
        player_cells: dict[int, tuple[int, int]] = {}
        projectile_cells: dict[int, tuple[int, int]] = {}
        breakable_cells: dict[int, tuple[int, int]] = {}
        letterbox_cells: dict[int, tuple[int, int]] = {}

        for obstacle in tick.snapshot.obstacles:
            cells = self._cache_cells_for_obstacle(obstacle)
            obstacle_cells[obstacle.obstacle_id] = cells
            if obstacle.solid:
                self._mark_cells(solid_mask, cells)

        for pickup in tick.snapshot.pickups:
            cell = self._point_to_cell(pickup.position)
            if cell is None:
                continue
            pickup_cells[pickup.pickup_id] = cell
            _set_mask_cell(pickup_mask, cell[0], cell[1], True)

        for player in tick.snapshot.players:
            cell = self._point_to_cell(player.position)
            if cell is None:
                continue
            player_cells[player.player_id] = cell
            _set_mask_cell(player_mask, cell[0], cell[1], True)

        for projectile in tick.snapshot.projectiles:
            cell = self._point_to_cell(projectile.position)
            if cell is None:
                continue
            projectile_cells[projectile.projectile_id] = cell
            _set_mask_cell(projectile_mask, cell[0], cell[1], True)

        for breakable in tick.snapshot.breakables:
            cell = self._point_to_cell(breakable.center)
            if cell is None:
                continue
            breakable_cells[breakable.breakable_id] = cell
            _set_mask_cell(breakable_mask, cell[0], cell[1], True)

        for letterbox in tick.snapshot.letterboxes:
            cell = self._point_to_cell(letterbox.position)
            if cell is None:
                continue
            letterbox_cells[letterbox.obstacle_id] = cell
            _set_mask_cell(letterbox_mask, cell[0], cell[1], True)

        walkable_mask = _and_not(self.floor_mask, solid_mask)
        return CellMapSnapshot(
            level_identifier=self.level_identifier,
            cell_size=self.cell_size,
            width_cells=self.width_cells,
            height_cells=self.height_cells,
            floor_mask=_copy_mask(self.floor_mask),
            solid_mask=solid_mask,
            walkable_mask=walkable_mask,
            pickup_mask=pickup_mask,
            player_mask=player_mask,
            projectile_mask=projectile_mask,
            breakable_mask=breakable_mask,
            letterbox_mask=letterbox_mask,
            obstacle_cells=obstacle_cells,
            pickup_cells=pickup_cells,
            player_cells=player_cells,
            projectile_cells=projectile_cells,
            breakable_cells=breakable_cells,
            letterbox_cells=letterbox_cells,
        )

    def _point_to_cell(self, position: "Vec2") -> tuple[int, int] | None:
        cell_x = int(position.x // self.cell_size)
        cell_y = int(position.y // self.cell_size)
        if 0 <= cell_x < self.width_cells and 0 <= cell_y < self.height_cells:
            return (cell_x, cell_y)
        return None

    def _mark_cells(self, mask: GridMask, cells: tuple[tuple[int, int], ...]) -> None:
        for cell_x, cell_y in cells:
            if 0 <= cell_x < self.width_cells and 0 <= cell_y < self.height_cells:
                _set_mask_cell(mask, cell_x, cell_y, True)

    def _cache_cells_for_obstacle(self, obstacle: "ObstacleView") -> tuple[tuple[int, int], ...]:
        cached = self._obstacle_cells_cache.get(obstacle.obstacle_id)
        if cached is not None:
            return cached

        left = obstacle.center.x - obstacle.half_size.x
        right = obstacle.center.x + obstacle.half_size.x
        top = obstacle.center.y - obstacle.half_size.y
        bottom = obstacle.center.y + obstacle.half_size.y
        max_x = self.width_cells - 1
        max_y = self.height_cells - 1

        min_cell_x = max(0, int(math.floor(left / self.cell_size)))
        max_cell_x = min(max_x, int(math.floor(max(left, right - 1e-6) / self.cell_size)))
        min_cell_y = max(0, int(math.floor(top / self.cell_size)))
        max_cell_y = min(max_y, int(math.floor(max(top, bottom - 1e-6) / self.cell_size)))

        cells: list[tuple[int, int]] = []
        for cell_y in range(min_cell_y, max_cell_y + 1):
            for cell_x in range(min_cell_x, max_cell_x + 1):
                cells.append((cell_x, cell_y))

        cached_cells = tuple(cells)
        self._obstacle_cells_cache[obstacle.obstacle_id] = cached_cells
        return cached_cells
