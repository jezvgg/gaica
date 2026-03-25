from __future__ import annotations

import math
import unittest

from game import config
from game.models import ObstacleRect, Vec2
from game.physics import clamp, resolve_circle_world


def _penetration(center: Vec2, rect: ObstacleRect) -> float:
    min_x = rect.center.x - rect.half_size.x
    max_x = rect.center.x + rect.half_size.x
    min_y = rect.center.y - rect.half_size.y
    max_y = rect.center.y + rect.half_size.y
    nearest_x = clamp(center.x, min_x, max_x)
    nearest_y = clamp(center.y, min_y, max_y)
    return config.PLAYER_RADIUS - math.hypot(center.x - nearest_x, center.y - nearest_y)


class TestPhysics(unittest.TestCase):
    def test_resolve_circle_world_fully_resolves_tight_corner_overlap(self) -> None:
        wall = ObstacleRect(
            obstacle_id=1,
            kind="wall",
            center=Vec2(160.0, 128.0),
            half_size=Vec2(2.0, 32.0),
        )
        box = ObstacleRect(
            obstacle_id=2,
            kind="box",
            center=Vec2(176.0, 128.0),
            half_size=Vec2(8.0, 8.0),
        )
        starting_point = Vec2(160.25, 124.0)

        for solids in ([wall, box], [box, wall]):
            with self.subTest(order=[rect.obstacle_id for rect in solids]):
                resolved = resolve_circle_world(starting_point, config.PLAYER_RADIUS, list(solids))
                max_penetration = max(_penetration(resolved, rect) for rect in solids)
                self.assertLessEqual(max_penetration, 1e-6)

