from __future__ import annotations

import json
from pathlib import Path
import unittest

from game import config
from game.level_loader import get_levels_count, load_level


class TestLevelLoader(unittest.TestCase):
    def _ldtk_path(self) -> Path:
        root = Path(__file__).resolve().parents[2]
        return root / "web_port" / "assets" / "levels" / "test_ldtk_project.ldtk"

    def test_load_level0_core_data(self) -> None:
        ldtk = self._ldtk_path()

        level = load_level(ldtk, level_index=0)

        self.assertEqual(level.identifier, "Level_0")
        self.assertEqual(level.width, 512)
        self.assertEqual(level.height, 320)

        self.assertGreaterEqual(len(level.player_spawns), 2)
        self.assertGreaterEqual(len(level.weapon_spawns), 2)
        self.assertGreaterEqual(len(level.floor_tiles), 1)

        kinds = {o.kind for o in level.obstacles}
        self.assertIn("wall", kinds)
        self.assertIn("letterbox", kinds)

        variants = {b.variant for b in level.breakables}
        self.assertIn("Box", variants)

    def test_load_all_levels(self) -> None:
        ldtk = self._ldtk_path()
        total = get_levels_count(ldtk)
        self.assertGreaterEqual(total, 4)

        for idx in range(total):
            level = load_level(ldtk, level_index=idx)
            self.assertGreaterEqual(len(level.player_spawns), 2)
            self.assertGreater(level.width, 0)
            self.assertGreater(level.height, 0)
            self.assertGreater(len(level.floor_tiles), 0)

    def test_random_level_selection(self) -> None:
        ldtk = self._ldtk_path()
        level_a = load_level(ldtk, level_index=None, seed=1)
        level_b = load_level(ldtk, level_index=None, seed=2)
        self.assertTrue(level_a.identifier.startswith("Level_"))
        self.assertTrue(level_b.identifier.startswith("Level_"))

    def test_box_spawns_shifted_by_half_cell(self) -> None:
        ldtk = self._ldtk_path()
        raw = json.loads(ldtk.read_text(encoding="utf-8"))
        level0 = raw["levels"][0]

        raw_box_centers: list[tuple[float, float]] = []
        for layer in level0.get("layerInstances", []):
            if layer.get("__type") != "Entities":
                continue
            grid_size = int(layer.get("__gridSize", 64))
            for ent in layer.get("entityInstances", []):
                if ent.get("__identifier") != "BoxSpawnPoint":
                    continue
                px = ent["px"]
                width = float(ent.get("width", grid_size))
                height = float(ent.get("height", grid_size))
                raw_box_centers.append((float(px[0]) + width / 2.0, float(px[1]) + height / 2.0))

        level = load_level(ldtk, level_index=0)
        loaded_box_centers = {(round(v.x, 3), round(v.y, 3)) for v in level.box_spawns}
        expected_centers = {
            (round(x - config.TILE_SIZE * 0.5, 3), round(y - config.TILE_SIZE * 0.5, 3))
            for x, y in raw_box_centers
        }

        self.assertGreater(len(raw_box_centers), 0)
        self.assertEqual(loaded_box_centers, expected_centers)


if __name__ == "__main__":
    unittest.main()
