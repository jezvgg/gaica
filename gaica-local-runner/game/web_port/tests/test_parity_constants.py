from __future__ import annotations

import unittest

from game import config


class TestParityConstants(unittest.TestCase):
    def test_core_values_match_rust(self) -> None:
        self.assertEqual(config.PLAYER_SPEED, 255.0)
        self.assertEqual(config.WEAPON_PICKUP_DISTANCE, 25.0)
        self.assertEqual(config.PLAYER_KICK_COOLDOWN, 1.0)

        self.assertEqual(config.REVOLVER_MAX_AMMO, 4)
        self.assertEqual(config.UZI_MAX_AMMO, 18)
        self.assertEqual(config.REVOLVER_SHOT_COOLDOWN, 0.4)
        self.assertEqual(config.UZI_SHOT_COOLDOWN, 0.1)

        self.assertEqual(config.BULLET_LIFETIME, 2.0)
        self.assertEqual(config.BULLET_SPEED, 500.0)

        self.assertEqual(config.STUN_DURATION, 0.5)
        self.assertEqual(config.STUN_SPEED, 350.0)

        self.assertEqual(config.BOX_BREAK_THRESHOLD, 1.0)
        self.assertEqual(config.GLASS_BREAK_THRESHOLD, 0.1)
        self.assertEqual(config.LETTERBOX_COOLDOWN_SECONDS, 1.5)


if __name__ == "__main__":
    unittest.main()
