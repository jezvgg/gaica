from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from urllib.request import urlopen


class TestManualBotRun(unittest.TestCase):
    def test_logs_show_bot_movement(self) -> None:
        main_script = Path(__file__).resolve().parents[1] / "main.py"
        cmd = [
            sys.executable,
            str(main_script),
            "--with-test-bots",
            "--test-bot-a",
            "aggressive",
            "--test-bot-b",
            "defensive",
            "--bot-port",
            "9160",
            "--web-port",
            "8160",
            "--round-time-limit",
            "8",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1", "SMART_BOT_LOG": "1"},
        )

        try:
            states: list[dict] = []
            deadline = time.time() + 10.0
            while time.time() < deadline:
                try:
                    with urlopen("http://127.0.0.1:8160/api/state", timeout=1.5) as response:
                        state = json.loads(response.read().decode("utf-8"))
                    states.append(state)
                except Exception:
                    pass
                time.sleep(0.25)

            self.assertGreater(len(states), 3)

            connected = any(len(s.get("bots_connected", [])) == 2 for s in states)
            ticks = [int(s.get("tick", 0)) for s in states]
            tick_progressed = max(ticks) - min(ticks) >= 5

            positions = []
            for s in states:
                players = s.get("players", [])
                if players:
                    positions.append(players[0]["position"])
            moved = False
            if len(positions) >= 2:
                first = positions[0]
                last = positions[-1]
                moved = abs(first[0] - last[0]) > 0.01 or abs(first[1] - last[1]) > 0.01

            result_seen = any(s.get("result") is not None for s in states)

            self.assertTrue(connected)
            self.assertTrue(tick_progressed)
            self.assertTrue(moved or result_seen)
        finally:
            proc.terminate()
            try:
                out, _ = proc.communicate(timeout=4.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate(timeout=2.0)

            self.assertNotIn("ModuleNotFoundError", out)
            self.assertNotIn("Traceback", out)


if __name__ == "__main__":
    unittest.main()
