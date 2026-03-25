#!/usr/bin/env python3
import argparse
import json
import os
import socket
import sys


def clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _run_stream(reader, writer) -> int:
    for line in reader:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(message, dict):
            continue
        if message.get("type") != "tick":
            continue
        
        print(message)
        tick = int(message.get("tick", 0) or 0)
        me = message.get("you", {})
        enemy = message.get("enemy", {})

        if not me.get("alive", False):
            cmd = {
                "type": "command",
                "seq": tick,
                "move": {"x": 0.0, "y": 0.0},
                "aim": {"x": 1.0, "y": 0.0},
                "shoot": False,
                "kick": False,
                "pickup": False,
                "drop": False,
                "throw": False,
            }
        else:
            me_pos = me.get("position", {})
            enemy_pos = enemy.get("position", {})

            dx = enemy_pos[0] - me_pos[0]
            dy = enemy_pos[1] - me_pos[1]
            print(dx, dy)
            norm = (dx * dx + dy * dy) ** 0.5 or 1.0

            cmd = {
                "type": "command",
                "seq": tick,
                "move": {"x": clamp(dx / norm), "y": clamp(dy / norm)},
                "aim": {"x": clamp(dx / norm), "y": clamp(dy / norm)},
                "shoot": bool(enemy.get("alive", False)),
                "kick": False,
                "pickup": True,
                "drop": False,
                "throw": False,
            }

        writer.write(json.dumps(cmd, ensure_ascii=False) + "\n")
        writer.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggressive sample bot for GAICA local runner")
    parser.add_argument("host", nargs="?", default=None)
    parser.add_argument("port", nargs="?", type=int, default=None)
    args = parser.parse_args()
    print(args)

    host = args.host or os.getenv("GAICA_BOT_HOST", os.getenv("BOT_HOST", "127.0.0.1"))
    raw_port = args.port
    if raw_port is None:
        raw_port = int(os.getenv("GAICA_BOT_PORT", os.getenv("BOT_PORT", "0")) or "0")
    if raw_port <= 0:
        return _run_stream(sys.stdin, sys.stdout)

    with socket.create_connection((host, raw_port)) as sock:
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")
        writer.write(json.dumps({"type": "register", "name": "bot_aggressive", "protocol": "standalone"}) + "\n")
        writer.flush()
        return _run_stream(reader, writer)


if __name__ == "__main__":
    raise SystemExit(main())