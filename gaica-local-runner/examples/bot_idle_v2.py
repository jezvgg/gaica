#!/usr/bin/env python3
import argparse
import json
import os
import socket
import sys


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

        tick = int(message.get("tick", 0) or 0)
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
        writer.write(json.dumps(cmd, ensure_ascii=False) + "\n")
        writer.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Idle sample bot for GAICA local runner")
    parser.add_argument("host", nargs="?", default=None)
    parser.add_argument("port", nargs="?", type=int, default=None)
    args = parser.parse_args()

    host = args.host or os.getenv("GAICA_BOT_HOST", os.getenv("BOT_HOST", "127.0.0.1"))
    raw_port = args.port
    if raw_port is None:
        raw_port = int(os.getenv("GAICA_BOT_PORT", os.getenv("BOT_PORT", "0")) or "0")

    if raw_port <= 0:
        return _run_stream(sys.stdin, sys.stdout)

    with socket.create_connection((host, raw_port)) as sock:
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")
        writer.write(json.dumps({"type": "register", "name": "bot_idle", "protocol": "standalone"}) + "\n")
        writer.flush()
        return _run_stream(reader, writer)


if __name__ == "__main__":
    raise SystemExit(main())