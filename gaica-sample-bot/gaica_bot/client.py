from __future__ import annotations

import json
import socket
from typing import Protocol

from gaica_bot.models import (
    BotCommand,
    BotState,
    HelloMessage,
    RoundEndMessage,
    RoundStartMessage,
    TickMessage,
)


class SocketBot(Protocol):
    state: BotState

    def on_hello(self, message: HelloMessage) -> None: ...
    def on_round_start(self, message: RoundStartMessage) -> None: ...
    def on_tick(self, message: TickMessage) -> BotCommand: ...
    def on_round_end(self, message: RoundEndMessage) -> None: ...


def _registration_payload(bot: SocketBot) -> dict[str, str]:
    raw_name = getattr(bot, "name", "") or bot.__class__.__name__
    name = str(raw_name).strip()[:40]
    if not name:
        name = "gaica-bot"
    return {
        "type": "register",
        "name": name,
        "protocol": "standalone",
    }


def run_socket_bot(host: str, port: int, bot: SocketBot) -> int:
    with socket.create_connection((host, port), timeout=15) as sock:
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")
        writer.write(json.dumps(_registration_payload(bot), ensure_ascii=False) + "\n")
        writer.flush()

        for line in reader:
            raw = line.strip()
            if not raw:
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            message_type = str(payload.get("type") or "")
            if message_type == "hello":
                bot.on_hello(HelloMessage.from_payload(payload))
                continue
            if message_type == "round_start":
                bot.on_round_start(RoundStartMessage.from_payload(payload))
                continue
            if message_type == "round_end":
                bot.on_round_end(RoundEndMessage.from_payload(payload))
                continue
            if message_type != "tick":
                continue

            command = bot.on_tick(TickMessage.from_payload(payload))
            writer.write(json.dumps(command.to_payload(), ensure_ascii=False) + "\n")
            writer.flush()

    return 0
