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
        sock.settimeout(15)

        payload = json.dumps(_registration_payload(bot), ensure_ascii=False) + "\n"
        sock.sendall(payload.encode("utf-8"))

        buffer = b""
        while True:
            try:
                data = sock.recv(4096)
                if not data:
                    break
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    raw = line.decode("utf-8").strip()
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
                        print("ROUND STARTED" +"="*25)
                        bot.on_round_start(RoundStartMessage.from_payload(payload))
                        continue
                    if message_type == "round_end":
                        print("ROUND ENDED" +"="*25)
                        bot.on_round_end(RoundEndMessage.from_payload(payload))
                        continue
                    if message_type != "tick":
                        continue

                    command = bot.on_tick(TickMessage.from_payload(payload))
                    response = json.dumps(command.to_payload(), ensure_ascii=False) + "\n"
                    sock.sendall(response.encode("utf-8"))
            except socket.timeout:
                break

    return 0
