from __future__ import annotations

import json
import socketserver
import socket
import threading

from .coordinator import MatchCoordinator


class _BotTCPHandler(socketserver.StreamRequestHandler):
    coordinator: MatchCoordinator

    def handle(self) -> None:
        player_id = self.coordinator.connect_bot(self.wfile)
        if player_id is None:
            try:
                self.wfile.write(b'{"type":"error","message":"server full"}\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError, socket.timeout):
                pass
            return

        try:
            registered = False
            while True:
                try:
                    raw = self.rfile.readline()
                except (BrokenPipeError, ConnectionResetError, OSError, socket.timeout):
                    break
                if not raw:
                    break

                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                msg_type = payload.get("type", "command")
                if msg_type == "register":
                    self.coordinator.register_bot(player_id, payload)
                    registered = True
                    continue
                if msg_type != "command":
                    continue
                if not registered:
                    try:
                        self.wfile.write(b'{"type":"error","message":"register required before command"}\n')
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError, socket.timeout):
                        pass
                    break

                self.coordinator.update_command(player_id, payload)
        finally:
            self.coordinator.disconnect_bot(player_id)


class BotTCPServer:
    def __init__(self, host: str, port: int, coordinator: MatchCoordinator) -> None:
        self.host = host
        self.port = port
        self.coordinator = coordinator

        class _Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True

        self._server: socketserver.ThreadingTCPServer = _Server((host, port), _BotTCPHandler)
        _BotTCPHandler.coordinator = coordinator
        self.actual_host, self.actual_port = self._server.server_address

        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._server.serve_forever, name="bot-tcp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
