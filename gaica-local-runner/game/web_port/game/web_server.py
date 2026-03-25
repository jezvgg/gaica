from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
from pathlib import Path
from urllib.parse import unquote, urlparse
import threading

from .coordinator import MatchCoordinator


class _WebHandler(BaseHTTPRequestHandler):
    coordinator: MatchCoordinator
    static_dir: Path
    assets_dir: Path

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        # Keep stdout clean.
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/state":
            snapshot = self.coordinator.get_snapshot()
            body = json.dumps(snapshot).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._safe_write(body)
            return

        if path == "/favicon.ico":
            # Browsers request it automatically; ignore when absent.
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        if path == "/":
            return self._serve_file(self.static_dir / "index.html")

        if path.startswith("/assets/"):
            relative = path.removeprefix("/assets/")
            return self._serve_file(self.assets_dir / relative)

        relative = path.removeprefix("/")
        return self._serve_file(self.static_dir / relative)

    def _serve_file(self, file_path: Path) -> None:
        base = self.assets_dir if str(file_path).startswith(str(self.assets_dir)) else self.static_dir

        try:
            resolved = file_path.resolve(strict=True)
        except FileNotFoundError:
            self._safe_send_error(HTTPStatus.NOT_FOUND)
            return

        if not str(resolved).startswith(str(base.resolve())):
            self._safe_send_error(HTTPStatus.FORBIDDEN)
            return

        data = resolved.read_bytes()
        mime, _ = mimetypes.guess_type(str(resolved))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self._safe_write(data)

    def _safe_send_error(self, status: HTTPStatus) -> None:
        try:
            self.send_error(status)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return

    def _safe_write(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return


class WebServer:
    def __init__(
        self,
        host: str,
        port: int,
        coordinator: MatchCoordinator,
        static_dir: Path,
        assets_dir: Path,
    ) -> None:
        self.host = host
        self.port = port

        class _Server(ThreadingHTTPServer):
            allow_reuse_address = True

        _WebHandler.coordinator = coordinator
        _WebHandler.static_dir = static_dir
        _WebHandler.assets_dir = assets_dir

        self._server: ThreadingHTTPServer = _Server((host, port), _WebHandler)
        self.actual_host, self.actual_port = self._server.server_address
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._server.serve_forever, name="web-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
