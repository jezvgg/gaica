from __future__ import annotations

import json
import os
import random
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from runner.protocol import (
    BotCommand,
    build_standalone_hello_message,
    build_standalone_round_end_message,
    build_standalone_round_start_message,
    build_standalone_tick_message,
    parse_bot_command,
)

ROUND_END_HOLD_SECONDS = 0.65
_BOT_LIBRARY_THREAD_LIMIT_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
    "TORCH_NUM_THREADS",
    "TORCH_NUM_INTEROP_THREADS",
)
_BOT_GPU_DISABLE_ENV = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "HIP_VISIBLE_DEVICES": "-1",
    "ROCR_VISIBLE_DEVICES": "-1",
}


def _bool_from_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_or(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _int_or(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _point_from_seq(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _float_or(value[0], 0.0), _float_or(value[1], 0.0)
    return 0.0, 0.0


def _point_dict(value: Any) -> dict[str, float]:
    x, y = _point_from_seq(value)
    return {"x": x, "y": y}


def _apply_bot_runtime_resource_env(env: dict[str, str]) -> None:
    thread_limit = os.getenv("GAICA_BOT_LIBRARY_THREAD_LIMIT", "1").strip() or "1"
    for name in _BOT_LIBRARY_THREAD_LIMIT_ENV:
        env[name] = thread_limit

    for name, default in _BOT_GPU_DISABLE_ENV.items():
        env[name] = os.getenv(f"GAICA_{name}", default).strip() or default


def _command_to_player_command(command: BotCommand, player_command_cls: Any, vec2_cls: Any) -> Any:
    return player_command_cls(
        seq=max(0, _int_or(getattr(command, "seq", 0), 0)),
        move=vec2_cls(command.move.x, command.move.y),
        aim=vec2_cls(command.aim.x, command.aim.y),
        shoot=bool(command.shoot),
        kick=bool(command.kick),
        pickup=bool(command.pickup),
        drop=bool(command.drop),
        throw=bool(command.throw),
    )


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for info in members:
            name = info.filename
            path = PurePosixPath(name)
            if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
                raise RuntimeError(f"Unsafe zip path detected: {name}")
            mode = (int(info.external_attr) >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if file_type and file_type not in {stat.S_IFREG, stat.S_IFDIR}:
                raise RuntimeError(f"Unsafe zip member type detected: {name}")
        archive.extractall(destination)

    main_py = destination / "main.py"
    if not main_py.exists() or not main_py.is_file():
        raise RuntimeError("Bot archive must contain main.py at zip root")


@dataclass(slots=True)
class BotRuntimeError:
    slot: int
    message: str


class BotProcess:
    def __init__(
        self,
        *,
        slot: int,
        work_dir: Path,
        stderr_path: Path,
        max_cpu_seconds: float,
        tick_response_timeout_seconds: float = 1.0,
        match_response_budget_seconds: float = 60.0,
    ) -> None:
        self.slot = slot
        self.work_dir = work_dir
        self.stderr_path = stderr_path
        self.max_cpu_seconds = max_cpu_seconds
        self.tick_response_timeout_seconds = max(0.01, float(tick_response_timeout_seconds))
        self.match_response_budget_seconds = max(
            self.tick_response_timeout_seconds,
            float(match_response_budget_seconds),
        )

        self._process: subprocess.Popen[str] | None = None
        self._stderr_fp = None
        self._reader_thread: threading.Thread | None = None
        self._latest_command = BotCommand.default()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._runtime_error: str | None = None
        self._started_at = 0.0
        self._ended_at = 0.0
        self._server_socket: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._socket_reader = None
        self._socket_writer = None
        self._connected = threading.Event()
        self._registered = threading.Event()
        self._terminating = False
        self._command_version = 0
        self._response_wait_seconds = 0.0
        self._failure_started_at = 0.0

    def start(self) -> None:
        env = dict(os.environ)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(min(10.0, max(1.0, self.match_response_budget_seconds)))
        host, port = listener.getsockname()
        self._server_socket = listener

        env["GAICA_BOT_HOST"] = host
        env["BOT_HOST"] = host
        env["GAICA_BOT_PORT"] = str(port)
        env["BOT_PORT"] = str(port)
        env["PYTHONUNBUFFERED"] = "1"
        env["GAICA_BOT_SLOT"] = str(self.slot)
        _apply_bot_runtime_resource_env(env)

        self._stderr_fp = self.stderr_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            [sys.executable, "main.py", host, str(port)],
            cwd=str(self.work_dir),
            stdin=subprocess.DEVNULL,
            stdout=self._stderr_fp,
            stderr=self._stderr_fp,
            text=True,
            bufsize=1,
            env=env,
        )
        self._started_at = time.monotonic()
        self._reader_thread = threading.Thread(target=self._read_socket_loop, name=f"bot-{self.slot}-socket", daemon=True)
        self._reader_thread.start()

    def _mark_runtime_error_locked(self, message: str) -> None:
        if self._runtime_error is None:
            self._runtime_error = message[:1500]
            if self._failure_started_at <= 0.0:
                self._failure_started_at = time.monotonic()

    def _mark_process_exit_locked(self, return_code: int | None) -> None:
        if return_code is None:
            return
        if self._ended_at <= 0.0:
            self._ended_at = time.monotonic()
        if return_code == 0:
            self._mark_runtime_error_locked("bot disconnected from control socket")
        else:
            self._mark_runtime_error_locked(f"bot exited with code {return_code}")

    def _set_error(self, message: str) -> None:
        with self._condition:
            self._mark_runtime_error_locked(message)
            self._condition.notify_all()

    def _looks_like_command(self, payload: dict[str, Any]) -> bool:
        if payload.get("type") == "command":
            return True
        return any(
            key in payload
            for key in ("move", "aim", "shoot", "kick", "pickup", "drop", "throw")
        )

    def _read_socket_loop(self) -> None:
        process = self._process
        listener = self._server_socket
        if process is None or listener is None:
            return

        try:
            conn, _ = listener.accept()
        except socket.timeout:
            if process.poll() is None and not self._terminating:
                self._set_error("bot failed to connect to control socket")
            return
        except OSError:
            return
        finally:
            try:
                listener.close()
            except OSError:
                pass
            self._server_socket = None

        self._connection = conn
        self._socket_reader = conn.makefile("r", encoding="utf-8", newline="\n")
        self._socket_writer = conn.makefile("w", encoding="utf-8", newline="\n")
        self._connected.set()

        while True:
            try:
                line = self._socket_reader.readline()
            except (OSError, ValueError):
                break
            if line == "":
                break
            payload_raw = line.strip()
            if not payload_raw:
                continue
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                self._set_error("bot returned non-JSON command")
                continue
            if not isinstance(payload, dict):
                self._set_error("bot returned command with unsupported type")
                continue
            if payload.get("type") == "register":
                self._registered.set()
                continue
            if not self._registered.is_set():
                self._set_error("bot must send register before any commands")
                break
            if not self._looks_like_command(payload):
                continue
            has_explicit_seq = payload.get("seq") is not None or payload.get("tick") is not None
            command = parse_bot_command(payload)
            with self._condition:
                self._command_version += 1
                if not has_explicit_seq:
                    command.seq = self._command_version
                self._latest_command = command
                self._condition.notify_all()

        rc = process.poll()
        if rc is not None and self._ended_at <= 0.0:
            self._ended_at = time.monotonic()
        if rc not in (None, 0):
            self._set_error(f"bot exited with code {rc}")
        elif not self._terminating:
            self._set_error("bot disconnected from control socket")

    def get_command(self) -> BotCommand:
        with self._lock:
            return self._latest_command

    def command_version(self) -> int:
        with self._lock:
            return self._command_version

    def reset_latest_command(self) -> None:
        with self._condition:
            self._latest_command = BotCommand.default()
            self._condition.notify_all()

    def wait_for_registration(self, *, timeout: float) -> bool:
        timeout = max(0.01, float(timeout))
        started = time.monotonic()
        if not self._connected.wait(timeout=timeout):
            process = self._process
            if process is not None and process.poll() is not None:
                self._set_error(f"bot exited with code {process.returncode}")
            else:
                self._set_error("bot failed to connect to control socket")
            return False

        remaining = max(0.0, timeout - (time.monotonic() - started))
        if self._registered.wait(timeout=remaining):
            return True

        process = self._process
        if self.runtime_error() is not None:
            return False
        if process is not None and process.poll() is not None:
            self._set_error(f"bot exited with code {process.returncode}")
        else:
            self._set_error("bot did not send required register handshake")
        return False

    def wait_for_command(self, *, after_version: int, timeout: float) -> tuple[int, BotCommand]:
        timeout = max(0.0, min(float(timeout), self.tick_response_timeout_seconds))
        should_terminate = False
        with self._condition:
            remaining_budget = max(0.0, self.match_response_budget_seconds - self._response_wait_seconds)
            if remaining_budget <= 0.0:
                if self._runtime_error is None:
                    self._mark_runtime_error_locked(
                        "bot response wait budget exceeded "
                        f"({self.match_response_budget_seconds:.2f}s)"
                    )
                    should_terminate = True
                    self._condition.notify_all()
                version = self._command_version
                latest = self._latest_command
            else:
                deadline = time.monotonic() + min(timeout, remaining_budget)
                started_waiting_at = time.monotonic()
                while self._command_version <= after_version:
                    process = self._process
                    if self._runtime_error is not None:
                        break
                    if process is not None and process.poll() is not None:
                        self._mark_process_exit_locked(process.returncode)
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    self._condition.wait(timeout=remaining)
                self._response_wait_seconds += max(0.0, time.monotonic() - started_waiting_at)
                if (
                    self._command_version <= after_version
                    and self._response_wait_seconds >= self.match_response_budget_seconds
                    and self._runtime_error is None
                ):
                    self._mark_runtime_error_locked(
                        "bot response wait budget exceeded "
                        f"({self.match_response_budget_seconds:.2f}s)"
                    )
                    should_terminate = True
                    self._condition.notify_all()
                version = self._command_version
                latest = self._latest_command
        if should_terminate:
            self.terminate(force=True)
        return version, latest

    def send_message(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None:
            self._set_error("bot process is not initialized")
            return
        if process.poll() is not None:
            self._set_error(f"bot exited with code {process.returncode}")
            return

        if not self._connected.wait(timeout=min(10.0, max(1.0, self.match_response_budget_seconds))):
            if process.poll() is not None:
                self._set_error(f"bot exited with code {process.returncode}")
            else:
                self._set_error("bot failed to connect to control socket")
            return

        writer = self._socket_writer
        if writer is None:
            self._set_error("bot control socket is not initialized")
            return

        try:
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            writer.flush()
        except OSError:
            self._set_error("failed to write message to bot socket")

    def runtime_error(self) -> str | None:
        with self._lock:
            return self._runtime_error

    def failure_started_at(self) -> float | None:
        process = self._process
        with self._lock:
            if self._failure_started_at > 0.0:
                return self._failure_started_at
            if process is not None and process.poll() is not None:
                self._failure_started_at = self._ended_at or time.monotonic()
                return self._failure_started_at
            return None

    def response_wait_seconds(self) -> float:
        with self._lock:
            return max(0.0, self._response_wait_seconds)

    def process_wall_time_seconds(self) -> float:
        if self._started_at <= 0.0:
            return 0.0
        end_time = self._ended_at or time.monotonic()
        return max(0.0, end_time - self._started_at)

    def is_alive(self) -> bool:
        process = self._process
        if process is None:
            return False
        return_code = process.poll()
        if return_code is None:
            return True
        with self._condition:
            self._mark_process_exit_locked(return_code)
            self._condition.notify_all()
        return False

    def terminate(self, *, force: bool = False) -> None:
        process = self._process
        self._terminating = True

        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass

        if process is None:
            return

        if process.poll() is None:
            if force:
                process.kill()
            else:
                process.terminate()

        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        if self._ended_at <= 0.0:
            self._ended_at = time.monotonic()

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)

        with self._condition:
            self._condition.notify_all()

        if self._socket_reader is not None:
            try:
                self._socket_reader.close()
            except (OSError, ValueError):
                pass
        if self._socket_writer is not None:
            try:
                self._socket_writer.close()
            except (OSError, ValueError):
                pass
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:
                pass
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._stderr_fp is not None:
            self._stderr_fp.close()


class ExternalBotSession:
    def __init__(
        self,
        *,
        slot: int,
        bind_host: str,
        bind_port: int,
        tick_response_timeout_seconds: float = 1.0,
        match_response_budget_seconds: float = 60.0,
    ) -> None:
        self.slot = slot
        self.bind_host = bind_host
        self.bind_port = max(1, int(bind_port))
        self.tick_response_timeout_seconds = max(0.01, float(tick_response_timeout_seconds))
        self.match_response_budget_seconds = max(
            self.tick_response_timeout_seconds,
            float(match_response_budget_seconds),
        )

        self._latest_command = BotCommand.default()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._runtime_error: str | None = None
        self._started_at = 0.0
        self._ended_at = 0.0
        self._server_socket: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._socket_reader = None
        self._socket_writer = None
        self._connected = threading.Event()
        self._registered = threading.Event()
        self._terminating = False
        self._command_version = 0
        self._response_wait_seconds = 0.0
        self._reader_thread: threading.Thread | None = None
        self._failure_started_at = 0.0

    def start(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.bind_host, self.bind_port))
        listener.listen(1)
        self._server_socket = listener
        self._started_at = time.monotonic()
        self._reader_thread = threading.Thread(
            target=self._read_socket_loop,
            name=f"external-bot-{self.slot}-socket",
            daemon=True,
        )
        self._reader_thread.start()

    def endpoint(self) -> tuple[str, int]:
        return self.bind_host, self.bind_port

    def _mark_runtime_error_locked(self, message: str) -> None:
        if self._runtime_error is None:
            self._runtime_error = message[:1500]
            if self._failure_started_at <= 0.0:
                self._failure_started_at = time.monotonic()
            if self._ended_at <= 0.0:
                self._ended_at = self._failure_started_at

    def _set_error(self, message: str) -> None:
        with self._condition:
            self._mark_runtime_error_locked(message)
            self._condition.notify_all()

    def _looks_like_command(self, payload: dict[str, Any]) -> bool:
        if payload.get("type") == "command":
            return True
        return any(
            key in payload
            for key in ("move", "aim", "shoot", "kick", "pickup", "drop", "throw")
        )

    def _read_socket_loop(self) -> None:
        listener = self._server_socket
        if listener is None:
            return

        try:
            conn, _ = listener.accept()
        except OSError:
            return
        finally:
            try:
                listener.close()
            except OSError:
                pass
            self._server_socket = None

        self._connection = conn
        self._socket_reader = conn.makefile("r", encoding="utf-8", newline="\n")
        self._socket_writer = conn.makefile("w", encoding="utf-8", newline="\n")
        self._connected.set()

        while True:
            try:
                line = self._socket_reader.readline()
            except (OSError, ValueError):
                break
            if line == "":
                break
            payload_raw = line.strip()
            if not payload_raw:
                continue
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                self._set_error("bot returned non-JSON command")
                continue
            if not isinstance(payload, dict):
                self._set_error("bot returned command with unsupported type")
                continue
            if payload.get("type") == "register":
                self._registered.set()
                continue
            if not self._registered.is_set():
                self._set_error("bot must send register before any commands")
                break
            if not self._looks_like_command(payload):
                continue
            has_explicit_seq = payload.get("seq") is not None or payload.get("tick") is not None
            command = parse_bot_command(payload)
            with self._condition:
                self._command_version += 1
                if not has_explicit_seq:
                    command.seq = self._command_version
                self._latest_command = command
                self._condition.notify_all()

        if not self._terminating:
            self._set_error("bot disconnected from control socket")

    def get_command(self) -> BotCommand:
        with self._lock:
            return self._latest_command

    def command_version(self) -> int:
        with self._lock:
            return self._command_version

    def reset_latest_command(self) -> None:
        with self._condition:
            self._latest_command = BotCommand.default()
            self._condition.notify_all()

    def wait_for_registration(self, *, timeout: float) -> bool:
        timeout = max(0.01, float(timeout))
        started = time.monotonic()
        if not self._connected.wait(timeout=timeout):
            self._set_error("bot failed to connect to control socket")
            return False

        remaining = max(0.0, timeout - (time.monotonic() - started))
        if self._registered.wait(timeout=remaining):
            return True

        if self.runtime_error() is not None:
            return False
        self._set_error("bot did not send required register handshake")
        return False

    def wait_for_command(self, *, after_version: int, timeout: float) -> tuple[int, BotCommand]:
        timeout = max(0.0, min(float(timeout), self.tick_response_timeout_seconds))
        with self._condition:
            remaining_budget = max(0.0, self.match_response_budget_seconds - self._response_wait_seconds)
            if remaining_budget <= 0.0:
                if self._runtime_error is None:
                    self._mark_runtime_error_locked(
                        "bot response wait budget exceeded "
                        f"({self.match_response_budget_seconds:.2f}s)"
                    )
                    self._condition.notify_all()
                return self._command_version, self._latest_command

            deadline = time.monotonic() + min(timeout, remaining_budget)
            started_waiting_at = time.monotonic()
            while self._command_version <= after_version:
                if self._runtime_error is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            self._response_wait_seconds += max(0.0, time.monotonic() - started_waiting_at)
            if (
                self._command_version <= after_version
                and self._response_wait_seconds >= self.match_response_budget_seconds
                and self._runtime_error is None
            ):
                self._mark_runtime_error_locked(
                    "bot response wait budget exceeded "
                    f"({self.match_response_budget_seconds:.2f}s)"
                )
                self._condition.notify_all()
            return self._command_version, self._latest_command

    def send_message(self, payload: dict[str, Any]) -> None:
        if not self._connected.wait(timeout=min(10.0, max(1.0, self.match_response_budget_seconds))):
            self._set_error("bot failed to connect to control socket")
            return

        writer = self._socket_writer
        if writer is None:
            self._set_error("bot control socket is not initialized")
            return

        try:
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            writer.flush()
        except OSError:
            self._set_error("failed to write message to bot socket")

    def runtime_error(self) -> str | None:
        with self._lock:
            return self._runtime_error

    def failure_started_at(self) -> float | None:
        with self._lock:
            if self._failure_started_at > 0.0:
                return self._failure_started_at
            if self._ended_at > 0.0:
                self._failure_started_at = self._ended_at
                return self._failure_started_at
            return None

    def response_wait_seconds(self) -> float:
        with self._lock:
            return max(0.0, self._response_wait_seconds)

    def process_wall_time_seconds(self) -> float:
        if self._started_at <= 0.0:
            return 0.0
        end_time = self._ended_at or time.monotonic()
        return max(0.0, end_time - self._started_at)

    def is_alive(self) -> bool:
        return not self._terminating and self.runtime_error() is None

    def terminate(self, *, force: bool = False) -> None:
        del force
        self._terminating = True

        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        if self._socket_reader is not None:
            try:
                self._socket_reader.close()
            except (OSError, ValueError):
                pass
        if self._socket_writer is not None:
            try:
                self._socket_writer.close()
            except (OSError, ValueError):
                pass
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:
                pass
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._ended_at <= 0.0:
            self._ended_at = time.monotonic()
        with self._condition:
            self._condition.notify_all()


def _import_web_port(game_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    game_root = game_root.resolve()
    import_roots = [game_root, game_root / "web_port"]
    for import_root in reversed(import_roots):
        import_root_str = str(import_root)
        if import_root.exists() and import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)

    from web_port.game import config as web_config
    from web_port.game.level_loader import get_levels_count, load_level
    from web_port.game.models import PlayerCommand, Vec2
    from web_port.game.simulation import GameSimulation

    return web_config, get_levels_count, load_level, GameSimulation, (PlayerCommand, Vec2)


def _resolve_game_root() -> Path:
    env_path = os.getenv("GAICA_GAME_ROOT", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[2] / "game",
            Path("/platform/game"),
            Path.cwd() / "game",
        ]
    )

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()
        except OSError:
            continue

    raise RuntimeError("Unable to resolve game root with web_port assets")


def _material_for_kind(kind: str) -> str:
    lowered = kind.lower()
    if lowered == "glass":
        return "Glass"
    if lowered in {"box", "letterbox"}:
        return "Wood"
    if lowered in {"wall", "door"}:
        return "Metal"
    return "Metal"


def _weapon_payload(raw_weapon: Any, *, player_id: int) -> dict[str, Any]:
    if not isinstance(raw_weapon, dict):
        return {
            "id": f"p{player_id}-none",
            "kind": "none",
            "variant": "none",
            "ammo": 0,
            "cooldowns": {"fire": 0.0, "kick": 0.0},
        }

    weapon_type = str(raw_weapon.get("type", "revolver")).strip().lower()
    ammo = max(0, _int_or(raw_weapon.get("ammo"), 0))
    return {
        "id": f"p{player_id}-{weapon_type}",
        "kind": "ranged",
        "variant": weapon_type,
        "ammo": ammo,
        "cooldowns": {"fire": 0.0, "kick": 0.0},
    }


def _snapshot_players(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    players: dict[int, dict[str, Any]] = {}
    for payload in snapshot.get("players", []):
        player_id = _int_or(payload.get("id"), 0)
        if player_id not in {1, 2}:
            continue
        players[player_id] = payload
    return players


def _build_map_state(snapshot: dict[str, Any], level_identifier: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    level_payload = snapshot.get("level") or {}
    obstacles = []
    replay_obstacles = []
    for obstacle in snapshot.get("obstacles", []):
        center = _point_dict(obstacle.get("center"))
        kind = str(obstacle.get("kind", "wall"))
        half_size = _point_dict(obstacle.get("half_size"))
        obstacle_payload = {
            "entity": _int_or(obstacle.get("id"), 0),
            "position": center,
            "half_size": half_size,
            "solid": bool(obstacle.get("solid", True)),
            "kind": kind,
            "material": _material_for_kind(kind),
        }
        obstacles.append(obstacle_payload)
        replay_obstacles.append(obstacle_payload)

    map_state = {
        "id": level_identifier,
        "width": _float_or(level_payload.get("width"), 0.0),
        "height": _float_or(level_payload.get("height"), 0.0),
        "floor_tiles": list(level_payload.get("floor_tiles", [])),
        "top_tiles": list(level_payload.get("top_tiles", [])),
        "small_tiles": list(level_payload.get("small_tiles", [])),
        "player_spawns": list(level_payload.get("player_spawns", [])),
        "obstacles": obstacles,
    }
    return map_state, replay_obstacles


def _floor_layer_payload(level_payload: dict[str, Any]) -> dict[str, Any]:
    raw_tiles = level_payload.get("floor_tiles") or []
    width = max(0.0, _float_or(level_payload.get("width"), 0.0))
    height = max(0.0, _float_or(level_payload.get("height"), 0.0))
    if not raw_tiles:
        return {"grid_size": 64, "cells": []}

    first_size = max(1, _int_or(raw_tiles[0].get("size"), 64))
    cells: set[tuple[int, int]] = set()
    for tile in raw_tiles:
        tile_x = _int_or(tile.get("x"), 0)
        tile_y = _int_or(tile.get("y"), 0)
        if tile_x < 0 or tile_y < 0 or tile_x >= width or tile_y >= height:
            continue
        cells.add((tile_x // first_size, tile_y // first_size))

    return {
        "grid_size": first_size,
        "cells": [[cell_x, cell_y] for cell_x, cell_y in sorted(cells, key=lambda item: (item[1], item[0]))],
    }


def _build_standalone_level_payload(snapshot: dict[str, Any], level_identifier: str) -> dict[str, Any]:
    level_payload = snapshot.get("level") or {}
    static_obstacles = []
    for obstacle in snapshot.get("obstacles", []):
        static_obstacles.append(
            {
                "id": _int_or(obstacle.get("id"), 0),
                "kind": str(obstacle.get("kind") or ""),
                "center": list(obstacle.get("center") or [0.0, 0.0]),
                "half_size": list(obstacle.get("half_size") or [0.0, 0.0]),
                "solid": bool(obstacle.get("solid", True)),
            }
        )

    return {
        "identifier": level_identifier,
        "width": _float_or(level_payload.get("width"), 0.0),
        "height": _float_or(level_payload.get("height"), 0.0),
        "floor": _floor_layer_payload(level_payload),
        "player_spawns": [list(item) for item in (level_payload.get("player_spawns") or [])],
        "static_obstacles": static_obstacles,
    }


def _build_standalone_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": str(snapshot.get("status") or ""),
        "tick": _int_or(snapshot.get("tick"), 0),
        "time_seconds": _float_or(snapshot.get("time_seconds"), 0.0),
        "time_limit_seconds": _float_or(snapshot.get("time_limit_seconds"), 0.0),
        "result": None if snapshot.get("result") is None else dict(snapshot.get("result") or {}),
        "players": [dict(player) for player in (snapshot.get("players") or []) if isinstance(player, dict)],
        "pickups": [dict(item) for item in (snapshot.get("pickups") or []) if isinstance(item, dict)],
        "projectiles": [dict(item) for item in (snapshot.get("projectiles") or []) if isinstance(item, dict)],
        "obstacles": [dict(item) for item in (snapshot.get("obstacles") or []) if isinstance(item, dict)],
        "breakables": [dict(item) for item in (snapshot.get("breakables") or []) if isinstance(item, dict)],
        "letterboxes": [dict(item) for item in (snapshot.get("letterboxes") or []) if isinstance(item, dict)],
    }


def _build_series_payload(
    *,
    round_index: int,
    total_rounds: int,
    completed_rounds: int,
    series_score: dict[str, int],
) -> dict[str, Any]:
    current_round = 1 if total_rounds <= 1 else max(1, min(total_rounds, round_index))
    return {
        "enabled": total_rounds > 1,
        "round_index": current_round,
        "total_rounds": total_rounds,
        "completed_rounds": max(0, completed_rounds),
        "score": {
            "1": int(series_score.get("1", 0)),
            "2": int(series_score.get("2", 0)),
        },
    }


def _build_standalone_round_result(
    *,
    round_result: dict[str, Any],
    round_index: int,
    total_rounds: int,
    level_identifier: str,
    series_score: dict[str, int],
) -> dict[str, Any]:
    result_payload = dict(round_result)
    if total_rounds <= 1:
        return result_payload

    result_payload["series_round"] = round_index
    result_payload["series_total_rounds"] = total_rounds
    result_payload["series_score"] = {
        "1": int(series_score.get("1", 0)),
        "2": int(series_score.get("2", 0)),
    }
    result_payload["series_finished"] = round_index >= total_rounds
    result_payload["level_identifier"] = level_identifier

    return result_payload


def _build_pickups(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in snapshot.get("pickups", []):
        payloads.append(
            {
                "entity": _int_or(item.get("id"), 0),
                "kind": "weapon",
                "position": _point_dict(item.get("position")),
                "weapon": {
                    "variant": str(item.get("type", "revolver")).strip().lower(),
                    "ammo": max(0, _int_or(item.get("ammo"), 0)),
                },
            }
        )
    return payloads


def _build_projectiles(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in snapshot.get("projectiles", []):
        payloads.append(
            {
                "entity": _int_or(item.get("id"), 0),
                "owner": _int_or(item.get("owner"), 0),
                "kind": str(item.get("type", "bullet")).strip().lower(),
                "position": _point_dict(item.get("position")),
                "velocity": _point_dict(item.get("velocity")),
                "remaining_life": _float_or(item.get("remaining_life"), 0.0),
            }
        )
    return payloads


def _build_breakables(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in snapshot.get("breakables", []):
        payloads.append(
            {
                "entity": _int_or(item.get("id"), 0),
                "obstacle_entity": _int_or(item.get("obstacle_id"), 0),
                "variant": str(item.get("variant", "unknown")),
                "current": _float_or(item.get("current"), 0.0),
                "threshold": _float_or(item.get("threshold"), 0.0),
                "alive": bool(item.get("alive", True)),
                "position": _point_dict(item.get("center")),
                "half_size": _point_dict(item.get("half_size")),
            }
        )
    return payloads


def _build_interactives(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in snapshot.get("letterboxes", []):
        payloads.append(
            {
                "entity": _int_or(item.get("id"), 0),
                "kind": "letterbox",
                "position": _point_dict(item.get("position")),
                "cooldown": _float_or(item.get("cooldown"), 0.0),
                "ready": bool(item.get("ready", False)),
            }
        )
    return payloads


def _build_runtime_events(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for effect in snapshot.get("effects", []):
        if not isinstance(effect, dict):
            continue
        events.append(dict(effect))
    return events


def _build_state_event(
    *,
    global_tick: int,
    round_index: int,
    map_identifier: str,
    map_index: int,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    players = []
    for player in snapshot.get("players", []):
        pid = _int_or(player.get("id"), 0)
        players.append(
            {
                "entity": pid,
                "id": pid,
                "slot": str(pid),
                "position": _point_dict(player.get("position")),
                "facing": _point_dict(player.get("facing")),
                "hp": 100 if bool(player.get("alive", False)) else 0,
                "alive": bool(player.get("alive", False)),
                "downed": False,
                "color": str(player.get("color") or ""),
                "character": str(player.get("character") or ""),
                "weapon": _weapon_payload(player.get("weapon"), player_id=pid),
            }
        )

    pickups = _build_pickups(snapshot)
    projectiles = _build_projectiles(snapshot)
    breakables = _build_breakables(snapshot)
    interactives = _build_interactives(snapshot)
    sliding_doors = []
    for obstacle in snapshot.get("obstacles", []):
        if str(obstacle.get("kind", "")).lower() != "door":
            continue
        center = _point_dict(obstacle.get("center"))
        sliding_doors.append(
            {
                "entity": _int_or(obstacle.get("id"), 0),
                "position": center,
                "is_closed": bool(obstacle.get("solid", True)),
            }
        )

    return {
        "tick": global_tick,
        "type": "state",
        "series_round": round_index,
        "map_id": map_identifier,
        "map_index": map_index,
        "time_seconds": _float_or(snapshot.get("time_seconds"), 0.0),
        "players": players,
        "projectiles": projectiles,
        "pickups": pickups,
        "interactives": interactives,
        "letterboxes": [
            {
                "entity": item["entity"],
                "position": item["position"],
                "cooldown": item["cooldown"],
                "ready": item["ready"],
            }
            for item in interactives
        ],
        # Keep state frames compact: static geometry is communicated via round_start.
        "obstacles": [],
        "breakables": breakables,
        "sliding_doors": sliding_doors,
        "effects": _build_runtime_events(snapshot),
        "debris": list(snapshot.get("debris", [])),
    }


def _log_line(lines: list[str], message: str) -> None:
    lines.append(message)


def _wait_for_path(path: Path, *, timeout_seconds: float, poll_interval: float = 0.05) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(max(0.01, float(poll_interval)))
    return path.exists()


def _run_series_match_core(
    *,
    output_dir: Path,
    seed: int,
    round_timeout_seconds: float,
    series_rounds: int,
    match_id: str | None,
    tick_response_timeout_seconds: float,
    match_response_budget_seconds: float,
    register_timeout_seconds: float,
    bot_a_runtime: Any,
    bot_b_runtime: Any,
    startup_errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    game_root = _resolve_game_root()
    web_config, get_levels_count, load_level, GameSimulation, models = _import_web_port(game_root)
    PlayerCommand, Vec2 = models

    ldtk_path = game_root / "web_port" / "assets" / "levels" / "test_ldtk_project.ldtk"
    if not ldtk_path.exists() or not ldtk_path.is_file():
        raise RuntimeError(f"LDtk project not found: {ldtk_path}")

    level_count = _int_or(get_levels_count(ldtk_path), 0)
    if level_count <= 0:
        raise RuntimeError("No levels available for series run")

    rng = random.Random(seed)
    all_indices = list(range(level_count))
    rng.shuffle(all_indices)

    requested_rounds = max(1, int(series_rounds))
    if requested_rounds <= len(all_indices):
        selected_indices = all_indices[:requested_rounds]
    else:
        selected_indices = []
        while len(selected_indices) < requested_rounds:
            selected_indices.extend(all_indices)
        selected_indices = selected_indices[:requested_rounds]

    events: list[dict[str, Any]] = []
    log_lines: list[str] = []
    bot_errors: dict[str, str] = {}

    match_started_at = time.monotonic()
    global_tick = 0
    tick_rate = max(1, _int_or(getattr(web_config, "TICK_RATE", 30), 30))
    tick_dt = 1.0 / tick_rate
    round_end_hold_ticks = max(1, int(round(tick_rate * ROUND_END_HOLD_SECONDS)))
    match_wall_limit_seconds = (
        round_timeout_seconds * len(selected_indices)
        + ROUND_END_HOLD_SECONDS * len(selected_indices)
        + (max(0.0, float(match_response_budget_seconds)) * 2.0)
        + 120.0
    )
    command_wait_timeout = max(0.01, float(tick_response_timeout_seconds))
    runtimes = ((1, bot_a_runtime), (2, bot_b_runtime))
    startup_errors_by_slot = {
        str(key): str(value)
        for key, value in (startup_errors or {}).items()
        if value
    }

    def _runtime_failure_message(slot: int, runtime: Any) -> str:
        return (
            startup_errors_by_slot.get(str(slot))
            or runtime.runtime_error()
            or "bot process terminated"
        )

    def _collect_unavailable_slots() -> list[int]:
        unavailable_slots: list[int] = []
        for slot, runtime in runtimes:
            if runtime.is_alive():
                continue
            unavailable_slots.append(slot)
            if str(slot) not in bot_errors:
                bot_errors[str(slot)] = _runtime_failure_message(slot, runtime)
        return unavailable_slots

    def _resolve_forfeit_winner(unavailable_slots: list[int]) -> tuple[int, list[int]] | None:
        if not unavailable_slots:
            return None

        failure_order: list[tuple[float, float, int]] = []
        for slot, runtime in runtimes:
            if slot not in unavailable_slots:
                continue
            failure_started_at = runtime.failure_started_at()
            failure_order.append(
                (
                    runtime.process_wall_time_seconds(),
                    failure_started_at if failure_started_at is not None else float("inf"),
                    slot,
                )
            )

        if not failure_order:
            return None

        failure_order.sort()
        losing_slot = failure_order[0][2]
        winner_slot = 2 if losing_slot == 1 else 1
        return winner_slot, [losing_slot]

    def _append_round_start_event(
        *,
        round_idx: int,
        map_index: int,
        level_identifier: str,
        snapshot: dict[str, Any],
        obstacles: list[dict[str, Any]],
        simulation_seed: int,
    ) -> None:
        events.append(
            {
                "tick": global_tick,
                "type": "round_start",
                "series_round": round_idx,
                "series_total_rounds": len(selected_indices),
                "map_id": level_identifier,
                "map_index": map_index,
                "map_width": _float_or((snapshot.get("level") or {}).get("width"), 0.0),
                "map_height": _float_or((snapshot.get("level") or {}).get("height"), 0.0),
                "obstacles": obstacles,
            }
        )
        _log_line(
            log_lines,
            f"round_start round={round_idx}/{len(selected_indices)} map={level_identifier} map_index={map_index} seed={simulation_seed}",
        )

    def _append_forfeit_round(
        *,
        round_idx: int,
        map_index: int,
        level_identifier: str,
        snapshot: dict[str, Any],
        winner_slot: int,
        reason: str,
        failed_slots: list[int],
        round_started: bool,
        obstacles: list[dict[str, Any]],
        simulation_seed: int,
    ) -> dict[str, Any]:
        nonlocal global_tick, final_hp, final_map_identifier, final_map_index

        if not round_started:
            _append_round_start_event(
                round_idx=round_idx,
                map_index=map_index,
                level_identifier=level_identifier,
                snapshot=snapshot,
                obstacles=obstacles,
                simulation_seed=simulation_seed,
            )

        global_tick += 1
        series_score[str(winner_slot)] += 1
        final_map_identifier = str(level_identifier)
        final_map_index = map_index
        final_hp = {
            "1": 100 if winner_slot == 1 else 0,
            "2": 100 if winner_slot == 2 else 0,
        }
        round_end_event = {
            "tick": global_tick,
            "type": "round_end",
            "series_round": round_idx,
            "series_total_rounds": len(selected_indices),
            "map_id": level_identifier,
            "map_index": map_index,
            "winner_slot": winner_slot,
            "draw": False,
            "reason": reason,
            "hp": dict(final_hp),
            "series_score": dict(series_score),
            "series_finished": round_idx >= len(selected_indices),
        }
        standalone_round_result = _build_standalone_round_result(
            round_result={
                "winner_id": winner_slot,
                "reason": reason,
                "duration_seconds": _float_or(snapshot.get("time_seconds"), 0.0),
            },
            round_index=round_idx,
            total_rounds=len(selected_indices),
            level_identifier=str(level_identifier),
            series_score=series_score,
        )
        events.append(round_end_event)
        round_results.append(
            {
                "series_round": round_idx,
                "series_total_rounds": len(selected_indices),
                "tick": global_tick,
                "map_id": level_identifier,
                "map_index": map_index,
                "winner_slot": winner_slot,
                "draw": False,
                "reason": reason,
                "hp": dict(final_hp),
                "series_score": dict(series_score),
                "series_finished": round_idx >= len(selected_indices),
            }
        )
        _log_line(
            log_lines,
            (
                f"round_end round={round_idx}/{len(selected_indices)} map={level_identifier} "
                f"winner_slot={winner_slot} draw=False reason={reason} "
                f"failed_slots={','.join(str(slot) for slot in failed_slots)} "
                f"score={series_score['1']}:{series_score['2']}"
            ),
        )
        return standalone_round_result

    def _award_forfeit_rounds(
        *,
        start_round_idx: int,
        winner_slot: int,
        reason: str,
        failed_slots: list[int],
    ) -> None:
        for future_round_idx, future_map_index in enumerate(
            selected_indices[start_round_idx - 1 :],
            start=start_round_idx,
        ):
            level = load_level(ldtk_path, level_index=future_map_index)
            simulation_seed = rng.randrange(1, 2_000_000_000)
            simulation = GameSimulation(
                level,
                seed=simulation_seed,
                round_time_limit_seconds=round_timeout_seconds,
            )
            simulation.set_match_characters(match_characters)
            initial_snapshot = simulation.get_snapshot()
            _, round_obstacles = _build_map_state(initial_snapshot, level.identifier)
            _append_forfeit_round(
                round_idx=future_round_idx,
                map_index=future_map_index,
                level_identifier=str(level.identifier),
                snapshot=initial_snapshot,
                winner_slot=winner_slot,
                reason=reason,
                failed_slots=failed_slots,
                round_started=False,
                obstacles=round_obstacles,
                simulation_seed=simulation_seed,
            )

    series_score = {"1": 0, "2": 0}
    round_results: list[dict[str, Any]] = []
    final_hp = {"1": 0, "2": 0}
    final_map_identifier = ""
    final_map_index: int | None = None
    match_characters = GameSimulation.sample_match_characters(rng)
    registration_ok: dict[int, bool] = {}

    for slot, runtime in runtimes:
        registration_ok[slot] = runtime.wait_for_registration(
            timeout=max(0.01, float(register_timeout_seconds)),
        )
        if not registration_ok[slot]:
            bot_errors[str(slot)] = _runtime_failure_message(slot, runtime)

    registered_slots = [slot for slot, ok in registration_ok.items() if ok]
    failed_registration_slots = [slot for slot, ok in registration_ok.items() if not ok]

    if len(registered_slots) == 1 and len(failed_registration_slots) == 1:
        winner_slot = registered_slots[0]
        _log_line(
            log_lines,
            (
                f"series_forfeit winner_slot={winner_slot} "
                f"reason=bot_forfeit failed_slots={failed_registration_slots[0]}"
            ),
        )
        _award_forfeit_rounds(
            start_round_idx=1,
            winner_slot=winner_slot,
            reason="bot_forfeit",
            failed_slots=failed_registration_slots,
        )
    elif not registered_slots:
        failure_messages = "; ".join(
            f"slot {slot}: {_runtime_failure_message(slot, runtime)}"
            for slot, runtime in runtimes
        )
        raise RuntimeError(
            "Both bots failed required register handshake: "
            f"{failure_messages}"
        )
    else:
        for slot, runtime in runtimes:
            runtime.send_message(
                build_standalone_hello_message(
                    player_id=slot,
                    tick_rate=tick_rate,
                )
            )

    try:
        if len(registered_slots) == 2:
            for round_idx, map_index in enumerate(selected_indices, start=1):
                unavailable_slots = _collect_unavailable_slots()
                forfeit_resolution = _resolve_forfeit_winner(unavailable_slots)
                if forfeit_resolution is not None:
                    winner_slot, failed_slots = forfeit_resolution
                    _award_forfeit_rounds(
                        start_round_idx=round_idx,
                        winner_slot=winner_slot,
                        reason="bot_forfeit",
                        failed_slots=failed_slots,
                    )
                    break

                level = load_level(ldtk_path, level_index=map_index)
                simulation_seed = rng.randrange(1, 2_000_000_000)
                simulation = GameSimulation(
                    level,
                    seed=simulation_seed,
                    round_time_limit_seconds=round_timeout_seconds,
                )
                simulation.set_match_characters(match_characters)
                initial_snapshot = simulation.get_snapshot()
                _, round_obstacles = _build_map_state(initial_snapshot, level.identifier)
                level_payload = _build_standalone_level_payload(initial_snapshot, str(level.identifier))
                standalone_series = _build_series_payload(
                    round_index=round_idx,
                    total_rounds=len(selected_indices),
                    completed_rounds=round_idx - 1,
                    series_score=series_score,
                )

                final_map_identifier = str(level.identifier)
                final_map_index = map_index

                _append_round_start_event(
                    round_idx=round_idx,
                    map_index=map_index,
                    level_identifier=str(level.identifier),
                    snapshot=initial_snapshot,
                    obstacles=round_obstacles,
                    simulation_seed=simulation_seed,
                )
                for slot, runtime in runtimes:
                    enemy_slot = 2 if slot == 1 else 1
                    runtime.send_message(
                        build_standalone_round_start_message(
                            player_id=slot,
                            enemy_id=enemy_slot,
                            tick_rate=tick_rate,
                            level=level_payload,
                            series=standalone_series,
                        )
                    )
                    runtime.reset_latest_command()

                hard_round_tick_limit = max(1, int(round_timeout_seconds * tick_rate))
                next_commands = {
                    1: PlayerCommand(),
                    2: PlayerCommand(),
                }
                series_resolved_by_forfeit = False

                while not simulation.is_finished():
                    unavailable_slots = _collect_unavailable_slots()
                    forfeit_resolution = _resolve_forfeit_winner(unavailable_slots)
                    if forfeit_resolution is not None:
                        winner_slot, failed_slots = forfeit_resolution
                        current_snapshot = simulation.get_snapshot()
                        standalone_round_result = _append_forfeit_round(
                            round_idx=round_idx,
                            map_index=map_index,
                            level_identifier=str(level.identifier),
                            snapshot=current_snapshot,
                            winner_slot=winner_slot,
                            reason="bot_forfeit",
                            failed_slots=failed_slots,
                            round_started=True,
                            obstacles=round_obstacles,
                            simulation_seed=simulation_seed,
                        )
                        for _, runtime in runtimes:
                            runtime.send_message(
                                build_standalone_round_end_message(result=standalone_round_result)
                            )
                        if round_idx < len(selected_indices):
                            _award_forfeit_rounds(
                                start_round_idx=round_idx + 1,
                                winner_slot=winner_slot,
                                reason="bot_forfeit",
                                failed_slots=failed_slots,
                            )
                        series_resolved_by_forfeit = True
                        break

                    simulation.step(dict(next_commands), dt=tick_dt)
                    frame_snapshot = simulation.get_snapshot()

                    if simulation.tick >= hard_round_tick_limit and not simulation.is_finished():
                        simulation.result = type("RoundResult", (), {
                            "winner_id": None,
                            "reason": "external_round_timeout",
                            "duration_seconds": _float_or(frame_snapshot.get("time_seconds"), 0.0),
                        })()
                        simulation.status = "finished"
                        frame_snapshot = simulation.get_snapshot()

                    global_tick += 1
                    events.append(
                        _build_state_event(
                            global_tick=global_tick,
                            round_index=round_idx,
                            map_identifier=level.identifier,
                            map_index=map_index,
                            snapshot=frame_snapshot,
                        )
                    )

                    command_versions = {
                        1: bot_a_runtime.command_version(),
                        2: bot_b_runtime.command_version(),
                    }

                    for slot, runtime in runtimes:
                        enemy_slot = 2 if slot == 1 else 1
                        runtime.send_message(
                            build_standalone_tick_message(
                                snapshot=_build_standalone_snapshot(frame_snapshot),
                                player_id=slot,
                                enemy_id=enemy_slot,
                            )
                        )

                    updated_commands = dict(next_commands)
                    for slot, runtime in runtimes:
                        if not runtime.is_alive():
                            continue
                        current_version = command_versions[slot]
                        version, parsed = runtime.wait_for_command(
                            after_version=current_version,
                            timeout=command_wait_timeout,
                        )
                        if version <= current_version:
                            continue
                        updated_commands[slot] = _command_to_player_command(
                            parsed,
                            player_command_cls=PlayerCommand,
                            vec2_cls=Vec2,
                        )
                    next_commands = updated_commands

                    if (time.monotonic() - match_started_at) > match_wall_limit_seconds:
                        raise RuntimeError("match execution exceeded hard wall limit")

                if series_resolved_by_forfeit:
                    break

                final_snapshot = simulation.get_snapshot()
                result_payload = final_snapshot.get("result") or {}
                winner_id = result_payload.get("winner_id")
                winner_slot = _int_or(winner_id, 0) if winner_id is not None else None
                if winner_slot == 1:
                    series_score["1"] += 1
                elif winner_slot == 2:
                    series_score["2"] += 1
                draw = winner_slot is None

                players_by_id = _snapshot_players(final_snapshot)
                final_hp = {
                    "1": 100 if bool((players_by_id.get(1) or {}).get("alive", False)) else 0,
                    "2": 100 if bool((players_by_id.get(2) or {}).get("alive", False)) else 0,
                }

                for _ in range(round_end_hold_ticks):
                    global_tick += 1
                    events.append(
                        _build_state_event(
                            global_tick=global_tick,
                            round_index=round_idx,
                            map_identifier=level.identifier,
                            map_index=map_index,
                            snapshot=final_snapshot,
                        )
                    )

                round_end_event = {
                    "tick": global_tick,
                    "type": "round_end",
                    "series_round": round_idx,
                    "series_total_rounds": len(selected_indices),
                    "map_id": level.identifier,
                    "map_index": map_index,
                    "winner_slot": winner_slot,
                    "draw": draw,
                    "reason": str(result_payload.get("reason", "unknown")),
                    "hp": dict(final_hp),
                    "series_score": dict(series_score),
                    "series_finished": round_idx >= len(selected_indices),
                }
                standalone_round_result = _build_standalone_round_result(
                    round_result=result_payload,
                    round_index=round_idx,
                    total_rounds=len(selected_indices),
                    level_identifier=str(level.identifier),
                    series_score=series_score,
                )
                events.append(round_end_event)
                round_results.append(
                    {
                        "series_round": round_idx,
                        "series_total_rounds": len(selected_indices),
                        "tick": global_tick,
                        "map_id": level.identifier,
                        "map_index": map_index,
                        "winner_slot": winner_slot,
                        "draw": draw,
                        "reason": str(result_payload.get("reason", "unknown")),
                        "hp": dict(final_hp),
                        "series_score": dict(series_score),
                        "series_finished": round_idx >= len(selected_indices),
                    }
                )
                _log_line(
                    log_lines,
                    (
                        f"round_end round={round_idx}/{len(selected_indices)} map={level.identifier} "
                        f"winner_slot={winner_slot} draw={draw} reason={round_end_event['reason']} "
                        f"score={series_score['1']}:{series_score['2']}"
                    ),
                )
                for _, runtime in runtimes:
                    runtime.send_message(
                        build_standalone_round_end_message(result=standalone_round_result)
                    )

        if series_score["1"] > series_score["2"]:
            final_winner_slot = 1
        elif series_score["2"] > series_score["1"]:
            final_winner_slot = 2
        else:
            final_winner_slot = None

        final_draw = final_winner_slot is None
        wall_time_seconds = max(0.0001, time.monotonic() - match_started_at)
        simulated_seconds = global_tick / float(tick_rate)
        speedup = simulated_seconds / wall_time_seconds

        for slot, runtime in runtimes:
            runtime_error = runtime.runtime_error()
            if runtime_error and str(slot) not in bot_errors:
                bot_errors[str(slot)] = runtime_error
            if runtime_error:
                events.append(
                    {
                        "tick": global_tick,
                        "type": "bot_error",
                        "slot": str(slot),
                        "message": runtime_error,
                    }
                )
                _log_line(log_lines, f"bot_error slot={slot} message={runtime_error}")

        bot_timings = {
            "1": {
                "response_wait_seconds": round(bot_a_runtime.response_wait_seconds(), 6),
                "response_wait_budget_seconds": float(match_response_budget_seconds),
                "tick_response_timeout_seconds": float(command_wait_timeout),
                "process_wall_time_seconds": round(bot_a_runtime.process_wall_time_seconds(), 6),
            },
            "2": {
                "response_wait_seconds": round(bot_b_runtime.response_wait_seconds(), 6),
                "response_wait_budget_seconds": float(match_response_budget_seconds),
                "tick_response_timeout_seconds": float(command_wait_timeout),
                "process_wall_time_seconds": round(bot_b_runtime.process_wall_time_seconds(), 6),
            },
        }
        for slot, timing in bot_timings.items():
            _log_line(
                log_lines,
                (
                    f"bot_timing slot={slot} "
                    f"response_wait_seconds={timing['response_wait_seconds']} "
                    f"process_wall_time_seconds={timing['process_wall_time_seconds']}"
                ),
            )

        outcome = {
            "protocol_version": "3.0",
            "runner": "gaica_python_web_runner",
            "draw": final_draw,
            "winner_slot": final_winner_slot,
            "ticks": global_tick,
            "tick_rate": tick_rate,
            "simulated_seconds": simulated_seconds,
            "wall_time_seconds": wall_time_seconds,
            "speedup": speedup,
            "round_end_reason": "series_score",
            "map_id": final_map_identifier,
            "map_index": final_map_index,
            "map_identifier": final_map_identifier,
            "map_iid": None,
            "series_total_rounds": len(selected_indices),
            "series_score": dict(series_score),
            "round_results": round_results,
            "series_levels": [
                {
                    "series_round": i + 1,
                    "map_index": idx,
                }
                for i, idx in enumerate(selected_indices)
            ],
            "hp": dict(final_hp),
            "bot_errors": bot_errors,
            "bot_timings": bot_timings,
            "events_count": len(events),
        }

        replay_payload = {
            "meta": {
                "match_id": match_id or "",
                "seed": seed,
                "runner": outcome["runner"],
                "protocol_version": outcome["protocol_version"],
                "tick_rate": tick_rate,
                "simulated_seconds": simulated_seconds,
                "wall_time_seconds": wall_time_seconds,
                "speedup": speedup,
                "series_total_rounds": len(selected_indices),
                "series_score": dict(series_score),
            },
            "events": events,
            "result": {
                "winner_slot": final_winner_slot,
                "draw": final_draw,
                "reason": "series_score",
                "series_total_rounds": len(selected_indices),
                "series_score": dict(series_score),
                "ticks": global_tick,
                "tick_rate": tick_rate,
                "simulated_seconds": simulated_seconds,
                "wall_time_seconds": wall_time_seconds,
                "speedup": speedup,
                "map_id": final_map_identifier,
                "map_index": final_map_index,
            },
        }

        (output_dir / "replay.json").write_text(
            json.dumps(replay_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "outcome.json").write_text(
            json.dumps(outcome, ensure_ascii=False),
            encoding="utf-8",
        )

        _log_line(
            log_lines,
            (
                f"series_end winner_slot={final_winner_slot} draw={final_draw} "
                f"score={series_score['1']}:{series_score['2']} ticks={global_tick}"
            ),
        )
        (output_dir / "match.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return outcome
    finally:
        graceful_stop = _bool_from_env("GAICA_GRACEFUL_BOT_STOP", default=True)
        bot_a_runtime.terminate(force=not graceful_stop)
        bot_b_runtime.terminate(force=not graceful_stop)


def run_series_match(
    *,
    bot_a_zip: Path,
    bot_b_zip: Path,
    output_dir: Path,
    seed: int,
    round_timeout_seconds: float,
    max_cpu_seconds: float,
    series_rounds: int,
    match_id: str | None,
    tick_response_timeout_seconds: float = 1.0,
    match_response_budget_seconds: float = 60.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gaica_web_series_run_") as temp_dir:
        temp_root = Path(temp_dir)
        bot_a_dir = temp_root / "bot_a"
        bot_b_dir = temp_root / "bot_b"
        bot_a_dir.mkdir(parents=True, exist_ok=True)
        bot_b_dir.mkdir(parents=True, exist_ok=True)

        _safe_extract_zip(bot_a_zip, bot_a_dir)
        _safe_extract_zip(bot_b_zip, bot_b_dir)

        bot_a_runtime = BotProcess(
            slot=1,
            work_dir=bot_a_dir,
            stderr_path=output_dir / "bot_a.stderr.log",
            max_cpu_seconds=max_cpu_seconds,
            tick_response_timeout_seconds=tick_response_timeout_seconds,
            match_response_budget_seconds=match_response_budget_seconds,
        )
        bot_b_runtime = BotProcess(
            slot=2,
            work_dir=bot_b_dir,
            stderr_path=output_dir / "bot_b.stderr.log",
            max_cpu_seconds=max_cpu_seconds,
            tick_response_timeout_seconds=tick_response_timeout_seconds,
            match_response_budget_seconds=match_response_budget_seconds,
        )
        bot_a_runtime.start()
        bot_b_runtime.start()
        return _run_series_match_core(
            output_dir=output_dir,
            seed=seed,
            round_timeout_seconds=round_timeout_seconds,
            series_rounds=series_rounds,
            match_id=match_id,
            tick_response_timeout_seconds=tick_response_timeout_seconds,
            match_response_budget_seconds=match_response_budget_seconds,
            register_timeout_seconds=min(10.0, max(1.0, tick_response_timeout_seconds)),
            bot_a_runtime=bot_a_runtime,
            bot_b_runtime=bot_b_runtime,
            startup_errors=None,
        )


def run_series_match_external(
    *,
    output_dir: Path,
    seed: int,
    round_timeout_seconds: float,
    series_rounds: int,
    match_id: str | None,
    bot_a_bind_host: str,
    bot_a_port: int,
    bot_b_bind_host: str,
    bot_b_port: int,
    tick_response_timeout_seconds: float = 1.0,
    match_response_budget_seconds: float = 60.0,
    register_timeout_seconds: float = 5.0,
    ready_path: Path | None = None,
    start_barrier_path: Path | None = None,
    start_barrier_timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bot_a_runtime = ExternalBotSession(
        slot=1,
        bind_host=bot_a_bind_host,
        bind_port=bot_a_port,
        tick_response_timeout_seconds=tick_response_timeout_seconds,
        match_response_budget_seconds=match_response_budget_seconds,
    )
    bot_b_runtime = ExternalBotSession(
        slot=2,
        bind_host=bot_b_bind_host,
        bind_port=bot_b_port,
        tick_response_timeout_seconds=tick_response_timeout_seconds,
        match_response_budget_seconds=match_response_budget_seconds,
    )
    bot_a_runtime.start()
    bot_b_runtime.start()

    if ready_path is not None:
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_payload = {
            "bot_a": {"host": bot_a_bind_host, "port": int(bot_a_port)},
            "bot_b": {"host": bot_b_bind_host, "port": int(bot_b_port)},
        }
        ready_path.write_text(json.dumps(ready_payload, ensure_ascii=False), encoding="utf-8")

    if start_barrier_path is not None:
        if not _wait_for_path(
            start_barrier_path,
            timeout_seconds=max(0.01, float(start_barrier_timeout_seconds)),
        ):
            raise RuntimeError(
                "Bot containers did not become ready before the register handshake window."
            )
    startup_errors: dict[str, str] = {}
    if start_barrier_path is not None and start_barrier_path.exists():
        try:
            start_barrier_payload = json.loads(start_barrier_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            start_barrier_payload = {}
        if isinstance(start_barrier_payload, dict):
            raw_errors = start_barrier_payload.get("startup_errors") or {}
            if isinstance(raw_errors, dict):
                startup_errors = {
                    str(key): str(value)
                    for key, value in raw_errors.items()
                    if value
                }

    return _run_series_match_core(
        output_dir=output_dir,
        seed=seed,
        round_timeout_seconds=round_timeout_seconds,
        series_rounds=series_rounds,
        match_id=match_id,
        tick_response_timeout_seconds=tick_response_timeout_seconds,
        match_response_budget_seconds=match_response_budget_seconds,
        register_timeout_seconds=max(0.01, float(register_timeout_seconds)),
        bot_a_runtime=bot_a_runtime,
        bot_b_runtime=bot_b_runtime,
        startup_errors=startup_errors,
    )
