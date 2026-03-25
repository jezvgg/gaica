from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Vector2:
    x: float
    y: float


@dataclass(slots=True)
class BotCommand:
    seq: int
    move: Vector2
    aim: Vector2
    shoot: bool
    kick: bool
    pickup: bool
    drop: bool
    throw: bool

    @staticmethod
    def default() -> "BotCommand":
        return BotCommand(
            seq=0,
            move=Vector2(0.0, 0.0),
            aim=Vector2(1.0, 0.0),
            shoot=False,
            kick=False,
            pickup=False,
            drop=False,
            throw=False,
        )


def _parse_vector(payload: Any, *, default_x: float = 0.0, default_y: float = 0.0) -> Vector2:
    if isinstance(payload, dict):
        raw_x = payload.get("x", default_x)
        raw_y = payload.get("y", default_y)
    elif isinstance(payload, (list, tuple)) and len(payload) >= 2:
        raw_x = payload[0]
        raw_y = payload[1]
    else:
        return Vector2(default_x, default_y)

    try:
        x = float(raw_x or default_x)
    except (TypeError, ValueError):
        x = default_x
    try:
        y = float(raw_y or default_y)
    except (TypeError, ValueError):
        y = default_y

    x = max(-1.0, min(1.0, x))
    y = max(-1.0, min(1.0, y))
    return Vector2(x=x, y=y)


def _parse_flag(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key, False)
    return value if isinstance(value, bool) else False


def parse_bot_command(payload: Any) -> BotCommand:
    if not isinstance(payload, dict):
        return BotCommand.default()

    raw_seq = payload.get("seq")
    if raw_seq is None:
        raw_seq = payload.get("tick", 0)
    try:
        seq = int(raw_seq)
    except (TypeError, ValueError):
        seq = 0

    move = _parse_vector(payload.get("move"), default_x=0.0, default_y=0.0)
    aim = _parse_vector(payload.get("aim"), default_x=move.x, default_y=move.y)
    if abs(aim.x) <= 1e-9 and abs(aim.y) <= 1e-9:
        aim = Vector2(move.x, move.y)
    if abs(aim.x) <= 1e-9 and abs(aim.y) <= 1e-9:
        aim = Vector2(1.0, 0.0)

    return BotCommand(
        seq=seq,
        move=move,
        aim=aim,
        shoot=_parse_flag(payload, "shoot"),
        kick=_parse_flag(payload, "kick"),
        pickup=_parse_flag(payload, "pickup"),
        drop=_parse_flag(payload, "drop"),
        throw=_parse_flag(payload, "throw"),
    )


def build_standalone_hello_message(*, player_id: int, tick_rate: int) -> dict[str, Any]:
    return {
        "type": "hello",
        "player_id": int(player_id),
        "tick_rate": int(tick_rate),
    }


def build_standalone_round_start_message(
    *,
    player_id: int,
    enemy_id: int,
    tick_rate: int,
    level: dict[str, Any],
    series: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "round_start",
        "player_id": int(player_id),
        "enemy_id": int(enemy_id),
        "tick_rate": int(tick_rate),
        "level": dict(level),
        "series": dict(series),
    }


def build_standalone_tick_message(
    *,
    snapshot: dict[str, Any],
    player_id: int,
    enemy_id: int,
) -> dict[str, Any]:
    players_by_id = {
        int(player.get("id", 0)): player
        for player in snapshot.get("players", [])
        if isinstance(player, dict)
    }
    return {
        "type": "tick",
        "tick": int(snapshot.get("tick", 0) or 0),
        "time_seconds": float(snapshot.get("time_seconds", 0.0) or 0.0),
        "you": players_by_id.get(int(player_id)),
        "enemy": players_by_id.get(int(enemy_id)),
        "snapshot": dict(snapshot),
    }


def build_standalone_round_end_message(*, result: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "type": "round_end",
        "result": None if result is None else dict(result),
    }
