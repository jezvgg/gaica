from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    @classmethod
    def from_payload(cls, payload: Any) -> "Vec2":
        if isinstance(payload, dict):
            return cls(
                x=_as_float(payload.get("x"), 0.0),
                y=_as_float(payload.get("y"), 0.0),
            )
        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
            return cls(x=_as_float(payload[0], 0.0), y=_as_float(payload[1], 0.0))
        return cls()

    def length(self) -> float:
        return (self.x * self.x + self.y * self.y) ** 0.5

    def normalized(self) -> "Vec2":
        length = self.length()
        if length <= 1e-9:
            return Vec2()
        return Vec2(self.x / length, self.y / length)

    def distance_to(self, other: "Vec2") -> float:
        dx = other.x - self.x
        dy = other.y - self.y
        return (dx * dx + dy * dy) ** 0.5

    def clamp_unit(self) -> "Vec2":
        return Vec2(
            x=max(-1.0, min(1.0, self.x)),
            y=max(-1.0, min(1.0, self.y)),
        )

    def to_list(self) -> list[float]:
        return [self.x, self.y]


@dataclass(slots=True)
class WeaponView:
    weapon_type: str = "none"
    ammo: int = 0

    @classmethod
    def from_payload(cls, payload: Any) -> "WeaponView | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            weapon_type=str(payload.get("type") or payload.get("variant") or "none"),
            ammo=max(0, _as_int(payload.get("ammo"), 0)),
        )


@dataclass(slots=True)
class PlayerView:
    player_id: int
    position: Vec2
    facing: Vec2
    alive: bool
    color: str
    character: str
    weapon: WeaponView | None
    shoot_cooldown: float
    kick_cooldown: float
    stun_remaining: float
    velocity: Vec2 = field(default_factory=Vec2)

    @classmethod
    def from_payload(cls, payload: Any) -> "PlayerView":
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            player_id=_as_int(payload.get("id"), 0),
            position=Vec2.from_payload(payload.get("position")),
            facing=Vec2.from_payload(payload.get("facing")),
            alive=bool(payload.get("alive", False)),
            color=str(payload.get("color") or ""),
            character=str(payload.get("character") or ""),
            weapon=WeaponView.from_payload(payload.get("weapon")),
            shoot_cooldown=max(0.0, _as_float(payload.get("shoot_cooldown"), 0.0)),
            kick_cooldown=max(0.0, _as_float(payload.get("kick_cooldown"), 0.0)),
            stun_remaining=max(0.0, _as_float(payload.get("stun_remaining"), 0.0)),
            velocity=Vec2.from_payload(payload.get("velocity")),
        )


@dataclass(slots=True)
class PickupView:
    pickup_id: int
    weapon_type: str
    ammo: int
    position: Vec2
    cooldown: float
    despawn_remaining: float | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "PickupView":
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            pickup_id=_as_int(payload.get("id"), 0),
            weapon_type=str(payload.get("type") or ""),
            ammo=max(0, _as_int(payload.get("ammo"), 0)),
            position=Vec2.from_payload(payload.get("position")),
            cooldown=max(0.0, _as_float(payload.get("cooldown"), 0.0)),
            despawn_remaining=(
                max(0.0, _as_float(payload.get("despawn_remaining"), 0.0))
                if payload.get("despawn_remaining") is not None
                else None
            ),
        )


@dataclass(slots=True)
class ProjectileView:
    projectile_id: int
    owner_id: int
    projectile_type: str
    position: Vec2
    velocity: Vec2
    remaining_life: float

    @classmethod
    def from_payload(cls, payload: Any) -> "ProjectileView":
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            projectile_id=_as_int(payload.get("id"), 0),
            owner_id=_as_int(payload.get("owner"), 0),
            projectile_type=str(payload.get("type") or ""),
            position=Vec2.from_payload(payload.get("position")),
            velocity=Vec2.from_payload(payload.get("velocity")),
            remaining_life=max(0.0, _as_float(payload.get("remaining_life"), 0.0)),
        )


@dataclass(slots=True)
class ObstacleView:
    obstacle_id: int
    kind: str
    center: Vec2
    half_size: Vec2
    solid: bool

    @classmethod
    def from_payload(cls, payload: Any) -> "ObstacleView":
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            obstacle_id=_as_int(payload.get("id"), 0),
            kind=str(payload.get("kind") or ""),
            center=Vec2.from_payload(payload.get("center")),
            half_size=Vec2.from_payload(payload.get("half_size")),
            solid=bool(payload.get("solid", True)),
        )


@dataclass(slots=True)
class FloorGrid:
    grid_size: int
    cells: list[tuple[int, int]]
    _cell_lookup: set[tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cell_lookup = set(self.cells)

    @classmethod
    def from_payload(cls, payload: Any) -> "FloorGrid":
        if not isinstance(payload, dict):
            payload = {}
        cells: list[tuple[int, int]] = []
        for item in (payload.get("cells") or []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                cells.append((_as_int(item[0], 0), _as_int(item[1], 0)))
        return cls(
            grid_size=max(1, _as_int(payload.get("grid_size"), 64)),
            cells=cells,
        )

    def contains(self, grid_x: int, grid_y: int) -> bool:
        return (grid_x, grid_y) in self._cell_lookup


@dataclass(slots=True)
class BreakableView:
    breakable_id: int
    obstacle_id: int
    variant: str
    current: float
    threshold: float
    alive: bool
    center: Vec2
    half_size: Vec2

    @classmethod
    def from_payload(cls, payload: Any) -> "BreakableView":
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            breakable_id=_as_int(payload.get("id"), 0),
            obstacle_id=_as_int(payload.get("obstacle_id"), 0),
            variant=str(payload.get("variant") or ""),
            current=_as_float(payload.get("current"), 0.0),
            threshold=_as_float(payload.get("threshold"), 0.0),
            alive=bool(payload.get("alive", False)),
            center=Vec2.from_payload(payload.get("center")),
            half_size=Vec2.from_payload(payload.get("half_size")),
        )


@dataclass(slots=True)
class LetterboxView:
    obstacle_id: int
    position: Vec2
    cooldown: float
    ready: bool

    @classmethod
    def from_payload(cls, payload: Any) -> "LetterboxView":
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            obstacle_id=_as_int(payload.get("id"), 0),
            position=Vec2.from_payload(payload.get("position")),
            cooldown=max(0.0, _as_float(payload.get("cooldown"), 0.0)),
            ready=bool(payload.get("ready", False)),
        )


@dataclass(slots=True)
class LevelInfo:
    identifier: str
    width: float
    height: float
    floor: FloorGrid
    player_spawns: list[Vec2]
    static_obstacles: list[ObstacleView]

    @classmethod
    def from_payload(cls, payload: Any) -> "LevelInfo":
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            identifier=str(payload.get("identifier") or ""),
            width=max(0.0, _as_float(payload.get("width"), 0.0)),
            height=max(0.0, _as_float(payload.get("height"), 0.0)),
            floor=FloorGrid.from_payload(payload.get("floor")),
            player_spawns=[Vec2.from_payload(item) for item in (payload.get("player_spawns") or [])],
            static_obstacles=[ObstacleView.from_payload(item) for item in (payload.get("static_obstacles") or [])],
        )


@dataclass(slots=True)
class SeriesInfo:
    enabled: bool
    round_index: int
    total_rounds: int
    completed_rounds: int
    score: dict[str, int]

    @classmethod
    def from_payload(cls, payload: Any) -> "SeriesInfo":
        if not isinstance(payload, dict):
            payload = {}
        score = payload.get("score") or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            round_index=max(0, _as_int(payload.get("round_index"), 0)),
            total_rounds=max(0, _as_int(payload.get("total_rounds"), 0)),
            completed_rounds=max(0, _as_int(payload.get("completed_rounds"), 0)),
            score={
                "1": max(0, _as_int(score.get("1"), 0)),
                "2": max(0, _as_int(score.get("2"), 0)),
            },
        )


@dataclass(slots=True)
class RoundResultInfo:
    winner_id: int | None
    reason: str
    duration_seconds: float
    series_round: int
    series_total_rounds: int
    series_score: dict[str, int]
    series_finished: bool
    level_identifier: str

    @classmethod
    def from_payload(cls, payload: Any) -> "RoundResultInfo":
        if not isinstance(payload, dict):
            payload = {}
        raw_winner = payload.get("winner_id")
        winner_id = None if raw_winner is None else _as_int(raw_winner, 0)
        score = payload.get("series_score") or {}
        return cls(
            winner_id=winner_id,
            reason=str(payload.get("reason") or ""),
            duration_seconds=max(0.0, _as_float(payload.get("duration_seconds"), 0.0)),
            series_round=max(0, _as_int(payload.get("series_round"), 0)),
            series_total_rounds=max(0, _as_int(payload.get("series_total_rounds"), 0)),
            series_score={
                "1": max(0, _as_int(score.get("1"), 0)),
                "2": max(0, _as_int(score.get("2"), 0)),
            },
            series_finished=bool(payload.get("series_finished", False)),
            level_identifier=str(payload.get("level_identifier") or ""),
        )


@dataclass(slots=True)
class SnapshotView:
    status: str
    tick: int
    time_seconds: float
    time_limit_seconds: float
    players: list[PlayerView]
    pickups: list[PickupView]
    projectiles: list[ProjectileView]
    obstacles: list[ObstacleView]
    breakables: list[BreakableView]
    letterboxes: list[LetterboxView]
    result: RoundResultInfo | None

    @classmethod
    def from_payload(cls, payload: Any) -> "SnapshotView":
        if not isinstance(payload, dict):
            payload = {}
        result_payload = payload.get("result")
        return cls(
            status=str(payload.get("status") or ""),
            tick=max(0, _as_int(payload.get("tick"), 0)),
            time_seconds=max(0.0, _as_float(payload.get("time_seconds"), 0.0)),
            time_limit_seconds=max(0.0, _as_float(payload.get("time_limit_seconds"), 0.0)),
            players=[PlayerView.from_payload(item) for item in (payload.get("players") or [])],
            pickups=[PickupView.from_payload(item) for item in (payload.get("pickups") or [])],
            projectiles=[ProjectileView.from_payload(item) for item in (payload.get("projectiles") or [])],
            obstacles=[ObstacleView.from_payload(item) for item in (payload.get("obstacles") or [])],
            breakables=[BreakableView.from_payload(item) for item in (payload.get("breakables") or [])],
            letterboxes=[LetterboxView.from_payload(item) for item in (payload.get("letterboxes") or [])],
            result=RoundResultInfo.from_payload(result_payload) if result_payload is not None else None,
        )


@dataclass(slots=True)
class HelloMessage:
    player_id: int
    tick_rate: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "HelloMessage":
        return cls(
            player_id=max(0, _as_int(payload.get("player_id"), 0)),
            tick_rate=max(1, _as_int(payload.get("tick_rate"), 30)),
        )


@dataclass(slots=True)
class RoundStartMessage:
    player_id: int
    enemy_id: int
    tick_rate: int
    level: LevelInfo
    series: SeriesInfo

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RoundStartMessage":
        return cls(
            player_id=max(0, _as_int(payload.get("player_id"), 0)),
            enemy_id=max(0, _as_int(payload.get("enemy_id"), 0)),
            tick_rate=max(1, _as_int(payload.get("tick_rate"), 30)),
            level=LevelInfo.from_payload(payload.get("level")),
            series=SeriesInfo.from_payload(payload.get("series")),
        )


@dataclass(slots=True)
class TickMessage:
    tick: int
    time_seconds: float
    you: PlayerView
    enemy: PlayerView
    snapshot: SnapshotView

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TickMessage":
        return cls(
            tick=max(0, _as_int(payload.get("tick"), 0)),
            time_seconds=max(0.0, _as_float(payload.get("time_seconds"), 0.0)),
            you=PlayerView.from_payload(payload.get("you")),
            enemy=PlayerView.from_payload(payload.get("enemy")),
            snapshot=SnapshotView.from_payload(payload.get("snapshot")),
        )


@dataclass(slots=True)
class RoundEndMessage:
    result: RoundResultInfo

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RoundEndMessage":
        return cls(result=RoundResultInfo.from_payload(payload.get("result")))


@dataclass(slots=True)
class BotCommand:
    seq: int
    move: Vec2 = field(default_factory=Vec2)
    aim: Vec2 = field(default_factory=lambda: Vec2(1.0, 0.0))
    shoot: bool = False
    kick: bool = False
    pickup: bool = False
    drop: bool = False
    throw_item: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "command",
            "seq": self.seq,
            "move": self.move.clamp_unit().to_list(),
            "aim": self.aim.clamp_unit().to_list(),
            "shoot": bool(self.shoot),
            "kick": bool(self.kick),
            "pickup": bool(self.pickup),
            "drop": bool(self.drop),
            "throw": bool(self.throw_item),
        }


@dataclass(slots=True)
class BotState:
    hello: HelloMessage | None = None
    current_round: RoundStartMessage | None = None
    last_tick: TickMessage | None = None
    last_round_end: RoundEndMessage | None = None
    command_seq: int = 0

    @property
    def level(self) -> LevelInfo | None:
        return self.current_round.level if self.current_round else None

    @property
    def enemy(self) -> PlayerView | None:
        return self.last_tick.enemy if self.last_tick else None

    @property
    def me(self) -> PlayerView | None:
        return self.last_tick.you if self.last_tick else None

    @property
    def current_obstacles(self) -> list[ObstacleView]:
        if self.last_tick is not None:
            return self.last_tick.snapshot.obstacles
        return self.level.static_obstacles if self.level else []

    @property
    def current_obstacles_by_id(self) -> dict[int, ObstacleView]:
        return {obstacle.obstacle_id: obstacle for obstacle in self.current_obstacles}

    def next_command_seq(self) -> int:
        current = self.command_seq
        self.command_seq += 1
        return current
