from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from . import config


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def distance_to(self, other: Vec2) -> float:
        return (self - other).length()

    def normalize(self) -> Vec2:
        mag = self.length()
        if mag <= 1e-8:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / mag, self.y / mag)

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def to_list(self) -> list[float]:
        return [self.x, self.y]

    @staticmethod
    def from_any(raw: Any) -> Vec2:
        if isinstance(raw, Vec2):
            return raw
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            return Vec2(float(raw[0]), float(raw[1]))
        return Vec2()


class WeaponType(str, Enum):
    REVOLVER = "Revolver"
    UZI = "Uzi"


@dataclass
class WeaponStats:
    max_ammo: int
    shot_cooldown: float
    spread_x: tuple[float, float]
    spread_y: tuple[float, float]
    breakability: float


WEAPON_STATS: dict[WeaponType, WeaponStats] = {
    WeaponType.REVOLVER: WeaponStats(
        max_ammo=config.REVOLVER_MAX_AMMO,
        shot_cooldown=config.REVOLVER_SHOT_COOLDOWN,
        spread_x=config.REVOLVER_SPREAD_X,
        spread_y=config.REVOLVER_SPREAD_Y,
        breakability=config.REVOLVER_BREAKABILITY,
    ),
    WeaponType.UZI: WeaponStats(
        max_ammo=config.UZI_MAX_AMMO,
        shot_cooldown=config.UZI_SHOT_COOLDOWN,
        spread_x=config.UZI_SPREAD_X,
        spread_y=config.UZI_SPREAD_Y,
        breakability=config.UZI_BREAKABILITY,
    ),
}


@dataclass
class PlayerCommand:
    seq: int = 0
    move: Vec2 = field(default_factory=Vec2)
    aim: Vec2 = field(default_factory=lambda: Vec2(1.0, 0.0))
    shoot: bool = False
    kick: bool = False
    pickup: bool = False
    drop: bool = False
    throw: bool = False

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> PlayerCommand:
        return PlayerCommand(
            seq=int(payload.get("seq", 0)),
            move=Vec2.from_any(payload.get("move", [0, 0])),
            aim=Vec2.from_any(payload.get("aim", [1, 0])),
            shoot=bool(payload.get("shoot", False)),
            kick=bool(payload.get("kick", False)),
            pickup=bool(payload.get("pickup", False)),
            drop=bool(payload.get("drop", False)),
            throw=bool(payload.get("throw", False)),
        )


@dataclass
class WeaponInstance:
    weapon_type: WeaponType
    ammo: int


@dataclass
class PlayerState:
    player_id: int
    position: Vec2
    facing: Vec2
    color: str
    character: str = "lemon"
    velocity: Vec2 = field(default_factory=Vec2)
    alive: bool = True
    current_weapon: WeaponInstance | None = None
    shoot_cooldown: float = 0.0
    kick_cooldown: float = 0.0
    is_kicking: bool = False
    stun_remaining: float = 0.0
    stun_direction: Vec2 = field(default_factory=Vec2)
    last_kick_seq: int = -1
    last_pickup_seq: int = -1
    last_drop_seq: int = -1
    last_throw_seq: int = -1


@dataclass
class Projectile:
    projectile_id: int
    owner_id: int
    weapon_type: WeaponType
    position: Vec2
    velocity: Vec2
    remaining_life: float


@dataclass
class PickupWeapon:
    pickup_id: int
    weapon_type: WeaponType
    ammo: int
    position: Vec2
    velocity: Vec2 = field(default_factory=Vec2)
    cooldown: float = config.WEAPON_PICKUP_COOLDOWN
    empty_remove_remaining: float | None = None


@dataclass
class BreakableState:
    breakable_id: int
    obstacle_id: int
    variant: str
    threshold: float
    current_value: float
    rect_center: Vec2
    rect_half_size: Vec2
    alive: bool = True


@dataclass
class ObstacleRect:
    obstacle_id: int
    kind: str
    center: Vec2
    half_size: Vec2
    solid: bool = True


@dataclass
class TileDraw:
    x: int
    y: int
    tile_id: int
    src_x: int
    src_y: int
    layer: str
    size: int


@dataclass
class LevelData:
    identifier: str
    width: int
    height: int
    floor_tiles: list[TileDraw]
    top_tiles: list[TileDraw]
    small_tiles: list[TileDraw]
    player_spawns: list[Vec2]
    weapon_spawns: list[tuple[Vec2, WeaponType]]
    box_spawns: list[Vec2]
    obstacles: list[ObstacleRect]
    breakables: list[BreakableState]
    letterboxes: list[ObstacleRect]


@dataclass
class RoundResult:
    winner_id: int | None
    reason: str
    duration_seconds: float
