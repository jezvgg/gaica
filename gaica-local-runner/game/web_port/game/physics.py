from __future__ import annotations

import math

from .models import Vec2, ObstacleRect


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def circle_intersects_rect(center: Vec2, radius: float, rect: ObstacleRect) -> bool:
    nearest_x = clamp(center.x, rect.center.x - rect.half_size.x, rect.center.x + rect.half_size.x)
    nearest_y = clamp(center.y, rect.center.y - rect.half_size.y, rect.center.y + rect.half_size.y)
    dx = center.x - nearest_x
    dy = center.y - nearest_y
    return (dx * dx + dy * dy) <= radius * radius


def resolve_circle_rect(center: Vec2, radius: float, rect: ObstacleRect) -> Vec2:
    """Resolve penetration by pushing the circle out along the smallest overlap axis."""
    min_x = rect.center.x - rect.half_size.x
    max_x = rect.center.x + rect.half_size.x
    min_y = rect.center.y - rect.half_size.y
    max_y = rect.center.y + rect.half_size.y

    nearest_x = clamp(center.x, min_x, max_x)
    nearest_y = clamp(center.y, min_y, max_y)
    dx = center.x - nearest_x
    dy = center.y - nearest_y
    d2 = dx * dx + dy * dy

    if d2 > radius * radius:
        return center

    if d2 > 1e-10:
        dist = math.sqrt(d2)
        push = radius - dist
        nx = dx / dist
        ny = dy / dist
        return Vec2(center.x + nx * push, center.y + ny * push)

    # center is inside rectangle; push out by minimal axis
    left = abs(center.x - min_x)
    right = abs(max_x - center.x)
    top = abs(center.y - min_y)
    bottom = abs(max_y - center.y)
    smallest = min(left, right, top, bottom)

    if smallest == left:
        return Vec2(min_x - radius, center.y)
    if smallest == right:
        return Vec2(max_x + radius, center.y)
    if smallest == top:
        return Vec2(center.x, min_y - radius)
    return Vec2(center.x, max_y + radius)


def resolve_circle_world(center: Vec2, radius: float, solids: list[ObstacleRect]) -> Vec2:
    active_solids = [rect for rect in solids if rect.solid]
    if not active_solids:
        return center

    # Re-run the sequential solver until a full pass stops moving the circle.
    # A single pass can push the circle out of one obstacle straight back into
    # an earlier neighbor when two colliders form a tight corner.
    fixed = center
    max_passes = max(1, len(active_solids))
    for _ in range(max_passes):
        changed = False
        for rect in active_solids:
            resolved = resolve_circle_rect(fixed, radius, rect)
            if abs(resolved.x - fixed.x) > 1e-9 or abs(resolved.y - fixed.y) > 1e-9:
                fixed = resolved
                changed = True
        if not changed:
            break
    return fixed


def ray_segment_circle_intersection(start: Vec2, end: Vec2, center: Vec2, radius: float) -> float | None:
    """Returns t in [0,1] if segment intersects circle, nearest hit first."""
    d = end - start
    f = start - center

    a = d.dot(d)
    b = 2.0 * f.dot(d)
    c = f.dot(f) - radius * radius

    disc = b * b - 4.0 * a * c
    if disc < 0.0 or abs(a) < 1e-9:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    candidates = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
    if not candidates:
        return None
    return min(candidates)


def ray_segment_aabb_intersection(start: Vec2, end: Vec2, rect: ObstacleRect) -> float | None:
    """Returns t in [0,1] if segment intersects axis-aligned rectangle."""
    dir_vec = end - start

    min_x = rect.center.x - rect.half_size.x
    max_x = rect.center.x + rect.half_size.x
    min_y = rect.center.y - rect.half_size.y
    max_y = rect.center.y + rect.half_size.y

    tmin = 0.0
    tmax = 1.0

    for axis in ("x", "y"):
        origin = getattr(start, axis)
        direction = getattr(dir_vec, axis)
        slab_min = min_x if axis == "x" else min_y
        slab_max = max_x if axis == "x" else max_y

        if abs(direction) < 1e-9:
            if origin < slab_min or origin > slab_max:
                return None
            continue

        ood = 1.0 / direction
        t1 = (slab_min - origin) * ood
        t2 = (slab_max - origin) * ood

        if t1 > t2:
            t1, t2 = t2, t1

        tmin = max(tmin, t1)
        tmax = min(tmax, t2)

        if tmin > tmax:
            return None

    if 0.0 <= tmin <= 1.0:
        return tmin
    if 0.0 <= tmax <= 1.0:
        return tmax
    return None


def kick_target_in_front(player_pos: Vec2, forward: Vec2, target_pos: Vec2, max_distance: float, min_dot: float) -> bool:
    to_target = target_pos - player_pos
    dist = to_target.length()
    if dist <= 1e-6 or dist > max_distance:
        return False

    return forward.normalize().dot(to_target.normalize()) >= min_dot
