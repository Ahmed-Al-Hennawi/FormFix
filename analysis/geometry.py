"""
Geometry helpers for the analysers. Points are 2D in pixel space: MediaPipe
scales x and y separately, so convert with pixel_xy first or the angles are
wrong. Missing input gives NaN back.
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def _is_bad(point: Point | None) -> bool:
    return point is None or any(p is None or math.isnan(p) for p in point)


def calculate_angle(a: Point | None, b: Point | None, c: Point | None) -> float:
    """Angle A-B-C at vertex b, 0 to 180 degrees. NaN if a point is missing."""
    if _is_bad(a) or _is_bad(b) or _is_bad(c):
        return float("nan")

    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")

    # clamp - float noise can push the cosine just outside [-1, 1] and acos raises
    cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def inclination_from_vertical(lower: Point | None, upper: Point | None) -> float:
    """Angle of the lower->upper segment from vertical. 0 upright, 90 horizontal."""
    if _is_bad(lower) or _is_bad(upper):
        return float("nan")

    dx = upper[0] - lower[0]
    dy = upper[1] - lower[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(dy) / length))))


def distance(a: Point | None, b: Point | None) -> float:
    """Euclidean distance in pixels; NaN when a point is missing."""
    if _is_bad(a) or _is_bad(b):
        return float("nan")
    return math.hypot(a[0] - b[0], a[1] - b[1])


def horizontal_distance(a: Point | None, b: Point | None) -> float:
    """|ax - bx| in pixels. Separate from distance so a straight-line distance
    can't stand in for a frontal-plane one."""
    if _is_bad(a) or _is_bad(b):
        return float("nan")
    return abs(a[0] - b[0])


def vertical_offset(a: Point | None, b: Point | None) -> float:
    """b.y - a.y in pixels. Image y grows downward, so positive means b is below a."""
    if _is_bad(a) or _is_bad(b):
        return float("nan")
    return b[1] - a[1]


def midpoint(a: Point | None, b: Point | None) -> Point | None:
    """Midpoint of two points; None when either is missing."""
    if _is_bad(a) or _is_bad(b):
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def signed_inclination_from_vertical(lower: Point | None, upper: Point | None) -> float:
    """Signed version of inclination_from_vertical. Pulldowns need the direction
    (leaning back is the fault), squats don't."""
    if _is_bad(lower) or _is_bad(upper):
        return float("nan")

    dx = upper[0] - lower[0]
    dy = upper[1] - lower[1]
    if math.hypot(dx, dy) < 1e-9:
        return float("nan")
    # Image y grows downward, so the "up" direction is -dy.
    return math.degrees(math.atan2(dx, -dy))


def angular_velocity(values, timestamps) -> list[float]:
    """Rate of change of an angle series, deg/s. Uses timestamps, not frame counts,
    so fps doesn't change the answer."""
    n = len(values)
    out = [float("nan")] * n
    for i in range(1, n):
        a, b = values[i - 1], values[i]
        dt = float(timestamps[i]) - float(timestamps[i - 1])
        if dt <= 1e-9 or a is None or b is None:
            continue
        if math.isnan(a) or math.isnan(b):
            continue
        out[i] = (b - a) / dt
    return out
