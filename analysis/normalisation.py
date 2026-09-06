"""
Pixel distances -> body-relative ones. A raw pixel distance depends on the
camera as much as the athlete, so distances are divided by a body dimension
from the same frame: torso length, else hip width, else lower leg. Recomputed
every frame. Angles are already scale-free and aren't touched.
"""

from __future__ import annotations

import math

from .geometry import Point, distance, midpoint

# Under this many pixels a "body dimension" is noise, not a measurement.
MIN_USABLE_SCALE_PX = 1e-6


def body_scale(
    mid_shoulder: Point | None,
    mid_hip: Point | None,
    left_hip: Point | None = None,
    right_hip: Point | None = None,
    knee: Point | None = None,
    ankle: Point | None = None,
) -> float:
    """Scale reference for this frame in pixels: torso, else hip width, else lower
    leg. NaN if none of them are usable."""
    torso = distance(mid_hip, mid_shoulder)
    if math.isfinite(torso) and torso > MIN_USABLE_SCALE_PX:
        return torso

    hips = distance(left_hip, right_hip)
    if math.isfinite(hips) and hips > MIN_USABLE_SCALE_PX:
        return hips

    lower_leg = distance(knee, ankle)
    if math.isfinite(lower_leg) and lower_leg > MIN_USABLE_SCALE_PX:
        return lower_leg

    return float("nan")


def normalise(value: float, scale: float) -> float:
    """value / scale, or NaN if either side is unusable."""
    if not (math.isfinite(value) and math.isfinite(scale)) or scale <= MIN_USABLE_SCALE_PX:
        return float("nan")
    return value / scale


def hip_centred(point: Point | None, mid_hip: Point | None, scale: float) -> Point | None:
    """point relative to the mid-hip, in body-scale units. Drops where the athlete
    is standing and how big they look, keeps direction."""
    if point is None or mid_hip is None:
        return None
    if not math.isfinite(scale) or scale <= MIN_USABLE_SCALE_PX:
        return None
    if any(p is None or math.isnan(p) for p in (*point, *mid_hip)):
        return None
    return ((point[0] - mid_hip[0]) / scale, (point[1] - mid_hip[1]) / scale)


def torso_reference(
    left_shoulder: Point | None,
    right_shoulder: Point | None,
    left_hip: Point | None,
    right_hip: Point | None,
) -> tuple[Point | None, Point | None]:
    """Mid-shoulder and mid-hip, falling back to whichever side is visible, so a
    side view with the far shoulder hidden still works."""
    mid_shoulder = midpoint(left_shoulder, right_shoulder) or left_shoulder or right_shoulder
    mid_hip = midpoint(left_hip, right_hip) or left_hip or right_hip
    return mid_shoulder, mid_hip
