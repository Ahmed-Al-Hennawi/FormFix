"""
Landmark stabilisation, between detection and measurement.

MediaPipe's Tasks API dropped the landmark smoothing the old solutions API had
(google-ai-edge/mediapipe issues 4507 and 4670), so every frame is estimated on
its own and one bad frame can throw a joint across the body. Smoothing alone
does not fix that: a low-pass filter spreads a spike over its neighbours instead
of removing it, so the spike has to go first.

    raw landmarks -> reject outliers (here) -> interpolate the gaps
                  -> smooth (smoothing.py / filters.py) -> measure or draw

Outliers are found with a Hampel test: compare each frame to the median of its
neighbours and allow a few MADs of scatter. The MAD adapts on its own, so a fast
movement is kept (its neighbours move too) while a teleporting joint is dropped.
Distances are in torso lengths, so it behaves the same close up and far away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .models import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    FramePoseData,
    VideoMetadata,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MIN_JUMP_TORSOS",
    "DEFAULT_WINDOW",
    "HAMPEL_SIGMAS",
    "StabiliserReport",
    "median_filter_track",
    "reject_outliers",
    "torso_pixels",
]

# frames compared either side of the one being tested
DEFAULT_WINDOW = 2

# MADs of scatter allowed before a landmark counts as an outlier. 1.4826 turns
# the MAD into a standard-deviation estimate, so this reads as sigmas.
HAMPEL_SIGMAS = 4.0

# smallest jump that can ever be called an outlier, in torso lengths. Without a
# floor a perfectly still body has MAD ~ 0 and normal sub-pixel noise gets cut.
DEFAULT_MIN_JUMP_TORSOS = 0.12

# if this share of a frame's landmarks look wrong it is the whole body that
# moved (or a camera cut), not one joint, so nothing is rejected
FRAME_REJECT_LIMIT = 0.5

_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class StabiliserReport:
    """What the stabiliser did, for the debug panel and the report."""

    rejected_cells: int
    frames: int
    landmarks: int
    torso_pixels: float

    @property
    def rejected_ratio(self) -> float:
        total = self.frames * max(self.landmarks, 1)
        return float(self.rejected_cells / total) if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rejected_cells": self.rejected_cells,
            "rejected_ratio": round(self.rejected_ratio, 5),
            "median_torso_px": round(self.torso_pixels, 1),
        }


def torso_pixels(pose: FramePoseData, video: VideoMetadata) -> float:
    """
    Median shoulder-centre to hip-centre distance in pixels - the scale every
    threshold here is expressed in. Falls back to a fraction of the frame when
    the torso was never tracked.
    """
    shoulders, hips = (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP)
    xy = pose.xy_raw
    usable = pose.valid[:, [*shoulders, *hips]].all(axis=1)
    if usable.any():
        frames = xy[usable]
        shoulder = frames[:, shoulders, :].mean(axis=1)
        hip = frames[:, hips, :].mean(axis=1)
        lengths = np.hypot(
            (shoulder[:, 0] - hip[:, 0]) * float(video.width),
            (shoulder[:, 1] - hip[:, 1]) * float(video.height),
        )
        lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
        if lengths.size:
            return float(np.median(lengths))
    # last resort: a quarter of the short edge is roughly a torso in frame
    short_edge = max(min(int(video.width), int(video.height)), 1)
    return 0.25 * short_edge


def _outlier_mask(values: np.ndarray, window: int, limit: np.ndarray) -> np.ndarray:
    """
    Hampel test on one coordinate series. The centre frame is left out of its own
    median so a single bad frame can't hide inside it. limit is the per-frame
    floor, in the same units as values.
    """
    n = values.size
    mask = np.zeros(n, dtype=bool)
    finite = np.isfinite(values)
    for i in range(n):
        if not finite[i]:
            continue
        lo, hi = max(0, i - window), min(n, i + window + 1)
        neighbours = values[lo:hi]
        keep = np.isfinite(neighbours).copy()
        keep[i - lo] = False  # leave the tested frame out
        neighbours = neighbours[keep]
        if neighbours.size < 2:
            continue
        median = float(np.median(neighbours))
        mad = float(np.median(np.abs(neighbours - median)))
        allowed = max(HAMPEL_SIGMAS * _MAD_TO_SIGMA * mad, float(limit[i]))
        if abs(values[i] - median) > allowed:
            mask[i] = True
    return mask


def reject_outliers(
    pose: FramePoseData,
    video: VideoMetadata,
    *,
    window: int = DEFAULT_WINDOW,
    min_jump_torsos: float = DEFAULT_MIN_JUMP_TORSOS,
    landmarks: frozenset[int] | None = None,
) -> StabiliserReport:
    """
    Drop landmark positions that cannot be real movement, in place on pose.xy and
    pose.valid. pose.xy_raw keeps the untouched track and visibility is left
    alone - it is the detector's own confidence and smoothing it would hide a
    tracking failure. Rejected cells become NaN, so interpolate_short_gaps fills
    the short ones and the long ones stay missing.
    """
    frames, count, _ = pose.xy.shape
    if frames == 0:
        return StabiliserReport(0, frames, count, 0.0)

    torso = torso_pixels(pose, video)
    wanted = range(count) if landmarks is None else sorted(landmarks)
    width, height = float(video.width or 1), float(video.height or 1)
    # the floor is a distance in pixels, so convert it per axis
    floor_x = np.full(frames, min_jump_torsos * torso / max(width, 1.0))
    floor_y = np.full(frames, min_jump_torsos * torso / max(height, 1.0))

    flagged = np.zeros((frames, count), dtype=bool)
    for landmark in wanted:
        if landmark >= count:
            continue
        bad_x = _outlier_mask(pose.xy[:, landmark, 0], window, floor_x)
        bad_y = _outlier_mask(pose.xy[:, landmark, 1], window, floor_y)
        # a landmark is one point, so either axis being wrong drops both
        flagged[:, landmark] = bad_x | bad_y

    # a frame where most of the body looks wrong is the body moving, not the
    # detector failing on one joint
    tracked = pose.valid.sum(axis=1)
    busy = flagged.sum(axis=1) > np.maximum(tracked * FRAME_REJECT_LIMIT, 1.0)
    flagged[busy, :] = False

    rejected = int(flagged.sum())
    if rejected:
        pose.xy[flagged] = np.nan
        pose.valid[flagged] = False
        logger.info(
            "Stabiliser rejected %d landmark position(s) of %d (%.2f%%)",
            rejected,
            frames * len(list(wanted)),
            100.0 * rejected / max(frames * len(list(wanted)), 1),
        )
    return StabiliserReport(rejected, frames, count, torso)


def median_filter_track(values: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Running median over one series, skipping NaN gaps. A median has no lag and
    removes what is left of the single-frame noise without rounding off the
    turning points the way a longer average would. Used for drawing.
    """
    if size < 3 or size % 2 == 0:
        return np.asarray(values, dtype=np.float64).copy()
    out = np.asarray(values, dtype=np.float64).copy()
    half = size // 2
    finite = np.isfinite(out)
    source = out.copy()
    for i in range(out.size):
        if not finite[i]:
            continue
        lo, hi = max(0, i - half), min(out.size, i + half + 1)
        chunk = source[lo:hi]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size >= 2:
            out[i] = float(np.median(chunk))
    return out
