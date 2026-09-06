"""
Landmark cleaning: interpolate short gaps, then EMA smooth. xy_raw keeps the
untouched track, pose.xy is what the metrics read. Visibility is left alone -
smoothing it would hide tracking failures. Gaps longer than max_gap stay NaN
and reset the EMA.
"""

from __future__ import annotations

import numpy as np

from .models import FramePoseData


def interpolate_short_gaps(pose: FramePoseData, max_gap: int) -> int:
    """Fill gaps of at most max_gap frames, in place on pose.xy. Returns how many
    cells were filled (the debug view shows it)."""
    if max_gap <= 0:
        return 0

    frames = pose.xy.shape[0]
    filled = 0

    for landmark in range(pose.xy.shape[1]):
        xs = pose.xy[:, landmark, 0]
        observed = np.flatnonzero(np.isfinite(xs))
        if observed.size < 2:
            continue

        for left, right in zip(observed[:-1], observed[1:], strict=False):
            gap = right - left - 1
            if gap == 0 or gap > max_gap:
                continue
            for axis in range(2):
                start = pose.xy[left, landmark, axis]
                end = pose.xy[right, landmark, axis]
                steps = np.linspace(start, end, gap + 2)[1:-1]
                pose.xy[left + 1 : right, landmark, axis] = steps
            pose.valid[left + 1 : right, landmark] = True
            # interpolated confidence = the lower of the two anchors
            anchor_conf = min(pose.visibility[left, landmark], pose.visibility[right, landmark])
            pose.visibility[left + 1 : right, landmark] = np.maximum(
                pose.visibility[left + 1 : right, landmark], anchor_conf
            )
            filled += gap

    del frames
    pose.interpolated_frames = filled
    return filled


def ema_smooth(pose: FramePoseData, alpha: float) -> None:
    """
    In-place EMA on each landmark: smoothed[t] = a*obs[t] + (1-a)*smoothed[t-1].
    Higher alpha follows the raw signal. Invalid frames stay NaN and reset it.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")

    frames, landmarks, _ = pose.xy.shape
    for landmark in range(landmarks):
        previous: np.ndarray | None = None
        for frame in range(frames):
            current = pose.xy[frame, landmark]
            if not np.isfinite(current[0]):
                previous = None  # gap -> reset instead of dragging stale data
                continue
            if previous is None:
                previous = current.copy()
            else:
                previous = alpha * current + (1.0 - alpha) * previous
                pose.xy[frame, landmark] = previous


def smooth_series(values: np.ndarray, alpha: float) -> np.ndarray:
    """EMA over a 1-D series with NaN gaps. Resets after a gap, like ema_smooth."""
    out = values.astype(np.float64).copy()
    previous = np.nan
    for i, value in enumerate(out):
        if not np.isfinite(value):
            previous = np.nan
            continue
        previous = alpha * value + (1.0 - alpha) * previous if np.isfinite(previous) else value
        out[i] = previous
    return out
