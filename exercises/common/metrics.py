"""
Small measurement helpers shared by the three exercises. The important two:

    plausible    sanity check - MediaPipe can be confident about a limb it is
                 guessing, giving an angle that is anatomically impossible
    frames_for   phase scoping, so a rule only reads the part of the rep it
                 is about
"""

from __future__ import annotations

import numpy as np

from analysis.models import RepetitionSegmentation

# generic phase keys the rules use; each exercise renames them for display
PHASE_START = "start"
PHASE_TOWARDS = "towards"  # the pull, the press, the descent
PHASE_EXTREME = "extreme"  # short window around the turning point
PHASE_RETURN = "return"
PHASE_WORKING = "working"  # towards + extreme
PHASE_MOVEMENT = "movement"  # whole rep
PHASE_COMPLETION = "completion"  # the frame the rep finished on


def frames_for(segmentation: RepetitionSegmentation, phase: str) -> range:
    """Frame range a rule should read for this phase of one rep."""
    seg = segmentation
    if phase == PHASE_START:
        return range(seg.start_frame, max(seg.towards_start_frame, seg.start_frame) + 1)
    if phase == PHASE_TOWARDS:
        return range(
            seg.towards_start_frame,
            max(seg.extreme_start_frame, seg.towards_start_frame),
        )
    if phase == PHASE_EXTREME:
        return range(seg.extreme_start_frame, seg.extreme_end_frame + 1)
    if phase == PHASE_RETURN:
        return range(min(seg.return_start_frame, seg.end_frame), seg.end_frame + 1)
    if phase == PHASE_WORKING:
        return range(seg.towards_start_frame, seg.extreme_end_frame + 1)
    if phase == PHASE_COMPLETION:
        return range(seg.end_frame, seg.end_frame + 1)
    # PHASE_MOVEMENT or anything unknown: the whole rep
    return range(seg.start_frame, seg.end_frame + 1)


def segment_at(segmentation: RepetitionSegmentation, frame: int) -> str | None:
    """Which part of a rep a frame is in, or None if it's outside one."""
    seg = segmentation
    if not (seg.start_frame <= frame <= seg.end_frame):
        return None
    if frame < seg.extreme_start_frame:
        return PHASE_TOWARDS
    if frame <= seg.extreme_end_frame:
        return PHASE_EXTREME
    return PHASE_RETURN


def series(metrics: list, attribute: str, frames: range | None = None) -> np.ndarray:
    """One measurement as a NaN-gapped float series over frames."""
    frames = frames if frames is not None else range(len(metrics))
    values: list[float] = []
    for i in frames:
        if 0 <= i < len(metrics) and getattr(metrics[i], "valid", False):
            values.append(float(getattr(metrics[i], attribute, float("nan"))))
        else:
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def raw_series(metrics: list, attribute: str, frames: range | None = None) -> np.ndarray:
    """Like series but without the validity check, for measurements that have their
    own reliability flag."""
    frames = frames if frames is not None else range(len(metrics))
    values: list[float] = []
    for i in frames:
        if 0 <= i < len(metrics):
            values.append(float(getattr(metrics[i], attribute, float("nan"))))
        else:
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def median(values) -> float:
    """Median ignoring NaN, NaN if nothing is measurable."""
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def safe_min(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmin(values)) if np.isfinite(values).any() else float("nan")


def safe_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")


def plausible(value: float, low: float, high: float) -> float:
    """value if it falls inside an anatomically possible band, else NaN."""
    if not np.isfinite(value):
        return float("nan")
    return float(value) if low <= value <= high else float("nan")


def effective_fps(timestamps: np.ndarray) -> float:
    """FPS from the timestamps themselves, so nothing assumes 30 fps."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        return 30.0
    step = float(np.median(np.diff(timestamps)))
    return 1.0 / max(step, 1e-6)


def span_seconds(timestamps: np.ndarray, start: int, end: int) -> float:
    if start < 0 or end < 0 or start >= len(timestamps) or end >= len(timestamps):
        return float("nan")
    return float(max(timestamps[end] - timestamps[start], 0.0))
