"""
Small measurement helpers used by all three metrics layers. The two worth
knowing about:

    plausible    an anatomical sanity gate. MediaPipe reports a confident
                 position while extrapolating a limb it can't see, and the
                 angle is geometrically fine and anatomically impossible.
    frames_for   phase scoping. An elbow angle during the pull is meant to
                 differ from one at the top, so asking whether a measurement
                 was correct everywhere invents findings.

series returns NaN for invalid frames rather than 0.
"""

from __future__ import annotations

import numpy as np

from analysis.models import RepetitionSegmentation

# Generic phase keys. Rules are declared against these; each exercise renames
# them for display in its own metrics module.

# Rest position at the start of the rep.
PHASE_START = "start"
# Travel away from rest - the pull, the press, the descent.
PHASE_TOWARDS = "towards"
# Short window around the turning point.
PHASE_EXTREME = "extreme"
# Controlled return to rest.
PHASE_RETURN = "return"
# Travel plus turning point, i.e. the working part of the rep.
PHASE_WORKING = "working"
# Whole rep, first frame to last.
PHASE_MOVEMENT = "movement"
# Just the frame the rep finished on.
PHASE_COMPLETION = "completion"


def frames_for(segmentation: RepetitionSegmentation, phase: str) -> range:
    """Frame range a rule should read for phase on one rep."""
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
    # PHASE_MOVEMENT and anything unrecognised: the whole repetition.
    return range(seg.start_frame, seg.end_frame + 1)


def segment_at(segmentation: RepetitionSegmentation, frame: int) -> str | None:
    """
    Which part of a rep a frame belongs to, or None if it is outside one. The
    video uses this so the caption inside a committed rep comes from that rep's
    segmentation - an abandoned partial movement can leave labels sitting inside
    the span of the real rep that followed.
    """
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
    """series without the frame-level validity gate, for measurements that carry
    their own reliability flag."""
    frames = frames if frames is not None else range(len(metrics))
    values: list[float] = []
    for i in frames:
        if 0 <= i < len(metrics):
            values.append(float(getattr(metrics[i], attribute, float("nan"))))
        else:
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def median(values) -> float:
    """NaN-tolerant median; NaN when nothing is measurable."""
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
    """
    Frames per second from the timestamps themselves. Nothing assumes 30 fps.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        return 30.0
    step = float(np.median(np.diff(timestamps)))
    return 1.0 / max(step, 1e-6)


def span_seconds(timestamps: np.ndarray, start: int, end: int) -> float:
    if start < 0 or end < 0 or start >= len(timestamps) or end >= len(timestamps):
        return float("nan")
    return float(max(timestamps[end] - timestamps[start], 0.0))
