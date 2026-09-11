"""
Picks which frames are worth rendering: first rep start to last rep end, plus
some padding. People film the walk-up and walk back, so the set is often less
than half the clip.

The source is never re-cut, so frame numbers and feedback timestamps still
match the original. If anything looks off it returns the full video.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# seconds kept either side of the set
DEFAULT_PADDING_SECONDS: float = 1.0

# below this the cut saves nothing
MIN_TRIM_SAVING_SECONDS: float = 2.0

# a shorter window is a detection artefact, so render the lot
MIN_WINDOW_SECONDS: float = 1.0


@runtime_checkable
class _HasFrameRange(Protocol):
    """Anything with rep boundaries - RawRep, SquatRep, PulldownRep, PressRep."""

    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class RenderWindow:
    """Frames worth rendering, inclusive, in ORIGINAL frame numbers. trimmed is
    False when the window is the whole video."""

    start_frame: int
    end_frame: int
    trimmed: bool = False
    reason: str = ""

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    def contains(self, index: int) -> bool:
        return self.start_frame <= index <= self.end_frame

    def seconds_saved(self, total_frames: int, fps: float) -> float:
        """How much footage the window skips. 0.0 when nothing was trimmed."""
        if not self.trimmed or fps <= 0:
            return 0.0
        return max(0.0, (total_frames - self.frame_count) / fps)

    def as_dict(self) -> dict[str, Any]:
        """For the debug panel and the export."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "trimmed": self.trimmed,
            "reason": self.reason,
        }


def full_video(frame_count: int, reason: str = "") -> RenderWindow:
    """The untrimmed window. Every fallback path ends up here."""
    return RenderWindow(
        start_frame=0,
        end_frame=max(0, frame_count - 1),
        trimmed=False,
        reason=reason,
    )


def compute_render_window(
    reps: Sequence[Any],
    frame_count: int,
    fps: float,
    *,
    padding_seconds: float = DEFAULT_PADDING_SECONDS,
    min_saving_seconds: float = MIN_TRIM_SAVING_SECONDS,
) -> RenderWindow:
    """
    Work out the render window from the detected reps. reps is anything with
    start_frame and end_frame; order doesn't matter. Returns the full video when
    there is nothing sensible to cut, and never raises.
    """
    if frame_count <= 0:
        return full_video(frame_count, "video has no frames")
    if fps <= 0:
        return full_video(frame_count, "no usable frame rate")
    if not reps:
        return full_video(frame_count, "no reps detected")

    last_index = frame_count - 1

    starts: list[int] = []
    ends: list[int] = []
    for rep in reps:
        start = _frame_attr(rep, "start_frame")
        end = _frame_attr(rep, "end_frame")
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        starts.append(start)
        ends.append(end)

    if not starts:
        return full_video(frame_count, "reps carried no usable frame numbers")

    first = min(starts)
    last = max(ends)

    # reps and video disagree, so a guessed window is worse than no trim
    if first > last_index or last < 0:
        return full_video(frame_count, "rep frames fall outside the video")

    first = max(0, min(first, last_index))
    last = max(0, min(last, last_index))

    pad = int(round(max(0.0, padding_seconds) * fps))
    start_frame = max(0, first - pad)
    end_frame = min(last_index, last + pad)

    window_seconds = (end_frame - start_frame + 1) / fps
    if window_seconds < MIN_WINDOW_SECONDS:
        return full_video(frame_count, "detected set is too short to trim to")

    saved_seconds = (frame_count - (end_frame - start_frame + 1)) / fps
    if saved_seconds < min_saving_seconds:
        return full_video(frame_count, "not enough setup footage to be worth trimming")

    logger.info(
        "Trimming render to frames %d-%d of %d (%.1fs saved)",
        start_frame,
        end_frame,
        frame_count,
        saved_seconds,
    )
    return RenderWindow(
        start_frame=start_frame,
        end_frame=end_frame,
        trimmed=True,
        reason=f"trimmed to the detected set ({saved_seconds:.1f}s of setup skipped)",
    )


def _frame_attr(rep: Any, name: str) -> int | None:
    """Read a frame number off a rep. None if it's missing, negative, or the
    property raises."""
    try:
        value = getattr(rep, name)
    except Exception:  # a property that raises shouldn't cost the run
        return None
    if value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if index < 0:  # the "not segmented" sentinel
        return None
    return index
