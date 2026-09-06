"""
Squat phase detection and rep counting, as a state machine over the smoothed
knee angle:

    STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING

Counting threshold crossings would turn a knee angle fluttering around one
number into several reps, so this uses four separate thresholds, a persistence
requirement, a minimum range of motion, a reversal delta and duration bounds in
seconds. All of them live in config.py. Movements that don't qualify are
counted as partial and reported with a reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from analysis.models import Phase

from .config import SquatConfig

logger = logging.getLogger(__name__)


@dataclass
class RawRep:
    """A detected rep as frame indices; metrics get added later."""

    start_frame: int
    bottom_frame: int
    end_frame: int
    min_knee_angle: float


@dataclass
class PhaseDetectionResult:
    """Everything the state machine worked out about the movement."""

    # Per-frame phase, aligned with the input series.
    phases: list[Phase]
    # Completed, sanity-checked reps in order.
    reps: list[RawRep] = field(default_factory=list)
    # Movements that started but never qualified as a full rep.
    partial_movements: int = 0
    # Why each partial was rejected, for the debug view.
    partial_reasons: list[str] = field(default_factory=list)


def detect_reps(
    knee_angles: np.ndarray,
    timestamps: np.ndarray,
    config: SquatConfig,
) -> PhaseDetectionResult:
    """Run the state machine over a smoothed knee-angle series. A NaN reaching here
    means the frame really had no measurement - short gaps are filled upstream."""
    n = len(knee_angles)
    phases = [Phase.UNKNOWN] * n
    result = PhaseDetectionResult(phases=phases)
    if n == 0:
        return result

    state = Phase.UNKNOWN
    below_start_streak = 0  # frames in a row under REP_START while standing
    descent_first_frame = -1
    min_angle = float("inf")
    min_frame = -1
    above_end_streak = 0
    rep_start_frame = -1
    nan_streak = 0

    def abandon(reason: str, count_partial: bool = True) -> None:
        nonlocal state, below_start_streak, descent_first_frame, min_angle
        nonlocal min_frame, above_end_streak, rep_start_frame
        if count_partial and state in (Phase.DESCENDING, Phase.BOTTOM, Phase.ASCENDING):
            result.partial_movements += 1
            result.partial_reasons.append(reason)
            logger.debug("Partial movement: %s", reason)
        state = Phase.UNKNOWN
        below_start_streak = 0
        descent_first_frame = -1
        min_angle = float("inf")
        min_frame = -1
        above_end_streak = 0
        rep_start_frame = -1

    for i in range(n):
        angle = knee_angles[i]

        # ---------- missing measurement ----------------------------------
        if not np.isfinite(angle):
            nan_streak += 1
            phases[i] = state if nan_streak <= config.MAX_TRACKING_LOSS_FRAMES else Phase.UNKNOWN
            if nan_streak > config.MAX_TRACKING_LOSS_FRAMES and state is not Phase.UNKNOWN:
                abandon("tracking was lost during the movement")
            continue
        nan_streak = 0

        # ---------- state transitions ------------------------------------
        if state is Phase.UNKNOWN:
            if angle >= config.STANDING_KNEE_ANGLE:
                state = Phase.STANDING
            # Otherwise stay UNKNOWN: the video might start mid-squat, and we
            # can't count a movement whose beginning we never saw.

        elif state is Phase.STANDING:
            if angle < config.REP_START_KNEE_ANGLE:
                below_start_streak += 1
                if below_start_streak == 1:
                    descent_first_frame = i
                if below_start_streak >= config.PHASE_MIN_FRAMES:
                    # Commit the descent, backdated to where it began.
                    state = Phase.DESCENDING
                    rep_start_frame = max(descent_first_frame - 1, 0)
                    min_angle = angle
                    min_frame = i
                    # Re-label the streak's frames as DESCENDING.
                    for j in range(descent_first_frame, i):
                        phases[j] = Phase.DESCENDING
            else:
                below_start_streak = 0
                descent_first_frame = -1

        elif state is Phase.DESCENDING:
            if angle < min_angle:
                min_angle = angle
                min_frame = i
            if angle > min_angle + config.BOTTOM_REVERSAL_DELTA:
                # Rising again - was the descent deep enough for a real bottom?
                if min_angle <= config.BOTTOM_CANDIDATE_ANGLE:
                    state = Phase.ASCENDING
                    above_end_streak = 0
                    # Mark the bottom window around the true minimum.
                    lo = max(min_frame - config.BOTTOM_WINDOW_FRAMES, 0)
                    hi = min(min_frame + config.BOTTOM_WINDOW_FRAMES, n - 1)
                    for j in range(lo, hi + 1):
                        phases[j] = Phase.BOTTOM
                elif angle >= config.REP_END_KNEE_ANGLE:
                    # All the way back up with no real bottom: a shallow dip.
                    abandon("movement was too shallow to count as a squat")
                    state = Phase.STANDING

        elif state is Phase.ASCENDING:
            if angle < min_angle:
                # Deeper than before, so that "ascent" was a bounce.
                state = Phase.DESCENDING
                min_angle = angle
                min_frame = i
            elif angle >= config.REP_END_KNEE_ANGLE:
                above_end_streak += 1
                if above_end_streak >= config.PHASE_MIN_FRAMES:
                    end_frame = i
                    _finish_rep(
                        result,
                        rep_start_frame,
                        min_frame,
                        end_frame,
                        min_angle,
                        knee_angles,
                        timestamps,
                        config,
                    )
                    state = Phase.STANDING
                    below_start_streak = 0
                    descent_first_frame = -1
                    min_angle = float("inf")
                    min_frame = -1
                    above_end_streak = 0
                    rep_start_frame = -1
            else:
                above_end_streak = 0

        # ---------- phase labelling for this frame ------------------------
        if phases[i] is Phase.UNKNOWN or phases[i] is not Phase.BOTTOM:
            phases[i] = state

    # Video ended mid-movement.
    if state in (Phase.DESCENDING, Phase.BOTTOM, Phase.ASCENDING):
        abandon("the video ended before the repetition finished")

    logger.info(
        "Rep detection: %d complete, %d partial",
        len(result.reps),
        result.partial_movements,
    )
    return result


def _finish_rep(
    result: PhaseDetectionResult,
    start_frame: int,
    bottom_frame: int,
    end_frame: int,
    min_angle: float,
    knee_angles: np.ndarray,
    timestamps: np.ndarray,
    config: SquatConfig,
) -> None:
    """Duration and range checks before a candidate rep is accepted."""
    duration = float(timestamps[end_frame] - timestamps[start_frame])
    start_angle = knee_angles[start_frame]
    rom = float(start_angle - min_angle) if np.isfinite(start_angle) else float("nan")

    if duration < config.MIN_REP_DURATION:
        result.partial_movements += 1
        result.partial_reasons.append(
            f"movement at {timestamps[start_frame]:.1f}s was too fast ({duration:.2f}s) to be a controlled squat"
        )
        return
    if duration > config.MAX_REP_DURATION:
        result.partial_movements += 1
        result.partial_reasons.append(
            f"movement at {timestamps[start_frame]:.1f}s took {duration:.1f}s - outside the plausible repetition window"
        )
        return
    if not np.isfinite(rom) or rom < config.MIN_RANGE_OF_MOTION:
        result.partial_movements += 1
        result.partial_reasons.append(
            f"movement at {timestamps[start_frame]:.1f}s had too little range of motion ({rom:.0f} deg)"
        )
        return

    result.reps.append(
        RawRep(
            start_frame=start_frame,
            bottom_frame=bottom_frame,
            end_frame=end_frame,
            min_knee_angle=float(min_angle),
        )
    )
