"""
Squat phase detection and rep counting, as a state machine over the smoothed
knee angle:

    STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING

Simple threshold crossing counted a jittery knee angle as several reps, so it
uses four thresholds, a persistence check, a minimum range of motion, a
reversal delta and duration limits (all in config.py). Movements that don't
qualify are counted as partial, with a reason.
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
    """A detected rep as frame indices, metrics added later."""

    start_frame: int
    bottom_frame: int
    end_frame: int
    min_knee_angle: float


@dataclass
class PhaseDetectionResult:
    """Everything the state machine worked out about the movement."""

    phases: list[Phase]
    reps: list[RawRep] = field(default_factory=list)
    # movements that started but never became a full rep
    partial_movements: int = 0
    partial_reasons: list[str] = field(default_factory=list)


def detect_reps(
    knee_angles: np.ndarray,
    timestamps: np.ndarray,
    config: SquatConfig,
) -> PhaseDetectionResult:
    """Run the state machine over the smoothed knee angle. Short gaps are already
    filled, so a NaN here means there really was no measurement."""
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

        # --- missing measurement ---
        if not np.isfinite(angle):
            nan_streak += 1
            phases[i] = state if nan_streak <= config.MAX_TRACKING_LOSS_FRAMES else Phase.UNKNOWN
            if nan_streak > config.MAX_TRACKING_LOSS_FRAMES and state is not Phase.UNKNOWN:
                abandon("tracking was lost during the movement")
            continue
        nan_streak = 0

        # --- state transitions ---
        if state is Phase.UNKNOWN:
            if angle >= config.STANDING_KNEE_ANGLE:
                state = Phase.STANDING
            # otherwise stay UNKNOWN - the video might start mid-squat

        elif state is Phase.STANDING:
            if angle < config.REP_START_KNEE_ANGLE:
                below_start_streak += 1
                if below_start_streak == 1:
                    descent_first_frame = i
                if below_start_streak >= config.PHASE_MIN_FRAMES:
                    # commit the descent, backdated to where it started
                    state = Phase.DESCENDING
                    rep_start_frame = max(descent_first_frame - 1, 0)
                    min_angle = angle
                    min_frame = i
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
                # rising again - was it deep enough for a real bottom?
                if min_angle <= config.BOTTOM_CANDIDATE_ANGLE:
                    state = Phase.ASCENDING
                    above_end_streak = 0
                    # bottom window around the true minimum
                    lo = max(min_frame - config.BOTTOM_WINDOW_FRAMES, 0)
                    hi = min(min_frame + config.BOTTOM_WINDOW_FRAMES, n - 1)
                    for j in range(lo, hi + 1):
                        phases[j] = Phase.BOTTOM
                elif angle >= config.REP_END_KNEE_ANGLE:
                    # back up without a real bottom, so just a shallow dip
                    abandon("movement was too shallow to count as a squat")
                    state = Phase.STANDING

        elif state is Phase.ASCENDING:
            if angle < min_angle:
                # deeper than before, so that "ascent" was just a bounce
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

        # --- phase label for this frame ---
        if phases[i] is Phase.UNKNOWN or phases[i] is not Phase.BOTTOM:
            phases[i] = state

    # video ended mid-movement
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
