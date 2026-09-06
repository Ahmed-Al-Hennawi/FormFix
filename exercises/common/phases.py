"""
The rep state machine, shared by all three exercises. They have the same
temporal shape - leave rest, travel to an extreme, turn around, come back - so
each hands in a 1-D signal that is high at rest and falls into the rep:

    squat        knee angle          high standing   -> low at depth
    lat pulldown elbow angle         high extended   -> low contracted
    press        elbow flexion       high at the     -> low at lockout
                 (180 - elbow angle) shoulders

    REST -> TOWARDS -> EXTREME -> RETURN -> REST

Counting threshold crossings would turn a signal fluttering around one number
into several reps, so there are five guards: hysteresis (four levels, never
one), a transition holding for min_phase_frames, a reversal delta, a minimum
range of motion, and duration bounds from timestamps rather than frame counts.

The machine won't start mid-movement either: until it has seen the signal at
rest the state stays UNKNOWN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class MovementPhase(str, Enum):
    """Phase of a rep, before an exercise gives it a nicer name."""

    REST = "rest"
    TOWARDS = "towards"
    EXTREME = "extreme"
    RETURN = "return"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RepDetectionConfig:
    """
    Engineering settings for the state machine. None of these say anything about
    technique. Signal convention is high at rest, falling into the rep:

        rest_level  >  start_level  >  extreme_level
        start_level <  end_level    <= rest_level
    """

    rest_level: float
    start_level: float
    extreme_level: float
    end_level: float
    min_phase_frames: int = 3
    # how far the signal must rise above its running minimum before the turning
    # point is accepted
    reversal_delta: float = 5.0
    # minimum excursion (rest minus extreme) to count as a rep attempt
    min_range: float = 30.0
    # frames either side of the raw minimum, so the extreme is a window
    extreme_window_frames: int = 3
    min_duration: float = 0.8
    max_duration: float = 12.0
    max_tracking_loss_frames: int = 10


@dataclass
class RawRep:
    """A detected rep as frame indices. Measuring it happens later."""

    start_frame: int
    extreme_frame: int
    end_frame: int
    extreme_value: float
    start_value: float = float("nan")

    @property
    def excursion(self) -> float:
        return float(self.start_value - self.extreme_value)


@dataclass
class RepDetectionResult:
    """Everything the state machine worked out about the movement."""

    phases: list[MovementPhase]
    reps: list[RawRep] = field(default_factory=list)
    partial_movements: int = 0
    partial_reasons: list[str] = field(default_factory=list)


def detect_repetitions(
    signal: np.ndarray,
    timestamps: np.ndarray,
    config: RepDetectionConfig,
) -> RepDetectionResult:
    """Run the state machine over a smoothed movement signal. A NaN reaching here
    means the data really is missing - short gaps are filled upstream."""
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    phases = [MovementPhase.UNKNOWN] * n
    result = RepDetectionResult(phases=phases)
    if n == 0:
        return result

    state = MovementPhase.UNKNOWN
    below_start_streak = 0
    travel_first_frame = -1
    min_value = float("inf")
    min_frame = -1
    above_end_streak = 0
    rep_start_frame = -1
    nan_streak = 0

    def reset(count_partial: bool, reason: str = "") -> None:
        nonlocal state, below_start_streak, travel_first_frame, min_value
        nonlocal min_frame, above_end_streak, rep_start_frame
        if count_partial and state in (
            MovementPhase.TOWARDS,
            MovementPhase.EXTREME,
            MovementPhase.RETURN,
        ):
            result.partial_movements += 1
            result.partial_reasons.append(reason)
            logger.debug("Partial movement: %s", reason)
        state = MovementPhase.UNKNOWN
        below_start_streak = 0
        travel_first_frame = -1
        min_value = float("inf")
        min_frame = -1
        above_end_streak = 0
        rep_start_frame = -1

    for i in range(n):
        value = signal[i]

        # --- missing measurement ---
        if not np.isfinite(value):
            nan_streak += 1
            phases[i] = (
                state if nan_streak <= config.max_tracking_loss_frames else MovementPhase.UNKNOWN
            )
            if nan_streak > config.max_tracking_loss_frames and state is not MovementPhase.UNKNOWN:
                reset(True, "tracking was lost during the movement")
            continue
        nan_streak = 0

        # --- transitions ---
        if state is MovementPhase.UNKNOWN:
            if value >= config.rest_level:
                state = MovementPhase.REST
            # Otherwise stay UNKNOWN: the clip might start mid-rep, and we can't
            # count a movement whose beginning we never saw.

        elif state is MovementPhase.REST:
            if value < config.start_level:
                below_start_streak += 1
                if below_start_streak == 1:
                    travel_first_frame = i
                if below_start_streak >= config.min_phase_frames:
                    state = MovementPhase.TOWARDS
                    rep_start_frame = max(travel_first_frame - 1, 0)
                    min_value = value
                    min_frame = i
                    for j in range(travel_first_frame, i):
                        phases[j] = MovementPhase.TOWARDS
            else:
                below_start_streak = 0
                travel_first_frame = -1

        elif state is MovementPhase.TOWARDS:
            if value < min_value:
                min_value = value
                min_frame = i
            if value > min_value + config.reversal_delta:
                # Rising again - did it go deep enough for a real turning point?
                if min_value <= config.extreme_level:
                    state = MovementPhase.RETURN
                    above_end_streak = 0
                    lo = max(min_frame - config.extreme_window_frames, 0)
                    hi = min(min_frame + config.extreme_window_frames, n - 1)
                    for j in range(lo, hi + 1):
                        phases[j] = MovementPhase.EXTREME
                elif value >= config.end_level:
                    reset(True, "movement was too shallow to count as a repetition")
                    state = MovementPhase.REST

        elif state is MovementPhase.RETURN:
            if value < min_value:
                # Deeper than before, so that "return" was a bounce.
                state = MovementPhase.TOWARDS
                min_value = value
                min_frame = i
            elif value >= config.end_level:
                above_end_streak += 1
                if above_end_streak >= config.min_phase_frames:
                    _finish(
                        result, rep_start_frame, min_frame, i, min_value, signal, timestamps, config
                    )
                    state = MovementPhase.REST
                    below_start_streak = 0
                    travel_first_frame = -1
                    min_value = float("inf")
                    min_frame = -1
                    above_end_streak = 0
                    rep_start_frame = -1
            else:
                above_end_streak = 0

        # --- label this frame ---
        if phases[i] is not MovementPhase.EXTREME:
            phases[i] = state

    if state in (MovementPhase.TOWARDS, MovementPhase.EXTREME, MovementPhase.RETURN):
        reset(True, "the video ended before the repetition finished")

    logger.info("Rep detection: %d complete, %d partial", len(result.reps), result.partial_movements)
    return result


def _finish(
    result: RepDetectionResult,
    start_frame: int,
    extreme_frame: int,
    end_frame: int,
    extreme_value: float,
    signal: np.ndarray,
    timestamps: np.ndarray,
    config: RepDetectionConfig,
) -> None:
    """Duration and range checks before a candidate rep is accepted."""
    duration = float(timestamps[end_frame] - timestamps[start_frame])
    start_value = float(signal[start_frame])
    excursion = start_value - extreme_value if np.isfinite(start_value) else float("nan")
    at = float(timestamps[start_frame])

    if duration < config.min_duration:
        result.partial_movements += 1
        result.partial_reasons.append(
            f"movement at {at:.1f}s was too fast ({duration:.2f}s) to be a controlled repetition"
        )
        return
    if duration > config.max_duration:
        result.partial_movements += 1
        result.partial_reasons.append(
            f"movement at {at:.1f}s took {duration:.1f}s - outside the plausible repetition window"
        )
        return
    if not np.isfinite(excursion) or excursion < config.min_range:
        result.partial_movements += 1
        result.partial_reasons.append(
            f"movement at {at:.1f}s had too little range of motion ({excursion:.0f} deg)"
        )
        return

    result.reps.append(
        RawRep(
            start_frame=start_frame,
            extreme_frame=extreme_frame,
            end_frame=end_frame,
            extreme_value=float(extreme_value),
            start_value=start_value,
        )
    )
