"""
Press phase detection and rep counting:

    READY (at the shoulders) -> PRESSING -> TOP -> LOWERING -> READY

The state machine is the shared one in exercises/common/phases.py, run on elbow
flexion (180 - elbow angle), which is high at the shoulders and falls towards
zero overhead. Framing it that way is what lets the press reuse the machine
unchanged even though this movement travels up where the other two go down.

The thresholds are adaptive, fixed margins below the flexion this person rests
at between presses.
"""

from __future__ import annotations

import numpy as np

from analysis.models import PressPhase
from exercises.common.phases import (
    MovementPhase,
    RepDetectionConfig,
    RepDetectionResult,
    detect_repetitions,
)

from .config import PressConfig

# Generic phase -> what the press calls it.
PHASE_NAMES: dict[MovementPhase, PressPhase] = {
    MovementPhase.REST: PressPhase.READY,
    MovementPhase.TOWARDS: PressPhase.PRESSING,
    MovementPhase.EXTREME: PressPhase.TOP,
    MovementPhase.RETURN: PressPhase.LOWERING,
    MovementPhase.UNKNOWN: PressPhase.UNKNOWN,
}


def resting_flexion(flexion: np.ndarray, config: PressConfig) -> float:
    """The elbow flexion this person rests at between presses. A high percentile
   , not the maximum, then clamped, so one over-flexed frame can't set it."""
    finite = flexion[np.isfinite(flexion)]
    if finite.size < 5:
        return config.READY_ELBOW_FLEXION
    reference = float(np.percentile(finite, config.REST_REFERENCE_PERCENTILE))
    return float(np.clip(reference, config.REST_REFERENCE_MIN, config.REST_REFERENCE_MAX))


def detection_config(config: PressConfig, reference: float) -> RepDetectionConfig:
    """The press's thresholds in the shared machine's vocabulary."""
    return RepDetectionConfig(
        rest_level=reference - config.REST_MARGIN,
        start_level=reference - config.PRESS_START_MARGIN,
        extreme_level=reference - config.TOP_MARGIN,
        end_level=reference - config.REP_END_MARGIN,
        min_phase_frames=config.PHASE_MIN_FRAMES,
        reversal_delta=config.TOP_REVERSAL_DELTA,
        min_range=config.MIN_RANGE_OF_MOTION,
        extreme_window_frames=config.TOP_WINDOW_FRAMES,
        min_duration=config.MIN_REP_DURATION,
        max_duration=config.MAX_REP_DURATION,
        max_tracking_loss_frames=config.MAX_TRACKING_LOSS_FRAMES,
    )


def detect_reps(
    flexion: np.ndarray,
    timestamps: np.ndarray,
    config: PressConfig,
    reference: float | None = None,
) -> RepDetectionResult:
    """Run the state machine over a smoothed elbow-flexion series. A NaN reaching
    here means the frame really had no measurement."""
    if reference is None:
        reference = resting_flexion(flexion, config)
    return detect_repetitions(flexion, timestamps, detection_config(config, reference))


def named_phases(phases: list[MovementPhase]) -> list[PressPhase]:
    """Rename the generic per-frame phases for the press."""
    return [PHASE_NAMES.get(phase, PressPhase.UNKNOWN) for phase in phases]
