"""
Press phase detection and rep counting:

    READY (at the shoulders) -> PRESSING -> TOP -> LOWERING -> READY

Uses the shared state machine (exercises/common/phases.py) on elbow flexion
(180 - elbow angle), which is high at the shoulders and near zero overhead.
That way the press can reuse it even though it moves up instead of down.
Thresholds are margins below the person's own resting flexion.
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

# generic phase -> press phase
PHASE_NAMES: dict[MovementPhase, PressPhase] = {
    MovementPhase.REST: PressPhase.READY,
    MovementPhase.TOWARDS: PressPhase.PRESSING,
    MovementPhase.EXTREME: PressPhase.TOP,
    MovementPhase.RETURN: PressPhase.LOWERING,
    MovementPhase.UNKNOWN: PressPhase.UNKNOWN,
}


def resting_flexion(flexion: np.ndarray, config: PressConfig) -> float:
    """Resting elbow flexion between presses. A high percentile instead of the max,
    then clamped, so one odd frame can't set it."""
    finite = flexion[np.isfinite(flexion)]
    if finite.size < 5:
        return config.READY_ELBOW_FLEXION
    reference = float(np.percentile(finite, config.REST_REFERENCE_PERCENTILE))
    return float(np.clip(reference, config.REST_REFERENCE_MIN, config.REST_REFERENCE_MAX))


def detection_config(config: PressConfig, reference: float) -> RepDetectionConfig:
    """The press thresholds in the format the shared state machine expects."""
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
    """Run the state machine over the smoothed elbow flexion."""
    if reference is None:
        reference = resting_flexion(flexion, config)
    return detect_repetitions(flexion, timestamps, detection_config(config, reference))


def named_phases(phases: list[MovementPhase]) -> list[PressPhase]:
    return [PHASE_NAMES.get(phase, PressPhase.UNKNOWN) for phase in phases]
