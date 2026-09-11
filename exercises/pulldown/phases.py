"""
Pulldown phase detection and rep counting:

    TOP (extended) -> PULLING -> BOTTOM (contracted) -> RETURNING -> TOP

Uses the shared state machine (exercises/common/phases.py) on the mean elbow
angle. This file only sets the thresholds and renames the phases.
"""

from __future__ import annotations

import numpy as np

from analysis.models import PulldownPhase
from exercises.common.phases import (
    MovementPhase,
    RepDetectionConfig,
    RepDetectionResult,
    detect_repetitions,
)

from .config import PulldownConfig

# generic phase -> pulldown phase
PHASE_NAMES: dict[MovementPhase, PulldownPhase] = {
    MovementPhase.REST: PulldownPhase.TOP,
    MovementPhase.TOWARDS: PulldownPhase.PULLING,
    MovementPhase.EXTREME: PulldownPhase.BOTTOM,
    MovementPhase.RETURN: PulldownPhase.RETURNING,
    MovementPhase.UNKNOWN: PulldownPhase.UNKNOWN,
}


def resting_extension(elbow_angles: np.ndarray, config: PulldownConfig) -> float:
    """
    Resting elbow angle between pulls. A high percentile instead of the max,
    then clamped, so one odd frame can't set it. Only used for rep detection,
    not technique (that's ROM_TOP_EXTENSION_PASS).
    """
    finite = elbow_angles[np.isfinite(elbow_angles)]
    if finite.size < 5:
        return config.TOP_ELBOW_ANGLE
    reference = float(np.percentile(finite, config.REST_REFERENCE_PERCENTILE))
    return float(np.clip(reference, config.REST_REFERENCE_MIN, config.REST_REFERENCE_MAX))


def detection_config(config: PulldownConfig, reference: float) -> RepDetectionConfig:
    """The pulldown thresholds in the format the shared state machine expects."""
    return RepDetectionConfig(
        rest_level=reference - config.REST_MARGIN,
        start_level=reference - config.PULL_START_MARGIN,
        extreme_level=reference - config.CONTRACTED_MARGIN,
        end_level=reference - config.REP_END_MARGIN,
        min_phase_frames=config.PHASE_MIN_FRAMES,
        reversal_delta=config.CONTRACTION_REVERSAL_DELTA,
        min_range=config.MIN_RANGE_OF_MOTION,
        extreme_window_frames=config.CONTRACTED_WINDOW_FRAMES,
        min_duration=config.MIN_REP_DURATION,
        max_duration=config.MAX_REP_DURATION,
        max_tracking_loss_frames=config.MAX_TRACKING_LOSS_FRAMES,
    )


def detect_reps(
    elbow_angles: np.ndarray,
    timestamps: np.ndarray,
    config: PulldownConfig,
    reference: float | None = None,
) -> RepDetectionResult:
    """Run the state machine over the smoothed mean elbow angle. reference is
    estimated from the series if not given."""
    if reference is None:
        reference = resting_extension(elbow_angles, config)
    return detect_repetitions(elbow_angles, timestamps, detection_config(config, reference))


def named_phases(phases: list[MovementPhase]) -> list[PulldownPhase]:
    return [PHASE_NAMES.get(phase, PulldownPhase.UNKNOWN) for phase in phases]
