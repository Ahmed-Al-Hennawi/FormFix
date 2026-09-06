"""
Pulldown phase detection and rep counting:

    TOP (extended) -> PULLING -> BOTTOM (contracted) -> RETURNING -> TOP

The state machine is the shared one in exercises/common/phases.py, run on the
mean elbow angle. This module only supplies thresholds and renames the generic
phases; the guards that make the counting trustworthy live in the shared
module, and the knobs live in config.py.
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

# Generic phase -> what the pulldown calls it.
PHASE_NAMES: dict[MovementPhase, PulldownPhase] = {
    MovementPhase.REST: PulldownPhase.TOP,
    MovementPhase.TOWARDS: PulldownPhase.PULLING,
    MovementPhase.EXTREME: PulldownPhase.BOTTOM,
    MovementPhase.RETURN: PulldownPhase.RETURNING,
    MovementPhase.UNKNOWN: PulldownPhase.UNKNOWN,
}


def resting_extension(elbow_angles: np.ndarray, config: PulldownConfig) -> float:
    """
    The elbow angle this person rests at between pulls. A high percentile rather
    than the maximum, so one over-extended frame can't set the reference, then
    clamped. Every segmentation threshold is measured from this - it is not a
    technique criterion, that is ROM_TOP_EXTENSION_PASS.
    """
    finite = elbow_angles[np.isfinite(elbow_angles)]
    if finite.size < 5:
        return config.TOP_ELBOW_ANGLE
    reference = float(np.percentile(finite, config.REST_REFERENCE_PERCENTILE))
    return float(np.clip(reference, config.REST_REFERENCE_MIN, config.REST_REFERENCE_MAX))


def detection_config(config: PulldownConfig, reference: float) -> RepDetectionConfig:
    """The pulldown's thresholds in the shared machine's vocabulary. The four
    hysteresis levels are fixed margins below reference."""
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
    """Run the state machine over a smoothed mean-elbow-angle series. reference is
    the resting extension, estimated from the series if not supplied."""
    if reference is None:
        reference = resting_extension(elbow_angles, config)
    return detect_repetitions(elbow_angles, timestamps, detection_config(config, reference))


def named_phases(phases: list[MovementPhase]) -> list[PulldownPhase]:
    """Rename the generic per-frame phases for the pulldown."""
    return [PHASE_NAMES.get(phase, PulldownPhase.UNKNOWN) for phase in phases]
