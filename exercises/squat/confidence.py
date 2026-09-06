"""
The squat's adapter over the shared confidence engine. All it adds is what the
engine can't know: which landmarks a squat metric needs, which frames a squat
rule reads, and where the bands sit.
"""

from __future__ import annotations

from analysis.models import (
    CameraOrientation,
    FramePoseData,
    Reliability,
    RepRuleOutcome,
    SquatRep,
)
from exercises.common.confidence import (
    MetricEvidence,
    ReliabilityBands,
    overall_reliability,
    view_support,
)
from exercises.common.confidence import (
    gather_evidence as _gather_evidence,
)
from exercises.common.confidence import (
    reliability_for as _reliability_for,
)

from .config import SquatConfig
from .landmarks import is_frontal_plane, required_ids
from .metrics import phase_frames

__all__ = [
    "MetricEvidence",
    "bands_for",
    "gather_evidence",
    "overall_reliability",
    "reliability_for",
    "view_support",
]


def bands_for(config: SquatConfig) -> ReliabilityBands:
    """Reshape the squat config's banding into what the engine expects."""
    return ReliabilityBands(
        high_visibility=config.RELIABILITY_HIGH_VISIBILITY,
        high_measurable_ratio=config.RELIABILITY_HIGH_MEASURABLE_RATIO,
        high_view_support=config.RELIABILITY_HIGH_VIEW_SUPPORT,
        high_min_reps=config.RELIABILITY_HIGH_MIN_REPS,
        medium_visibility=config.RELIABILITY_MEDIUM_VISIBILITY,
        medium_measurable_ratio=config.RELIABILITY_MEDIUM_MEASURABLE_RATIO,
        medium_view_support=config.RELIABILITY_MEDIUM_VIEW_SUPPORT,
        min_subject_pixels=config.RELIABILITY_MIN_SUBJECT_PIXELS,
        poor_subject_pixels=config.RELIABILITY_POOR_SUBJECT_PIXELS,
    )


def gather_evidence(
    metric: str,
    reps: list[SquatRep],
    outcomes: list[RepRuleOutcome],
    pose: FramePoseData,
    phase: str,
    supported_views: tuple[str, ...],
    orientation: CameraOrientation,
    side_view_confidence: float,
    side: str,
    subject_scale_px: float = float("nan"),
    recording_quality: str = "good",
) -> MetricEvidence:
    """Evidence behind one squat metric, across all detected reps."""
    return _gather_evidence(
        metric,
        reps,
        outcomes,
        pose,
        required_ids(metric, side),
        lambda rep: phase_frames(rep, phase),
        supported_views,
        orientation,
        side_view_confidence,
        subject_scale_px=subject_scale_px,
        recording_quality=recording_quality,
        frontal_plane=is_frontal_plane(metric),
    )


def reliability_for(evidence: MetricEvidence, config: SquatConfig) -> Reliability:
    """Squat evidence -> one of the four reliability levels."""
    return _reliability_for(evidence, bands_for(config))
