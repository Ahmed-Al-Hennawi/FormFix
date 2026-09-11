"""
How much to trust a measurement. Four bands rather than a probability, since a
probability would suggest a calibrated model I don't have:

    HIGH           several reps, visible landmarks, suitable view
    MEDIUM         measurable, on thinner evidence
    LOW            measurable but weak; treat the finding as a hint
    CANNOT_ASSESS  nothing usable, so the rule reports nothing

It's worked out per measurement, not per video - e.g. a pulldown's range of
motion can be reliable while trunk lean in the same clip isn't.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from analysis.models import (
    CameraOrientation,
    FramePoseData,
    Reliability,
    RepRuleOutcome,
    RuleStatus,
)


@dataclass(frozen=True)
class ReliabilityBands:
    """Boundaries between the four levels (engineering settings, not technique)."""

    high_visibility: float = 0.75
    high_measurable_ratio: float = 0.85
    high_view_support: float = 0.85
    high_min_reps: int = 2
    medium_visibility: float = 0.55
    medium_measurable_ratio: float = 0.6
    medium_view_support: float = 0.5
    # median body scale in pixels. Someone filmed from far away has more error
    # per landmark than the visibility score shows
    min_subject_pixels: float = 140.0
    poor_subject_pixels: float = 70.0


@dataclass(frozen=True)
class MetricEvidence:
    """What stands behind one metric's verdict."""

    metric: str
    landmark_visibility: float
    measurable_ratio: float
    view_support: float
    evaluable_reps: int
    total_reps: int
    subject_scale_px: float = float("nan")
    recording_quality: str = "good"

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "landmark_visibility": round(self.landmark_visibility, 3),
            "measurable_ratio": round(self.measurable_ratio, 3),
            "view_support": round(self.view_support, 3),
            "evaluable_reps": self.evaluable_reps,
            "total_reps": self.total_reps,
            "subject_scale_px": (
                round(self.subject_scale_px, 1) if np.isfinite(self.subject_scale_px) else None
            ),
            "recording_quality": self.recording_quality,
        }


def view_support(
    supported_views: tuple[str, ...],
    orientation: CameraOrientation,
    side_view_confidence: float,
    frontal_plane: bool = False,
) -> float:
    """
    How well this camera view supports a metric, 0 to 1. Diagonal views score the
    view confidence (flipped for front-on metrics), unknown views get 0.5.
    """
    if orientation.value not in supported_views:
        return 0.0
    if orientation is CameraOrientation.SIDE:
        return 0.0 if frontal_plane else 1.0
    if orientation is CameraOrientation.DIAGONAL_SIDE:
        confidence = float(np.clip(side_view_confidence, 0.0, 1.0))
        return 1.0 - confidence if frontal_plane else confidence
    if orientation is CameraOrientation.FRONTAL:
        # only reached for frontal-plane or view-independent metrics
        return 1.0
    return 0.5  # UNKNOWN - analysed, but never treated as strong evidence


def gather_evidence(
    metric: str,
    reps: list,
    outcomes: list[RepRuleOutcome],
    pose: FramePoseData,
    required_ids: tuple[int, ...],
    frames_for_rep: Callable[[object], range],
    supported_views: tuple[str, ...],
    orientation: CameraOrientation,
    side_view_confidence: float,
    subject_scale_px: float = float("nan"),
    recording_quality: str = "good",
    frontal_plane: bool = False,
) -> MetricEvidence:
    """Collect the evidence for one metric across all reps. required_ids is just the
    landmarks it depends on - averaging all 33 would make everything look better."""
    visibilities: list[float] = []
    measurable: list[float] = []

    for rep in reps:
        frames = [f for f in frames_for_rep(rep) if 0 <= f < pose.frame_count]
        if not frames or not required_ids:
            continue
        block = pose.visibility[np.ix_(frames, list(required_ids))]
        visibilities.append(float(np.mean(block)) if block.size else 0.0)
        valid_block = pose.valid[np.ix_(frames, list(required_ids))]
        measurable.append(float(np.mean(np.all(valid_block, axis=1))) if valid_block.size else 0.0)

    evaluable = sum(1 for o in outcomes if o.status is not RuleStatus.NOT_EVALUABLE)
    return MetricEvidence(
        metric=metric,
        landmark_visibility=float(np.mean(visibilities)) if visibilities else 0.0,
        measurable_ratio=float(np.mean(measurable)) if measurable else 0.0,
        view_support=view_support(supported_views, orientation, side_view_confidence, frontal_plane),
        evaluable_reps=evaluable,
        total_reps=len(reps),
        subject_scale_px=subject_scale_px,
        recording_quality=recording_quality,
    )


def reliability_for(evidence: MetricEvidence, bands: ReliabilityBands) -> Reliability:
    """Turn the evidence into one of the four levels. Every floor has to be passed,
    not averaged - great visibility with the wrong camera angle is still unusable."""
    if evidence.evaluable_reps == 0 or evidence.view_support <= 0.0:
        return Reliability.CANNOT_ASSESS

    if (
        evidence.landmark_visibility >= bands.high_visibility
        and evidence.measurable_ratio >= bands.high_measurable_ratio
        and evidence.view_support >= bands.high_view_support
        and evidence.evaluable_reps >= bands.high_min_reps
    ):
        return apply_ceilings(Reliability.HIGH, evidence, bands)

    if (
        evidence.landmark_visibility >= bands.medium_visibility
        and evidence.measurable_ratio >= bands.medium_measurable_ratio
        and evidence.view_support >= bands.medium_view_support
    ):
        level = Reliability.MEDIUM
    else:
        level = Reliability.LOW

    return apply_ceilings(level, evidence, bands)


def apply_ceilings(
    level: Reliability, evidence: MetricEvidence, bands: ReliabilityBands
) -> Reliability:
    """
    Two caps visibility can't capture: a person filmed from across the gym (small
    but still "visible"), and a clip validation already marked as limited.
    """
    scale = evidence.subject_scale_px
    if np.isfinite(scale):
        if scale < bands.poor_subject_pixels:
            level = Reliability.LOW
        elif scale < bands.min_subject_pixels and level is Reliability.HIGH:
            level = Reliability.MEDIUM

    if evidence.recording_quality != "good" and level is Reliability.HIGH:
        level = Reliability.MEDIUM

    return level


def overall_reliability(levels: list[Reliability]) -> Reliability:
    """Median across the metrics that gave a verdict, so one weak or strong check
    doesn't decide the whole session."""
    usable = [level for level in levels if level is not Reliability.CANNOT_ASSESS]
    if not usable:
        return Reliability.CANNOT_ASSESS
    ranks = sorted(level.rank for level in usable)
    median_rank = ranks[len(ranks) // 2] if len(ranks) % 2 else ranks[len(ranks) // 2 - 1]
    return next(level for level in Reliability if level.rank == median_rank)
