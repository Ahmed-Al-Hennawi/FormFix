"""
Metric-level reliability: how each verdict gets banded, and how the per-metric
bands roll up into one figure for the run.
"""

from __future__ import annotations

from analysis.models import CameraOrientation, Reliability
from exercises.squat.confidence import (
    MetricEvidence,
    overall_reliability,
    reliability_for,
    view_support,
)
from exercises.squat.config import ANY_VIEW, FRONTAL_VIEWS, SAGITTAL_VIEWS, SquatConfig

CONFIG = SquatConfig()


def ev(**kwargs) -> MetricEvidence:
    base = {
        "metric": "squat_depth",
        "landmark_visibility": 0.9,
        "measurable_ratio": 0.95,
        "view_support": 1.0,
        "evaluable_reps": 3,
        "total_reps": 3,
    }
    base.update(kwargs)
    return MetricEvidence(**base)


class TestViewSupport:
    def test_side_view_fully_supports_sagittal_metrics(self):
        assert view_support(SAGITTAL_VIEWS, CameraOrientation.SIDE, 1.0) == 1.0

    def test_side_view_does_not_support_a_frontal_metric(self):
        assert view_support(FRONTAL_VIEWS, CameraOrientation.SIDE, 1.0) == 0.0

    def test_diagonal_view_scores_its_estimated_confidence(self):
        assert view_support(SAGITTAL_VIEWS, CameraOrientation.DIAGONAL_SIDE, 0.6) == 0.6

    def test_frontal_view_supports_the_metrics_designed_for_it(self):
        assert view_support(FRONTAL_VIEWS, CameraOrientation.FRONTAL, 0.0) == 1.0
        assert view_support(ANY_VIEW, CameraOrientation.FRONTAL, 0.0) == 1.0

    def test_unknown_view_is_analysed_but_never_at_full_support(self):
        support = view_support(SAGITTAL_VIEWS, CameraOrientation.UNKNOWN, 0.0)
        assert 0.0 < support < 1.0


class TestBanding:
    def test_clear_evidence_is_high(self):
        assert reliability_for(ev(), CONFIG) is Reliability.HIGH

    def test_a_single_repetition_is_not_high(self):
        assert reliability_for(ev(evaluable_reps=1), CONFIG) is Reliability.MEDIUM

    def test_partial_occlusion_drops_to_medium(self):
        assert reliability_for(ev(landmark_visibility=0.6), CONFIG) is Reliability.MEDIUM

    def test_thin_evidence_drops_to_low(self):
        assert reliability_for(ev(landmark_visibility=0.3), CONFIG) is Reliability.LOW

    def test_no_camera_support_cannot_be_assessed(self):
        assert reliability_for(ev(view_support=0.0), CONFIG) is Reliability.CANNOT_ASSESS

    def test_no_evaluable_repetition_cannot_be_assessed(self):
        assert reliability_for(ev(evaluable_reps=0), CONFIG) is Reliability.CANNOT_ASSESS

    def test_excellent_visibility_cannot_rescue_an_unsupported_view(self):
        # The bands are floors that all have to be met, not an average - an
        # average here would report "medium" for something the camera can't see.
        strong_but_blind = ev(landmark_visibility=1.0, measurable_ratio=1.0, view_support=0.0)
        assert reliability_for(strong_but_blind, CONFIG) is Reliability.CANNOT_ASSESS


class TestOverall:
    def test_unassessable_metrics_are_excluded(self):
        levels = [Reliability.HIGH, Reliability.CANNOT_ASSESS, Reliability.HIGH]
        assert overall_reliability(levels) is Reliability.HIGH

    def test_one_weak_check_does_not_condemn_the_run(self):
        levels = [Reliability.HIGH, Reliability.HIGH, Reliability.LOW]
        assert overall_reliability(levels) is Reliability.HIGH

    def test_one_strong_check_does_not_rescue_the_run(self):
        levels = [Reliability.LOW, Reliability.LOW, Reliability.HIGH]
        assert overall_reliability(levels) is Reliability.LOW

    def test_nothing_assessable_reports_cannot_assess(self):
        assert overall_reliability([]) is Reliability.CANNOT_ASSESS
        assert overall_reliability([Reliability.CANNOT_ASSESS]) is Reliability.CANNOT_ASSESS

    def test_labels_are_user_facing(self):
        assert Reliability.HIGH.label == "High"
        assert Reliability.CANNOT_ASSESS.label == "Cannot assess"


class TestFrontalPlaneMetricsOnADiagonalCamera:
    """
    side_view_confidence says how side-on the camera is, which means
    opposite things to different metrics: good for trunk lean, bad for
    left/right knee evenness where the legs line up behind each other.
    Feeding both the same number gave the worst diagonals the highest
    confidence, so frontal-plane metrics take the complement.
    """

    def test_a_sagittal_metric_prefers_a_side_on_diagonal(self):
        assert view_support(SAGITTAL_VIEWS, CameraOrientation.DIAGONAL_SIDE, 0.9) > view_support(
            SAGITTAL_VIEWS, CameraOrientation.DIAGONAL_SIDE, 0.2
        )

    def test_a_frontal_metric_prefers_a_front_on_diagonal(self):
        nearly_side_on = view_support(
            FRONTAL_VIEWS, CameraOrientation.DIAGONAL_SIDE, 0.9, frontal_plane=True
        )
        nearly_front_on = view_support(
            FRONTAL_VIEWS, CameraOrientation.DIAGONAL_SIDE, 0.2, frontal_plane=True
        )
        assert nearly_side_on < nearly_front_on

    def test_the_squat_declares_knee_symmetry_as_a_frontal_plane_metric(self):
        from analysis.models import METRIC_DEPTH, METRIC_KNEE_SYMMETRY, METRIC_TORSO_LEAN
        from exercises.squat.landmarks import is_frontal_plane

        assert is_frontal_plane(METRIC_KNEE_SYMMETRY)
        assert not is_frontal_plane(METRIC_DEPTH)
        assert not is_frontal_plane(METRIC_TORSO_LEAN)

    def test_a_frontal_view_still_fully_supports_a_frontal_metric(self):
        assert view_support(FRONTAL_VIEWS, CameraOrientation.FRONTAL, 0.0, frontal_plane=True) == 1.0
