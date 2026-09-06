"""
How a validation verdict reaches the interface. A limited recording still
renders as an analysis, with warnings and "not assessed" rows; a rejected one
carries a typed reason and the developer diagnostics.
"""

from __future__ import annotations

from pathlib import Path

from analysis.models import (
    METRIC_DEPTH,
    AnalysisFailure,
    CameraOrientation,
    FailureCode,
    RejectionCode,
    RepRuleOutcome,
    RuleResult,
    RuleStatus,
    SessionSummary,
    SquatAnalysisResult,
    ValidationResult,
    VideoMetadata,
)
from utils.analysis import _failure_result, _to_ui_result

VIDEO = VideoMetadata(
    path=Path("clip.mp4"), width=720, height=1280, fps=30.0, frame_count=300, duration=10.0
)


def rule(rule_id: str, status: RuleStatus) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        title=rule_id.replace("_", " ").title(),
        status=status,
        explanation="explanation",
        per_rep=[RepRuleOutcome(rep_number=1, status=status)],
    )


def analysis(validation: ValidationResult, rules: list[RuleResult]) -> SquatAnalysisResult:
    return SquatAnalysisResult(
        success=True,
        exercise="Squat",
        analysis_side="left",
        video=VIDEO,
        validation=validation,
        reps=[],
        rule_results=rules,
        summary=SessionSummary(complete_reps=3, partial_movements=0, score=80, score_formula="f"),
        annotated_video_path=None,
    )


def test_limited_recording_is_rendered_as_an_analysis():
    validation = ValidationResult()
    validation.add_warning("The camera is not fully side-on.")
    validation.orientation = CameraOrientation.DIAGONAL_SIDE
    validation.side_view_confidence = 0.6

    ui = _to_ui_result(analysis(validation, [rule("squat_depth", RuleStatus.PASS)]), "squat", "c.mp4")

    assert ui.success
    assert ui.recording_quality == "limited"
    assert ui.camera_orientation == "diagonal_side"
    assert ui.score_available
    assert ui.warnings


def test_unobservable_metric_becomes_a_not_assessed_row_without_a_score():
    validation = ValidationResult()
    validation.add_warning("Front-on recording.", METRIC_DEPTH)
    validation.orientation = CameraOrientation.FRONTAL

    depth = rule("squat_depth", RuleStatus.NOT_EVALUABLE)
    depth.limitation = "Depth needs a side view."
    ui = _to_ui_result(analysis(validation, [depth]), "squat", "c.mp4")

    row = next(r for r in ui.rows if r.rule_id == "squat_depth")
    assert row.badge == "Not assessed"
    assert row.detail == "Depth needs a side view."
    assert ui.score_available is False  # nothing measurable, so no score
    assert ui.limited_metrics == [METRIC_DEPTH]


def test_rejection_carries_reason_code_and_diagnostics():
    validation = ValidationResult()
    validation.add_error("Not enough usable frames.", RejectionCode.IMPORTANT_LANDMARKS_MISSING)

    failure = AnalysisFailure(
        FailureCode.INSUFFICIENT_VISIBILITY,
        "Not enough usable frames.",
        suggestions=["Record from the side."],
        validation=validation,
    )
    ui = _failure_result(failure, "squat", "c.mp4")

    assert ui.success is False
    assert ui.recording_quality == "unusable"
    assert ui.debug["failure_code"] == "insufficient_visibility"
    assert ui.debug["validation"]["reason_codes"] == ["important_landmarks_missing"]


def test_per_rep_results_and_reliability_reach_the_interface():
    """Per-rep verdicts, the overview and the not-assessed block, all three."""
    from analysis.models import NotAssessedItem, Reliability, RepSummary

    validation = ValidationResult()
    depth = rule("squat_depth", RuleStatus.PASS)
    depth.reliability = Reliability.HIGH
    depth.feedback_key = "depth"
    heel = rule("heel_lift", RuleStatus.NOT_EVALUABLE)
    heel.reliability = Reliability.CANNOT_ASSESS
    heel.limitation = "Your heels were not visible enough in this recording."

    result = analysis(validation, [depth, heel])
    result.summary.rep_summaries = [
        RepSummary(
            rep_number=1,
            status=RuleStatus.PASS,
            headline="Good repetition",
            passed=["Depth"],
            not_assessed=["Heel stability"],
            reliability=Reliability.HIGH,
            bottom_time=4.2,
        )
    ]
    result.summary.overview = ["Depth: 1 of 1 repetitions acceptable"]
    result.summary.not_assessed = [
        NotAssessedItem(metric="heel_lift", title="Heel stability", reason="heels not visible")
    ]
    result.summary.reliability = Reliability.HIGH

    ui = _to_ui_result(result, "squat", "clip.mp4")

    assert ui.reliability == "High"
    assert [r.number for r in ui.rep_results] == [1]
    assert ui.rep_results[0].headline == "Good repetition"
    assert ui.rep_results[0].not_assessed == ["Heel stability"]
    assert ui.rep_results[0].timestamp == "00:04.2"
    assert ui.overview == ["Depth: 1 of 1 repetitions acceptable"]
    assert [n.title for n in ui.not_assessed] == ["Heel stability"]
    # Reliability is shown per check, not just once for the run.
    assert next(r for r in ui.rows if r.rule_id == "squat_depth").reliability == "High"
    assert next(r for r in ui.rows if r.rule_id == "heel_lift").reliability == "Cannot assess"
