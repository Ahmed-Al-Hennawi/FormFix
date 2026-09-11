"""
Tests for the rule engine: a rule only runs if the camera view supports it,
only fires if the violation lasted, and says why it was skipped instead of
guessing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from analysis.models import (
    CameraOrientation,
    FrameMetrics,
    Reliability,
    RuleStatus,
)
from exercises.squat.config import DEFAULT_RULES, SquatConfig, rule_specs
from exercises.squat.feedback import (
    FEEDBACK_TEMPLATES,
    build_findings,
    build_not_assessed,
    build_overview,
    build_rep_summaries,
    build_summary,
    template_for,
)
from exercises.squat.rules import (
    evaluate_all,
    rule_descent_control,
    rule_torso_lean,
)
from tests.test_rules import make_rep

CONFIG = SquatConfig()


def rep_with(**changes):
    """A good repetition with specific fields overridden."""
    base = make_rep(
        number=changes.pop("number", 1),
        min_knee_angle=changes.pop("min_knee_angle", 92.0),
    )
    defaults = {
        "descent_start_frame": base.start_frame,
        "bottom_start_frame": base.bottom_frame - 3,
        "bottom_end_frame": base.bottom_frame + 3,
        "ascent_start_frame": base.bottom_frame + 4,
        "descent_duration": 1.2,
        "bottom_duration": 0.2,
        "ascent_duration": 1.2,
        "symmetry_reliable": True,
        "max_knee_asymmetry": 3.0,
        "max_knee_asymmetry_frame": base.bottom_frame,
        "valid_frame_ratio": 1.0,
        "mean_landmark_visibility": 0.9,
    }
    defaults.update(changes)
    return dataclasses.replace(base, **defaults)


def frame_series(rep, attribute: str, values) -> list[FrameMetrics]:
    """A per-frame track over one rep, so persistence gating has a real series."""
    metrics = [
        FrameMetrics(frame_index=i, timestamp=i / 30.0, valid=False) for i in range(rep.end_frame + 5)
    ]
    for offset, value in enumerate(values):
        index = rep.start_frame + offset
        if index > rep.end_frame:
            break
        metrics[index] = FrameMetrics(
            frame_index=index,
            timestamp=index / 30.0,
            valid=True,
            **{attribute: float(value)},
        )
    return metrics


# --- Persistence ---


class TestPersistentErrorDetection:
    def _torso_spec(self):
        return next(s for s in rule_specs(CONFIG) if s.rule_id == "torso_lean")

    def test_one_noisy_frame_does_not_produce_a_finding(self):
        rep = rep_with(max_torso_lean=CONFIG.TORSO_LEAN_WARN + 8)
        span = rep.end_frame - rep.start_frame + 1
        values = np.full(span, 20.0)
        values[10] = CONFIG.TORSO_LEAN_WARN + 8  # one jittered frame
        metrics = frame_series(rep, "torso_lean", values)

        result = rule_torso_lean([rep], 5.0, CONFIG, self._torso_spec(), metrics)
        assert result.status is RuleStatus.PASS
        assert result.per_rep[0].violating_frames == 1

    def test_a_sustained_lean_does_produce_a_finding(self):
        rep = rep_with(max_torso_lean=CONFIG.TORSO_LEAN_WARN + 8)
        span = rep.end_frame - rep.start_frame + 1
        values = np.full(span, 20.0)
        values[10:40] = CONFIG.TORSO_LEAN_WARN + 8
        metrics = frame_series(rep, "torso_lean", values)

        result = rule_torso_lean([rep], 5.0, CONFIG, self._torso_spec(), metrics)
        assert result.status is RuleStatus.WARNING
        assert result.per_rep[0].violating_frames == 30
        assert result.per_rep[0].violation_ratio > CONFIG.TORSO_LEAN_MIN_VIOLATION_RATIO

    def test_persistence_evidence_is_recorded_even_when_the_rule_passes(self):
        rep = rep_with(max_torso_lean=20.0)
        span = rep.end_frame - rep.start_frame + 1
        metrics = frame_series(rep, "torso_lean", np.full(span, 20.0))
        result = rule_torso_lean([rep], 5.0, CONFIG, self._torso_spec(), metrics)
        assert result.per_rep[0].phase_frames == span
        assert result.per_rep[0].violating_frames == 0


# --- Timing rule ---


class TestDescentControlRule:
    def test_controlled_descent_passes(self):
        result = rule_descent_control([rep_with(descent_duration=1.1)], CONFIG)
        assert result.status is RuleStatus.PASS

    def test_quick_descent_warns(self):
        duration = (CONFIG.DESCENT_MIN_DURATION + CONFIG.DESCENT_FAST_DURATION) / 2
        result = rule_descent_control([rep_with(descent_duration=duration)], CONFIG)
        assert result.status is RuleStatus.WARNING

    def test_dropped_descent_fails(self):
        result = rule_descent_control(
            [rep_with(descent_duration=CONFIG.DESCENT_FAST_DURATION / 2)], CONFIG
        )
        assert result.status is RuleStatus.FAIL

    def test_unmeasured_timing_is_not_evaluable(self):
        result = rule_descent_control([rep_with(descent_duration=float("nan"))], CONFIG)
        assert result.status is RuleStatus.NOT_EVALUABLE

    def test_wording_avoids_injury_claims(self):
        result = rule_descent_control([rep_with(descent_duration=0.1)], CONFIG)
        text = f"{result.explanation} {result.correction}".lower()
        assert not any(word in text for word in ("injur", "damage", "danger", "harm"))


# --- View gating ---


def evaluate(orientation: CameraOrientation, confidence: float = 1.0, reps=None):
    reps = reps if reps is not None else [rep_with(number=1), rep_with(number=2)]
    return {
        rule.rule_id: rule
        for rule in evaluate_all(
            reps,
            standing_knee_angle=178.0,
            standing_torso_lean=5.0,
            config=CONFIG,
            orientation=orientation,
            side_view_confidence=confidence,
        )
    }


class TestViewGating:
    def test_side_view_runs_the_sagittal_rules(self):
        rules = evaluate(CameraOrientation.SIDE)
        for rule_id in ("squat_depth", "torso_lean", "return_to_standing", "descent_control"):
            assert rules[rule_id].status is not RuleStatus.NOT_EVALUABLE, rule_id

    def test_front_view_gates_out_the_sagittal_rules(self):
        rules = evaluate(CameraOrientation.FRONTAL, confidence=0.0)
        for rule_id in ("squat_depth", "torso_lean", "return_to_standing", "descent_control"):
            assert rules[rule_id].status is RuleStatus.NOT_EVALUABLE, rule_id
            assert rules[rule_id].limitation
            assert rules[rule_id].reliability is Reliability.CANNOT_ASSESS
        # heel lift doesn't depend on the camera position
        assert rules["heel_lift"].status is not RuleStatus.NOT_EVALUABLE

    def test_heel_check_is_view_independent(self):
        for orientation in (
            CameraOrientation.SIDE,
            CameraOrientation.FRONTAL,
            CameraOrientation.DIAGONAL_SIDE,
        ):
            rules = evaluate(orientation)
            assert rules["heel_lift"].status is not RuleStatus.NOT_EVALUABLE

    def test_diagonal_view_still_runs_the_sagittal_rules(self):
        rules = evaluate(CameraOrientation.DIAGONAL_SIDE, confidence=0.6)
        assert rules["squat_depth"].status is not RuleStatus.NOT_EVALUABLE
        assert rules["return_to_standing"].status is not RuleStatus.NOT_EVALUABLE

    def test_unknown_view_is_analysed_conservatively(self):
        rules = evaluate(CameraOrientation.UNKNOWN, confidence=0.0)
        assert rules["squat_depth"].status is not RuleStatus.NOT_EVALUABLE

    def test_a_gated_rule_never_scores_as_a_failure(self):
        rules = evaluate(CameraOrientation.FRONTAL, confidence=0.0)
        depth = rules["squat_depth"]
        assert all(o.status is RuleStatus.NOT_EVALUABLE for o in depth.per_rep)


# --- Declared specifications ---


class TestRuleSpecifications:
    def test_every_rule_declares_its_traceability_metadata(self):
        for spec in DEFAULT_RULES:
            assert spec.metric
            assert spec.phase
            assert spec.supported_views
            assert spec.feedback_key in FEEDBACK_TEMPLATES
            assert spec.minimum_persistence_frames >= 1
            assert 0.0 <= spec.min_violation_ratio <= 1.0
            assert 0.0 <= spec.minimum_visibility <= 1.0

    def test_thresholds_are_ranges_not_exact_targets(self):
        for spec in DEFAULT_RULES:
            assert spec.acceptable_min is not None or spec.acceptable_max is not None
            assert spec.tolerance >= 0.0

    def test_threshold_provenance_is_recorded(self):
        assert all(spec.threshold_source for spec in DEFAULT_RULES)

    def test_results_carry_the_declared_metadata(self):
        rules = evaluate(CameraOrientation.SIDE)
        depth = rules["squat_depth"]
        assert depth.metric == "squat_depth"
        assert depth.phase == "bottom"
        assert "side" in depth.supported_views
        assert depth.feedback_key == "depth"


# --- Feedback ---


class TestFeedbackGeneration:
    def test_every_rule_maps_to_a_feedback_template(self):
        rules = evaluate(CameraOrientation.DIAGONAL_SIDE, confidence=0.7)
        for rule in rules.values():
            template = template_for(rule)
            assert template.title
            assert template.rep_pass and template.rep_issue and template.overview

    def test_findings_carry_their_feedback_key(self):
        reps = [rep_with(number=1, min_knee_angle=130.0, hip_above_knee_at_bottom=0.6)]
        rules = list(evaluate(CameraOrientation.SIDE, reps=reps).values())
        findings = build_findings(rules)
        assert findings
        assert findings[0].feedback_key == "depth"
        assert findings[0].title == FEEDBACK_TEMPLATES["depth"].title

    def test_findings_never_use_internal_vocabulary(self):
        reps = [rep_with(number=1, min_knee_angle=130.0, hip_above_knee_at_bottom=0.6)]
        rules = list(evaluate(CameraOrientation.SIDE, reps=reps).values())
        for finding in build_findings(rules):
            text = f"{finding.title} {finding.issue} {finding.correction}".lower()
            for banned in ("metric", "threshold 2", "not_evaluable", "rule_id", "nan"):
                assert banned not in text

    def test_no_contradictory_message_for_the_same_check(self):
        reps = [rep_with(number=1, min_knee_angle=130.0, hip_above_knee_at_bottom=0.6)]
        rules = list(evaluate(CameraOrientation.SIDE, reps=reps).values())
        summary = build_summary(reps, 0, rules)
        flagged = {f.rule_id for f in build_findings(rules)}
        praised = set(summary.positives)
        # depth is flagged, so there shouldn't be depth praise as well
        assert "squat_depth" in flagged
        assert "Good squat depth" not in praised

    def test_cannot_assess_is_reported_separately_from_a_failure(self):
        reps = [rep_with(number=1)]
        rules = list(evaluate(CameraOrientation.FRONTAL, confidence=0.0, reps=reps).values())
        not_assessed = build_not_assessed(rules)
        assert any(item.metric == "squat_depth" for item in not_assessed)
        assert all(item.reason for item in not_assessed)
        # ...and it doesn't also turn up as something to improve.
        assert all(f.rule_id != "squat_depth" for f in build_findings(rules))


class TestPerRepetitionReporting:
    def test_each_repetition_gets_its_own_verdict(self):
        reps = [
            rep_with(number=1),
            rep_with(number=2, min_knee_angle=130.0, hip_above_knee_at_bottom=0.6),
            rep_with(number=3),
        ]
        rules = list(evaluate(CameraOrientation.SIDE, reps=reps).values())
        summaries = build_rep_summaries(reps, rules)

        assert [s.rep_number for s in summaries] == [1, 2, 3]
        assert summaries[0].headline == "Good repetition"
        assert summaries[1].status is RuleStatus.FAIL
        assert "Depth" in summaries[1].headline
        assert summaries[2].headline == "Good repetition"

    def test_a_repetition_lists_what_it_could_not_be_judged_on(self):
        reps = [rep_with(number=1)]
        rules = list(evaluate(CameraOrientation.FRONTAL, confidence=0.0, reps=reps).values())
        summary = build_rep_summaries(reps, rules)[0]
        assert "Depth" in summary.not_assessed
        assert "Heels planted" in summary.passed

    def test_overview_counts_how_often_each_check_was_acceptable(self):
        reps = [
            rep_with(number=1),
            rep_with(number=2, min_knee_angle=130.0, hip_above_knee_at_bottom=0.6),
        ]
        rules = list(evaluate(CameraOrientation.SIDE, reps=reps).values())
        overview = build_overview(rules)
        assert "Depth: 1 of 2 repetitions acceptable" in overview

    def test_a_check_that_could_not_be_assessed_is_left_out_of_the_overview(self):
        reps = [rep_with(number=1), rep_with(number=2)]
        rules = list(evaluate(CameraOrientation.FRONTAL, confidence=0.0, reps=reps).values())
        overview = build_overview(rules)
        # left out instead of showing "0 of 2 acceptable"
        assert not any(line.startswith("Depth") for line in overview)

    def test_summary_aggregates_reliability_and_not_assessed(self):
        reps = [rep_with(number=1), rep_with(number=2)]
        rules = list(evaluate(CameraOrientation.SIDE, reps=reps).values())
        summary = build_summary(reps, 0, rules)
        assert summary.complete_reps == 2
        assert summary.rep_summaries
        assert summary.overview
        assert summary.reliability is not Reliability.CANNOT_ASSESS


@pytest.mark.parametrize("orientation", list(CameraOrientation))
def test_some_check_always_survives_a_recognised_view(orientation):
    """No camera view should switch off every check at once."""
    rules = evaluate(orientation, confidence=0.5)
    assert any(rule.status is not RuleStatus.NOT_EVALUABLE for rule in rules.values())
