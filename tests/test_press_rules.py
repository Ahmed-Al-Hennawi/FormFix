"""
Shoulder-press rules, driven from hand-built repetitions so each test isolates
one comparison between a measured value and a configured range. The clean
cases matter as much as the faulty ones: a rule set only ever tested on faults
will happily flag everything.
"""

from __future__ import annotations

from analysis.models import (
    ROM_COMPLETE,
    ROM_LIMITED_BOTTOM,
    ROM_LIMITED_OVERALL,
    ROM_LIMITED_TOP,
    ROM_NOT_ASSESSABLE,
    CameraOrientation,
    PressRep,
    Reliability,
    RepetitionSegmentation,
    RuleStatus,
)
from exercises.press.config import DEFAULT_CONFIG, rule_specs
from exercises.press.rules import (
    classify_rom,
    classify_symmetry,
    evaluate_all,
    rule_alignment,
    rule_range_of_motion,
    rule_symmetry,
)

CONFIG = DEFAULT_CONFIG
SPECS = {spec.rule_id: spec for spec in rule_specs(CONFIG)}


def make_rep(
    number: int = 1,
    top_elbow: float = 168.0,
    bottom_elbow: float = 85.0,
    angle_difference: float = 2.0,
    height_difference: float = 0.02,
    rom_difference: float = 2.0,
    left_offset: float = 0.15,
    right_offset: float = 0.15,
    symmetry_reliable: bool = True,
    alignment_reliable: bool = True,
    higher_side: str = "",
) -> PressRep:
    """A rep with the measurements a rule reads set directly."""
    return PressRep(
        number=number,
        segmentation=RepetitionSegmentation(
            start_frame=0,
            towards_start_frame=0,
            extreme_start_frame=10,
            extreme_end_frame=14,
            return_start_frame=15,
            end_frame=30,
        ),
        start_time=0.0,
        extreme_time=0.6,
        end_time=1.4,
        duration=1.4,
        left_top_elbow_angle=top_elbow,
        right_top_elbow_angle=top_elbow,
        left_bottom_elbow_angle=bottom_elbow,
        right_bottom_elbow_angle=bottom_elbow,
        top_elbow_angle=top_elbow,
        bottom_elbow_angle=bottom_elbow,
        left_rom_degrees=top_elbow - bottom_elbow,
        right_rom_degrees=top_elbow - bottom_elbow - rom_difference,
        rom_degrees=top_elbow - bottom_elbow,
        max_elbow_angle_difference=angle_difference,
        max_elbow_angle_difference_frame=12,
        max_wrist_height_difference=height_difference,
        max_wrist_height_difference_frame=12,
        rom_difference=rom_difference,
        top_timing_difference=0.05,
        higher_side=higher_side,
        max_left_alignment_offset=left_offset,
        max_right_alignment_offset=right_offset,
        max_left_alignment_frame=12,
        max_right_alignment_frame=12,
        valid_frame_ratio=1.0,
        mean_landmark_visibility=0.9,
        both_arms_ratio=1.0,
        symmetry_reliable=symmetry_reliable,
        alignment_reliable=alignment_reliable,
    )


# --- Rule 1 - arm symmetry ---


class TestSymmetryClassification:
    def test_a_symmetrical_press_passes(self):
        signals, status = classify_symmetry(make_rep(), CONFIG)
        assert status is RuleStatus.PASS
        assert signals == []

    def test_a_small_natural_difference_is_not_flagged(self):
        # Nobody presses perfectly evenly, and a few degrees is inside
        # MediaPipe's own error anyway, so the bar sits well above 0.
        _, status = classify_symmetry(make_rep(angle_difference=9.0), CONFIG)
        assert status is RuleStatus.PASS

    def test_a_clear_angle_difference_is_flagged(self):
        signals, status = classify_symmetry(make_rep(angle_difference=18.0), CONFIG)
        assert status is RuleStatus.WARNING
        assert "elbow angle" in signals

    def test_a_large_angle_difference_is_a_failure(self):
        _, status = classify_symmetry(make_rep(angle_difference=30.0), CONFIG)
        assert status is RuleStatus.FAIL

    def test_a_height_difference_alone_is_enough_to_flag(self):
        # Both elbows can sit at the same angle with the wrists at very
        # different heights.
        signals, status = classify_symmetry(make_rep(height_difference=0.16), CONFIG)
        assert status is RuleStatus.WARNING
        assert signals == ["wrist height"]

    def test_a_range_difference_alone_is_enough_to_flag(self):
        signals, status = classify_symmetry(make_rep(rom_difference=19.0), CONFIG)
        assert status is RuleStatus.WARNING
        assert signals == ["range of motion"]

    def test_several_signals_are_all_named(self):
        signals, _ = classify_symmetry(
            make_rep(angle_difference=18.0, height_difference=0.16, rom_difference=19.0), CONFIG
        )
        assert set(signals) == {"elbow angle", "wrist height", "range of motion"}

    def test_one_hidden_arm_is_not_assessed_rather_than_passed(self):
        # "Couldn't compare them" is not the same claim as "they were even".
        _, status = classify_symmetry(make_rep(symmetry_reliable=False), CONFIG)
        assert status is RuleStatus.NOT_EVALUABLE


class TestSymmetryRule:
    def test_a_clean_set_passes_and_says_what_was_checked(self):
        result = rule_symmetry([make_rep(1), make_rep(2)], CONFIG, SPECS["press_symmetry"])
        assert result.status is RuleStatus.PASS
        assert "symmetrical" in result.explanation
        assert result.correction == ""

    def test_the_explanation_names_the_leading_arm(self):
        reps = [
            make_rep(1, angle_difference=20.0, higher_side="right"),
            make_rep(2, angle_difference=22.0, higher_side="right"),
        ]
        result = rule_symmetry(reps, CONFIG, SPECS["press_symmetry"])
        assert "right arm stayed higher than your left arm" in result.explanation
        assert "controlled pace" in result.correction

    def test_the_explanation_avoids_a_medical_claim(self):
        reps = [make_rep(1, angle_difference=30.0), make_rep(2, angle_difference=30.0)]
        text = (rule_symmetry(reps, CONFIG, SPECS["press_symmetry"]).explanation).lower()
        for word in ("injury", "mobility", "impingement", "damage", "shoulder problem"):
            assert word not in text

    def test_the_evidence_carries_every_signal_the_rule_read(self):
        result = rule_symmetry([make_rep(angle_difference=20.0)], CONFIG, SPECS["press_symmetry"])
        evidence = result.per_rep[0].evidence
        assert evidence["max_elbow_angle_difference"] == 20.0
        assert evidence["max_wrist_height_difference"] is not None
        assert evidence["rom_difference"] is not None
        assert evidence["top_timing_difference"] is not None

    def test_unassessable_repetitions_do_not_become_failures(self):
        result = rule_symmetry(
            [make_rep(1, angle_difference=40.0, symmetry_reliable=False)],
            CONFIG,
            SPECS["press_symmetry"],
        )
        assert result.status is RuleStatus.NOT_EVALUABLE
        assert result.correction == ""


# --- Rule 2 - elbow / wrist alignment ---


class TestAlignmentRule:
    def test_stacked_wrists_pass(self):
        result = rule_alignment([make_rep(1), make_rep(2)], CONFIG, SPECS["press_alignment"])
        assert result.status is RuleStatus.PASS
        assert result.correction == ""

    def test_a_clear_drift_on_one_side_is_flagged_and_named(self):
        reps = [make_rep(1, left_offset=0.45), make_rep(2, left_offset=0.44)]
        result = rule_alignment(reps, CONFIG, SPECS["press_alignment"])
        assert result.status is RuleStatus.WARNING
        assert "left wrist" in result.explanation
        assert result.per_rep[0].evidence["worst_side"] == "left"

    def test_a_large_drift_is_a_failure(self):
        reps = [make_rep(1, right_offset=0.60), make_rep(2, right_offset=0.58)]
        result = rule_alignment(reps, CONFIG, SPECS["press_alignment"])
        assert result.status is RuleStatus.FAIL
        assert "right wrist" in result.explanation

    def test_the_two_sides_are_judged_independently(self):
        result = rule_alignment(
            [make_rep(1, left_offset=0.60, right_offset=0.05)], CONFIG, SPECS["press_alignment"]
        )
        assert result.status is RuleStatus.FAIL
        assert result.per_rep[0].evidence["worst_side"] == "left"

    def test_the_tolerance_is_not_unrealistically_narrow(self):
        # A forearm is never perfectly vertical, so a modest offset passes.
        result = rule_alignment(
            [make_rep(1, left_offset=0.30, right_offset=0.28)], CONFIG, SPECS["press_alignment"]
        )
        assert result.status is RuleStatus.PASS

    def test_unreliable_landmarks_are_not_assessed(self):
        result = rule_alignment(
            [make_rep(1, left_offset=0.8, alignment_reliable=False)],
            CONFIG,
            SPECS["press_alignment"],
        )
        assert result.status is RuleStatus.NOT_EVALUABLE

    def test_the_evidence_reports_the_offset_in_shoulder_widths(self):
        result = rule_alignment([make_rep(1, left_offset=0.45)], CONFIG, SPECS["press_alignment"])
        assert result.evidence["warn_shoulder_widths"] == CONFIG.ALIGNMENT_OFFSET_WARN
        assert result.per_rep[0].evidence["max_alignment_offset"] == 0.45


# --- Rule 3 - range of motion ---


class TestRangeOfMotionClassification:
    def test_a_full_press_passes(self):
        category, status = classify_rom(make_rep(), CONFIG)
        assert category == ROM_COMPLETE
        assert status is RuleStatus.PASS

    def test_a_press_that_stops_short_overhead_is_limited_at_the_top(self):
        category, status = classify_rom(make_rep(top_elbow=148.0), CONFIG)
        assert category == ROM_LIMITED_TOP
        assert status is RuleStatus.WARNING

    def test_a_clearly_short_press_is_a_failure(self):
        category, status = classify_rom(make_rep(top_elbow=135.0), CONFIG)
        assert category == ROM_LIMITED_TOP
        assert status is RuleStatus.FAIL

    def test_dumbbells_not_returning_to_the_shoulders_are_limited_at_the_bottom(self):
        category, status = classify_rom(make_rep(bottom_elbow=108.0), CONFIG)
        assert category == ROM_LIMITED_BOTTOM
        assert status is RuleStatus.WARNING

    def test_short_at_both_ends_is_reported_as_an_overall_limit(self):
        category, _ = classify_rom(make_rep(top_elbow=148.0, bottom_elbow=108.0), CONFIG)
        assert category == ROM_LIMITED_OVERALL

    def test_a_locked_elbow_is_not_required(self):
        # 157 deg is short of straight but inside the configured range.
        _, status = classify_rom(make_rep(top_elbow=157.0), CONFIG)
        assert status is RuleStatus.PASS

    def test_a_missing_measurement_is_not_assessed(self):
        category, status = classify_rom(make_rep(top_elbow=float("nan")), CONFIG)
        assert category == ROM_NOT_ASSESSABLE
        assert status is RuleStatus.NOT_EVALUABLE


class TestRangeOfMotionRule:
    def test_the_wording_avoids_implying_that_deeper_is_always_better(self):
        reps = [make_rep(1, top_elbow=135.0), make_rep(2, top_elbow=136.0)]
        text = rule_range_of_motion(reps, CONFIG, SPECS["press_rom"]).explanation.lower()
        for word in ("must", "always", "safer", "dangerous", "injury"):
            assert word not in text
        assert "configured" in text

    def test_the_correction_offers_a_controlled_range_not_a_forced_one(self):
        reps = [make_rep(1, top_elbow=135.0), make_rep(2, top_elbow=136.0)]
        correction = rule_range_of_motion(reps, CONFIG, SPECS["press_rom"]).correction.lower()
        assert "comfortable" in correction or "control" in correction

    def test_one_short_repetition_among_good_ones_is_a_warning(self):
        reps = [make_rep(1), make_rep(2, top_elbow=135.0), make_rep(3), make_rep(4)]
        result = rule_range_of_motion(reps, CONFIG, SPECS["press_rom"])
        assert result.status is RuleStatus.WARNING


# --- View gating and assembly ---


class TestViewGating:
    def test_a_side_on_recording_cannot_compare_the_arms(self):
        results = {
            r.rule_id: r
            for r in evaluate_all(
                [make_rep(1), make_rep(2)],
                CONFIG,
                orientation=CameraOrientation.SIDE,
                side_view_confidence=1.0,
            )
        }
        assert results["press_symmetry"].status is RuleStatus.NOT_EVALUABLE
        assert results["press_alignment"].status is RuleStatus.NOT_EVALUABLE
        assert "one arm hides the other" in results["press_symmetry"].limitation

    def test_a_side_on_recording_still_measures_range_of_motion(self):
        results = {
            r.rule_id: r
            for r in evaluate_all(
                [make_rep(1), make_rep(2)],
                CONFIG,
                orientation=CameraOrientation.SIDE,
                side_view_confidence=1.0,
            )
        }
        assert results["press_rom"].status is RuleStatus.PASS

    def test_a_front_on_recording_supports_every_rule(self):
        results = evaluate_all(
            [make_rep(1), make_rep(2)], CONFIG, orientation=CameraOrientation.FRONTAL
        )
        assert all(r.status is not RuleStatus.NOT_EVALUABLE for r in results)

    def test_an_unassessable_rule_reports_no_reliability(self):
        results = {
            r.rule_id: r for r in evaluate_all([make_rep()], CONFIG, orientation=CameraOrientation.SIDE)
        }
        assert results["press_symmetry"].reliability is Reliability.CANNOT_ASSESS

    def test_every_rule_is_returned_in_a_declared_order(self):
        results = evaluate_all([make_rep()], CONFIG)
        assert [r.rule_id for r in results] == [spec.rule_id for spec in rule_specs(CONFIG)]


class TestFrontalPlaneViewSupport:
    def test_a_nearly_side_on_diagonal_weakens_a_left_right_comparison(self):
        """
        A nearly side-on diagonal scores high on side_view_confidence,
        which is right for trunk lean and backwards for comparing two arms.
        Frontal-plane metrics take the complement.
        """
        from exercises.common.confidence import view_support
        from exercises.common.spec import FRONTAL_VIEWS

        nearly_side_on = view_support(
            FRONTAL_VIEWS, CameraOrientation.DIAGONAL_SIDE, 0.9, frontal_plane=True
        )
        nearly_front_on = view_support(
            FRONTAL_VIEWS, CameraOrientation.DIAGONAL_SIDE, 0.1, frontal_plane=True
        )
        assert nearly_side_on < nearly_front_on
