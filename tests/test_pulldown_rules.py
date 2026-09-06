"""
Lat-pulldown rules, driven from hand-built repetitions. A rule only compares a
measured value against a configured range, so setting the value directly keeps
these independent of the measurement layer and makes the false-positive cases
easy to write.
"""

from __future__ import annotations

from analysis.models import (
    ROM_COMPLETE,
    ROM_LIMITED_BOTTOM,
    ROM_LIMITED_OVERALL,
    ROM_LIMITED_TOP,
    ROM_NOT_ASSESSABLE,
    CameraOrientation,
    PulldownRep,
    Reliability,
    RepetitionSegmentation,
    RuleStatus,
)
from exercises.pulldown.config import DEFAULT_CONFIG, rule_specs
from exercises.pulldown.rules import (
    classify_rom,
    evaluate_all,
    rule_range_of_motion,
    rule_torso_movement,
)

CONFIG = DEFAULT_CONFIG
SPECS = {spec.rule_id: spec for spec in rule_specs(CONFIG)}


def make_rep(
    number: int = 1,
    top_elbow: float = 172.0,
    bottom_elbow: float = 80.0,
    torso_excursion: float = 3.0,
    torso_at_bottom: float = 14.0,
    arms_reliable: bool = True,
    torso_reliable: bool = True,
    torso_mode: str = "posterior",
) -> PulldownRep:
    """A rep with the measurements a rule reads set directly."""
    rom = top_elbow - bottom_elbow
    return PulldownRep(
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
        extreme_time=0.5,
        end_time=1.2,
        duration=1.2,
        top_elbow_angle=top_elbow,
        bottom_elbow_angle=bottom_elbow,
        rom_degrees=rom,
        wrist_rise_at_top=0.6,
        wrist_rise_at_bottom=0.1,
        wrist_travel=0.5,
        elbow_travel=0.4,
        torso_at_top=12.0,
        torso_at_bottom=torso_at_bottom,
        max_torso_excursion=torso_excursion,
        max_torso_excursion_frame=12,
        torso_mode=torso_mode,
        peak_torso_velocity=8.0,
        valid_frame_ratio=1.0,
        mean_landmark_visibility=0.9,
        arms_reliable=arms_reliable,
        torso_reliable=torso_reliable,
    )


# --- Rule 1 - range of motion ---


class TestRangeOfMotionClassification:
    def test_a_full_repetition_passes(self):
        category, status = classify_rom(make_rep(), CONFIG)
        assert category == ROM_COMPLETE
        assert status is RuleStatus.PASS

    def test_arms_that_do_not_return_to_extension_are_limited_at_the_top(self):
        category, status = classify_rom(make_rep(top_elbow=142.0), CONFIG)
        assert category == ROM_LIMITED_TOP
        assert status is RuleStatus.WARNING

    def test_a_clearly_restricted_top_is_a_failure_not_a_warning(self):
        category, status = classify_rom(make_rep(top_elbow=125.0), CONFIG)
        assert category == ROM_LIMITED_TOP
        assert status is RuleStatus.FAIL

    def test_a_pull_that_stops_early_is_limited_at_the_bottom(self):
        category, status = classify_rom(make_rep(bottom_elbow=108.0), CONFIG)
        assert category == ROM_LIMITED_BOTTOM
        assert status is RuleStatus.WARNING

    def test_a_clearly_short_pull_is_a_failure(self):
        category, status = classify_rom(make_rep(bottom_elbow=128.0), CONFIG)
        assert category == ROM_LIMITED_BOTTOM
        assert status is RuleStatus.FAIL

    def test_short_at_both_ends_is_reported_as_an_overall_limit(self):
        category, status = classify_rom(make_rep(top_elbow=142.0, bottom_elbow=108.0), CONFIG)
        assert category == ROM_LIMITED_OVERALL

    def test_unreliable_arms_are_not_assessed_rather_than_failed(self):
        category, status = classify_rom(make_rep(arms_reliable=False), CONFIG)
        assert category == ROM_NOT_ASSESSABLE
        assert status is RuleStatus.NOT_EVALUABLE

    def test_a_missing_measurement_is_not_assessed(self):
        category, status = classify_rom(make_rep(top_elbow=float("nan")), CONFIG)
        assert category == ROM_NOT_ASSESSABLE
        assert status is RuleStatus.NOT_EVALUABLE

    def test_the_top_criterion_does_not_demand_a_locked_elbow(self):
        # 152 deg is short of straight but inside the configured range. No
        # point asking anyone to hyperextend under load.
        _, status = classify_rom(make_rep(top_elbow=152.0), CONFIG)
        assert status is RuleStatus.PASS


class TestRangeOfMotionRule:
    def test_a_clean_set_passes_and_says_so_without_a_correction(self):
        result = rule_range_of_motion(
            [make_rep(1), make_rep(2), make_rep(3)], CONFIG, SPECS["pulldown_rom"]
        )
        assert result.status is RuleStatus.PASS
        assert result.correction == ""
        assert "configured movement range" in result.explanation

    def test_the_explanation_names_the_end_that_fell_short(self):
        reps = [make_rep(1, bottom_elbow=125.0), make_rep(2, bottom_elbow=124.0)]
        result = rule_range_of_motion(reps, CONFIG, SPECS["pulldown_rom"])
        assert result.status is RuleStatus.FAIL
        assert "pull stopped early" in result.explanation
        assert "elbows" in result.correction

    def test_a_single_bad_repetition_among_good_ones_is_a_warning_not_a_failure(self):
        reps = [make_rep(1), make_rep(2, bottom_elbow=125.0), make_rep(3), make_rep(4)]
        result = rule_range_of_motion(reps, CONFIG, SPECS["pulldown_rom"])
        assert result.status is RuleStatus.WARNING
        flagged = [o.rep_number for o in result.per_rep if o.status is not RuleStatus.PASS]
        assert flagged == [2]

    def test_the_evidence_carries_the_numbers_the_rule_compared(self):
        result = rule_range_of_motion([make_rep(bottom_elbow=125.0)], CONFIG, SPECS["pulldown_rom"])
        evidence = result.per_rep[0].evidence
        assert evidence["bottom_elbow_angle"] == 125.0
        assert evidence["rom_status"] == ROM_LIMITED_BOTTOM
        assert result.evidence["bottom_flexion_pass_deg"] == CONFIG.ROM_BOTTOM_FLEXION_PASS

    def test_the_rule_carries_its_traceability_metadata(self):
        result = rule_range_of_motion([make_rep()], CONFIG, SPECS["pulldown_rom"])
        assert result.metric == SPECS["pulldown_rom"].metric
        assert result.phase == SPECS["pulldown_rom"].phase
        assert result.supported_views == SPECS["pulldown_rom"].supported_views
        assert result.feedback_key == "pulldown_rom"


# --- Rule 2 - torso movement ---


class TestTorsoRule:
    def test_a_stable_torso_passes(self):
        result = rule_torso_movement([make_rep(1), make_rep(2)], CONFIG, SPECS["pulldown_torso"])
        assert result.status is RuleStatus.PASS
        assert result.correction == ""

    def test_a_modest_natural_lean_is_not_flagged(self):
        # The rule isn't asking for a vertical trunk - a seated pulldown is
        # done with some recline and a few degrees of movement in it.
        result = rule_torso_movement(
            [make_rep(1, torso_excursion=9.0, torso_at_bottom=22.0)], CONFIG, SPECS["pulldown_torso"]
        )
        assert result.status is RuleStatus.PASS

    def test_a_clear_backward_swing_is_flagged(self):
        result = rule_torso_movement(
            [make_rep(1, torso_excursion=19.0), make_rep(2, torso_excursion=20.0)],
            CONFIG,
            SPECS["pulldown_torso"],
        )
        assert result.status is RuleStatus.WARNING
        assert "moved backward" in result.explanation

    def test_a_large_swing_is_a_failure(self):
        result = rule_torso_movement(
            [make_rep(1, torso_excursion=30.0), make_rep(2, torso_excursion=32.0)],
            CONFIG,
            SPECS["pulldown_torso"],
        )
        assert result.status is RuleStatus.FAIL
        assert "driving your elbows" in result.correction

    def test_an_extreme_absolute_posture_is_flagged_even_with_a_small_change(self):
        # 50 deg from vertical the whole way through isn't the exercise we
        # think we're analysing, however still it stayed.
        result = rule_torso_movement(
            [make_rep(1, torso_excursion=4.0, torso_at_bottom=50.0)],
            CONFIG,
            SPECS["pulldown_torso"],
        )
        assert result.status is RuleStatus.FAIL

    def test_unreliable_trunk_landmarks_are_not_assessed(self):
        result = rule_torso_movement(
            [make_rep(1, torso_excursion=40.0, torso_reliable=False)],
            CONFIG,
            SPECS["pulldown_torso"],
        )
        assert result.status is RuleStatus.NOT_EVALUABLE
        assert "not visible clearly enough" in result.explanation
        assert result.correction == ""

    def test_the_wording_drops_the_direction_when_it_could_not_be_established(self):
        result = rule_torso_movement(
            [make_rep(1, torso_excursion=20.0, torso_mode="unsigned")],
            CONFIG,
            SPECS["pulldown_torso"],
        )
        assert "moved backward" not in result.explanation
        assert "away from its starting position" in result.explanation

    def test_the_evidence_records_which_measurement_mode_was_used(self):
        result = rule_torso_movement([make_rep()], CONFIG, SPECS["pulldown_torso"])
        assert result.evidence["measurement_mode"] == "posterior"
        assert result.evidence["warn_deg"] == CONFIG.TORSO_EXCURSION_WARN


# --- View gating and assembly ---


class TestViewGating:
    def test_a_front_on_recording_cannot_assess_torso_movement(self):
        results = {
            r.rule_id: r
            for r in evaluate_all(
                [make_rep(1), make_rep(2)],
                CONFIG,
                orientation=CameraOrientation.FRONTAL,
                side_view_confidence=0.0,
            )
        }
        assert results["pulldown_torso"].status is RuleStatus.NOT_EVALUABLE
        assert results["pulldown_torso"].reliability is Reliability.CANNOT_ASSESS
        assert "front-on" in results["pulldown_torso"].limitation

    def test_a_front_on_recording_still_assesses_range_of_motion(self):
        results = {
            r.rule_id: r
            for r in evaluate_all(
                [make_rep(1), make_rep(2)],
                CONFIG,
                orientation=CameraOrientation.FRONTAL,
                side_view_confidence=0.0,
            )
        }
        assert results["pulldown_rom"].status is RuleStatus.PASS

    def test_a_side_view_supports_both_rules(self):
        results = evaluate_all([make_rep(1), make_rep(2)], CONFIG, orientation=CameraOrientation.SIDE)
        assert all(r.status is not RuleStatus.NOT_EVALUABLE for r in results)

    def test_every_rule_is_returned_in_a_declared_order(self):
        results = evaluate_all([make_rep()], CONFIG)
        assert [r.rule_id for r in results] == [spec.rule_id for spec in rule_specs(CONFIG)]


class TestScoringExcludesUnassessedChecks:
    def test_an_unassessed_check_is_not_scored_as_a_failure(self):
        from exercises.pulldown.feedback import transparent_score

        results = evaluate_all(
            [make_rep(1), make_rep(2)],
            CONFIG,
            orientation=CameraOrientation.FRONTAL,
            side_view_confidence=0.0,
        )
        score, _, evaluable = transparent_score(results)
        # Only the two range-of-motion checks had evidence, both passed.
        assert evaluable == 2
        assert score == 100
