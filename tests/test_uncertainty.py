"""
Tests for the measurement uncertainty layer: the published figures reach the
verdict correctly, and by default it only annotates - nothing changes unless
strict mode is on.
"""

from __future__ import annotations

import pytest

from analysis.models import RepRuleOutcome, RuleResult, RuleStatus
from exercises.common.uncertainty import (
    GONIOMETRY_REFERENCE,
    MEASURED_KEYS,
    METRIC_BANDS,
    PUBLISHED_BANDS,
    annotate_uncertainty,
    band_for,
)
from exercises.squat.config import DEFAULT_CONFIG, rule_specs


def _specs():
    return {spec.rule_id: spec for spec in rule_specs(DEFAULT_CONFIG)}


def _depth_result(*angles: float) -> RuleResult:
    """A depth result with one rep per supplied minimum knee angle."""
    pass_bar = DEFAULT_CONFIG.DEPTH_KNEE_ANGLE_PASS
    warn_bar = DEFAULT_CONFIG.DEPTH_KNEE_ANGLE_WARN
    per_rep = []
    for number, angle in enumerate(angles, start=1):
        if angle <= pass_bar:
            status = RuleStatus.PASS
        elif angle <= warn_bar:
            status = RuleStatus.WARNING
        else:
            status = RuleStatus.FAIL
        per_rep.append(
            RepRuleOutcome(rep_number=number, status=status, evidence={"min_knee_angle": angle})
        )
    worst = max(
        (o.status for o in per_rep),
        key=lambda s: [RuleStatus.PASS, RuleStatus.WARNING, RuleStatus.FAIL].index(s),
    )
    return RuleResult(
        rule_id="squat_depth",
        title="Squat depth",
        status=worst,
        explanation="Your squat was shallow.",
        correction="Sit lower.",
        per_rep=per_rep,
        metric="squat_depth",
    )


class TestTheBandsThemselves:
    def test_every_rule_that_can_be_measured_declares_a_band(self):
        for rule_id in MEASURED_KEYS:
            assert band_for("", rule_id=rule_id) is not None, rule_id

    def test_every_declared_band_resolves_to_a_published_figure(self):
        for rule_id, key in METRIC_BANDS.items():
            assert key in PUBLISHED_BANDS, f"{rule_id} points at an unknown band {key!r}"

    def test_every_derived_band_shows_its_working(self):
        # a derived figure nobody can check is worse than none
        for band in PUBLISHED_BANDS.values():
            if band.derived:
                assert band.derivation.strip(), band.metric

    def test_every_band_cites_something(self):
        for band in PUBLISHED_BANDS.values():
            assert band.source.strip()

    def test_the_occluded_limb_band_is_far_wider_than_the_near_one(self):
        near = band_for("", rule_id="squat_depth")
        far = band_for("", rule_id="squat_depth", far_side=True)
        assert far.sigma > 2 * near.sigma

    def test_a_worse_camera_view_widens_the_band(self):
        good = band_for("", rule_id="squat_depth", view_support=1.0)
        poor = band_for("", rule_id="squat_depth", view_support=0.0)
        assert poor.sigma > good.sigma
        # capped at the 1.58x that Dill et al. measured
        assert poor.sigma == pytest.approx(good.sigma * 1.584, rel=0.01)

    def test_a_systematic_term_is_added_in_quadrature_not_arithmetically(self):
        band = band_for("", rule_id="squat_depth", systematic=4.53)
        assert band.total < band.sigma + 4.53
        assert band.total > band.sigma

    def test_normalised_and_timing_bands_are_not_scaled_by_camera_view(self):
        # Dill et al. measured this for joint angles only, not durations
        for rule_id in ("heel_lift", "descent_control"):
            good = band_for("", rule_id=rule_id, view_support=1.0)
            poor = band_for("", rule_id=rule_id, view_support=0.0)
            assert good.sigma == poor.sigma

    def test_the_goniometry_reference_is_ordered_as_published(self):
        assert (
            GONIOMETRY_REFERENCE["digital_inclinometer"]
            < GONIOMETRY_REFERENCE["long_arm_goniometer"]
            < GONIOMETRY_REFERENCE["visual_estimation"]
        )


class TestAnnotationDoesNotChangeVerdicts:
    def test_a_clear_fault_keeps_its_status(self):
        result = _depth_result(150.0)
        annotate_uncertainty([result], _specs())
        assert result.status is RuleStatus.FAIL

    def test_a_marginal_fault_also_keeps_its_status_by_default(self):
        # 108 vs a 100 deg pass bar is 8 deg, inside the +/-10.7 band - still
        # reported, just labelled
        result = _depth_result(108.0)
        annotate_uncertainty([result], _specs())
        assert result.status is RuleStatus.WARNING

    def test_a_passing_rule_gains_a_band_but_no_sentence(self):
        result = _depth_result(85.0)
        annotate_uncertainty([result], _specs())
        assert result.status is RuleStatus.PASS
        assert "measurement_uncertainty" in result.evidence
        assert "Measurement note" not in result.explanation

    def test_a_rule_that_could_not_be_evaluated_is_left_alone(self):
        result = RuleResult(
            rule_id="squat_depth",
            title="Squat depth",
            status=RuleStatus.NOT_EVALUABLE,
            explanation="Not assessed.",
            metric="squat_depth",
        )
        annotate_uncertainty([result], _specs())
        assert "measurement_uncertainty" not in result.evidence
        assert result.explanation == "Not assessed."


class TestWhatTheNoteSays:
    def test_a_marginal_finding_is_called_indicative(self):
        result = _depth_result(108.0)
        annotate_uncertainty([result], _specs())
        assert "indicative rather than established" in result.explanation
        assert result.evidence["measurement_uncertainty"]["beyond_measurement_error"] is False

    def test_a_large_deviation_is_reported_as_clearing_the_band(self):
        result = _depth_result(150.0)
        annotate_uncertainty([result], _specs())
        assert "clears" in result.explanation
        assert result.evidence["measurement_uncertainty"]["beyond_measurement_error"] is True

    def test_the_margin_is_measured_against_the_pass_bar(self):
        result = _depth_result(130.0)
        annotate_uncertainty([result], _specs())
        note = result.evidence["measurement_uncertainty"]
        assert note["worst_margin"] == pytest.approx(130.0 - DEFAULT_CONFIG.DEPTH_KNEE_ANGLE_PASS)

    def test_the_worst_repetition_drives_the_note(self):
        result = _depth_result(105.0, 150.0, 102.0)
        annotate_uncertainty([result], _specs())
        note = result.evidence["measurement_uncertainty"]
        assert note["worst_margin"] == pytest.approx(50.0)
        assert note["beyond_measurement_error"] is True
        assert set(note["inconclusive_reps"]) == {1, 3}

    def test_the_note_survives_serialisation_for_the_export(self):
        import json

        result = _depth_result(118.0)
        annotate_uncertainty([result], _specs())
        json.dumps(result.evidence["measurement_uncertainty"])


class TestStrictMode:
    def test_it_downgrades_only_the_marginal_repetitions(self):
        result = _depth_result(108.0, 150.0)
        annotate_uncertainty([result], _specs(), strict=True)
        assert result.per_rep[0].status is RuleStatus.PASS  # 8 deg, inside the band
        assert result.per_rep[1].status is RuleStatus.FAIL  # 50 deg, well clear
        assert result.status is RuleStatus.FAIL

    def test_a_rule_whose_findings_are_all_marginal_becomes_a_pass(self):
        result = _depth_result(108.0, 106.0)
        annotate_uncertainty([result], _specs(), strict=True)
        assert result.status is RuleStatus.PASS
        assert result.correction == ""

    def test_depth_failures_can_never_be_marginal_and_that_is_by_design(self):
        # the 15 deg gap between the pass (100) and fail (115) bars is wider than
        # the 10.7 deg band, so strict mode can soften a depth warning but never
        # a fail
        result = _depth_result(120.0)
        assert result.status is RuleStatus.FAIL
        annotate_uncertainty([result], _specs(), strict=True)
        assert result.status is RuleStatus.FAIL

    def test_strict_mode_never_makes_a_finding_worse(self):
        for angles in [(85.0,), (108.0,), (150.0,), (105.0, 150.0)]:
            order = [RuleStatus.PASS, RuleStatus.WARNING, RuleStatus.FAIL]
            lenient = _depth_result(*angles)
            annotate_uncertainty([lenient], _specs(), strict=False)
            strict = _depth_result(*angles)
            annotate_uncertainty([strict], _specs(), strict=True)
            assert order.index(strict.status) <= order.index(lenient.status)


class TestWhatTheLayerRevealsAboutTheThresholds:
    """If one of these thresholds is tightened, this should flag that it's now
    inside the measurement noise."""

    def test_the_heel_lift_bar_clears_its_own_measurement_error(self):
        # it used to sit inside the band, which is why planted heels were warned
        # about; this fails again if anyone tightens it back below the noise
        band = band_for("", rule_id="heel_lift")
        assert band.total <= DEFAULT_CONFIG.HEEL_LIFT_THRESHOLD

    def test_the_depth_warning_band_is_comparable_to_its_measurement_error(self):
        band = band_for("", rule_id="squat_depth")
        width = DEFAULT_CONFIG.DEPTH_KNEE_ANGLE_WARN - DEFAULT_CONFIG.DEPTH_KNEE_ANGLE_PASS
        assert 0.5 * band.total < width < 2.0 * band.total


class TestAcceptabilityCriterion:
    """
    Mercadal-Baudart et al. (2024): under 12 deg beats a physio's by-eye
    assessment, under 6 deg is "very good". Each band reports where it sits,
    even when it isn't flattering.
    """

    def test_the_bars_are_the_published_ones(self):
        from exercises.common.uncertainty import ACCEPTABILITY_THRESHOLDS

        assert ACCEPTABILITY_THRESHOLDS["good"] == 12.0
        assert ACCEPTABILITY_THRESHOLDS["very_good"] == 6.0

    @pytest.mark.parametrize(
        "sigma, expected",
        [
            (3.0, "very good"),
            (6.0, "good"),
            (10.7, "good"),
            (12.0, "worse than by-eye assessment"),
            (25.1, "worse than by-eye assessment"),
        ],
    )
    def test_it_labels_an_error_against_those_bars(self, sigma, expected):
        from exercises.common.uncertainty import acceptability

        assert acceptability(sigma) == expected

    def test_the_depth_band_still_beats_by_eye_assessment(self):
        # includes the filter's own bias, since that's what reaches the verdict
        from analysis.filters import depth_bias_for

        band = band_for("", rule_id="squat_depth", systematic=depth_bias_for("ema"))
        assert band.as_dict()["acceptability"] == "good"

    def test_the_bilateral_band_does_not_and_the_export_says_so(self):
        # still published, since press symmetry uses the same left/right comparison
        band = band_for("", rule_id="press_symmetry")
        assert band.as_dict()["acceptability"] == "worse than by-eye assessment"

    def test_a_non_angular_band_is_not_given_an_angular_verdict(self):
        band = band_for("", rule_id="heel_lift")
        assert "acceptability" not in band.as_dict()
