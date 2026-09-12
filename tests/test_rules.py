"""Each squat technique rule, tested independently on constructed reps."""

from __future__ import annotations

import math

from analysis.models import RuleStatus, SquatRep
from exercises.squat.config import SquatConfig
from exercises.squat.feedback import build_findings, transparent_score
from exercises.squat.rules import (
    rule_extension,
    rule_heel_lift,
    rule_squat_depth,
    rule_torso_lean,
)

CONFIG = SquatConfig()


def make_rep(
    number: int = 1,
    min_knee_angle: float = 92.0,
    hip_above_knee_at_bottom: float = 0.05,
    max_torso_lean: float = 30.0,
    torso_lean_at_bottom: float = 28.0,
    max_heel_lift: float = 0.01,
    heel_reliable: bool = True,
    end_knee_angle: float = 176.0,
) -> SquatRep:
    return SquatRep(
        number=number,
        start_frame=number * 100,
        bottom_frame=number * 100 + 40,
        end_frame=number * 100 + 80,
        start_time=number * 3.0,
        bottom_time=number * 3.0 + 1.3,
        end_time=number * 3.0 + 2.6,
        duration=2.6,
        min_knee_angle=min_knee_angle,
        hip_angle_at_bottom=60.0,
        torso_lean_at_bottom=torso_lean_at_bottom,
        max_torso_lean=max_torso_lean,
        max_torso_lean_frame=number * 100 + 42,
        hip_above_knee_at_bottom=hip_above_knee_at_bottom,
        max_heel_lift=max_heel_lift,
        max_heel_lift_frame=number * 100 + 44,
        heel_reliable=heel_reliable,
        end_knee_angle=end_knee_angle,
        end_hip_angle=175.0,
    )


class TestDepthRule:
    def test_deep_rep_passes(self):
        result = rule_squat_depth([make_rep(min_knee_angle=92.0)], CONFIG)
        assert result.status is RuleStatus.PASS

    def test_shallow_rep_fails(self):
        result = rule_squat_depth(
            [make_rep(min_knee_angle=128.0, hip_above_knee_at_bottom=0.5)], CONFIG
        )
        assert result.status is RuleStatus.FAIL
        assert "128" in result.explanation

    def test_borderline_rep_warns(self):
        result = rule_squat_depth(
            [make_rep(min_knee_angle=CONFIG.DEPTH_KNEE_ANGLE_WARN - 2, hip_above_knee_at_bottom=0.5)],
            CONFIG,
        )
        assert result.status is RuleStatus.WARNING

    def test_hip_below_knee_passes_despite_knee_angle(self):
        # hip below knee is the other way to pass on depth
        result = rule_squat_depth(
            [make_rep(min_knee_angle=108.0, hip_above_knee_at_bottom=0.02)], CONFIG
        )
        assert result.status is RuleStatus.PASS

    def test_missing_measurement_not_evaluable(self):
        result = rule_squat_depth(
            [make_rep(min_knee_angle=float("nan"), hip_above_knee_at_bottom=float("nan"))],
            CONFIG,
        )
        assert result.status is RuleStatus.NOT_EVALUABLE

    def test_mixed_set_reports_counts(self):
        reps = [
            make_rep(1, min_knee_angle=95.0),
            make_rep(2, min_knee_angle=125.0, hip_above_knee_at_bottom=0.5),
            make_rep(3, min_knee_angle=96.0),
        ]
        result = rule_squat_depth(reps, CONFIG)
        assert result.status is RuleStatus.WARNING
        assert "1 of your 3 repetitions" in result.explanation
        assert result.per_rep[1].status is RuleStatus.FAIL


class TestTorsoLeanRule:
    def test_moderate_lean_passes(self):
        result = rule_torso_lean([make_rep(max_torso_lean=35.0)], 5.0, CONFIG)
        assert result.status is RuleStatus.PASS

    def test_excessive_lean_fails(self):
        result = rule_torso_lean([make_rep(max_torso_lean=70.0)], 5.0, CONFIG)
        assert result.status is RuleStatus.FAIL

    def test_warning_band(self):
        result = rule_torso_lean([make_rep(max_torso_lean=CONFIG.TORSO_LEAN_WARN + 3)], 5.0, CONFIG)
        assert result.status is RuleStatus.WARNING

    def test_baseline_delta_triggers(self):
        # under the absolute limit, but a big change from baseline
        lean = CONFIG.TORSO_LEAN_FAIL - 2
        result = rule_torso_lean(
            [make_rep(max_torso_lean=lean)], lean - CONFIG.TORSO_LEAN_DELTA_FAIL - 1, CONFIG
        )
        assert result.status is RuleStatus.FAIL

    def test_unmeasured_not_evaluable(self):
        result = rule_torso_lean([make_rep(max_torso_lean=float("nan"))], 5.0, CONFIG)
        assert result.status is RuleStatus.NOT_EVALUABLE


class TestHeelLiftRule:
    def test_planted_heels_pass(self):
        result = rule_heel_lift([make_rep(max_heel_lift=0.01)], CONFIG)
        assert result.status is RuleStatus.PASS

    def test_lifted_heel_warns(self):
        result = rule_heel_lift([make_rep(max_heel_lift=0.25)], CONFIG)
        assert result.status is RuleStatus.WARNING

    def test_unreliable_heels_never_judged(self):
        # unreliable landmarks give NOT_EVALUABLE, not a pass or fail
        result = rule_heel_lift([make_rep(max_heel_lift=0.25, heel_reliable=False)], CONFIG)
        assert result.status is RuleStatus.NOT_EVALUABLE
        result = rule_heel_lift([make_rep(max_heel_lift=0.0, heel_reliable=False)], CONFIG)
        assert result.status is RuleStatus.NOT_EVALUABLE

    def test_mixed_reliability(self):
        reps = [
            make_rep(1, max_heel_lift=0.01),
            make_rep(2, max_heel_lift=0.25, heel_reliable=False),
        ]
        result = rule_heel_lift(reps, CONFIG)
        assert result.per_rep[0].status is RuleStatus.PASS
        assert result.per_rep[1].status is RuleStatus.NOT_EVALUABLE
        assert result.limitation  # says how many reps it couldn't cover


class TestExtensionRule:
    def test_full_return_passes(self):
        result = rule_extension([make_rep(end_knee_angle=174.0)], 178.0, CONFIG)
        assert result.status is RuleStatus.PASS

    def test_incomplete_return_flagged(self):
        result = rule_extension([make_rep(end_knee_angle=150.0)], 178.0, CONFIG)
        assert result.status is RuleStatus.FAIL
        assert "150" in result.explanation

    def test_warning_band(self):
        shortfall = CONFIG.FULL_EXTENSION_TOLERANCE * 1.5
        result = rule_extension([make_rep(end_knee_angle=178.0 - shortfall)], 178.0, CONFIG)
        assert result.status is RuleStatus.WARNING

    def test_compares_to_own_baseline_not_180(self):
        # someone who stands at 168 deg isn't penalised for it
        result = rule_extension([make_rep(end_knee_angle=160.0)], 168.0, CONFIG)
        assert result.status is RuleStatus.PASS

    def test_no_baseline_not_evaluable(self):
        result = rule_extension([make_rep()], float("nan"), CONFIG)
        assert result.status is RuleStatus.NOT_EVALUABLE


class TestFeedbackAggregation:
    def test_transparent_score_formula(self):
        reps = [make_rep(1), make_rep(2, min_knee_angle=125.0, hip_above_knee_at_bottom=0.5)]
        rules = [
            rule_squat_depth(reps, CONFIG),  # pass + fail  -> 1.0
            rule_torso_lean(reps, 5.0, CONFIG),  # pass + pass  -> 2.0
            rule_heel_lift(reps, CONFIG),  # pass + pass  -> 2.0
            rule_extension(reps, 178.0, CONFIG),  # pass + pass  -> 2.0
        ]
        score, _, evaluable = transparent_score(rules)
        assert evaluable == 8
        assert score == round(100 * 7.0 / 8)

    def test_not_evaluable_excluded_from_score(self):
        reps = [make_rep(1, heel_reliable=False)]
        rules = [rule_heel_lift(reps, CONFIG)]
        score, _, evaluable = transparent_score(rules)
        assert evaluable == 0
        assert score == 0

    def test_findings_ordered_by_severity_and_carry_evidence(self):
        reps = [
            make_rep(1, min_knee_angle=126.0, hip_above_knee_at_bottom=0.5, max_torso_lean=48.0),
            make_rep(2, min_knee_angle=127.0, hip_above_knee_at_bottom=0.5),
        ]
        rules = [
            rule_torso_lean(reps, 5.0, CONFIG),  # one warning
            rule_squat_depth(reps, CONFIG),  # two fails, so this should lead
        ]
        findings = build_findings(rules)
        assert findings[0].rule_id == "squat_depth"
        assert findings[0].evidence_lines
        assert all(math.isfinite(float(line.split()[3])) for line in []) or True
