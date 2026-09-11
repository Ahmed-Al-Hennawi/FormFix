"""
Does the movement match the exercise the user picked? Stops e.g. a bench press
uploaded as "shoulder press" getting detailed press feedback.

Most tests here are about NOT rejecting - a real squat filmed awkwardly has to
pass, so the checks only look for contradictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from exercises.common.plausibility import (
    MIN_CONTRADICTING_REP_RATIO,
    MIN_REPS_TO_JUDGE,
    PRESS_MIN_WRIST_ABOVE_SHOULDER,
    PULLDOWN_MIN_WRIST_ABOVE_SHOULDER_AT_START,
    SQUAT_MAX_KNEE_ANGLE,
    check_press,
    check_pulldown,
    check_squat,
)


@dataclass
class SquatLike:
    min_knee_angle: float


@dataclass
class ArmLike:
    rom_degrees: float


def squats(*angles) -> list:
    return [SquatLike(a) for a in angles]


def arms(*roms) -> list:
    return [ArmLike(r) for r in roms]


# --- Squat ---


def test_a_real_squat_passes():
    assert check_squat(squats(95, 88, 101, 92)).plausible


def test_a_shallow_squat_is_still_a_squat():
    """A high squat is a depth finding, not a different exercise."""
    result = check_squat(squats(140, 138, 142))
    assert result.plausible


def test_standing_still_is_not_a_squat():
    result = check_squat(squats(178, 176, 179, 177))
    assert not result.plausible
    assert "squat" in result.message.lower()


def test_one_odd_rep_cannot_reject_a_squat():
    # three good reps and one mis-measured one
    result = check_squat(squats(95, 92, 179, 90))
    assert result.plausible


def test_arm_exercise_labelled_as_a_squat_is_caught():
    result = check_squat(squats(95, 92), wrist_travel_torsos=6.0)
    assert not result.plausible
    assert "arm exercise" in result.message.lower()


def test_normal_arm_travel_does_not_reject_a_squat():
    result = check_squat(squats(95, 92), wrist_travel_torsos=0.8)
    assert result.plausible


# --- Press - the bench-press case ---


def test_a_real_overhead_press_passes():
    # a locked-out press is around 1.5 shoulder widths
    assert check_press(arms(120, 118, 122), wrist_above_shoulder_at_top=1.45).plausible


def test_a_bench_press_is_rejected():
    """Good elbow range and clean reps, but the wrists never go above the shoulders."""
    result = check_press(arms(115, 118, 120), wrist_above_shoulder_at_top=0.05)
    assert not result.plausible
    assert "overhead" in result.message.lower()


def test_a_bicep_curl_is_rejected_as_a_press():
    result = check_press(arms(125, 130), wrist_above_shoulder_at_top=-0.30)
    assert not result.plausible


def test_a_press_stopped_short_of_lockout_still_passes():
    """Partial range is a technique finding, not the wrong exercise."""
    result = check_press(arms(70, 68, 72), wrist_above_shoulder_at_top=0.9)
    assert result.plausible


def test_press_with_no_arm_movement_is_rejected():
    result = check_press(arms(5, 4, 6), wrist_above_shoulder_at_top=1.4)
    assert not result.plausible


def test_press_threshold_boundary():
    just_over = check_press(arms(100, 100), PRESS_MIN_WRIST_ABOVE_SHOULDER + 0.05)
    just_under = check_press(arms(100, 100), PRESS_MIN_WRIST_ABOVE_SHOULDER - 0.05)
    assert just_over.plausible
    assert not just_under.plausible


# --- Pulldown ---


def test_a_real_pulldown_passes():
    assert check_pulldown(arms(75, 80, 78), wrist_above_shoulder_at_start=0.45).plausible


def test_a_row_is_rejected_as_a_pulldown():
    """A row pulls horizontally - the hands never get above the shoulders."""
    result = check_pulldown(arms(80, 85), wrist_above_shoulder_at_start=-0.20)
    assert not result.plausible
    assert "pulldown" in result.message.lower()


def test_a_partial_pulldown_still_passes():
    result = check_pulldown(arms(45, 48), wrist_above_shoulder_at_start=0.30)
    assert result.plausible


def test_pulldown_with_no_arm_movement_is_rejected():
    result = check_pulldown(arms(3, 5), wrist_above_shoulder_at_start=0.40)
    assert not result.plausible


# --- Abstaining - the gate must never guess ---


def test_no_reps_does_not_reject():
    assert check_squat([]).plausible
    assert check_press([]).plausible
    assert check_pulldown([]).plausible


def test_a_single_rep_is_too_few_to_judge():
    """One rep is not enough evidence to call the exercise wrong."""
    result = check_squat(squats(179))
    assert result.plausible
    assert "too few" in result.reason


def test_unmeasurable_reps_do_not_reject():
    result = check_squat(squats(float("nan"), float("nan"), float("nan")))
    assert result.plausible


def test_missing_geometry_signal_does_not_reject():
    """When the wrist-height signal is unavailable, the gate abstains on it."""
    result = check_press(arms(120, 118), wrist_above_shoulder_at_top=None)
    assert result.plausible


def test_partially_measurable_set_uses_only_what_it_can_read():
    result = check_squat(squats(179, 177, float("nan")))
    assert not result.plausible
    assert result.checked_reps == 2


@pytest.mark.parametrize("bad,total", [(1, 4), (2, 5), (2, 4)])
def test_minority_of_bad_reps_never_rejects(bad, total):
    angles = [179.0] * bad + [95.0] * (total - bad)
    result = check_squat(squats(*angles))
    assert result.plausible, f"{bad}/{total} rejected but ratio is under the threshold"


def test_the_ratio_threshold_is_a_majority():
    assert MIN_CONTRADICTING_REP_RATIO > 0.5
    assert MIN_REPS_TO_JUDGE >= 2


def test_squat_threshold_leaves_room_below_standing():
    """Must not reject squats that merely lack depth."""
    assert SQUAT_MAX_KNEE_ANGLE < 170.0
    assert SQUAT_MAX_KNEE_ANGLE > 130.0


def test_pulldown_threshold_is_permissive():
    assert PULLDOWN_MIN_WRIST_ABOVE_SHOULDER_AT_START < 0.2


def test_result_serialises_for_the_export():
    data = check_squat(squats(179, 178, 177)).as_dict()
    assert data["plausible"] is False
    assert data["checked_reps"] == 3
    assert data["contradicting_reps"] == 3
