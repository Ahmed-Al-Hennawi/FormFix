"""
Geometry the upper-body exercises added: signed inclination and angular rate.
Coordinates are built so the answer is known by hand, degenerate cases
included. A sign error here wouldn't crash, it would just report a backward
lean as a forward one.
"""

from __future__ import annotations

import math

import pytest

from analysis.geometry import (
    angular_velocity,
    calculate_angle,
    inclination_from_vertical,
    signed_inclination_from_vertical,
)


class TestSignedInclination:
    def test_vertical_segment_is_zero(self):
        # y grows downward, so "upper" is the smaller y.
        assert signed_inclination_from_vertical((0.0, 100.0), (0.0, 0.0)) == pytest.approx(0.0)

    def test_leaning_towards_positive_x_is_positive(self):
        angle = signed_inclination_from_vertical((0.0, 100.0), (50.0, 50.0))
        assert angle == pytest.approx(45.0)

    def test_leaning_towards_negative_x_is_negative(self):
        angle = signed_inclination_from_vertical((0.0, 100.0), (-50.0, 50.0))
        assert angle == pytest.approx(-45.0)

    def test_horizontal_segment_is_ninety(self):
        assert signed_inclination_from_vertical((0.0, 0.0), (50.0, 0.0)) == pytest.approx(90.0)

    def test_magnitude_matches_the_unsigned_version(self):
        for upper in ((30.0, 40.0), (-30.0, 40.0), (10.0, 90.0)):
            lower = (0.0, 100.0)
            assert abs(signed_inclination_from_vertical(lower, upper)) == pytest.approx(
                inclination_from_vertical(lower, upper)
            )

    def test_missing_point_is_not_measured(self):
        assert math.isnan(signed_inclination_from_vertical(None, (0.0, 0.0)))
        assert math.isnan(signed_inclination_from_vertical((0.0, 0.0), None))

    def test_coincident_points_are_not_measured(self):
        # NaN, not 0 - a zero-length segment has no direction.
        assert math.isnan(signed_inclination_from_vertical((5.0, 5.0), (5.0, 5.0)))


class TestElbowAngleCases:
    """Elbow angle, which both upper-body exercises are built on."""

    def test_straight_arm_is_one_eighty(self):
        assert calculate_angle((0.0, 0.0), (0.0, 100.0), (0.0, 200.0)) == pytest.approx(180.0)

    def test_right_angle(self):
        assert calculate_angle((0.0, 0.0), (0.0, 100.0), (100.0, 100.0)) == pytest.approx(90.0)

    def test_forty_five_degrees(self):
        # Vertex at (0, 100), one ray straight up to (0, 0) and the other up
        # and out to (100, 0).
        assert calculate_angle((0.0, 0.0), (0.0, 100.0), (100.0, 0.0)) == pytest.approx(45.0)

    def test_one_thirty_five_degrees(self):
        assert calculate_angle((0.0, 0.0), (0.0, 100.0), (100.0, 200.0)) == pytest.approx(135.0)

    def test_degenerate_points_are_not_measured(self):
        assert math.isnan(calculate_angle((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)))


class TestAngularVelocity:
    def test_constant_rate_from_timestamps_not_frames(self):
        slow = angular_velocity([0.0, 10.0, 20.0], [0.0, 0.5, 1.0])
        fast = angular_velocity([0.0, 5.0, 10.0, 15.0, 20.0], [0.0, 0.25, 0.5, 0.75, 1.0])
        assert slow[1] == pytest.approx(20.0)
        assert fast[1] == pytest.approx(20.0)

    def test_first_sample_has_no_rate(self):
        assert math.isnan(angular_velocity([1.0, 2.0], [0.0, 1.0])[0])

    def test_gap_produces_no_fabricated_rate(self):
        rates = angular_velocity([1.0, float("nan"), 3.0], [0.0, 1.0, 2.0])
        assert math.isnan(rates[1])
        assert math.isnan(rates[2])

    def test_zero_timestep_is_not_infinite(self):
        assert math.isnan(angular_velocity([1.0, 5.0], [1.0, 1.0])[1])
