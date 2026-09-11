"""Unit tests for the shared geometry utilities."""

from __future__ import annotations

import math

import pytest

from analysis.geometry import (
    calculate_angle,
    distance,
    inclination_from_vertical,
    midpoint,
)


class TestCalculateAngle:
    def test_straight_line_is_180(self):
        assert calculate_angle((0, 0), (1, 0), (2, 0)) == pytest.approx(180.0)

    def test_right_angle_is_90(self):
        assert calculate_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)

    def test_acute_angle(self):
        assert calculate_angle((1, 0), (0, 0), (1, 1)) == pytest.approx(45.0)

    def test_zero_angle(self):
        assert calculate_angle((1, 1), (0, 0), (2, 2)) == pytest.approx(0.0, abs=1e-4)

    def test_missing_point_returns_nan(self):
        assert math.isnan(calculate_angle(None, (0, 0), (1, 1)))
        assert math.isnan(calculate_angle((0, 0), None, (1, 1)))
        assert math.isnan(calculate_angle((0, 0), (1, 1), None))

    def test_nan_coordinate_returns_nan(self):
        assert math.isnan(calculate_angle((float("nan"), 0), (0, 0), (1, 1)))

    def test_identical_points_return_nan(self):
        # zero-length arm, so the angle is undefined, not 0
        assert math.isnan(calculate_angle((0, 0), (0, 0), (1, 1)))
        assert math.isnan(calculate_angle((1, 1), (0, 0), (0, 0)))

    def test_floating_point_cosine_clamped(self):
        # collinear points where float error puts the cosine just over 1.0
        a, b, c = (0.1, 0.1), (0.2, 0.2), (0.30000000000000004, 0.3)
        result = calculate_angle(a, b, c)
        assert not math.isnan(result)
        assert 0.0 <= result <= 180.0

    def test_result_always_within_0_180(self):
        import itertools

        points = [(-1, 2), (0.5, -3), (4, 4), (2, 0)]
        for a, b, c in itertools.permutations(points, 3):
            angle = calculate_angle(a, b, c)
            assert math.isnan(angle) or 0.0 <= angle <= 180.0


class TestInclination:
    def test_vertical_is_zero(self):
        assert inclination_from_vertical((0, 10), (0, 0)) == pytest.approx(0.0)

    def test_horizontal_is_90(self):
        assert inclination_from_vertical((0, 0), (10, 0)) == pytest.approx(90.0)

    def test_45_degrees(self):
        assert inclination_from_vertical((0, 10), (10, 0)) == pytest.approx(45.0)

    def test_direction_independent(self):
        # leaning left or right gives the same value, so a mirrored clip is fine
        left = inclination_from_vertical((0, 10), (-4, 0))
        right = inclination_from_vertical((0, 10), (4, 0))
        assert left == pytest.approx(right)

    def test_missing_point_nan(self):
        assert math.isnan(inclination_from_vertical(None, (0, 0)))

    def test_coincident_points_nan(self):
        assert math.isnan(inclination_from_vertical((1, 1), (1, 1)))


class TestSmallHelpers:
    def test_distance(self):
        assert distance((0, 0), (3, 4)) == pytest.approx(5.0)
        assert math.isnan(distance(None, (0, 0)))

    def test_midpoint(self):
        assert midpoint((0, 0), (2, 4)) == (1.0, 2.0)
        assert midpoint(None, (2, 4)) is None
