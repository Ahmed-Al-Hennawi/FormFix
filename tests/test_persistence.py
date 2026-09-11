"""
Tests for persistence filtering: a one-frame spike is just a jittery landmark,
but a violation that holds or keeps coming back is real.
"""

from __future__ import annotations

import numpy as np

from exercises.squat.persistence import assess, sustained_extreme


def evidence(values, threshold, prefer_max=True):
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        violates = values > threshold if prefer_max else values < threshold
    return assess(values, violates, prefer_max=prefer_max)


class TestSingleFrameSpikes:
    def test_one_frame_violation_does_not_trigger(self):
        series = [10.0] * 20
        series[7] = 90.0  # a single jittered landmark
        ev = evidence(series, 45.0)
        assert ev.violating_frames == 1
        assert ev.longest_run == 1
        assert not ev.triggers(min_frames=4, min_ratio=0.15)

    def test_two_frame_violation_below_the_run_floor_does_not_trigger(self):
        series = [10.0] * 20
        series[7] = series[8] = 90.0
        ev = evidence(series, 45.0)
        assert ev.longest_run == 2
        assert not ev.triggers(min_frames=4, min_ratio=0.0)


class TestSustainedViolations:
    def test_sustained_violation_triggers(self):
        series = [10.0] * 10 + [70.0] * 10
        ev = evidence(series, 45.0)
        assert ev.longest_run == 10
        assert ev.ratio == 0.5
        assert ev.triggers(min_frames=4, min_ratio=0.15)

    def test_ratio_floor_rejects_a_brief_but_long_enough_run(self):
        series = [10.0] * 96 + [70.0] * 4
        ev = evidence(series, 45.0)
        assert ev.longest_run == 4
        assert ev.ratio == 0.04
        assert ev.triggers(min_frames=4, min_ratio=0.0)
        assert not ev.triggers(min_frames=4, min_ratio=0.15)

    def test_peak_index_points_at_the_worst_frame_in_the_longest_run(self):
        series = [10.0, 80.0, 10.0, 60.0, 70.0, 95.0, 65.0, 10.0]
        ev = evidence(series, 45.0)
        assert ev.longest_run == 4
        assert ev.peak_index == 5
        assert ev.peak_value == 95.0


class TestMissingData:
    def test_unmeasurable_frames_are_not_evidence_either_way(self):
        series = [np.nan] * 10 + [70.0] * 5
        ev = evidence(series, 45.0)
        # NaN frames count as neither violations nor clean frames
        assert ev.measurable_frames == 5
        assert ev.violating_frames == 5
        assert ev.ratio == 1.0

    def test_no_measurable_frames_never_triggers(self):
        ev = evidence([np.nan] * 12, 45.0)
        assert not ev.has_evidence
        assert not ev.triggers(min_frames=1, min_ratio=0.0)


class TestSustainedExtreme:
    def test_spike_shorter_than_the_window_is_discarded(self):
        series = np.array([10.0] * 10)
        series[4] = 100.0
        held, _ = sustained_extreme(series, window=3)
        assert held == 10.0  # the spike never held for three frames

    def test_a_real_peak_is_reported(self):
        series = np.array([10.0] * 5 + [80.0, 82.0, 85.0, 83.0] + [10.0] * 5)
        held, at = sustained_extreme(series, window=3)
        assert 80.0 <= held <= 85.0
        assert 5 <= at <= 8

    def test_min_seeking_variant(self):
        series = np.array([170.0] * 5 + [95.0, 92.0, 90.0] + [170.0] * 5)
        held, at = sustained_extreme(series, window=3, prefer_max=False)
        assert 90.0 <= held <= 95.0
        assert 5 <= at <= 7

    def test_series_shorter_than_the_window_returns_nan(self):
        held, at = sustained_extreme(np.array([1.0, 2.0]), window=5)
        assert not np.isfinite(held)
        assert at == -1
