"""Gap interpolation and EMA smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.models import NUM_LANDMARKS, FramePoseData
from analysis.smoothing import ema_smooth, interpolate_short_gaps, smooth_series


def make_pose(n_frames: int) -> FramePoseData:
    xy = np.full((n_frames, NUM_LANDMARKS, 2), np.nan)
    return FramePoseData(
        xy_raw=xy.copy(),
        xy=xy,
        visibility=np.zeros((n_frames, NUM_LANDMARKS)),
        valid=np.zeros((n_frames, NUM_LANDMARKS), dtype=bool),
        pose_found=np.ones(n_frames, dtype=bool),
        n_poses=np.ones(n_frames, dtype=np.int16),
        timestamps=np.arange(n_frames) / 30.0,
    )


def set_landmark(pose: FramePoseData, frame: int, lm: int, x: float, y: float, vis: float = 0.9):
    pose.xy[frame, lm] = (x, y)
    pose.xy_raw[frame, lm] = (x, y)
    pose.valid[frame, lm] = True
    pose.visibility[frame, lm] = vis


class TestGapInterpolation:
    def test_short_gap_is_filled_linearly(self):
        pose = make_pose(5)
        set_landmark(pose, 0, 25, 0.2, 0.2)
        set_landmark(pose, 4, 25, 0.6, 0.6)  # 3-frame gap
        filled = interpolate_short_gaps(pose, max_gap=3)
        assert filled == 3
        assert pose.xy[2, 25, 0] == pytest.approx(0.4)
        assert pose.valid[1:4, 25].all()

    def test_long_gap_is_not_fabricated(self):
        pose = make_pose(12)
        set_landmark(pose, 0, 25, 0.2, 0.2)
        set_landmark(pose, 11, 25, 0.6, 0.6)  # 10-frame gap > max
        filled = interpolate_short_gaps(pose, max_gap=5)
        assert filled == 0
        assert np.isnan(pose.xy[5, 25, 0])
        assert not pose.valid[5, 25]

    def test_raw_landmarks_untouched(self):
        pose = make_pose(5)
        set_landmark(pose, 0, 25, 0.2, 0.2)
        set_landmark(pose, 4, 25, 0.6, 0.6)
        interpolate_short_gaps(pose, max_gap=3)
        assert np.isnan(pose.xy_raw[2, 25, 0])  # raw track keeps the gap


class TestEmaSmoothing:
    def test_stable_coordinates_stay_stable(self):
        pose = make_pose(20)
        for f in range(20):
            set_landmark(pose, f, 25, 0.5, 0.5)
        ema_smooth(pose, alpha=0.4)
        assert np.allclose(pose.xy[:, 25, 0], 0.5)

    def test_jitter_is_reduced(self):
        rng = np.random.default_rng(1)
        pose = make_pose(120)
        noisy = 0.5 + rng.normal(0, 0.02, 120)
        for f in range(120):
            set_landmark(pose, f, 25, float(noisy[f]), 0.5)
        ema_smooth(pose, alpha=0.3)
        assert np.std(pose.xy[:, 25, 0]) < np.std(noisy) * 0.8

    def test_gap_resets_instead_of_dragging(self):
        pose = make_pose(30)
        for f in range(10):
            set_landmark(pose, f, 25, 0.1, 0.1)
        # Long untracked gap, then it reappears somewhere else entirely.
        for f in range(25, 30):
            set_landmark(pose, f, 25, 0.9, 0.9)
        ema_smooth(pose, alpha=0.4)
        # First frame back is the observation itself, not a blend with 0.1.
        assert pose.xy[25, 25, 0] == pytest.approx(0.9)
        assert np.isnan(pose.xy[15, 25, 0])

    def test_invalid_alpha_rejected(self):
        pose = make_pose(3)
        with pytest.raises(ValueError):
            ema_smooth(pose, alpha=0.0)
        with pytest.raises(ValueError):
            ema_smooth(pose, alpha=1.5)

    def test_confidence_is_not_smoothed(self):
        pose = make_pose(10)
        for f in range(10):
            set_landmark(pose, f, 25, 0.5, 0.5, vis=0.9 if f % 2 else 0.1)
        before = pose.visibility.copy()
        ema_smooth(pose, alpha=0.4)
        assert np.array_equal(pose.visibility, before)


class TestSeriesSmoothing:
    def test_nan_gap_resets(self):
        series = np.array([10.0, 10.0, np.nan, np.nan, 50.0, 50.0])
        out = smooth_series(series, alpha=0.5)
        assert out[4] == pytest.approx(50.0)  # reset, not blended with 10
        assert np.isnan(out[2])
