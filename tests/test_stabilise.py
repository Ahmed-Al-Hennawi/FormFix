"""
Tests for the landmark stabiliser: it should remove tracking errors without
removing real movement.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from analysis import smoothing, stabilise
from analysis.models import LEFT_KNEE, RIGHT_KNEE, VideoMetadata
from tests.synthetic_squat import HEIGHT, WIDTH, SyntheticSpec, make_pose_data


def video_for(pose) -> VideoMetadata:
    return VideoMetadata(
        path=None,
        width=WIDTH,
        height=HEIGHT,
        fps=30.0,
        frame_count=pose.frame_count,
        duration=pose.frame_count / 30.0,
    )


@pytest.fixture
def clean_pose():
    pose = make_pose_data(SyntheticSpec(noise=0.0))
    return pose, video_for(pose)


class TestOutlierRejection:
    def test_a_single_frame_teleport_is_removed(self, clean_pose):
        pose, video = clean_pose
        frame = 40
        truth = pose.xy[frame, LEFT_KNEE].copy()
        pose.xy[frame, LEFT_KNEE] = (truth[0] + 0.35, truth[1] - 0.25)

        report = stabilise.reject_outliers(pose, video)

        assert report.rejected_cells >= 1
        assert not np.isfinite(pose.xy[frame, LEFT_KNEE]).any()
        assert not pose.valid[frame, LEFT_KNEE]
        # the neighbouring frames are untouched
        assert pose.valid[frame - 1, LEFT_KNEE]
        assert pose.valid[frame + 1, LEFT_KNEE]

    def test_it_never_changes_the_raw_track(self, clean_pose):
        pose, video = clean_pose
        pose.xy[20, RIGHT_KNEE] = (0.05, 0.05)
        pose.xy_raw[20, RIGHT_KNEE] = (0.05, 0.05)
        before = pose.xy_raw.copy()

        stabilise.reject_outliers(pose, video)

        assert np.allclose(pose.xy_raw, before, equal_nan=True)

    def test_a_clean_recording_is_left_alone(self, clean_pose):
        pose, video = clean_pose
        report = stabilise.reject_outliers(pose, video)
        assert report.rejected_cells == 0

    def test_normal_landmark_noise_survives(self):
        pose = make_pose_data(SyntheticSpec(noise=0.004), seed=3)
        report = stabilise.reject_outliers(pose, video_for(pose))
        # a little detector noise is not a tracking error
        assert report.rejected_ratio < 0.01

    def test_fast_movement_is_not_mistaken_for_an_error(self):
        # half-second reps: the fastest movement the state machine still accepts
        pose = make_pose_data(SyntheticSpec(rep_seconds=1.0, reps=4, noise=0.001))
        report = stabilise.reject_outliers(pose, video_for(pose))
        assert report.rejected_ratio < 0.01

    def test_the_whole_body_moving_is_not_an_error(self, clean_pose):
        pose, video = clean_pose
        # every landmark jumps together: a camera cut or a pan, not a bad joint
        pose.xy[50] += 0.3

        report = stabilise.reject_outliers(pose, video)

        assert report.rejected_cells == 0
        assert pose.valid[50].any()

    def test_a_rejected_frame_is_interpolated_back(self, clean_pose):
        pose, video = clean_pose
        frame = 60
        truth = pose.xy[frame, LEFT_KNEE].copy()
        pose.xy[frame, LEFT_KNEE] = (truth[0] + 0.4, truth[1])

        stabilise.reject_outliers(pose, video)
        smoothing.interpolate_short_gaps(pose, max_gap=5)

        assert pose.valid[frame, LEFT_KNEE]
        # back near where the knee really was, not where the detector put it
        assert abs(pose.xy[frame, LEFT_KNEE][0] - truth[0]) < 0.01

    def test_the_scale_is_the_persons_torso_not_the_frame(self, clean_pose):
        pose, video = clean_pose
        torso = stabilise.torso_pixels(pose, video)
        assert 0.15 * HEIGHT < torso < 0.45 * HEIGHT


class TestMedianFilter:
    def test_it_removes_a_one_frame_spike(self):
        values = np.zeros(21)
        values[10] = 5.0
        out = stabilise.median_filter_track(values, 3)
        assert out[10] == pytest.approx(0.0)

    def test_it_does_not_lag_a_ramp(self):
        values = np.arange(21, dtype=np.float64)
        out = stabilise.median_filter_track(values, 3)
        assert np.allclose(out[1:-1], values[1:-1])

    def test_gaps_stay_gaps(self):
        values = np.arange(21, dtype=np.float64)
        values[5:8] = np.nan
        out = stabilise.median_filter_track(values, 3)
        assert np.isnan(out[5:8]).all()


class TestTheDrawnSkeleton:
    """
    The display chain (reject -> interpolate -> smooth) should keep the drawn
    joints on the body when the detector throws one across the frame. Rejecting
    first is the point: a low-pass filter on its own spreads the bad frame over
    its neighbours instead of removing it.
    """

    def render_chain(self, pose, video, *, reject: bool):
        from analysis import filters
        from analysis.annotation import RENDER_MEDIAN_FRAMES, RENDER_SMOOTHING_CUTOFF_HZ

        if reject:
            stabilise.reject_outliers(pose, video)
        smoothing.interpolate_short_gaps(pose, 5)
        smoothing.ema_smooth(pose, 0.45)
        xy = pose.xy.copy()
        for axis in (0, 1):
            series = xy[:, LEFT_KNEE, axis]
            if reject:
                series = stabilise.median_filter_track(series, RENDER_MEDIAN_FRAMES)
            xy[:, LEFT_KNEE, axis] = filters.butterworth_lowpass(
                series, video.fps, cutoff_hz=RENDER_SMOOTHING_CUTOFF_HZ, order=4
            )
        return xy

    def test_a_teleported_joint_stays_near_the_body(self, clean_pose):
        pose, video = clean_pose
        truth = pose.xy[:, LEFT_KNEE].copy()
        frame = 45
        pose.xy[frame, LEFT_KNEE] += (0.2, -0.2)
        pose.xy_raw[frame, LEFT_KNEE] = pose.xy[frame, LEFT_KNEE]

        smoothed_only = self.render_chain(copy.deepcopy(pose), video, reject=False)
        stabilised = self.render_chain(copy.deepcopy(pose), video, reject=True)

        def worst_error(xy):
            window = slice(frame - 8, frame + 9)
            dx = (xy[window, LEFT_KNEE, 0] - truth[window, 0]) * WIDTH
            dy = (xy[window, LEFT_KNEE, 1] - truth[window, 1]) * HEIGHT
            return float(np.nanmax(np.hypot(dx, dy)))

        assert worst_error(stabilised) < 0.5 * worst_error(smoothed_only)

    def test_smoothing_does_not_lag_the_movement(self, clean_pose):
        pose, video = clean_pose
        truth = pose.xy[:, LEFT_KNEE, 1].copy()
        drawn = self.render_chain(copy.deepcopy(pose), video, reject=True)[:, LEFT_KNEE, 1]

        # the best-fitting shift between the drawn knee and the real one should be
        # no more than a frame either way
        def correlation(shift: int) -> float:
            a, b = np.roll(drawn, shift), truth
            ok = np.isfinite(a) & np.isfinite(b)
            return float(np.corrcoef(a[ok], b[ok])[0, 1])

        best = max(range(-4, 5), key=correlation)
        assert abs(best) <= 1
