"""
Tests for the squat measurements, phases and frame-rate independence. No
verdicts here, just that the numbers describe the movement correctly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from analysis.models import Phase, SquatRep, VideoMetadata
from exercises.squat.config import SquatConfig
from exercises.squat.metrics import (
    PHASE_ASCENT,
    PHASE_BOTTOM,
    PHASE_COMPLETION,
    PHASE_DESCENT,
    PHASE_MOVEMENT,
    build_reps,
    compute_frame_metrics,
    phase_frames,
    standing_baseline,
)
from exercises.squat.phases import detect_reps
from tests.synthetic_squat import FPS, HEIGHT, WIDTH, SyntheticSpec, make_pose_data

CONFIG = SquatConfig()


def video_meta(frames: int, fps: float = FPS) -> VideoMetadata:
    return VideoMetadata(
        path=Path("clip.mp4"),
        width=WIDTH,
        height=HEIGHT,
        fps=fps,
        frame_count=frames,
        duration=frames / fps,
    )


# three clean reps, used by most of these
CLEAN = SyntheticSpec(noise=0.001)


def analysed(spec: SyntheticSpec | None = None):
    spec = spec or CLEAN
    pose = make_pose_data(spec)
    video = video_meta(pose.frame_count)
    metrics = compute_frame_metrics(video, pose, "left", CONFIG)
    knee = np.array([m.knee_angle for m in metrics], dtype=float)
    detection = detect_reps(knee, pose.timestamps, CONFIG)
    reps = build_reps(detection.reps, metrics, knee, pose, "left", CONFIG)
    return pose, video, metrics, detection, reps


class TestMultiFeatureMeasurement:
    def test_both_legs_are_measured(self):
        _, _, metrics, _, _ = analysed()
        moving = [m for m in metrics if m.valid and math.isfinite(m.left_knee_angle)]
        assert moving
        sample = moving[len(moving) // 2]
        assert math.isfinite(sample.left_knee_angle)
        assert math.isfinite(sample.right_knee_angle)
        assert math.isfinite(sample.left_hip_angle)
        assert math.isfinite(sample.right_hip_angle)
        assert sample.both_sides_valid

    def test_shin_inclination_grows_with_depth(self):
        _, _, metrics, detection, reps = analysed()
        rep = reps[0]
        standing = [m for m in metrics if m.valid and m.frame_index < rep.start_frame]
        bottom = metrics[rep.bottom_frame]
        assert standing
        assert math.isfinite(bottom.shin_inclination)
        # the synthetic shin tilts forward on the way down
        assert bottom.shin_inclination > standing[-1].shin_inclination
        del detection

    def test_symmetry_is_near_zero_for_an_even_squat(self):
        _, _, metrics, _, _ = analysed()
        gaps = [m.knee_asymmetry for m in metrics if math.isfinite(m.knee_asymmetry)]
        assert gaps
        assert max(gaps) < 10.0

    def test_normalised_values_are_dimensionless(self):
        _, _, metrics, _, _ = analysed()
        sample = next(m for m in metrics if m.valid and math.isfinite(m.body_scale))
        assert sample.body_scale > 1.0  # pixels
        assert 0.0 <= sample.stance_width < 5.0  # a ratio, so not pixels
        assert math.isfinite(sample.knee_over_ankle_offset)

    def test_unmeasurable_values_are_nan_not_zero(self):
        spec = SyntheticSpec(noise=0.001)
        pose = make_pose_data(spec)
        pose.valid[:, :] = False
        pose.pose_found[:] = False
        metrics = compute_frame_metrics(video_meta(pose.frame_count), pose, "left", CONFIG)
        assert all(not m.valid for m in metrics)
        assert all(math.isnan(m.knee_angle) for m in metrics)


class TestPhaseSegmentation:
    def test_every_rep_carries_its_phase_boundaries(self):
        _, _, _, _, reps = analysed()
        assert reps
        for rep in reps:
            assert rep.start_frame <= rep.descent_start_frame <= rep.bottom_start_frame
            assert rep.bottom_start_frame <= rep.bottom_frame <= rep.bottom_end_frame
            assert rep.bottom_end_frame <= rep.ascent_start_frame <= rep.end_frame

    def test_phase_durations_are_seconds_and_add_up(self):
        _, _, _, _, reps = analysed()
        rep = reps[0]
        for value in (rep.descent_duration, rep.bottom_duration, rep.ascent_duration):
            assert math.isfinite(value) and value >= 0.0
        total = rep.descent_duration + rep.bottom_duration + rep.ascent_duration
        assert math.isclose(total, rep.duration, abs_tol=2.0 / FPS)

    def test_phase_frames_selects_the_right_span(self):
        rep = SquatRep(
            number=1,
            start_frame=100,
            bottom_frame=140,
            end_frame=180,
            start_time=0.0,
            bottom_time=1.3,
            end_time=2.6,
            duration=2.6,
            min_knee_angle=92.0,
            hip_angle_at_bottom=60.0,
            torso_lean_at_bottom=28.0,
            max_torso_lean=30.0,
            max_torso_lean_frame=142,
            hip_above_knee_at_bottom=0.05,
            max_heel_lift=0.01,
            max_heel_lift_frame=141,
            heel_reliable=True,
            end_knee_angle=176.0,
            end_hip_angle=175.0,
            descent_start_frame=100,
            bottom_start_frame=137,
            bottom_end_frame=143,
            ascent_start_frame=144,
        )
        assert phase_frames(rep, PHASE_DESCENT) == range(100, 137)
        assert phase_frames(rep, PHASE_BOTTOM) == range(137, 144)
        assert phase_frames(rep, PHASE_ASCENT) == range(144, 181)
        assert phase_frames(rep, PHASE_COMPLETION) == range(180, 181)
        assert phase_frames(rep, PHASE_MOVEMENT) == range(100, 181)

    def test_bottom_window_is_a_band_not_a_single_frame(self):
        _, _, _, _, reps = analysed()
        rep = reps[0]
        assert rep.bottom_end_frame > rep.bottom_start_frame


class TestBaseline:
    def test_baseline_is_measured_from_standing_frames(self):
        pose, video, metrics, detection, reps = analysed()
        baseline = standing_baseline(
            pose, metrics, detection.phases, reps[0].start_frame, video, "left", CONFIG
        )
        assert baseline["standing_frames"] >= CONFIG.BASELINE_MIN_FRAMES
        # synthetic stance is upright, so straight knee and vertical torso
        assert baseline["knee_angle"] > 160.0
        assert baseline["torso_lean"] < 15.0
        assert math.isfinite(baseline["lower_leg_px"])

    def test_baseline_prefers_frames_before_the_first_repetition(self):
        pose, video, metrics, detection, reps = analysed()
        baseline = standing_baseline(
            pose, metrics, detection.phases, reps[0].start_frame, video, "left", CONFIG
        )
        standing_before = sum(
            1
            for i, p in enumerate(detection.phases)
            if p is Phase.STANDING and i < reps[0].start_frame and metrics[i].valid
        )
        assert baseline["standing_frames"] == standing_before


class TestFrameRateIndependence:
    """Nothing should assume 30 fps - same movement at 24, 30 and 60 fps gives the
    same reps and durations."""

    @staticmethod
    def knee_trace(fps: float) -> tuple[np.ndarray, np.ndarray]:
        seconds = np.arange(0, 8.0, 1.0 / fps)
        # one 3-second squat with standing either side
        angle = np.where(
            (seconds < 2.0) | (seconds > 5.0),
            175.0,
            175.0 - 85.0 * np.sin(np.pi * np.clip((seconds - 2.0) / 3.0, 0, 1)),
        )
        return angle, seconds

    def test_same_movement_at_different_frame_rates(self):
        results = {}
        for fps in (24.0, 30.0, 60.0):
            angle, seconds = self.knee_trace(fps)
            detection = detect_reps(angle, seconds, CONFIG)
            assert len(detection.reps) == 1, fps
            rep = detection.reps[0]
            results[fps] = float(seconds[rep.end_frame] - seconds[rep.start_frame])
        durations = list(results.values())
        assert max(durations) - min(durations) < 0.25, results
