"""
Lat-pulldown measurement layer. Nothing here says whether a pulldown was any
good, only that the numbers describe the movement that was generated, so
re-tuning a threshold shouldn't need any of these to change.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.models import VideoMetadata
from exercises.common.phases import MovementPhase, RawRep
from exercises.pulldown import metrics as metrics_mod
from exercises.pulldown.config import DEFAULT_CONFIG
from exercises.pulldown.phases import detect_reps, resting_extension
from tests.synthetic_pulldown import (
    HEIGHT,
    WIDTH,
    PulldownSpec,
    make_pose_data,
    skeleton_at,
)

CONFIG = DEFAULT_CONFIG


def video_for(pose) -> VideoMetadata:
    from pathlib import Path

    return VideoMetadata(
        path=Path("synthetic.mp4"),
        width=WIDTH,
        height=HEIGHT,
        fps=30.0,
        frame_count=pose.frame_count,
        duration=pose.frame_count / 30.0,
    )


def measured(spec: PulldownSpec, side: str = "left"):
    pose = make_pose_data(spec)
    video = video_for(pose)
    metrics = metrics_mod.compute_frame_metrics(video, pose, side, CONFIG)
    return pose, video, metrics


class TestElbowAngle:
    def test_the_measured_elbow_angle_matches_the_generated_one(self):
        # The generator is aspect-corrected, so a 172-degree elbow measures 172.
        # If this drifts, every range threshold is compared at the wrong scale.
        spec = PulldownSpec(top_elbow=172.0, bottom_elbow=80.0)
        _, _, metrics = measured(spec)
        angles = [m.elbow_angle for m in metrics if m.valid]
        assert max(angles) == pytest.approx(172.0, abs=1.0)
        assert min(angles) == pytest.approx(80.0, abs=1.0)

    def test_both_arms_are_measured(self):
        _, _, metrics = measured(PulldownSpec())
        sample = next(m for m in metrics if m.valid)
        assert math.isfinite(sample.left_elbow_angle)
        assert math.isfinite(sample.right_elbow_angle)
        assert sample.both_arms_valid

    def test_an_invisible_arm_is_not_measured_rather_than_guessed(self):
        spec = PulldownSpec(visibility=0.15)
        _, _, metrics = measured(spec)
        assert all(not m.valid for m in metrics)

    def test_an_implausible_angle_is_discarded(self):
        # An extrapolated limb can give a geometrically valid but anatomically
        # impossible angle, which has to come out as "not measured".
        pose = make_pose_data(PulldownSpec())
        video = video_for(pose)
        # Wrist onto shoulder puts the elbow angle under the plausibility floor.
        pose.xy[40:50, 15] = pose.xy[40:50, 11]
        pose.xy[40:50, 16] = pose.xy[40:50, 12]
        metrics = metrics_mod.compute_frame_metrics(video, pose, "left", CONFIG)
        assert all(
            not math.isfinite(m.left_elbow_angle)
            or m.left_elbow_angle >= CONFIG.PLAUSIBLE_ELBOW_ANGLE_MIN
            for m in metrics[40:50]
        )


class TestTorsoMeasurement:
    def test_a_still_torso_shows_almost_no_excursion(self):
        spec = PulldownSpec(torso_gain_deg=0.0)
        pose, video, metrics = measured(spec)
        facing, _ = metrics_mod.estimate_facing(video, pose, CONFIG)
        metrics_mod.apply_posterior_lean(metrics, facing)
        signal = metrics_mod.movement_signal(metrics)
        detection = detect_reps(signal, pose.timestamps, CONFIG)
        baseline = metrics_mod.top_baseline(
            metrics, detection.phases, detection.reps[0].start_frame, CONFIG
        )
        metrics_mod.apply_torso_excursion(metrics, baseline, CONFIG)
        peak = max(m.torso_excursion for m in metrics if math.isfinite(m.torso_excursion))
        assert peak < 2.0

    def test_a_backward_lean_is_measured_at_roughly_its_true_size(self):
        spec = PulldownSpec(torso_gain_deg=20.0)
        pose, video, metrics = measured(spec)
        facing, _ = metrics_mod.estimate_facing(video, pose, CONFIG)
        metrics_mod.apply_posterior_lean(metrics, facing)
        signal = metrics_mod.movement_signal(metrics)
        detection = detect_reps(signal, pose.timestamps, CONFIG)
        baseline = metrics_mod.top_baseline(
            metrics, detection.phases, detection.reps[0].start_frame, CONFIG
        )
        mode = metrics_mod.apply_torso_excursion(metrics, baseline, CONFIG)
        assert mode == "posterior"
        peak = max(m.torso_excursion for m in metrics if math.isfinite(m.torso_excursion))
        assert peak == pytest.approx(20.0, abs=3.0)

    def test_facing_direction_is_estimated_from_the_head(self):
        right_facing, confidence = metrics_mod.estimate_facing(
            *_pose_and_video(PulldownSpec(facing=1)), CONFIG
        )
        left_facing, _ = metrics_mod.estimate_facing(*_pose_and_video(PulldownSpec(facing=-1)), CONFIG)
        assert right_facing == 1
        assert left_facing == -1
        assert confidence > 0.5

    def test_a_hidden_head_makes_the_facing_unknown(self):
        facing, _ = metrics_mod.estimate_facing(
            *_pose_and_video(PulldownSpec(head_visibility=0.1)), CONFIG
        )
        assert facing == 0

    def test_without_a_facing_direction_the_excursion_falls_back_to_unsigned(self):
        spec = PulldownSpec(head_visibility=0.1, torso_gain_deg=20.0)
        pose, video, metrics = measured(spec)
        facing, _ = metrics_mod.estimate_facing(video, pose, CONFIG)
        metrics_mod.apply_posterior_lean(metrics, facing)
        signal = metrics_mod.movement_signal(metrics)
        detection = detect_reps(signal, pose.timestamps, CONFIG)
        baseline = metrics_mod.top_baseline(
            metrics, detection.phases, detection.reps[0].start_frame, CONFIG
        )
        mode = metrics_mod.apply_torso_excursion(metrics, baseline, CONFIG)
        assert mode == "unsigned"
        # The size still comes out, it's only the direction we've lost.
        peak = max(m.torso_excursion for m in metrics if math.isfinite(m.torso_excursion))
        assert peak == pytest.approx(20.0, abs=3.0)

    def test_left_and_right_facing_recordings_measure_the_same_lean(self):
        peaks = []
        for facing in (1, -1):
            spec = PulldownSpec(facing=facing, torso_gain_deg=18.0)
            pose, video, metrics = measured(spec)
            direction, _ = metrics_mod.estimate_facing(video, pose, CONFIG)
            metrics_mod.apply_posterior_lean(metrics, direction)
            signal = metrics_mod.movement_signal(metrics)
            detection = detect_reps(signal, pose.timestamps, CONFIG)
            baseline = metrics_mod.top_baseline(
                metrics, detection.phases, detection.reps[0].start_frame, CONFIG
            )
            metrics_mod.apply_torso_excursion(metrics, baseline, CONFIG)
            peaks.append(max(m.torso_excursion for m in metrics if math.isfinite(m.torso_excursion)))
        assert peaks[0] == pytest.approx(peaks[1], abs=1.0)


def _pose_and_video(spec: PulldownSpec):
    pose = make_pose_data(spec)
    return video_for(pose), pose


class TestNormalisation:
    def test_measurements_do_not_depend_on_video_resolution(self):
        # Fail this and we're measuring the camera, not the lifter.
        pose = make_pose_data(PulldownSpec())
        small = VideoMetadata(
            path=__import__("pathlib").Path("s.mp4"),
            width=WIDTH // 2,
            height=HEIGHT // 2,
            fps=30.0,
            frame_count=pose.frame_count,
            duration=1.0,
        )
        big = video_for(pose)
        a = metrics_mod.compute_frame_metrics(big, pose, "left", CONFIG)
        b = metrics_mod.compute_frame_metrics(small, pose, "left", CONFIG)
        for x, y in zip(a, b, strict=True):
            if x.valid and y.valid:
                assert x.elbow_angle == pytest.approx(y.elbow_angle, abs=0.01)
                assert x.wrist_rise == pytest.approx(y.wrist_rise, abs=0.001)

    def test_wrist_rise_is_positive_at_the_top_and_falls_during_the_pull(self):
        _, _, metrics = measured(PulldownSpec())
        rises = [m.wrist_rise for m in metrics if math.isfinite(m.wrist_rise)]
        assert max(rises) > 0.4  # hands well above the shoulders at the top
        assert min(rises) < max(rises) - 0.3  # clearly lower when contracted


class TestRestingExtensionReference:
    def test_the_reference_follows_the_athletes_own_extension(self):
        full = resting_extension(np.asarray([172.0] * 40 + [80.0] * 20), CONFIG)
        limited = resting_extension(np.asarray([132.0] * 40 + [80.0] * 20), CONFIG)
        assert full == pytest.approx(172.0, abs=2.0)
        assert limited == pytest.approx(132.0, abs=2.0)

    def test_the_reference_is_clamped_to_a_plausible_band(self):
        assert resting_extension(np.full(40, 40.0), CONFIG) == CONFIG.REST_REFERENCE_MIN
        assert resting_extension(np.full(40, 250.0), CONFIG) == CONFIG.REST_REFERENCE_MAX

    def test_an_unusable_series_falls_back_rather_than_failing(self):
        assert resting_extension(np.array([np.nan, np.nan]), CONFIG) == CONFIG.TOP_ELBOW_ANGLE


class TestTopWindows:
    def test_the_top_is_read_outside_the_repetition_not_at_its_first_frame(self):
        # The state machine only commits a rep once the pull has begun, so
        # the extended position sits in the frames either side of it.
        raws = [RawRep(start_frame=40, extreme_frame=55, end_frame=70, extreme_value=80.0)]
        windows = metrics_mod.top_windows(raws, 0, 30.0, 200)
        assert windows[0].start < 40 <= windows[0].stop
        assert windows[1].start == 70

    def test_one_repetition_cannot_borrow_its_neighbours_extension(self):
        raws = [
            RawRep(start_frame=40, extreme_frame=55, end_frame=70, extreme_value=80.0),
            RawRep(start_frame=80, extreme_frame=95, end_frame=110, extreme_value=80.0),
        ]
        first = metrics_mod.top_windows(raws, 0, 30.0, 200)
        second = metrics_mod.top_windows(raws, 1, 30.0, 200)
        assert first[1].stop - 1 <= 80  # can't reach into the second rep
        assert second[0].start >= 70  # can't reach back into the first


class TestPhaseSegmentation:
    def test_a_clean_recording_produces_the_expected_repetitions(self):
        pose, _, metrics = measured(PulldownSpec(reps=3))
        detection = detect_reps(metrics_mod.movement_signal(metrics), pose.timestamps, CONFIG)
        assert len(detection.reps) == 3
        assert detection.partial_movements == 0

    def test_every_expected_phase_is_labelled(self):
        pose, _, metrics = measured(PulldownSpec())
        detection = detect_reps(metrics_mod.movement_signal(metrics), pose.timestamps, CONFIG)
        seen = set(detection.phases)
        assert MovementPhase.REST in seen
        assert MovementPhase.TOWARDS in seen
        assert MovementPhase.EXTREME in seen
        assert MovementPhase.RETURN in seen

    def test_a_restricted_athlete_still_has_their_repetitions_counted(self):
        # Why segmentation is adaptive: someone who never straightens their
        # arms still needs reps counted, or the range rule can't tell them.
        pose, _, metrics = measured(PulldownSpec(top_elbow=132.0, bottom_elbow=80.0))
        detection = detect_reps(metrics_mod.movement_signal(metrics), pose.timestamps, CONFIG)
        assert len(detection.reps) == 3


class TestRepMeasurements:
    def test_per_repetition_facts_describe_the_generated_movement(self):
        spec = PulldownSpec(top_elbow=172.0, bottom_elbow=80.0, torso_gain_deg=0.0)
        pose, video, metrics = measured(spec)
        facing, _ = metrics_mod.estimate_facing(video, pose, CONFIG)
        metrics_mod.apply_posterior_lean(metrics, facing)
        detection = detect_reps(metrics_mod.movement_signal(metrics), pose.timestamps, CONFIG)
        baseline = metrics_mod.top_baseline(
            metrics, detection.phases, detection.reps[0].start_frame, CONFIG
        )
        mode = metrics_mod.apply_torso_excursion(metrics, baseline, CONFIG)
        reps = metrics_mod.build_reps(detection.reps, metrics, pose, "left", mode, CONFIG)

        assert len(reps) == 3
        for rep in reps:
            assert rep.top_elbow_angle == pytest.approx(172.0, abs=3.0)
            assert rep.bottom_elbow_angle == pytest.approx(80.0, abs=6.0)
            assert rep.rom_degrees > 80.0
            assert rep.arms_reliable
            assert rep.torso_reliable
            assert rep.valid_frame_ratio > 0.95
            assert rep.wrist_travel > 0.3
            assert rep.duration > CONFIG.MIN_REP_DURATION

    def test_repetitions_are_numbered_in_order(self):
        pose, video, metrics = measured(PulldownSpec())
        detection = detect_reps(metrics_mod.movement_signal(metrics), pose.timestamps, CONFIG)
        reps = metrics_mod.build_reps(detection.reps, metrics, pose, "left", "unsigned", CONFIG)
        assert [rep.number for rep in reps] == [1, 2, 3]
        assert [rep.start_time for rep in reps] == sorted(rep.start_time for rep in reps)


def test_the_generator_itself_is_geometrically_faithful():
    """
    Fixture guard, not a test of the system. Everything above assumes the
    synthetic skeleton really has the elbow angle it was asked for, so a
    regression in the aspect correction would go unnoticed without this.
    """
    from analysis.geometry import calculate_angle

    spec = PulldownSpec(top_elbow=140.0, bottom_elbow=70.0)
    for depth, expected in ((0.0, 140.0), (1.0, 70.0)):
        points = skeleton_at(depth, spec)

        def px(lm, points=points):
            return (points[lm][0] * WIDTH, points[lm][1] * HEIGHT)

        assert calculate_angle(px(11), px(13), px(15)) == pytest.approx(expected, abs=0.5)


class TestRangeOfMotionNeedsOneArmNotTwo:
    """
    In the side view this exercise asks for, the far arm spends much of the
    pull behind the near one. Range of motion is a joint angle, not a
    comparison, so requiring both arms had the check reporting "not assessed"
    on the camera position the guidance recommends.
    """

    def _pose_with_a_hidden_far_arm(self):
        pose = make_pose_data(PulldownSpec())
        pose.visibility[:, [14, 16]] = 0.1  # far arm's elbow and wrist
        return pose

    def _reps_for(self, pose):
        video = video_for(pose)
        metrics = metrics_mod.compute_frame_metrics(video, pose, "left", CONFIG)
        detection = detect_reps(metrics_mod.movement_signal(metrics), pose.timestamps, CONFIG)
        return metrics_mod.build_reps(detection.reps, metrics, pose, "left", "unsigned", CONFIG)

    def test_a_hidden_far_arm_still_allows_the_range_check(self):
        reps = self._reps_for(self._pose_with_a_hidden_far_arm())
        assert reps
        assert reps[0].arms_reliable
        assert math.isfinite(reps[0].rom_degrees)

    def test_the_both_arm_share_is_still_recorded_for_the_export(self):
        reps = self._reps_for(self._pose_with_a_hidden_far_arm())
        assert reps[0].both_arms_ratio == pytest.approx(0.0, abs=0.05)

    def test_a_hidden_analysed_arm_does_block_the_range_check(self):
        pose = make_pose_data(PulldownSpec())
        pose.visibility[:, [13, 15]] = 0.1  # the arm being analysed
        video = video_for(pose)
        metrics = metrics_mod.compute_frame_metrics(video, pose, "left", CONFIG)
        detection = detect_reps(metrics_mod.movement_signal(metrics), pose.timestamps, CONFIG)
        reps = metrics_mod.build_reps(detection.reps, metrics, pose, "left", "unsigned", CONFIG)
        assert all(not rep.arms_reliable for rep in reps)
