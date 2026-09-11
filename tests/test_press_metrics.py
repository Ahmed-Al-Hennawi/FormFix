"""
Tests for the shoulder press measurements. No verdicts, just that the numbers
match the generated movement, including normalisation (so one threshold works
close up and from across the gym).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from analysis.models import VideoMetadata
from exercises.common.phases import MovementPhase, RawRep
from exercises.press import metrics as metrics_mod
from exercises.press.config import DEFAULT_CONFIG
from exercises.press.phases import detect_reps, resting_flexion
from tests.synthetic_press import HEIGHT, WIDTH, PressSpec, make_pose_data, skeleton_at

CONFIG = DEFAULT_CONFIG


def video_for(pose, width: int = WIDTH, height: int = HEIGHT) -> VideoMetadata:
    return VideoMetadata(
        path=Path("synthetic.mp4"),
        width=width,
        height=height,
        fps=30.0,
        frame_count=pose.frame_count,
        duration=pose.frame_count / 30.0,
    )


def measured(spec: PressSpec, side: str = "left"):
    pose = make_pose_data(spec)
    video = video_for(pose)
    return pose, video, metrics_mod.compute_frame_metrics(video, pose, side, CONFIG)


def reps_for(spec: PressSpec):
    pose, _, metrics = measured(spec)
    signal = metrics_mod.movement_signal(metrics)
    detection = detect_reps(signal, pose.timestamps, CONFIG)
    return metrics_mod.build_reps(detection.reps, metrics, pose, CONFIG), metrics, detection


class TestElbowAngles:
    def test_the_measured_angles_match_the_generated_ones(self):
        _, _, metrics = measured(PressSpec(bottom_elbow=85.0, top_elbow=168.0))
        left = [m.left_elbow_angle for m in metrics if m.valid]
        right = [m.right_elbow_angle for m in metrics if m.valid]
        assert max(left) == pytest.approx(168.0, abs=1.0)
        assert min(left) == pytest.approx(85.0, abs=1.0)
        assert max(right) == pytest.approx(168.0, abs=1.0)
        assert min(right) == pytest.approx(85.0, abs=1.0)

    def test_the_movement_signal_is_flexion_not_the_raw_angle(self):
        # high at the shoulders, near zero overhead - what the shared state
        # machine expects
        _, _, metrics = measured(PressSpec())
        signal = metrics_mod.movement_signal(metrics)
        finite = signal[np.isfinite(signal)]
        assert finite.max() == pytest.approx(180.0 - 85.0, abs=2.0)
        assert finite.min() == pytest.approx(180.0 - 168.0, abs=2.0)

    def test_an_implausible_angle_is_discarded_rather_than_reported(self):
        pose = make_pose_data(PressSpec())
        pose.xy[40:50, 15] = pose.xy[40:50, 11]  # wrist folded onto the shoulder
        metrics = metrics_mod.compute_frame_metrics(video_for(pose), pose, "left", CONFIG)
        for m in metrics[40:50]:
            assert (
                not math.isfinite(m.left_elbow_angle)
                or m.left_elbow_angle >= CONFIG.PLAUSIBLE_ELBOW_ANGLE_MIN
            )


class TestSymmetryMeasurements:
    def test_a_synchronised_press_shows_no_difference(self):
        _, _, metrics = measured(PressSpec())
        differences = [
            m.elbow_angle_difference for m in metrics if math.isfinite(m.elbow_angle_difference)
        ]
        heights = [
            m.wrist_height_difference for m in metrics if math.isfinite(m.wrist_height_difference)
        ]
        assert max(differences) < 1.0
        assert max(abs(h) for h in heights) < 0.01

    def test_a_trailing_arm_produces_a_measurable_difference(self):
        _, _, metrics = measured(PressSpec(right_lag=0.3))
        differences = [
            m.elbow_angle_difference for m in metrics if math.isfinite(m.elbow_angle_difference)
        ]
        assert max(differences) > CONFIG.SYMMETRY_ANGLE_WARN

    def test_the_sign_names_the_higher_arm(self):
        # positive = left wrist higher, and the right arm lags here
        _, _, metrics = measured(PressSpec(right_lag=0.3))
        heights = [
            m.wrist_height_difference for m in metrics if math.isfinite(m.wrist_height_difference)
        ]
        assert max(heights) > 0.05
        assert abs(min(heights)) < 0.02

    def test_one_hidden_arm_makes_the_comparison_unavailable(self):
        pose = make_pose_data(PressSpec())
        pose.visibility[:, [14, 16]] = 0.1  # right elbow and wrist barely seen
        metrics = metrics_mod.compute_frame_metrics(video_for(pose), pose, "left", CONFIG)
        assert not any(m.both_arms_valid for m in metrics)
        assert all(not math.isfinite(m.elbow_angle_difference) for m in metrics)


class TestAlignmentMeasurement:
    def test_a_stacked_forearm_shows_a_small_offset(self):
        _, _, metrics = measured(PressSpec())
        offsets = [m.left_alignment_offset for m in metrics if math.isfinite(m.left_alignment_offset)]
        assert max(offsets) < CONFIG.ALIGNMENT_OFFSET_WARN

    def test_a_drifting_wrist_shows_a_large_offset(self):
        _, _, metrics = measured(PressSpec(left_drift=0.6))
        left = max(m.left_alignment_offset for m in metrics if math.isfinite(m.left_alignment_offset))
        right = max(
            m.right_alignment_offset for m in metrics if math.isfinite(m.right_alignment_offset)
        )
        assert left > CONFIG.ALIGNMENT_OFFSET_FAIL
        assert right < CONFIG.ALIGNMENT_OFFSET_WARN  # other arm unaffected

    def test_the_offset_is_normalised_by_shoulder_width_not_pixels(self):
        # same movement at two resolutions gives the same offset
        pose = make_pose_data(PressSpec(left_drift=0.4))
        big = metrics_mod.compute_frame_metrics(video_for(pose), pose, "left", CONFIG)
        small = metrics_mod.compute_frame_metrics(
            video_for(pose, WIDTH // 2, HEIGHT // 2), pose, "left", CONFIG
        )
        for a, b in zip(big, small, strict=True):
            if math.isfinite(a.left_alignment_offset) and math.isfinite(b.left_alignment_offset):
                assert a.left_alignment_offset == pytest.approx(b.left_alignment_offset, abs=1e-6)

    def test_a_collapsed_shoulder_width_cannot_explode_the_offset(self):
        # nearly side-on the shoulder width gets tiny, but the numbers still have
        # to be finite
        _, _, metrics = measured(PressSpec(view_compression=0.05))
        offsets = [m.left_alignment_offset for m in metrics if math.isfinite(m.left_alignment_offset)]
        assert max(offsets) < CONFIG.ALIGNMENT_OFFSET_MAX_PLAUSIBLE


class TestRestingFlexionReference:
    def test_the_reference_follows_the_athletes_own_starting_position(self):
        deep = resting_flexion(np.asarray([95.0] * 40 + [12.0] * 20), CONFIG)
        shallow = resting_flexion(np.asarray([55.0] * 40 + [12.0] * 20), CONFIG)
        assert deep == pytest.approx(95.0, abs=2.0)
        assert shallow == pytest.approx(55.0, abs=2.0)

    def test_the_reference_is_clamped_to_a_plausible_band(self):
        assert resting_flexion(np.full(40, 10.0), CONFIG) == CONFIG.REST_REFERENCE_MIN
        assert resting_flexion(np.full(40, 200.0), CONFIG) == CONFIG.REST_REFERENCE_MAX

    def test_an_unusable_series_falls_back_rather_than_failing(self):
        assert resting_flexion(np.array([np.nan]), CONFIG) == CONFIG.READY_ELBOW_FLEXION


class TestPhaseSegmentation:
    def test_a_clean_recording_produces_the_expected_repetitions(self):
        reps, _, detection = reps_for(PressSpec(reps=3))
        assert len(reps) == 3
        assert detection.partial_movements == 0

    def test_every_expected_phase_is_labelled(self):
        _, _, detection = reps_for(PressSpec())
        seen = set(detection.phases)
        assert MovementPhase.REST in seen
        assert MovementPhase.TOWARDS in seen
        assert MovementPhase.EXTREME in seen
        assert MovementPhase.RETURN in seen

    def test_an_athlete_who_never_locks_out_still_has_reps_counted(self):
        # this is why rep detection is adaptive - with fixed thresholds the ROM
        # rule would never see the habit it's meant to catch
        reps, _, _ = reps_for(PressSpec(top_elbow=138.0))
        assert len(reps) == 3

    def test_an_athlete_who_never_lowers_fully_still_has_reps_counted(self):
        reps, _, _ = reps_for(PressSpec(bottom_elbow=125.0))
        assert len(reps) == 3


class TestRepMeasurements:
    def test_per_repetition_facts_describe_the_generated_movement(self):
        reps, _, _ = reps_for(PressSpec(bottom_elbow=85.0, top_elbow=168.0))
        assert len(reps) == 3
        for rep in reps:
            assert rep.top_elbow_angle == pytest.approx(168.0, abs=4.0)
            assert rep.bottom_elbow_angle == pytest.approx(85.0, abs=4.0)
            assert rep.rom_degrees > 70.0
            assert rep.symmetry_reliable
            assert rep.alignment_reliable
            assert rep.both_arms_ratio > 0.95

    def test_each_arm_gets_its_own_range(self):
        reps, _, _ = reps_for(PressSpec(right_top_elbow=130.0))
        rep = reps[0]
        assert rep.left_rom_degrees > rep.right_rom_degrees
        assert rep.rom_difference > CONFIG.SYMMETRY_ROM_WARN

    def test_a_lagging_arm_does_not_shorten_the_other_arms_reported_range(self):
        # each arm uses its own best top position, so a timing issue is only
        # reported once (by symmetry)
        reps, _, _ = reps_for(PressSpec(right_lag=0.25))
        assert reps[0].left_top_elbow_angle == pytest.approx(168.0, abs=4.0)

    def test_the_higher_side_is_named_from_the_median_not_one_frame(self):
        reps, _, _ = reps_for(PressSpec(right_lag=0.3))
        assert reps[0].higher_side == "left"

    def test_a_symmetrical_press_names_no_higher_side(self):
        reps, _, _ = reps_for(PressSpec())
        assert reps[0].higher_side == ""


class TestBottomWindows:
    def test_the_bottom_is_read_outside_the_repetition_not_at_its_first_frame(self):
        raws = [RawRep(start_frame=40, extreme_frame=55, end_frame=70, extreme_value=12.0)]
        windows = metrics_mod.bottom_windows(raws, 0, 30.0, 200)
        assert windows[0].start < 40 <= windows[0].stop
        assert windows[1].start == 70

    def test_one_repetition_cannot_borrow_its_neighbours_depth(self):
        raws = [
            RawRep(start_frame=40, extreme_frame=55, end_frame=70, extreme_value=12.0),
            RawRep(start_frame=80, extreme_frame=95, end_frame=110, extreme_value=12.0),
        ]
        assert metrics_mod.bottom_windows(raws, 0, 30.0, 200)[1].stop - 1 <= 80
        assert metrics_mod.bottom_windows(raws, 1, 30.0, 200)[0].start >= 70


def test_the_generator_itself_is_geometrically_faithful():
    """A synthetic 90 degree elbow should measure 90 in pixels."""
    from analysis.geometry import calculate_angle

    spec = PressSpec(bottom_elbow=90.0, top_elbow=160.0)
    for depth, expected in ((0.0, 90.0), (1.0, 160.0)):
        points = skeleton_at(depth, spec)

        def px(lm, points=points):
            return (points[lm][0] * WIDTH, points[lm][1] * HEIGHT)

        assert calculate_angle(px(11), px(13), px(15)) == pytest.approx(expected, abs=0.5)
        assert calculate_angle(px(12), px(14), px(16)) == pytest.approx(expected, abs=0.5)


class TestReliabilityGatesAreScopedToWhatEachCheckNeeds:
    """
    Only symmetry needs both arms. Alignment is per arm, so one visible arm is
    enough - averaging both let a hidden arm switch off the check for the other.
    """

    def _pose_with_one_hidden_arm(self):
        from tests.synthetic_press import make_pose_data

        pose = make_pose_data(PressSpec())
        # low enough that the mean of both arms drops under the floor
        pose.visibility[:, [14, 16]] = 0.02
        return pose

    def test_one_hidden_arm_still_allows_the_alignment_check(self):
        pose = self._pose_with_one_hidden_arm()
        metrics = metrics_mod.compute_frame_metrics(video_for(pose), pose, "left", CONFIG)
        signal = metrics_mod.movement_signal(metrics)
        detection = detect_reps(signal, pose.timestamps, CONFIG)
        reps = metrics_mod.build_reps(detection.reps, metrics, pose, CONFIG)
        assert reps
        assert reps[0].alignment_reliable
        assert math.isfinite(reps[0].max_left_alignment_offset)

    def test_one_hidden_arm_still_blocks_the_symmetry_check(self):
        pose = self._pose_with_one_hidden_arm()
        metrics = metrics_mod.compute_frame_metrics(video_for(pose), pose, "left", CONFIG)
        signal = metrics_mod.movement_signal(metrics)
        detection = detect_reps(signal, pose.timestamps, CONFIG)
        reps = metrics_mod.build_reps(detection.reps, metrics, pose, CONFIG)
        assert not reps[0].symmetry_reliable
