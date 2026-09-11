"""
Tests for validation and its three states:

    GOOD      enough evidence, clean side view      -> analyse
    LIMITED   enough evidence, imperfect recording  -> analyse and warn
    UNUSABLE  not enough evidence                   -> typed rejection

Most of these are about the middle case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analysis import validation
from analysis.models import (
    NUM_LANDMARKS,
    CameraOrientation,
    FramePoseData,
    RecordingQuality,
    RejectionCode,
    VideoMetadata,
)
from analysis.video_processor import probe_video
from exercises.squat.analyser import validation_config_for
from exercises.squat.config import SquatConfig

C = SquatConfig()
VCONFIG = validation_config_for(C)


def video_meta(**overrides) -> VideoMetadata:
    values = dict(
        path=Path("test.mp4"), width=720, height=1280, fps=30.0, frame_count=300, duration=10.0
    )
    values.update(overrides)
    return VideoMetadata(**values)


def empty_pose(n: int) -> FramePoseData:
    xy = np.full((n, NUM_LANDMARKS, 2), np.nan)
    return FramePoseData(
        xy_raw=xy.copy(),
        xy=xy,
        visibility=np.zeros((n, NUM_LANDMARKS)),
        valid=np.zeros((n, NUM_LANDMARKS), dtype=bool),
        pose_found=np.zeros(n, dtype=bool),
        n_poses=np.zeros(n, dtype=np.int16),
        timestamps=np.arange(n) / 30.0,
    )


def tracked_pose(
    n: int = 300,
    visibility: float = 0.9,
    shoulder_sep: float = 0.02,
    hip_sep: float = 0.02,
) -> FramePoseData:
    """
    Fully tracked side-on pose. shoulder_sep and hip_sep set the frontality
    (small = side-on). On this 720x1280 frame the ratio is
    s * 720 / (0.25 * 1280) = s * 2.25.
    """
    pose = empty_pose(n)
    pose.pose_found[:] = True
    pose.n_poses[:] = 1
    pose.visibility[:, :] = visibility
    pose.xy_raw[:, :, :] = 0.5
    pose.xy_raw[:, 11, 0] = 0.5 - shoulder_sep / 2
    pose.xy_raw[:, 12, 0] = 0.5 + shoulder_sep / 2
    pose.xy_raw[:, 23, 0] = 0.5 - hip_sep / 2
    pose.xy_raw[:, 24, 0] = 0.5 + hip_sep / 2
    pose.xy_raw[:, 11, 1] = pose.xy_raw[:, 12, 1] = 0.30
    pose.xy_raw[:, 23, 1] = pose.xy_raw[:, 24, 1] = 0.55
    pose.valid[:, :] = True
    pose.xy = pose.xy_raw.copy()
    return pose


# --- File-level checks ---


class TestFileValidation:
    def test_unreadable_file_rejected(self, tmp_path):
        bogus = tmp_path / "not_a_video.mp4"
        bogus.write_bytes(b"this is not a video at all")
        result = validation.validate_file(probe_video(bogus), VCONFIG)
        assert not result.valid
        assert "couldn't read" in result.errors[0]
        assert RejectionCode.VIDEO_READ_ERROR in result.reason_codes

    def test_empty_file_rejected(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        assert not validation.validate_file(probe_video(empty), VCONFIG).valid

    def test_too_short_rejected(self):
        result = validation.validate_file(video_meta(duration=1.2, frame_count=36), VCONFIG)
        assert not result.valid
        assert RejectionCode.VIDEO_TOO_SHORT in result.reason_codes

    def test_too_long_rejected(self):
        result = validation.validate_file(video_meta(duration=600.0), VCONFIG)
        assert RejectionCode.VIDEO_TOO_LONG in result.reason_codes

    def test_good_metadata_passes(self):
        result = validation.validate_file(video_meta(), VCONFIG)
        assert result.valid
        assert result.quality is RecordingQuality.GOOD

    def test_zero_fps_handled_without_crash(self, tmp_path):
        bogus = tmp_path / "zfps.mp4"
        bogus.write_bytes(b"\x00" * 2048)
        assert probe_video(bogus).duration == 0.0


# --- Stage A-C: is there enough evidence? ---


class TestPoseEvidence:
    def test_no_pose_frames_rejected(self):
        result = validation.validate_pose_track(video_meta(), empty_pose(300), VCONFIG)
        assert not result.valid
        assert RejectionCode.NO_POSE_DETECTED in result.reason_codes
        assert result.quality is RecordingQuality.UNUSABLE

    def test_sparse_detection_rejected(self):
        pose = empty_pose(300)
        pose.pose_found[:60] = True  # 20% of the clip
        pose.n_poses[:60] = 1
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert RejectionCode.INSUFFICIENT_POSE_COVERAGE in result.reason_codes

    def test_low_visibility_rejected(self):
        result = validation.validate_pose_track(video_meta(), tracked_pose(visibility=0.2), VCONFIG)
        assert RejectionCode.IMPORTANT_LANDMARKS_MISSING in result.reason_codes

    def test_clean_side_view_is_good(self):
        result = validation.validate_pose_track(video_meta(), tracked_pose(), VCONFIG)
        assert result.valid
        assert result.quality is RecordingQuality.GOOD
        assert result.orientation is CameraOrientation.SIDE
        assert result.side_view_confidence == pytest.approx(1.0)
        assert result.warnings == []

    def test_temporary_tracking_loss_still_analysed(self):
        """A few lost frames shouldn't reject the whole recording."""
        pose = tracked_pose()
        pose.visibility[100:130, :] = 0.0  # a second of lost tracking
        pose.valid[100:130, :] = False
        pose.pose_found[100:130] = False
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert result.valid
        assert result.metrics["usable_frame_ratio"] == pytest.approx(0.9)

    def test_usable_frame_ratio_boundary(self):
        """Clearly above MIN_USABLE_FRAME_RATIO analyses, clearly below rejects."""
        above = tracked_pose()
        cut = int(300 * (1 - (C.MIN_USABLE_FRAME_RATIO + 0.2)))
        above.visibility[:cut, :] = 0.0
        above.valid[:cut, :] = False
        assert validation.validate_pose_track(video_meta(), above, VCONFIG).valid

        below = tracked_pose()
        cut = int(300 * (1 - (C.MIN_USABLE_FRAME_RATIO - 0.05)))
        below.visibility[:cut, :] = 0.0
        below.valid[:cut, :] = False
        result = validation.validate_pose_track(video_meta(), below, VCONFIG)
        assert RejectionCode.IMPORTANT_LANDMARKS_MISSING in result.reason_codes

    def test_bystanders_warn_but_do_not_reject(self):
        """Someone else in shot is normal in a gym - as long as tracking stayed on one
        person it only gets a note."""
        pose = tracked_pose()
        pose.n_poses[:] = 2
        pose.identity_switches = 0
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert RejectionCode.MULTIPLE_PEOPLE not in result.reason_codes
        assert result.valid
        assert any("person" in w.lower() for w in result.warnings)

    def test_a_track_that_keeps_coming_apart_is_rejected(self):
        """The tracker keeps losing which person it's following, so reject it."""
        pose = tracked_pose()
        pose.n_poses[:] = 3
        pose.identity_switches = 60
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert RejectionCode.MULTIPLE_PEOPLE in result.reason_codes


# --- Stage D: framing ---


class TestFraming:
    def test_feet_at_edge_warns_and_limits_heel_only(self):
        pose = tracked_pose()
        pose.xy_raw[:, 27, 1] = 0.999
        pose.xy_raw[:, 28, 1] = 0.999
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert result.valid
        assert result.quality is RecordingQuality.LIMITED
        assert result.limited_metrics == ["heel_lift"]

    def test_partly_cropped_hips_warn_but_analyse(self):
        pose = tracked_pose()
        pose.xy_raw[:100, 23, 0] = 0.999  # hips clipped for a third of the clip
        pose.xy_raw[:100, 24, 0] = 0.999
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert result.valid
        assert any("hips" in w for w in result.warnings)

    def test_badly_cropped_hips_rejected(self):
        pose = tracked_pose()
        pose.xy_raw[:, 23, 0] = 0.999
        pose.xy_raw[:, 24, 0] = 0.999
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert RejectionCode.BODY_OUT_OF_FRAME in result.reason_codes


# --- Stage E: camera orientation is a confidence, not a gate ---


class TestCameraOrientation:
    def test_classification_bands(self):
        side, conf = validation.classify_orientation(0.10, VCONFIG)
        assert side is CameraOrientation.SIDE and conf == 1.0

        diagonal, conf = validation.classify_orientation(
            (C.SIDE_VIEW_GOOD_RATIO + C.SIDE_VIEW_FRONTAL_RATIO) / 2, VCONFIG
        )
        assert diagonal is CameraOrientation.DIAGONAL_SIDE
        assert 0.0 < conf < 1.0

        frontal, conf = validation.classify_orientation(1.4, VCONFIG)
        assert frontal is CameraOrientation.FRONTAL and conf == 0.0

        unknown, conf = validation.classify_orientation(float("nan"), VCONFIG)
        assert unknown is CameraOrientation.UNKNOWN and conf == 0.0

    def test_diagonal_view_is_analysed_with_a_warning(self):
        """A diagonal view gets a warning, not a rejection."""
        pose = tracked_pose(shoulder_sep=0.29, hip_sep=0.20)  # ~0.65 frontality
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert result.valid
        assert result.orientation is CameraOrientation.DIAGONAL_SIDE
        assert result.quality is RecordingQuality.LIMITED
        assert result.limited_metrics == []  # everything still gets measured
        assert any("not fully side-on" in w for w in result.warnings)

    def test_frontal_view_is_analysed_but_limits_sagittal_metrics(self):
        pose = tracked_pose(shoulder_sep=0.53, hip_sep=0.36)  # ~1.2 frontality
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert result.valid  # front-on isn't a rejection either
        assert result.orientation is CameraOrientation.FRONTAL
        assert set(result.limited_metrics) == {"squat_depth", "torso_lean", "return_to_standing"}
        assert RejectionCode.UNSUPPORTED_CAMERA_ANGLE not in result.reason_codes

    def test_frontal_view_with_unusable_landmarks_is_rejected(self):
        pose = tracked_pose(visibility=0.1, shoulder_sep=0.53, hip_sep=0.36)
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert not result.valid
        assert RejectionCode.IMPORTANT_LANDMARKS_MISSING in result.reason_codes


# --- Side selection ---


class TestSideSelection:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_better_side_wins(self, side):
        pose = tracked_pose(visibility=0.2)
        ids = {"left": [11, 23, 25, 27, 29, 31], "right": [12, 24, 26, 28, 30, 32]}[side]
        pose.visibility[:, ids] = 0.95
        chosen, scores = validation.select_analysis_side(pose, C.MIN_KEY_LANDMARK_VISIBILITY)
        assert chosen == side
        assert scores[side] == pytest.approx(1.0)

    def test_hidden_far_side_does_not_block_analysis(self):
        """A side view always hides half the body, so that can't fail it."""
        pose = tracked_pose()
        pose.visibility[:, [12, 24, 26, 28, 30, 32]] = 0.15  # far side occluded
        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert result.valid
        assert result.selected_side == "left"

    def test_diagnostics_expose_every_development_number(self):
        result = validation.validate_pose_track(video_meta(), tracked_pose(), VCONFIG)
        diagnostics = result.diagnostics()
        for key in (
            "pose_frame_ratio",
            "usable_frame_ratio",
            "average_landmark_visibility",
            "selected_side",
            "orientation",
            "side_view_confidence",
            "quality",
            "reason_codes",
        ):
            assert key in diagnostics


# --- Merge ---


def test_merge_keeps_codes_and_orientation():
    first = validation.validate_file(video_meta(), VCONFIG)
    second = validation.validate_pose_track(video_meta(), tracked_pose(shoulder_sep=0.29), VCONFIG)
    merged = validation.merge(first, second)
    assert merged.valid
    assert merged.orientation is CameraOrientation.DIAGONAL_SIDE
    assert merged.selected_side in ("left", "right")
    assert merged.quality is RecordingQuality.LIMITED


class TestViewStability:
    """The camera angle is estimated once per video, so a clip that changes angle
    gets a warning instead of an average."""

    def test_a_steady_camera_is_stable(self):
        pose = tracked_pose()
        spread = validation.view_stability(pose)
        assert spread < C.VIEW_STABILITY_SPREAD

    def test_a_changing_camera_angle_is_reported(self):

        pose = tracked_pose()
        half = pose.frame_count // 2
        # second half is filmed from the front
        pose.xy_raw[half:, 11, 0] -= 0.30
        pose.xy_raw[half:, 12, 0] += 0.30
        assert validation.view_stability(pose) > C.VIEW_STABILITY_SPREAD

        result = validation.validate_pose_track(video_meta(), pose, VCONFIG)
        assert result.valid  # a warning, not a rejection
        assert any("camera angle appears to change" in w for w in result.warnings)
        assert result.metrics["frontality_spread"] is not None

    def test_stability_is_unknown_without_enough_data(self):
        import numpy as np

        pose = tracked_pose()
        pose.pose_found[:] = False
        assert np.isnan(validation.view_stability(pose))
