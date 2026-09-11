"""
Tests for the annotated video writer. Mostly here because cv2.VideoWriter fails
silently if the folder doesn't exist - it took an FFmpeg error about a missing
input before I worked out what was happening.
"""

from __future__ import annotations

import pytest

from analysis.models import AnalysisFailure, FailureCode
from analysis.video_processor import open_writer, probe_video


class TestWriterGuards:
    def test_a_missing_parent_directory_is_created(self, tmp_path):
        target = tmp_path / "does" / "not" / "exist" / "clip.mp4"
        with open_writer(target, 30.0, (64, 64)) as writer:
            assert writer.isOpened()
        assert target.parent.is_dir()

    def test_the_written_file_is_readable(self, tmp_path):
        import numpy as np

        target = tmp_path / "out" / "clip.mp4"
        frame = np.full((64, 64, 3), 40, dtype=np.uint8)
        with open_writer(target, 30.0, (64, 64)) as writer:
            for _ in range(10):
                writer.write(frame)
        assert probe_video(target).readable

    def test_an_unopenable_target_raises_instead_of_silently_doing_nothing(self, tmp_path):
        # parent is a file, so the writer can't open the target
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        with pytest.raises(AnalysisFailure) as caught:
            with open_writer(blocker / "clip.mp4", 30.0, (64, 64)):
                pass
        assert caught.value.code is FailureCode.VIDEO_RENDER_ERROR


class TestAnalysersCreateTheirOwnOutputDirectory:
    """Each analyser is handed an output_dir that doesn't exist yet."""

    def test_pulldown(self, tmp_path):
        from unittest import mock

        from analysis import pose_detector
        from exercises.pulldown.analyser import analyse_lat_pulldown
        from tests.synthetic_pulldown import PulldownSpec, make_pose_data, write_plain_video

        pose = make_pose_data(PulldownSpec())
        video = tmp_path / "clip.mp4"
        write_plain_video(video, pose.frame_count)
        target = tmp_path / "not" / "created" / "yet"
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_lat_pulldown(video, output_dir=target)
        assert result.annotated_video_path is not None
        assert result.annotated_video_path.is_file()

    def test_press(self, tmp_path):
        from unittest import mock

        from analysis import pose_detector
        from exercises.press.analyser import analyse_shoulder_press
        from tests.synthetic_press import PressSpec, make_pose_data, write_plain_video

        pose = make_pose_data(PressSpec())
        video = tmp_path / "clip.mp4"
        write_plain_video(video, pose.frame_count)
        target = tmp_path / "not" / "created" / "yet"
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_shoulder_press(video, output_dir=target)
        assert result.annotated_video_path is not None
        assert result.annotated_video_path.is_file()

    def test_squat(self, tmp_path):
        from unittest import mock

        from analysis import pose_detector
        from exercises.squat.analyser import analyse_squat
        from tests.synthetic_squat import SyntheticSpec, make_pose_data, write_plain_video

        pose = make_pose_data(SyntheticSpec(noise=0.002))
        video = tmp_path / "clip.mp4"
        write_plain_video(video, pose.frame_count)
        target = tmp_path / "not" / "created" / "yet"
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_squat(video, output_dir=target)
        assert result.annotated_video_path is not None
        assert result.annotated_video_path.is_file()


class TestOverlayShowsWhatTheAnalysisUsed:
    """
    Interpolated frames are still measured, so they need a skeleton too. Using
    the raw detector flag made it vanish on 21 frames of a 710-frame clip.
    """

    def _pose_with_an_interpolated_gap(self):
        from analysis import smoothing
        from tests.synthetic_pulldown import PulldownSpec, make_pose_data

        pose = make_pose_data(PulldownSpec())
        gap = range(40, 43)
        for frame in gap:
            pose.xy[frame, :, :] = float("nan")
            pose.xy_raw[frame, :, :] = float("nan")
            pose.valid[frame, :] = False
            pose.pose_found[frame] = False
        smoothing.interpolate_short_gaps(pose, 5)
        return pose, gap

    def test_an_interpolated_frame_is_still_drawn(self):
        from analysis.annotation import _has_drawable_pose

        pose, gap = self._pose_with_an_interpolated_gap()
        for frame in gap:
            assert not pose.pose_found[frame]  # the detector never fired here
            assert pose.valid[frame].any()  # but the analysis has landmarks
            assert _has_drawable_pose(pose, frame)

    def test_a_frame_with_no_landmarks_at_all_is_not_drawn(self):
        from analysis.annotation import _has_drawable_pose
        from tests.synthetic_pulldown import PulldownSpec, make_pose_data

        pose = make_pose_data(PulldownSpec())
        pose.xy[10, :, :] = float("nan")
        pose.valid[10, :] = False
        pose.pose_found[10] = False
        assert not _has_drawable_pose(pose, 10)

    def test_a_detected_frame_is_drawn(self):
        from analysis.annotation import _has_drawable_pose
        from tests.synthetic_pulldown import PulldownSpec, make_pose_data

        pose = make_pose_data(PulldownSpec())
        assert _has_drawable_pose(pose, 0)


class TestPhaseCaptionMatchesTheRepetition:
    """
    An abandoned partial rep left old labels inside the next real rep, so the
    overlay captioned the start of a rep "RETURNING".
    """

    def _states_for(self, analyser_module, reps, detection, metrics, frames):
        return analyser_module._frame_states(detection, reps, metrics, frames)

    def _run(self, exercise: str):
        from unittest import mock

        from analysis import pose_detector

        if exercise == "pulldown":
            from exercises.pulldown import analyser as module
            from tests.synthetic_pulldown import PulldownSpec, make_pose_data

            pose = make_pose_data(PulldownSpec())
            entry = module.analyse_lat_pulldown
        else:
            from exercises.press import analyser as module
            from tests.synthetic_press import PressSpec, make_pose_data

            pose = make_pose_data(PressSpec())
            entry = module.analyse_shoulder_press
        return module, entry, pose, mock, pose_detector

    def _analyse(self, exercise, tmp_path):
        module, entry, pose, mock, pose_detector = self._run(exercise)
        if exercise == "pulldown":
            from tests.synthetic_pulldown import write_plain_video
        else:
            from tests.synthetic_press import write_plain_video
        tmp_path.mkdir(parents=True, exist_ok=True)
        video = tmp_path / "clip.mp4"
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            return entry(video, output_dir=tmp_path)

    @pytest.mark.parametrize(
        ("exercise", "towards", "extreme", "back"),
        [
            ("pulldown", "pulling", "bottom", "returning"),
            ("press", "pressing", "top", "lowering"),
        ],
    )
    def test_a_repetitions_frames_read_in_order(self, exercise, towards, extreme, back, tmp_path):
        from exercises.common.metrics import segment_at

        result = self._analyse(exercise, tmp_path)
        rep = result.reps[0]
        seg = rep.segmentation

        assert segment_at(seg, seg.start_frame) == "towards"
        assert segment_at(seg, seg.extreme_start_frame) == "extreme"
        assert segment_at(seg, seg.end_frame) == "return"
        assert segment_at(seg, seg.start_frame - 5) is None
        assert segment_at(seg, seg.end_frame + 5) is None
        del towards, extreme, back

    def test_the_squat_caption_follows_its_own_segmentation(self, tmp_path):
        from unittest import mock

        from analysis import pose_detector
        from analysis.models import Phase
        from exercises.squat import analyser as module
        from exercises.squat.analyser import analyse_squat
        from tests.synthetic_squat import SyntheticSpec, make_pose_data, write_plain_video

        pose = make_pose_data(SyntheticSpec(noise=0.002))
        tmp_path.mkdir(parents=True, exist_ok=True)
        video = tmp_path / "clip.mp4"
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_squat(video, output_dir=tmp_path)

        detection = mock.MagicMock()
        detection.phases = [Phase.UNKNOWN] * pose.frame_count
        states = module._frame_states(detection, result.reps, pose.frame_count)
        rep = result.reps[0]
        assert states[rep.start_frame].phase is Phase.DESCENDING
        assert states[rep.bottom_frame].phase is Phase.BOTTOM
        assert states[rep.end_frame].phase is Phase.ASCENDING


class TestTheOverlayDrawsOnlyWhatTheDetectorActuallySaw:
    """
    Side-on, MediaPipe guesses the hidden far limbs at 0.2-0.6 confidence and
    they jump hundreds of pixels between frames, so the skeleton thrashed while
    the numbers were calm. The overlay now needs higher confidence for a hidden
    limb and drops impossible jumps (not for front views or measured limbs).
    """

    FRAMES = 40

    def _pose(self, far_visibility: float = 0.45, jitter: float = 0.25):
        """A still figure whose right knee and wrist are guessed, not seen."""
        import numpy as np

        from analysis.models import (
            NUM_LANDMARKS,
            RIGHT_KNEE,
            RIGHT_WRIST,
            FramePoseData,
        )

        rng = np.random.default_rng(11)
        xy = np.full((self.FRAMES, NUM_LANDMARKS, 2), np.nan)
        visibility = np.zeros((self.FRAMES, NUM_LANDMARKS))
        for frame in range(self.FRAMES):
            for landmark in range(NUM_LANDMARKS):
                xy[frame, landmark] = (0.48, 0.20 + 0.012 * landmark)
                visibility[frame, landmark] = 0.95
            for landmark in (RIGHT_KNEE, RIGHT_WRIST):
                xy[frame, landmark] = (
                    0.5 + rng.uniform(-jitter, jitter),
                    0.5 + rng.uniform(-jitter, jitter),
                )
                visibility[frame, landmark] = far_visibility
        return FramePoseData(
            xy_raw=xy.copy(),
            xy=xy.copy(),
            visibility=visibility,
            valid=np.isfinite(xy[:, :, 0]),
            pose_found=np.ones(self.FRAMES, dtype=bool),
            n_poses=np.ones(self.FRAMES, dtype="int16"),
            timestamps=np.arange(self.FRAMES) / 30.0,
        )

    def _video(self):
        from pathlib import Path

        from analysis.models import VideoMetadata

        return VideoMetadata(
            path=Path("clip.mp4"),
            width=1080,
            height=1920,
            fps=30.0,
            frame_count=self.FRAMES,
            duration=self.FRAMES / 30.0,
        )

    def _drawn(self, orientation, side="left", focus=frozenset()):
        """How many frames each landmark survived to be drawn on."""
        from collections import Counter

        from analysis.annotation import _draw_policy, _drawable_points

        pose, video = self._pose(), self._video()
        policy = _draw_policy(pose, video, side, orientation, focus)
        counts: Counter = Counter()
        for frame in range(self.FRAMES):
            points, _ = _drawable_points(pose, frame, video, policy)
            counts.update(points.keys())
        return counts

    def test_a_guessed_far_limb_is_not_drawn_in_a_side_view(self):
        from analysis.models import RIGHT_KNEE, RIGHT_WRIST

        drawn = self._drawn("side")
        assert drawn[RIGHT_KNEE] == 0
        assert drawn[RIGHT_WRIST] == 0

    def test_the_measured_near_side_is_untouched(self):
        from analysis.models import LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER

        drawn = self._drawn("side")
        for landmark in (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE):
            assert drawn[landmark] == self.FRAMES

    def test_the_far_shoulder_and_hip_stay_so_the_torso_keeps_its_width(self):
        from analysis.models import RIGHT_HIP, RIGHT_SHOULDER

        drawn = self._drawn("side")
        assert drawn[RIGHT_SHOULDER] == self.FRAMES
        assert drawn[RIGHT_HIP] == self.FRAMES

    def test_a_frontal_view_keeps_both_sides(self):
        from analysis.models import RIGHT_KNEE, RIGHT_WRIST

        # only the jump gate applies here, and it trips on some frames, hence "> 0"
        drawn = self._drawn("frontal")
        assert drawn[RIGHT_KNEE] > 0
        assert drawn[RIGHT_WRIST] > 0

    def test_an_unknown_view_behaves_as_it_always_did(self):
        from analysis.models import RIGHT_KNEE

        assert self._drawn("")[RIGHT_KNEE] == self._drawn("frontal")[RIGHT_KNEE]

    def test_a_limb_the_exercise_measures_is_never_suppressed(self):
        from analysis.models import RIGHT_KNEE, RIGHT_WRIST

        # the press uses both arms, so the far arm stays drawn even side-on
        drawn = self._drawn("side", focus=frozenset({RIGHT_KNEE, RIGHT_WRIST}))
        assert drawn[RIGHT_KNEE] > 0
        assert drawn[RIGHT_WRIST] > 0

    def test_a_landmark_that_teleports_is_refused_and_then_recovers(self):
        from analysis.annotation import MAX_CONSECUTIVE_JUMP_REJECTS, _DrawPolicy

        policy = _DrawPolicy(
            trust_visibility=True,
            far_limbs=frozenset(),
            max_jump_px=50.0,
        )
        assert policy.accept_position(1, 0, 100.0, 100.0)
        for offset in range(1, MAX_CONSECUTIVE_JUMP_REJECTS + 1):
            assert not policy.accept_position(1, offset, 900.0, 900.0)
        # after enough rejects it has to come back, or a camera cut would hide it
        # for the rest of the clip
        assert policy.accept_position(1, MAX_CONSECUTIVE_JUMP_REJECTS + 1, 900.0, 900.0)

    def test_a_track_without_confidence_values_is_still_drawn_in_full(self):
        # synthetic data has no confidence, so gating on it would hide everything
        import numpy as np

        from analysis.annotation import _draw_policy, _drawable_points
        from analysis.models import RIGHT_KNEE

        pose, video = self._pose(jitter=0.0), self._video()
        pose.visibility = np.zeros_like(pose.visibility)
        policy = _draw_policy(pose, video, "left", "side")
        assert not policy.trust_visibility
        points, _ = _drawable_points(pose, 0, video, policy)
        assert RIGHT_KNEE in points
