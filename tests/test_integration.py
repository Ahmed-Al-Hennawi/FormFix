"""
End-to-end squat pipeline on synthetic recordings. Only MediaPipe's output is
faked, everything else runs for real, including rendering the MP4. The
optional test at the bottom runs real detection if FORMFIX_TEST_VIDEO points
to a side-view squat clip.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from analysis import pose_detector
from analysis.models import AnalysisFailure, RuleStatus
from exercises.squat.analyser import analyse_squat
from tests.synthetic_squat import SyntheticSpec, make_pose_data, write_plain_video


def run_synthetic(tmp_path: Path, spec: SyntheticSpec, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pose = make_pose_data(spec)
    video = tmp_path / "clip.mp4"
    write_plain_video(video, pose.frame_count)
    with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
        return analyse_squat(video, output_dir=tmp_path, **kwargs)


class TestFullPipeline:
    def test_clean_squat_full_result(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002))

        assert result.success
        assert result.summary.complete_reps == 3
        assert result.summary.partial_movements == 0
        assert result.analysis_side in ("left", "right")
        assert result.summary.score == 100
        statuses = {r.rule_id: r.status for r in result.rule_results}
        assert statuses["squat_depth"] is RuleStatus.PASS
        assert statuses["torso_lean"] is RuleStatus.PASS
        assert statuses["return_to_standing"] is RuleStatus.PASS
        assert result.annotated_video_path is not None
        assert result.annotated_video_path.is_file()
        from analysis.video_processor import probe_video

        rendered = probe_video(result.annotated_video_path)
        assert rendered.readable
        assert rendered.frame_count > 0
        # check these are real measurements, not placeholders
        for rep in result.reps:
            assert 50 < rep.min_knee_angle < 100
            assert rep.duration > 1.0

    def test_shallow_squat_flagged_with_evidence(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(depth=0.75, noise=0.002))
        depth = next(r for r in result.rule_results if r.rule_id == "squat_depth")
        assert depth.status in (RuleStatus.WARNING, RuleStatus.FAIL)
        worst = max(o.evidence["min_knee_angle"] for o in depth.per_rep)
        assert f"{worst:.0f}" in depth.explanation

    def test_heel_visibility_gates_heel_rule_only(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(heel_lift=0.5, heel_visibility=0.2, noise=0.002))
        statuses = {r.rule_id: r.status for r in result.rule_results}
        assert statuses["heel_lift"] is RuleStatus.NOT_EVALUABLE
        assert result.success
        assert statuses["squat_depth"] is RuleStatus.PASS

    def test_left_and_right_facing_equivalent(self, tmp_path):
        right = run_synthetic(tmp_path / "r", SyntheticSpec(noise=0.002))
        left = run_synthetic(tmp_path / "l", SyntheticSpec(facing=-1, noise=0.002))
        assert right.summary.complete_reps == left.summary.complete_reps == 3
        r_min = right.reps[0].min_knee_angle
        l_min = left.reps[0].min_knee_angle
        assert abs(r_min - l_min) < 5.0

    def test_no_complete_squat_fails_gracefully(self, tmp_path):
        with pytest.raises(AnalysisFailure) as excinfo:
            run_synthetic(tmp_path, SyntheticSpec(depth=0.3, noise=0.002))
        failure = excinfo.value
        assert failure.code.value == "no_complete_squat"
        assert "complete squat" in failure.message
        assert failure.suggestions

    def test_export_bundle_written(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002), export_root=tmp_path)
        export_dir = Path(result.debug["export_dir"])
        assert (export_dir / "summary.json").is_file()
        assert (export_dir / "frame_metrics.csv").is_file()
        assert (export_dir / "reps.csv").is_file()


@pytest.mark.skipif(
    not os.environ.get("FORMFIX_TEST_VIDEO"),
    reason="set FORMFIX_TEST_VIDEO to a real side-view squat clip to run",
)
def test_real_video_end_to_end(tmp_path):
    result = analyse_squat(Path(os.environ["FORMFIX_TEST_VIDEO"]), output_dir=tmp_path)
    assert result.success
    assert result.summary.complete_reps >= 1
    assert result.annotated_video_path is not None


class TestRecordingQualityStates:
    """
    Recording cases from the validation design (numbers match that table).

    1    clean side view         -> analysed, "good"
    2    diagonal camera         -> analysed, "limited" plus a warning
    4    front-on camera         -> analysed, sagittal checks not assessed
    5    feet cropped            -> analysed, only the heel check limited
    7    temporary tracking loss -> analysed
    6/8  no usable landmarks     -> rejected with a typed reason
    """

    def test_case1_side_view_is_good_quality(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002))
        assert result.validation.quality.value == "good"
        assert result.validation.orientation.value == "side"
        assert result.debug["side_view_confidence"] == 1.0

    def test_case2_diagonal_camera_still_analysed(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002, side_offset=0.26))
        assert result.success
        assert result.validation.orientation.value == "diagonal_side"
        assert result.validation.quality.value == "limited"
        assert result.summary.complete_reps == 3
        statuses = {r.rule_id: r.status for r in result.rule_results}
        assert statuses["squat_depth"] is not RuleStatus.NOT_EVALUABLE
        assert any("side-on" in w for w in result.validation.warnings)

    def test_case4_front_view_analyses_reps_and_limits_angles(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002, side_offset=0.45))
        assert result.success
        assert result.validation.orientation.value == "frontal"
        assert result.summary.complete_reps == 3  # reps are still counted
        statuses = {r.rule_id: r.status for r in result.rule_results}
        for rule_id in ("squat_depth", "torso_lean", "return_to_standing"):
            assert statuses[rule_id] is RuleStatus.NOT_EVALUABLE
        limited = next(r for r in result.rule_results if r.rule_id == "squat_depth")
        assert "side view" in limited.limitation

    def test_case5_cropped_feet_limit_only_the_heel_check(self, tmp_path):
        pose = make_pose_data(SyntheticSpec(noise=0.002))
        pose.xy_raw[:, [27, 28, 29, 30, 31, 32], 1] = 0.999
        pose.xy = pose.xy_raw.copy()
        video = tmp_path / "clip.mp4"
        tmp_path.mkdir(parents=True, exist_ok=True)
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_squat(video, output_dir=tmp_path)
        assert result.success
        statuses = {r.rule_id: r.status for r in result.rule_results}
        assert statuses["heel_lift"] is RuleStatus.NOT_EVALUABLE
        assert statuses["squat_depth"] is RuleStatus.PASS

    def test_case7_temporary_tracking_loss_survives(self, tmp_path):
        pose = make_pose_data(SyntheticSpec(noise=0.002))
        pose.visibility[40:70, :] = 0.0
        pose.valid[40:70, :] = False
        pose.pose_found[40:70] = False
        video = tmp_path / "clip.mp4"
        tmp_path.mkdir(parents=True, exist_ok=True)
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_squat(video, output_dir=tmp_path)
        assert result.success
        assert result.summary.complete_reps >= 2

    def test_case6_unusable_landmarks_rejected_with_reason(self, tmp_path):
        pose = make_pose_data(SyntheticSpec(noise=0.002, visibility=0.1, heel_visibility=0.1))
        video = tmp_path / "clip.mp4"
        tmp_path.mkdir(parents=True, exist_ok=True)
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            with pytest.raises(AnalysisFailure) as excinfo:
                analyse_squat(video, output_dir=tmp_path)
        failure = excinfo.value
        assert failure.validation is not None
        assert "important_landmarks_missing" in [c.value for c in failure.validation.reason_codes]
        assert failure.validation.quality.value == "unusable"


class TestExplainableOutput:
    """Each link of landmarks -> measurement -> phase -> rule -> evidence ->
    feedback, as it reaches the UI."""

    def test_per_repetition_results_are_produced(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002))
        summaries = result.summary.rep_summaries
        assert [s.rep_number for s in summaries] == [1, 2, 3]
        assert all(s.headline for s in summaries)
        assert all(s.status is RuleStatus.PASS for s in summaries)

    def test_overall_findings_state_how_often_each_check_passed(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002))
        assert any(line.startswith("Depth: 3 of 3") for line in result.summary.overview)

    def test_unavailable_measurements_are_reported_not_guessed(self, tmp_path):
        # front-on can't support the side-view checks, so they should be in
        # not_assessed with a reason
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002, side_offset=0.45))
        assert any(item.metric == "squat_depth" for item in result.summary.not_assessed)
        assert all(item.reason for item in result.summary.not_assessed)

    def test_every_rule_carries_its_traceability_metadata(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002))
        for rule in result.rule_results:
            assert rule.metric and rule.phase and rule.supported_views
            assert rule.feedback_key
            assert rule.reliability is not None

    def test_reliability_is_reported_per_check(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002))
        levels = {rule.rule_id: rule.reliability.value for rule in result.rule_results}
        assert levels["squat_depth"] in ("high", "medium")
        assert result.summary.reliability.value in ("high", "medium")

    def test_phase_durations_are_measured_per_repetition(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002))
        for rep in result.reps:
            assert rep.descent_duration > 0
            assert rep.ascent_duration > 0
            assert rep.bottom_end_frame >= rep.bottom_start_frame

    def test_debug_records_the_persistence_evidence(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002))
        persistence = result.debug["rule_persistence"]["torso_lean"]
        assert len(persistence) == 3
        assert all("violation_ratio" in row for row in persistence)
        assert "rule_specs" in result.debug


class TestFrontViewAnalysis:
    def test_front_view_still_segments_the_repetitions(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002, side_offset=0.45))
        assert result.summary.complete_reps == 3

    def test_front_view_still_produces_usable_feedback(self, tmp_path):
        result = run_synthetic(tmp_path, SyntheticSpec(noise=0.002, side_offset=0.45))
        assert result.summary.rep_summaries
        assert result.summary.overview  # something was still measured
        assert any(item.metric == "squat_depth" for item in result.summary.not_assessed)


class TestRecordingVariation:
    """Ways phone footage varies that shouldn't change the verdict."""

    def test_resolution_does_not_change_the_measurements(self, tmp_path):
        """Same framing at half the resolution gives the same angles (aspect ratio kept)."""
        import tests.synthetic_squat as syn

        full = run_synthetic(tmp_path / "hd", SyntheticSpec(noise=0.002))
        original = (syn.WIDTH, syn.HEIGHT)
        syn.WIDTH, syn.HEIGHT = original[0] // 2, original[1] // 2
        try:
            half = run_synthetic(tmp_path / "sd", SyntheticSpec(noise=0.002))
        finally:
            syn.WIDTH, syn.HEIGHT = original

        assert full.summary.complete_reps == half.summary.complete_reps
        assert abs(full.reps[0].min_knee_angle - half.reps[0].min_knee_angle) < 1.0
        assert abs(full.reps[0].max_torso_lean - half.reps[0].max_torso_lean) < 1.0
        assert full.summary.score == half.summary.score

    def test_frame_rate_does_not_change_the_repetition_count(self, tmp_path):
        import tests.synthetic_squat as syn

        original = syn.FPS
        counts = {}
        try:
            for fps in (24.0, 30.0, 60.0):
                syn.FPS = fps
                result = run_synthetic(tmp_path / f"f{int(fps)}", SyntheticSpec(noise=0.002))
                counts[fps] = result.summary.complete_reps
        finally:
            syn.FPS = original
        assert set(counts.values()) == {3}, counts

    def test_temporary_occlusion_of_one_leg_is_survivable(self, tmp_path):
        pose = make_pose_data(SyntheticSpec(noise=0.002, side_offset=0.45))
        # far leg disappears for a third of the clip
        n = pose.frame_count
        pose.visibility[n // 3 : 2 * n // 3, [26, 28, 30, 32]] = 0.05
        pose.valid[n // 3 : 2 * n // 3, [26, 28, 30, 32]] = False
        video = tmp_path / "clip.mp4"
        tmp_path.mkdir(parents=True, exist_ok=True)
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_squat(video, output_dir=tmp_path)
        assert result.success
        assert result.summary.complete_reps >= 2


class TestHeelMeasurementRobustness:
    """Heel lift is measured within the foot (heel vs toe), so a drifting camera
    doesn't affect it."""

    def test_heel_lift_is_detected_in_proportion(self, tmp_path):
        planted = run_synthetic(tmp_path / "flat", SyntheticSpec(noise=0.002, heel_lift=0.0))
        lifted = run_synthetic(tmp_path / "lift", SyntheticSpec(noise=0.002, heel_lift=0.25))

        def heel(result):
            return next(r for r in result.rule_results if r.rule_id == "heel_lift")

        assert heel(planted).status is RuleStatus.PASS
        assert heel(lifted).status is RuleStatus.WARNING
        assert max(rep.max_heel_lift for rep in lifted.reps) > 0.15

    def test_a_drifting_camera_is_not_a_heel_lift(self, tmp_path):
        import numpy as np

        pose = make_pose_data(SyntheticSpec(noise=0.002))
        drift = np.linspace(0.0, 0.10, pose.frame_count)  # the body slides down the frame
        pose.xy[:, :, 1] += drift[:, None]
        pose.xy_raw[:, :, 1] += drift[:, None]
        video = tmp_path / "clip.mp4"
        tmp_path.mkdir(parents=True, exist_ok=True)
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_squat(video, output_dir=tmp_path)

        heel = next(r for r in result.rule_results if r.rule_id == "heel_lift")
        assert heel.status is RuleStatus.PASS


class TestImplausibleTrackingIsNotAFinding:
    """
    MediaPipe can be confident about limbs it's guessing, so an impossible
    measurement has to be "not measured", not a technique fault.
    """

    def test_an_impossible_knee_angle_is_discarded(self, tmp_path):
        from analysis.models import VideoMetadata
        from exercises.squat.config import DEFAULT_CONFIG
        from exercises.squat.metrics import compute_frame_metrics

        pose = make_pose_data(SyntheticSpec(noise=0.002))
        # fold the far knee onto its hip - fine geometrically, impossible for a human
        pose.xy[:, 26] = pose.xy[:, 24]
        video = VideoMetadata(
            path=Path("clip.mp4"),
            width=720,
            height=1280,
            fps=30.0,
            frame_count=pose.frame_count,
            duration=pose.frame_count / 30.0,
        )
        metrics = compute_frame_metrics(video, pose, "left", DEFAULT_CONFIG)
        import math

        assert all(math.isnan(m.right_knee_angle) for m in metrics if m.valid)
        # ...and the left/right comparison won't use it either.
        assert all(math.isnan(m.knee_asymmetry) for m in metrics)
