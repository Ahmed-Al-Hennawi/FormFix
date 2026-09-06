"""
End-to-end shoulder-press pipeline on synthetic recordings. Only MediaPipe's
output is faked; everything after it runs for real, including the
annotated-video render to an actual MP4.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from analysis import pose_detector
from analysis.models import (
    ROM_LIMITED_BOTTOM,
    ROM_LIMITED_TOP,
    AnalysisFailure,
    CameraOrientation,
    FailureCode,
    Reliability,
    RuleStatus,
)
from exercises.press.analyser import analyse_shoulder_press
from tests.synthetic_press import PressSpec, make_pose_data, write_plain_video


def run(tmp_path: Path, spec: PressSpec, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pose = make_pose_data(spec)
    video = tmp_path / "clip.mp4"
    write_plain_video(video, pose.frame_count)
    with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
        return analyse_shoulder_press(video, output_dir=tmp_path, **kwargs)


def statuses(result) -> dict[str, RuleStatus]:
    return {rule.rule_id: rule.status for rule in result.rule_results}


class TestCleanRecording:
    def test_a_correct_set_is_analysed_and_passes(self, tmp_path):
        result = run(tmp_path, PressSpec())
        assert result.success
        assert result.exercise_id == "press"
        assert result.summary.complete_reps == 3
        assert result.summary.partial_movements == 0
        assert statuses(result) == {
            "press_symmetry": RuleStatus.PASS,
            "press_alignment": RuleStatus.PASS,
            "press_rom": RuleStatus.PASS,
        }
        assert result.summary.score == 100

    def test_a_square_front_view_is_recognised_as_frontal(self, tmp_path):
        # If this regresses, every front-on clip gets quietly downgraded to
        # "diagonal" reliability and all the left/right comparisons with it.
        result = run(tmp_path, PressSpec())
        assert result.validation.orientation is CameraOrientation.FRONTAL
        for rule in result.rule_results:
            assert rule.reliability is Reliability.HIGH

    def test_the_annotated_video_is_produced_and_readable(self, tmp_path):
        from analysis.video_processor import probe_video

        result = run(tmp_path, PressSpec())
        assert result.annotated_video_path is not None
        rendered = probe_video(result.annotated_video_path)
        assert rendered.readable
        assert rendered.frame_count > 0

    def test_the_positives_claim_only_what_was_measured(self, tmp_path):
        result = run(tmp_path, PressSpec())
        text = " ".join(result.summary.positives).lower()
        assert "perfect" not in text
        assert "symmetrical" in text


class TestDetectsTheIntendedFault:
    def test_a_trailing_arm_is_detected_as_asymmetry(self, tmp_path):
        result = run(tmp_path, PressSpec(right_lag=0.28))
        symmetry = next(r for r in result.rule_results if r.rule_id == "press_symmetry")
        assert symmetry.status is RuleStatus.FAIL
        assert "left arm stayed higher" in symmetry.explanation

    def test_a_shorter_range_on_one_arm_is_detected_as_asymmetry(self, tmp_path):
        result = run(tmp_path, PressSpec(right_top_elbow=130.0))
        symmetry = next(r for r in result.rule_results if r.rule_id == "press_symmetry")
        assert symmetry.status is RuleStatus.FAIL
        assert any("range of motion" in o.evidence["signals"] for o in symmetry.per_rep)

    def test_a_drifting_wrist_is_detected_as_misalignment(self, tmp_path):
        result = run(tmp_path, PressSpec(left_drift=0.6))
        alignment = next(r for r in result.rule_results if r.rule_id == "press_alignment")
        assert alignment.status is RuleStatus.FAIL
        assert alignment.per_rep[0].evidence["worst_side"] == "left"

    def test_a_press_stopping_short_overhead_is_reported_at_the_top(self, tmp_path):
        result = run(tmp_path, PressSpec(top_elbow=138.0))
        rom = next(r for r in result.rule_results if r.rule_id == "press_rom")
        assert rom.status is RuleStatus.FAIL
        assert {o.evidence["rom_status"] for o in rom.per_rep} == {ROM_LIMITED_TOP}
        assert "stopped before reaching" in rom.explanation

    def test_dumbbells_not_returning_to_the_shoulders_are_reported_at_the_bottom(self, tmp_path):
        result = run(tmp_path, PressSpec(bottom_elbow=125.0))
        rom = next(r for r in result.rule_results if r.rule_id == "press_rom")
        assert rom.status is RuleStatus.FAIL
        assert {o.evidence["rom_status"] for o in rom.per_rep} == {ROM_LIMITED_BOTTOM}

    def test_a_finding_can_be_traced_to_its_evidence(self, tmp_path):
        result = run(tmp_path, PressSpec(right_lag=0.28))
        symmetry = next(r for r in result.rule_results if r.rule_id == "press_symmetry")
        for outcome in symmetry.per_rep:
            if outcome.status in (RuleStatus.WARNING, RuleStatus.FAIL):
                assert outcome.evidence["signals"]
                assert outcome.evidence_frame >= 0
                assert outcome.phase_frames > 0
                assert outcome.violating_frames > 0


class TestFalsePositiveProtection:
    def test_landmark_noise_does_not_manufacture_a_fault(self, tmp_path):
        result = run(tmp_path, PressSpec(noise=0.004))
        assert set(statuses(result).values()) == {RuleStatus.PASS}

    def test_a_very_small_left_right_difference_is_not_flagged(self, tmp_path):
        result = run(tmp_path, PressSpec(right_lag=0.04))
        assert statuses(result)["press_symmetry"] is RuleStatus.PASS

    def test_a_brief_landmark_drop_does_not_manufacture_a_fault(self, tmp_path):
        pose = make_pose_data(PressSpec())
        pose.xy[50:54, 16] = float("nan")  # right wrist gone for four frames
        pose.valid[50:54, 16] = False
        tmp_path.mkdir(parents=True, exist_ok=True)
        video = tmp_path / "clip.mp4"
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_shoulder_press(video, output_dir=tmp_path)
        assert result.summary.complete_reps == 3
        assert statuses(result)["press_symmetry"] is RuleStatus.PASS

    def test_a_frame_rate_change_does_not_change_the_repetition_count(self, tmp_path):
        slow = run(tmp_path / "a", PressSpec(rep_seconds=2.4))
        fast = run(tmp_path / "b", PressSpec(rep_seconds=3.4))
        assert slow.summary.complete_reps == fast.summary.complete_reps == 3

    def test_only_the_generated_fault_is_reported(self, tmp_path):
        result = run(tmp_path, PressSpec(left_drift=0.6))
        assert statuses(result)["press_rom"] is RuleStatus.PASS


class TestGracefulDegradation:
    def test_a_side_on_recording_reports_range_and_names_what_it_cannot_check(self, tmp_path):
        result = run(tmp_path, PressSpec(view_compression=0.10))
        assert result.validation.orientation is CameraOrientation.SIDE
        assert statuses(result)["press_rom"] is not RuleStatus.NOT_EVALUABLE
        assert statuses(result)["press_symmetry"] is RuleStatus.NOT_EVALUABLE
        assert statuses(result)["press_alignment"] is RuleStatus.NOT_EVALUABLE
        titles = {item.title for item in result.summary.not_assessed}
        assert titles == {"Arm symmetry", "Elbow and wrist alignment"}

    def test_a_side_on_recording_is_not_scored_as_a_failure(self, tmp_path):
        result = run(tmp_path, PressSpec(view_compression=0.10))
        assert result.summary.score == 100

    def test_one_hidden_arm_costs_only_the_check_that_needs_both(self, tmp_path):
        # Only symmetry needs both arms. Alignment judges each side on its own
        # and range of motion is a single-arm joint angle.
        pose = make_pose_data(PressSpec())
        pose.visibility[:, [14, 16]] = 0.02  # right elbow and wrist
        tmp_path.mkdir(parents=True, exist_ok=True)
        video = tmp_path / "clip.mp4"
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_shoulder_press(video, output_dir=tmp_path)
        assert statuses(result)["press_symmetry"] is RuleStatus.NOT_EVALUABLE
        assert statuses(result)["press_rom"] is not RuleStatus.NOT_EVALUABLE
        assert statuses(result)["press_alignment"] is not RuleStatus.NOT_EVALUABLE
        alignment = next(r for r in result.rule_results if r.rule_id == "press_alignment")
        assert alignment.per_rep[0].evidence["worst_side"] == "left"


class TestFailureStates:
    def test_no_person_detected_is_explained_not_crashed(self, tmp_path):
        pose = make_pose_data(PressSpec())
        pose.pose_found[:] = False
        pose.valid[:] = False
        tmp_path.mkdir(parents=True, exist_ok=True)
        video = tmp_path / "clip.mp4"
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            with pytest.raises(AnalysisFailure) as caught:
                analyse_shoulder_press(video, output_dir=tmp_path)
        assert caught.value.code is FailureCode.NO_POSE

    def test_too_little_movement_is_reported_as_no_repetition(self, tmp_path):
        with pytest.raises(AnalysisFailure) as caught:
            run(tmp_path, PressSpec(top_elbow=95.0))
        assert caught.value.code is FailureCode.NO_COMPLETE_REPETITION
        assert "shoulder height" in " ".join(caught.value.suggestions)

    def test_the_recording_advice_names_the_front_on_view(self, tmp_path):
        with pytest.raises(AnalysisFailure) as caught:
            run(tmp_path, PressSpec(top_elbow=95.0))
        assert any("front" in tip for tip in caught.value.suggestions)


class TestExplainableOutput:
    def test_every_rule_carries_its_traceability_metadata(self, tmp_path):
        result = run(tmp_path, PressSpec())
        for rule in result.rule_results:
            assert rule.metric
            assert rule.phase
            assert rule.supported_views
            assert rule.feedback_key
            assert rule.evidence.get("threshold_source")

    def test_per_repetition_verdicts_name_the_affected_repetitions(self, tmp_path):
        result = run(tmp_path, PressSpec(right_lag=0.28))
        assert len(result.summary.rep_summaries) == 3
        assert all(rep.issues for rep in result.summary.rep_summaries)

    def test_the_overview_states_how_often_each_check_was_acceptable(self, tmp_path):
        result = run(tmp_path, PressSpec(right_lag=0.28))
        joined = " | ".join(result.summary.overview)
        assert "Arm symmetry:" in joined
        assert "Elbow and wrist alignment:" in joined
        assert "Range of motion:" in joined

    def test_the_findings_carry_measured_evidence_lines(self, tmp_path):
        from exercises.press.feedback import build_findings

        result = run(tmp_path, PressSpec(right_lag=0.28))
        findings = build_findings(result.rule_results)
        assert findings
        assert findings[0].evidence_lines
        assert "elbow difference" in findings[0].evidence_lines[0]

    def test_the_debug_block_records_the_thresholds_actually_used(self, tmp_path):
        result = run(tmp_path, PressSpec())
        assert result.debug["resting_flexion_reference_deg"] > 40
        assert result.debug["rep_detection_thresholds"]["extreme_level"] > 0
        assert len(result.debug["rep_boundaries"]) == 3
        assert result.debug["rule_persistence"]["press_symmetry"]


class TestEvaluationExport:
    def test_the_export_bundle_is_written_and_machine_readable(self, tmp_path):
        import json

        result = run(tmp_path, PressSpec(right_lag=0.28), export_root=tmp_path)
        run_dir = Path(result.debug["export_dir"])
        assert (run_dir / "frame_metrics.csv").is_file()
        assert (run_dir / "reps.csv").is_file()
        data = json.loads((run_dir / "summary.json").read_text())
        assert data["exercise"] == "Shoulder Press"
        detected = {rule["rule_id"]: rule["status"] for rule in data["rule_results"]}
        assert detected["press_symmetry"] == "fail"
