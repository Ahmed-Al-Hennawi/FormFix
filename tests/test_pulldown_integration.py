"""
End-to-end lat pulldown pipeline on synthetic recordings. Only MediaPipe's
output is faked, everything else runs for real, including rendering the MP4.
The main ones are TestDetectsTheIntendedFault and TestFalsePositiveProtection.
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
from exercises.pulldown.analyser import analyse_lat_pulldown
from tests.synthetic_pulldown import PulldownSpec, make_pose_data, write_plain_video


def run(tmp_path: Path, spec: PulldownSpec, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pose = make_pose_data(spec)
    video = tmp_path / "clip.mp4"
    write_plain_video(video, pose.frame_count)
    with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
        return analyse_lat_pulldown(video, output_dir=tmp_path, **kwargs)


def statuses(result) -> dict[str, RuleStatus]:
    return {rule.rule_id: rule.status for rule in result.rule_results}


class TestCleanRecording:
    def test_a_correct_set_is_analysed_and_passes(self, tmp_path):
        result = run(tmp_path, PulldownSpec())
        assert result.success
        assert result.exercise_id == "pulldown"
        assert result.summary.complete_reps == 3
        assert result.summary.partial_movements == 0
        assert statuses(result) == {
            "pulldown_rom": RuleStatus.PASS,
            "pulldown_torso": RuleStatus.PASS,
        }
        assert result.summary.score == 100

    def test_the_annotated_video_is_produced_and_readable(self, tmp_path):
        from analysis.video_processor import probe_video

        result = run(tmp_path, PulldownSpec())
        assert result.annotated_video_path is not None
        assert result.annotated_video_path.is_file()
        rendered = probe_video(result.annotated_video_path)
        assert rendered.readable
        assert rendered.frame_count > 0

    def test_the_numbers_are_real_measurements_not_placeholders(self, tmp_path):
        result = run(tmp_path, PulldownSpec(top_elbow=172.0, bottom_elbow=80.0))
        for rep in result.reps:
            assert rep.top_elbow_angle == pytest.approx(172.0, abs=4.0)
            assert rep.bottom_elbow_angle == pytest.approx(80.0, abs=8.0)
            assert rep.duration > 0.8

    def test_the_positives_claim_only_what_was_measured(self, tmp_path):
        result = run(tmp_path, PulldownSpec())
        text = " ".join(result.summary.positives).lower()
        assert "perfect" not in text
        assert any("range" in line or "torso" in line for line in result.summary.positives)


class TestDetectsTheIntendedFault:
    def test_an_excessive_backward_swing_is_detected(self, tmp_path):
        result = run(tmp_path, PulldownSpec(torso_gain_deg=28.0))
        assert statuses(result)["pulldown_torso"] is RuleStatus.FAIL
        assert statuses(result)["pulldown_rom"] is RuleStatus.PASS  # one fault, not two

    def test_a_moderate_swing_is_a_warning_rather_than_a_failure(self, tmp_path):
        result = run(tmp_path, PulldownSpec(torso_gain_deg=18.0))
        assert statuses(result)["pulldown_torso"] is RuleStatus.WARNING

    def test_arms_that_never_extend_are_reported_at_the_top(self, tmp_path):
        result = run(tmp_path, PulldownSpec(top_elbow=132.0))
        rule = next(r for r in result.rule_results if r.rule_id == "pulldown_rom")
        assert rule.status is RuleStatus.FAIL
        assert {o.evidence["rom_status"] for o in rule.per_rep} == {ROM_LIMITED_TOP}
        assert "extended position" in rule.explanation

    def test_a_pull_that_stops_early_is_reported_at_the_bottom(self, tmp_path):
        result = run(tmp_path, PulldownSpec(bottom_elbow=125.0))
        rule = next(r for r in result.rule_results if r.rule_id == "pulldown_rom")
        assert rule.status is RuleStatus.FAIL
        assert {o.evidence["rom_status"] for o in rule.per_rep} == {ROM_LIMITED_BOTTOM}
        assert "stopped early" in rule.explanation

    def test_a_finding_can_be_traced_to_its_evidence(self, tmp_path):
        # every flagged outcome needs a measurement, frame and timestamp for the UI
        result = run(tmp_path, PulldownSpec(torso_gain_deg=28.0))
        rule = next(r for r in result.rule_results if r.rule_id == "pulldown_torso")
        for outcome in rule.per_rep:
            if outcome.status in (RuleStatus.WARNING, RuleStatus.FAIL):
                assert outcome.evidence["max_torso_excursion"] is not None
                assert outcome.evidence_frame >= 0
                assert outcome.phase_frames > 0
                assert outcome.violating_frames > 0


class TestFalsePositiveProtection:
    def test_landmark_noise_does_not_manufacture_a_fault(self, tmp_path):
        result = run(tmp_path, PulldownSpec(noise=0.004))
        assert statuses(result) == {
            "pulldown_rom": RuleStatus.PASS,
            "pulldown_torso": RuleStatus.PASS,
        }

    def test_a_brief_landmark_drop_does_not_manufacture_a_fault(self, tmp_path):
        pose = make_pose_data(PulldownSpec())
        pose.xy[45:49, 13] = float("nan")  # elbow gone for four frames
        pose.valid[45:49, 13] = False
        video = tmp_path / "clip.mp4"
        tmp_path.mkdir(parents=True, exist_ok=True)
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            result = analyse_lat_pulldown(video, output_dir=tmp_path)
        assert result.summary.complete_reps == 3
        assert statuses(result)["pulldown_torso"] is RuleStatus.PASS

    def test_a_lean_just_under_the_tolerance_is_not_flagged(self, tmp_path):
        # sitting just under the threshold is how a frame-by-frame system
        # invents findings
        result = run(tmp_path, PulldownSpec(torso_gain_deg=12.0))
        assert statuses(result)["pulldown_torso"] is RuleStatus.PASS

    def test_a_mirrored_recording_gives_the_same_verdicts(self, tmp_path):
        left = run(tmp_path / "l", PulldownSpec(facing=-1, torso_gain_deg=28.0))
        right = run(tmp_path / "r", PulldownSpec(facing=1, torso_gain_deg=28.0))
        assert statuses(left) == statuses(right)

    def test_a_higher_frame_rate_does_not_change_the_repetition_count(self, tmp_path):
        # timing comes from timestamps, not frame counts
        slow = run(tmp_path / "a", PulldownSpec(rep_seconds=2.4))
        fast = run(tmp_path / "b", PulldownSpec(rep_seconds=3.2))
        assert slow.summary.complete_reps == fast.summary.complete_reps == 3


class TestGracefulDegradation:
    def test_a_front_on_recording_reports_what_it_can_and_names_what_it_cannot(self, tmp_path):
        result = run(tmp_path, PulldownSpec(side_offset=0.45))
        assert result.validation.orientation is CameraOrientation.FRONTAL
        assert statuses(result)["pulldown_rom"] is RuleStatus.PASS
        torso = next(r for r in result.rule_results if r.rule_id == "pulldown_torso")
        assert torso.status is RuleStatus.NOT_EVALUABLE
        assert torso.reliability is Reliability.CANNOT_ASSESS
        assert torso.limitation
        assert [item.title for item in result.summary.not_assessed] == ["Torso movement"]

    def test_a_diagonal_recording_is_analysed_with_reduced_reliability(self, tmp_path):
        result = run(tmp_path, PulldownSpec(side_offset=0.22))
        assert result.validation.orientation is CameraOrientation.DIAGONAL_SIDE
        torso = next(r for r in result.rule_results if r.rule_id == "pulldown_torso")
        assert torso.status is not RuleStatus.NOT_EVALUABLE
        assert torso.reliability.rank < Reliability.HIGH.rank

    def test_hidden_hips_cost_only_the_torso_check(self, tmp_path):
        # seated, the hips are often hidden - that should only cost the trunk check
        result = run(tmp_path, PulldownSpec(hip_visibility=0.15))
        assert statuses(result)["pulldown_rom"] is RuleStatus.PASS
        assert statuses(result)["pulldown_torso"] is RuleStatus.NOT_EVALUABLE


class TestFailureStates:
    def test_no_person_detected_is_explained_not_crashed(self, tmp_path):
        pose = make_pose_data(PulldownSpec())
        pose.pose_found[:] = False
        pose.valid[:] = False
        tmp_path.mkdir(parents=True, exist_ok=True)
        video = tmp_path / "clip.mp4"
        write_plain_video(video, pose.frame_count)
        with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
            with pytest.raises(AnalysisFailure) as caught:
                analyse_lat_pulldown(video, output_dir=tmp_path)
        assert caught.value.code is FailureCode.NO_POSE
        assert "couldn't detect a person" in caught.value.message

    def test_too_little_movement_is_reported_as_no_repetition(self, tmp_path):
        with pytest.raises(AnalysisFailure) as caught:
            run(tmp_path, PulldownSpec(bottom_elbow=165.0))
        assert caught.value.code is FailureCode.NO_COMPLETE_REPETITION
        assert caught.value.suggestions

    def test_a_video_that_is_too_short_is_rejected_with_advice(self, tmp_path):
        with pytest.raises(AnalysisFailure) as caught:
            run(tmp_path, PulldownSpec(reps=1, rep_seconds=1.0, top_seconds=0.3))
        assert caught.value.code in (
            FailureCode.INVALID_VIDEO,
            FailureCode.NO_COMPLETE_REPETITION,
        )
        assert caught.value.suggestions

    def test_a_failure_never_produces_technique_feedback(self, tmp_path):
        with pytest.raises(AnalysisFailure) as caught:
            run(tmp_path, PulldownSpec(bottom_elbow=165.0))
        assert not hasattr(caught.value, "summary")


class TestExplainableOutput:
    def test_every_rule_carries_its_traceability_metadata(self, tmp_path):
        result = run(tmp_path, PulldownSpec())
        for rule in result.rule_results:
            assert rule.metric
            assert rule.phase
            assert rule.supported_views
            assert rule.feedback_key
            assert rule.evidence.get("threshold_source")

    def test_per_repetition_verdicts_are_produced(self, tmp_path):
        result = run(tmp_path, PulldownSpec(torso_gain_deg=28.0))
        assert len(result.summary.rep_summaries) == 3
        for rep in result.summary.rep_summaries:
            assert rep.headline
            assert rep.issues or rep.passed

    def test_the_overview_states_how_often_each_check_was_acceptable(self, tmp_path):
        result = run(tmp_path, PulldownSpec(torso_gain_deg=28.0))
        joined = " | ".join(result.summary.overview)
        assert "Range of motion:" in joined
        assert "Torso movement:" in joined
        assert "of 3 repetitions acceptable" in joined

    def test_the_score_formula_travels_with_the_result(self, tmp_path):
        result = run(tmp_path, PulldownSpec())
        assert "evaluable checks" in result.summary.score_formula

    def test_the_debug_block_records_the_thresholds_actually_used(self, tmp_path):
        result = run(tmp_path, PulldownSpec())
        assert result.debug["rep_detection_thresholds"]["rest_level"] > 0
        assert result.debug["resting_extension_reference_deg"] > 100
        assert result.debug["torso_measurement_mode"] in ("posterior", "unsigned")
        assert len(result.debug["rep_boundaries"]) == 3
        assert result.debug["rule_persistence"]["pulldown_torso"]


class TestEvaluationExport:
    def test_the_export_bundle_is_written(self, tmp_path):
        result = run(tmp_path, PulldownSpec(), export_root=tmp_path)
        run_dir = Path(result.debug["export_dir"])
        assert (run_dir / "summary.json").is_file()
        assert (run_dir / "frame_metrics.csv").is_file()
        assert (run_dir / "reps.csv").is_file()

    def test_the_summary_json_is_machine_readable(self, tmp_path):
        import json

        result = run(tmp_path, PulldownSpec(torso_gain_deg=28.0), export_root=tmp_path)
        data = json.loads((Path(result.debug["export_dir"]) / "summary.json").read_text())
        assert data["exercise"] == "Lat Pulldown"
        detected = {rule["rule_id"]: rule["status"] for rule in data["rule_results"]}
        assert detected["pulldown_torso"] == "fail"
        assert "path" not in data["video"]  # don't store the uploader's file path
