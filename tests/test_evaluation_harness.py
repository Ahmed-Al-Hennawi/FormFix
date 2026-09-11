"""
Tests for the evaluation harness that makes the tables for the write-up. A bug
here would corrupt the evaluation, so it's run end to end on synthetic clips
with known faults.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest import mock

import pytest

from analysis import pose_detector
from tests.synthetic_press import PressSpec
from tests.synthetic_press import make_pose_data as make_press_pose
from tests.synthetic_press import write_plain_video as write_press_video
from tests.synthetic_pulldown import PulldownSpec
from tests.synthetic_pulldown import make_pose_data as make_pulldown_pose
from tests.synthetic_pulldown import write_plain_video as write_pulldown_video

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate_videos  # noqa: E402


@pytest.fixture
def labelled_set(tmp_path: Path):
    """Three synthetic recordings with known, deliberate faults."""
    clips = {
        "LP01_correct.mp4": ("pulldown", "correct", PulldownSpec()),
        "LP02_lean.mp4": ("pulldown", "pulldown_torso", PulldownSpec(torso_gain_deg=28.0)),
        "SP02_asym.mp4": ("press", "press_symmetry", PressSpec(right_lag=0.28)),
    }
    poses = {}
    for name, (exercise, _, spec) in clips.items():
        if exercise == "pulldown":
            pose = make_pulldown_pose(spec)
            write_pulldown_video(tmp_path / name, pose.frame_count)
        else:
            pose = make_press_pose(spec)
            write_press_video(tmp_path / name, pose.frame_count)
        poses[name] = pose

    manifest = tmp_path / "labels.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "exercise", "expected"])
        for name, (exercise, expected, _) in clips.items():
            writer.writerow([name, exercise, expected])

    def fake_detect(video, *args, **kwargs):
        return poses[Path(video.path).name]

    return manifest, poses, fake_detect


def run_harness(manifest: Path, fake_detect) -> int:
    with mock.patch.object(pose_detector, "detect_poses", side_effect=fake_detect):
        with mock.patch.object(sys, "argv", ["evaluate_videos.py", str(manifest)]):
            return evaluate_videos.main()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestManifest:
    def test_an_unknown_exercise_is_rejected_loudly(self, tmp_path):
        manifest = tmp_path / "labels.csv"
        manifest.write_text("video,exercise,expected\na.mp4,deadlift,correct\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            evaluate_videos.load_manifest(manifest)

    def test_an_unknown_rule_id_is_rejected_loudly(self, tmp_path):
        # a typo in a label would make every detection a false positive, so it
        # has to fail before anything runs
        manifest = tmp_path / "labels.csv"
        manifest.write_text("video,exercise,expected\na.mp4,press,press_symetry\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            evaluate_videos.load_manifest(manifest)

    def test_correct_and_none_both_mean_no_expected_fault(self, tmp_path):
        manifest = tmp_path / "labels.csv"
        manifest.write_text(
            "video,exercise,expected\na.mp4,press,correct\nb.mp4,press,none\nc.mp4,press,\n",
            encoding="utf-8",
        )
        entries = evaluate_videos.load_manifest(manifest)
        assert [expected for _, _, expected in entries] == [[], [], []]

    def test_several_expected_faults_are_read(self, tmp_path):
        manifest = tmp_path / "labels.csv"
        manifest.write_text(
            "video,exercise,expected\na.mp4,press,press_symmetry;press_rom\n", encoding="utf-8"
        )
        assert evaluate_videos.load_manifest(manifest)[0][2] == ["press_symmetry", "press_rom"]


class TestOutcomeClassification:
    def _row(self, statuses: dict[str, str], expected: list[str], analysed: bool = True):
        return evaluate_videos.Row(
            video="c.mp4",
            exercise="press",
            expected=expected,
            analysed=analysed,
            statuses=statuses,
        )

    def test_a_detected_expected_fault_is_a_true_positive(self):
        row = self._row({"press_symmetry": "fail"}, ["press_symmetry"])
        assert evaluate_videos.outcome_for(row, "press_symmetry") == "TP"

    def test_a_fault_reported_on_a_correct_recording_is_a_false_positive(self):
        row = self._row({"press_symmetry": "warning"}, [])
        assert evaluate_videos.outcome_for(row, "press_symmetry") == "FP"

    def test_a_missed_fault_is_a_false_negative(self):
        row = self._row({"press_symmetry": "pass"}, ["press_symmetry"])
        assert evaluate_videos.outcome_for(row, "press_symmetry") == "FN"

    def test_correct_silence_is_a_true_negative(self):
        row = self._row({"press_symmetry": "pass"}, [])
        assert evaluate_videos.outcome_for(row, "press_symmetry") == "TN"

    def test_a_declined_measurement_is_neither_a_detection_nor_a_miss(self):
        row = self._row({"press_symmetry": "not_evaluable"}, ["press_symmetry"])
        assert evaluate_videos.outcome_for(row, "press_symmetry") == "NOT_ASSESSED"

    def test_an_unanalysable_recording_is_recorded_as_such(self):
        row = self._row({}, ["press_symmetry"], analysed=False)
        assert evaluate_videos.outcome_for(row, "press_symmetry") == "NOT_ANALYSED"


class TestEndToEnd:
    def test_the_harness_writes_all_three_reports(self, labelled_set):
        manifest, _, fake_detect = labelled_set
        assert run_harness(manifest, fake_detect) == 0
        out = manifest.parent
        assert (out / "results.csv").is_file()
        assert (out / "error_matrix.csv").is_file()
        assert (out / "summary.md").is_file()

    def test_the_results_match_the_labels(self, labelled_set):
        manifest, _, fake_detect = labelled_set
        run_harness(manifest, fake_detect)
        rows = {row["video"]: row for row in read_csv(manifest.parent / "results.csv")}

        assert rows["LP01_correct.mp4"]["detected"] == "correct"
        assert rows["LP02_lean.mp4"]["detected"] == "pulldown_torso"
        assert rows["SP02_asym.mp4"]["expected"] == "press_symmetry"
        assert "press_symmetry" in rows["SP02_asym.mp4"]["detected"]

    def test_the_error_matrix_labels_every_rule_of_every_recording(self, labelled_set):
        manifest, _, fake_detect = labelled_set
        run_harness(manifest, fake_detect)
        rows = read_csv(manifest.parent / "error_matrix.csv")
        # 2 pulldown rules x 2 clips + 3 press rules x 1 clip
        assert len(rows) == 7
        matrix = {(row["video"], row["rule"]): row["result"] for row in rows}
        assert matrix[("LP01_correct.mp4", "pulldown_torso")] == "TN"
        assert matrix[("LP02_lean.mp4", "pulldown_torso")] == "TP"
        assert matrix[("SP02_asym.mp4", "press_symmetry")] == "TP"

    def test_the_summary_reports_counts_rather_than_an_invented_accuracy(self, labelled_set):
        manifest, _, fake_detect = labelled_set
        run_harness(manifest, fake_detect)
        summary = (manifest.parent / "summary.md").read_text()
        assert "| Rule | TP | FP | FN | TN | Not assessed |" in summary
        assert "No accuracy percentage is computed here" in summary
        assert "%" not in summary.split("## Per rule")[1]

    def test_the_results_carry_the_measurements_behind_each_verdict(self, labelled_set):
        import json

        manifest, _, fake_detect = labelled_set
        run_harness(manifest, fake_detect)
        rows = {row["video"]: row for row in read_csv(manifest.parent / "results.csv")}
        measurements = json.loads(rows["LP02_lean.mp4"]["measurements"])
        assert measurements["rep_boundaries"]
        assert measurements["rep_boundaries"][0]["max_torso_excursion"] > 20
        assert measurements["thresholds"]

    def test_a_missing_file_is_reported_rather_than_crashing_the_run(self, tmp_path):
        manifest = tmp_path / "labels.csv"
        manifest.write_text("video,exercise,expected\nmissing.mp4,press,correct\n", encoding="utf-8")
        with mock.patch.object(sys, "argv", ["evaluate_videos.py", str(manifest)]):
            assert evaluate_videos.main() == 0
        rows = read_csv(tmp_path / "results.csv")
        assert rows[0]["analysed"] == "False"
        assert rows[0]["failure"] == "file not found"


def test_the_example_manifest_ships_and_parses():
    """The example template has to load with the same loader."""
    example = Path(__file__).resolve().parent.parent / "evaluation" / "labels.example.csv"
    assert example.is_file()
    entries = evaluate_videos.load_manifest(example)
    assert len(entries) == 9
    assert {exercise for _, exercise, _ in entries} == {"squat", "pulldown", "press"}
