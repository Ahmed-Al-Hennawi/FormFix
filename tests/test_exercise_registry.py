"""
Checks the wiring between the exercises and the UI. The wrong analyser, the
wrong reference clip or a check that doesn't exist would all look normal on
screen, so the registries and metadata are checked against each other here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.models import (
    ExerciseAnalysisResult,
    Reliability,
    RepRuleOutcome,
    RepSummary,
    RuleResult,
    RuleStatus,
    SessionSummary,
    ValidationResult,
    VideoMetadata,
)
from exercises import ANALYSERS, DISPLAY_NAMES, FEEDBACK, has_analyser
from exercises.press.config import DEFAULT_CONFIG as PRESS_CONFIG
from exercises.press.config import rule_specs as press_rule_specs
from exercises.pulldown.config import DEFAULT_CONFIG as PULLDOWN_CONFIG
from exercises.pulldown.config import rule_specs as pulldown_rule_specs
from exercises.squat.config import DEFAULT_CONFIG as SQUAT_CONFIG
from exercises.squat.config import rule_specs as squat_rule_specs
from utils import assets
from utils.analysis import _to_ui_result
from utils.exercise_data import EXERCISES
from utils.exercise_data import get as get_exercise

EXERCISE_IDS = ("squat", "pulldown", "press")

RULE_SPECS = {
    "squat": squat_rule_specs(SQUAT_CONFIG),
    "pulldown": pulldown_rule_specs(PULLDOWN_CONFIG),
    "press": press_rule_specs(PRESS_CONFIG),
}


class TestRegistries:
    def test_every_exercise_has_an_analyser(self):
        assert set(ANALYSERS) == set(EXERCISE_IDS)
        assert all(has_analyser(eid) for eid in EXERCISE_IDS)

    def test_every_exercise_has_feedback_wording(self):
        assert set(FEEDBACK) == set(EXERCISE_IDS)

    def test_the_three_analysers_are_distinct(self):
        # catches one exercise running another's analyser
        assert len({id(fn) for fn in ANALYSERS.values()}) == 3

    def test_every_exercise_has_a_display_name(self):
        assert set(DISPLAY_NAMES) == set(EXERCISE_IDS)

    def test_the_explorer_lists_exactly_the_analysed_exercises(self):
        assert {exercise.id for exercise in EXERCISES} == set(EXERCISE_IDS)


class TestAdvertisedChecksMatchImplementedRules:
    @pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
    def test_the_explorer_advertises_only_rules_that_exist(self, exercise_id):
        implemented = {spec.rule_id for spec in RULE_SPECS[exercise_id]}
        advertised = set(get_exercise(exercise_id).rules)
        assert advertised <= implemented, f"{exercise_id} advertises checks it does not perform"

    @pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
    def test_every_rule_has_feedback_wording(self, exercise_id):
        templates = FEEDBACK[exercise_id].FEEDBACK_TEMPLATES
        for spec in RULE_SPECS[exercise_id]:
            assert spec.feedback_key in templates, f"{spec.rule_id} has no wording"

    @pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
    def test_every_rule_declares_its_provenance_and_scope(self, exercise_id):
        for spec in RULE_SPECS[exercise_id]:
            assert spec.metric
            assert spec.phase
            assert spec.supported_views
            assert spec.threshold_source


class TestSampleFallback:
    """The sample output shown when the CV libraries are missing."""

    @pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
    def test_the_sample_uses_only_real_rule_ids(self, exercise_id):
        from utils.analysis import demo_analysis

        implemented = {spec.rule_id for spec in RULE_SPECS[exercise_id]} | {"pose_detection"}
        sample = demo_analysis(exercise_id)
        assert {row.rule_id for row in sample.rows} <= implemented
        assert {item.rule_id for item in sample.improvements} <= implemented

    @pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
    def test_the_sample_is_flagged_as_a_sample(self, exercise_id):
        from utils.analysis import demo_analysis

        assert demo_analysis(exercise_id).is_demo

    @pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
    def test_the_sample_never_claims_perfection(self, exercise_id):
        from utils.analysis import demo_analysis

        sample = demo_analysis(exercise_id)
        text = " ".join([*sample.positives, sample.feedback]).lower()
        for word in ("perfect", "flawless", "injury", "diagnos"):
            assert word not in text


class TestRecordingQualityAdvice:
    """
    The recording note has to be per exercise - it used to be one sentence about
    a side view, which was wrong for two of the three.
    """

    @pytest.mark.parametrize(
        ("exercise_id", "expected"),
        [("squat", "side-on"), ("pulldown", "side or three-quarter"), ("press", "front-on")],
    )
    def test_the_note_names_the_view_this_exercise_needs(self, exercise_id, expected):
        from components.analysis_results import recording_quality_block
        from utils.analysis import AnalysisResult

        result = AnalysisResult(
            exercise_id=exercise_id,
            exercise_name=DISPLAY_NAMES[exercise_id],
            filename="clip.mp4",
            duration="00:12",
            reps=3,
            recording_quality="limited",
            camera_orientation="diagonal_side",
            warnings=["The camera is not fully square."],
        )
        markup = recording_quality_block(result, get_exercise(exercise_id))
        assert expected in markup
        assert "roughly level with your hips" not in markup

    def test_no_note_is_shown_for_a_good_recording(self):
        from components.analysis_results import recording_quality_block
        from utils.analysis import AnalysisResult

        result = AnalysisResult(
            exercise_id="press",
            exercise_name="Shoulder Press",
            filename="clip.mp4",
            duration="00:12",
            reps=3,
            recording_quality="good",
        )
        assert recording_quality_block(result, get_exercise("press")) == ""

    def test_a_failure_panel_falls_back_to_the_exercises_own_tips(self):
        from components.analysis_results import failure_markup
        from utils.analysis import AnalysisResult

        result = AnalysisResult(
            exercise_id="press",
            exercise_name="Shoulder Press",
            filename="clip.mp4",
            duration="",
            reps=0,
            success=False,
            error_title="We couldn't analyse this recording",
            error_message="No person found.",
            error_suggestions=[],
        )
        markup = failure_markup(result, get_exercise("press"))
        assert "Film from the front" in markup


class TestRecordingGuidance:
    @pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
    def test_each_exercise_states_its_camera_view_and_tips(self, exercise_id):
        exercise = get_exercise(exercise_id)
        assert exercise.camera_view
        assert len(exercise.recording_tips) >= 3

    def test_the_two_new_exercises_ask_for_different_camera_views(self):
        # get this wrong and checks disappear: a front-on pulldown loses its torso
        # check, a side-on press loses two of three
        assert "Side" in get_exercise("pulldown").camera_view
        assert "Front" in get_exercise("press").camera_view

    def test_the_guidance_matches_the_exercise_config(self):
        from exercises.press.config import RECORDING_TIPS as PRESS_TIPS
        from exercises.pulldown.config import RECORDING_TIPS as PULLDOWN_TIPS

        assert get_exercise("press").recording_tips == PRESS_TIPS
        assert get_exercise("pulldown").recording_tips == PULLDOWN_TIPS


class TestReferenceVideos:
    @pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
    def test_each_exercise_maps_to_its_own_reference_clip(self, exercise_id):
        assert exercise_id in assets.REFERENCE_VIDEOS

    def test_no_two_exercises_share_a_reference_clip(self):
        paths = [assets.REFERENCE_VIDEOS[eid] for eid in EXERCISE_IDS]
        assert len(set(paths)) == len(paths)

    def test_the_reference_filenames_name_their_exercise(self):
        assert "squat" in assets.REFERENCE_VIDEOS["squat"]
        assert "lat-pulldown" in assets.REFERENCE_VIDEOS["pulldown"]
        assert "shoulder-press" in assets.REFERENCE_VIDEOS["press"]

    @pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
    def test_the_reference_clip_is_present_on_disk(self, exercise_id):
        assert assets.asset_path(assets.REFERENCE_VIDEOS[exercise_id]) is not None


# --- The interface mapping ---


def fake_result(exercise_id: str, exercise_name: str, rule_id: str, feedback_key: str):
    """Smallest result that maps, so no video is needed."""
    outcome = RepRuleOutcome(rep_number=1, status=RuleStatus.WARNING, evidence={})
    rule = RuleResult(
        rule_id=rule_id,
        title="A check",
        status=RuleStatus.WARNING,
        explanation="Something was observed.",
        correction="Try this instead.",
        per_rep=[outcome],
        metric=rule_id,
        phase="movement",
        supported_views=("side",),
        feedback_key=feedback_key,
        reliability=Reliability.MEDIUM,
    )
    summary = SessionSummary(
        complete_reps=1,
        partial_movements=0,
        positives=["Did a thing"],
        score=50,
        score_formula="formula",
        rep_summaries=[
            RepSummary(
                rep_number=1,
                status=RuleStatus.WARNING,
                headline="Needs work",
                bottom_time=1.25,
            )
        ],
        overview=["A check: 0 of 1 repetitions acceptable"],
        reliability=Reliability.MEDIUM,
    )
    return ExerciseAnalysisResult(
        success=True,
        exercise=exercise_name,
        analysis_side="left",
        video=VideoMetadata(Path("c.mp4"), 720, 1280, 30.0, 90, 3.0),
        validation=ValidationResult(),
        reps=[],
        rule_results=[rule],
        summary=summary,
        annotated_video_path=None,
        exercise_id=exercise_id,
    )


class TestResultsMapping:
    @pytest.mark.parametrize(
        ("exercise_id", "name", "rule_id", "feedback_key"),
        [
            ("pulldown", "Lat Pulldown", "pulldown_torso", "pulldown_torso"),
            ("press", "Shoulder Press", "press_symmetry", "press_symmetry"),
            ("squat", "Squat", "squat_depth", "depth"),
        ],
    )
    def test_each_exercise_maps_onto_the_shared_results_model(
        self, exercise_id, name, rule_id, feedback_key
    ):
        ui = _to_ui_result(fake_result(exercise_id, name, rule_id, feedback_key), exercise_id, "c.mp4")
        assert ui.exercise_id == exercise_id
        assert ui.exercise_name == name
        assert ui.reps == 1
        assert ui.score == 50
        assert [row.rule_id for row in ui.rows] == ["pose_detection", rule_id]
        assert ui.rep_results[0].timestamp == "00:01.2"

    def test_the_finding_uses_the_exercises_own_wording(self):
        ui = _to_ui_result(
            fake_result("press", "Shoulder Press", "press_symmetry", "press_symmetry"),
            "press",
            "c.mp4",
        )
        assert ui.improvements[0].title == "Uneven arms"

    def test_the_exercise_id_comes_from_the_result_not_the_caller(self):
        # otherwise the results page could show another exercise's reference clip
        ui = _to_ui_result(
            fake_result("pulldown", "Lat Pulldown", "pulldown_rom", "pulldown_rom"),
            "squat",
            "c.mp4",
        )
        assert ui.exercise_id == "pulldown"

    def test_a_failure_names_the_exercise_the_user_selected(self):
        from analysis.models import AnalysisFailure, FailureCode
        from utils.analysis import _failure_result

        failure = AnalysisFailure(FailureCode.NO_POSE, "No person found.", ["Try again"])
        assert _failure_result(failure, "press", "c.mp4").exercise_name == "Shoulder Press"
        assert _failure_result(failure, "pulldown", "c.mp4").exercise_name == "Lat Pulldown"
