"""
What the UI calls to get a video analysed, and what it gets back.

    save_upload()            -> a real file on disk
    ANALYSERS[exercise_id]() -> pose detection, validation, measurement,
                                rep detection, rules, feedback, video
    process_uploaded_video() -> AnalysisResult for the results view

The UI only calls analyse() / process_uploaded_video() and shows an
AnalysisResult, so the pages don't need to know which exercise it is.
demo_analysis() is the fallback if the CV libraries aren't installed, and it is
labelled as a sample.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

Status = Literal["good", "attention", "successful"]

# gentle wording on purpose - this is for people still learning the lift
Severity = Literal["minor", "moderate", "important"]

# score band, sets the wording and the colour
Band = Literal["good", "moderate", "poor"]

SUPPORTED_VIDEO_TYPES: tuple[str, ...] = ("mp4", "mov", "avi")

# score thresholds used for every band, colour and verdict in the UI
BAND_THRESHOLDS: tuple[tuple[int, Band, str], ...] = (
    (80, "good", "Good Form"),
    (60, "moderate", "Needs Improvement"),
    (0, "poor", "Poor Form"),
)

SEVERITY_LABELS: dict[str, str] = {
    "minor": "Minor",
    "moderate": "Moderate",
    "important": "Important",
}

# how many corrections are shown first, the rest go further down
MAX_VISIBLE_CORRECTIONS = 3
MAX_VISIBLE_POSITIVES = 4


def band_for(score: float) -> Band:
    for threshold, band, _ in BAND_THRESHOLDS:
        if score >= threshold:
            return band
    return "poor"


def verdict_for(score: float) -> str:
    """Plain-English verdict for the score."""
    for threshold, _, verdict in BAND_THRESHOLDS:
        if score >= threshold:
            return verdict
    return "Poor Form"


@dataclass
class RuleResult:
    """One row of the analysis report."""

    label: str
    # 0-100, share of reps that passed (sets the meter width)
    score: float
    status: Status
    badge: str
    detail: str = ""
    rule_id: str = ""
    # "High" / "Medium" / "Low" / "Cannot assess"
    reliability: str = ""


@dataclass
class RepFeedback:
    """One rep's verdict, ready to show."""

    number: int
    status: str
    headline: str
    passed: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    not_assessed: list[str] = field(default_factory=list)
    reliability: str = ""
    # timestamp of the rep's bottom position, e.g. "00:04.2"
    timestamp: str = ""


@dataclass
class NotAssessed:
    """A skipped check and the reason."""

    title: str
    reason: str


@dataclass
class Improvement:
    """One detected issue: the three plain lines shown on the page, plus the detail
    for the "How did FormFix detect this?" panel."""

    title: str
    issue: str
    correction: str
    severity: Severity = "moderate"
    rule_id: str = ""
    # evidence lines ("Rep 2 - minimum knee angle 116 deg - 00:04.2")
    evidence: list[str] = field(default_factory=list)

    # --- what a beginner reads first ---
    simple_issue: str = ""
    why: str = ""
    simple_fix: str = ""

    # --- how FormFix detected it ---
    measures: str = ""
    measured_from: str = ""
    metric: str = ""
    phase: str = ""
    flagged_reps: int = 0
    evaluable_reps: int = 0
    reliability: str = ""
    settings: dict = field(default_factory=dict)

    @property
    def severity_label(self) -> str:
        return SEVERITY_LABELS.get(self.severity, "Moderate")

    @property
    def headline_issue(self) -> str:
        """The plain sentence if there is one, otherwise the rule wording."""
        return self.simple_issue or self.issue

    @property
    def headline_fix(self) -> str:
        """The one-line instruction, falling back to the rule's correction."""
        return self.simple_fix or self.correction


@dataclass
class AnalysisResult:
    """Everything the results section needs."""

    exercise_id: str
    exercise_name: str
    filename: str
    duration: str
    reps: int
    rows: list[RuleResult] = field(default_factory=list)
    feedback: str = ""
    # overall score, 0-100 (formula in exercises/common/feedback.py)
    score: int = 0
    positives: list[str] = field(default_factory=list)
    improvements: list[Improvement] = field(default_factory=list)
    is_demo: bool = False
    annotated_video: Path | None = None
    landmarks: Any = None

    # --- real-analysis fields ---
    # False when the recording couldn't be analysed; the error fields say why
    success: bool = True
    error_title: str = ""
    error_message: str = ""
    error_suggestions: list[str] = field(default_factory=list)
    analysis_side: str = ""
    warnings: list[str] = field(default_factory=list)
    recording_quality: str = "good"
    # estimated camera position, and how side-on it looked (0-1)
    camera_orientation: str = ""
    side_view_confidence: float = 0.0
    # rule ids the recording couldn't support, shown as "not assessed"
    limited_metrics: list[str] = field(default_factory=list)
    # False when no check had reliable evidence, so no score is claimed
    score_available: bool = True
    score_formula: str = ""
    partial_movements: int = 0
    debug: dict = field(default_factory=dict)
    rep_results: list[RepFeedback] = field(default_factory=list)
    # one line per evaluated check, e.g. "Depth: 3 of 4 reps acceptable"
    overview: list[str] = field(default_factory=list)
    # checks that weren't judged, kept separate so they don't look like failures
    not_assessed: list[NotAssessed] = field(default_factory=list)
    reliability: str = ""
    headline: str = ""

    @property
    def band(self) -> Band:
        return band_for(self.score)

    @property
    def verdict(self) -> str:
        return verdict_for(self.score)

    @property
    def key_improvements(self) -> list[Improvement]:
        """Top three corrections (the list is already sorted by priority)."""
        return self.improvements[:MAX_VISIBLE_CORRECTIONS]

    @property
    def further_improvements(self) -> list[Improvement]:
        """The rest, only shown in the details panel."""
        return self.improvements[MAX_VISIBLE_CORRECTIONS:]

    @property
    def key_positives(self) -> list[str]:
        """What went well, trimmed to fit."""
        return self.positives[:MAX_VISIBLE_POSITIVES]


# --- Upload handling ---


def save_upload(uploaded_file) -> Path | None:
    """Save the upload to a temp file, since cv2.VideoCapture needs a real file."""
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


# --- Progress reporting ---


@dataclass(frozen=True)
class Stage:
    """One step of the progress bar."""

    key: str
    label: str


# stages the user sees - the keys match the pipeline's progress callback so the
# bar follows the real work
ANALYSIS_STAGES: tuple[Stage, ...] = (
    Stage("prepare", "Preparing video"),
    Stage("validate", "Checking recording"),
    Stage("landmarks", "Detecting body landmarks"),
    Stage("measure", "Measuring movement"),
    Stage("reps", "Detecting repetitions"),
    Stage("rules", "Assessing technique"),
    Stage("feedback", "Building feedback"),
    Stage("render", "Rendering analysed video"),
)

STAGE_INDEX: dict[str, int] = {stage.key: i for i, stage in enumerate(ANALYSIS_STAGES)}

# callback(stage_key, fraction_0_to_1, message)
ProgressCallback = Callable[[str, float, str], None]


# --- The real pipeline ---


def process_uploaded_video(
    uploaded_file,
    exercise_id: str = "squat",
    progress: ProgressCallback | None = None,
    export_root: Path | None = None,
) -> AnalysisResult | None:
    """Entry point for the upload page. Returns None if there is no analyser or the
    CV libraries are missing, and the UI then shows the sample instead."""
    try:
        from analysis.models import AnalysisFailure
        from exercises import ANALYSERS
    except ImportError as exc:  # CV dependencies not installed
        logger.warning("Computer-vision backend unavailable: %s", exc)
        return None

    analyser = ANALYSERS.get(exercise_id)
    if analyser is None:
        return None  # -> demo fallback, labelled as sample output

    video_path = save_upload(uploaded_file)
    if video_path is None:
        return None

    filename = getattr(uploaded_file, "name", video_path.name)
    try:
        result = analyser(video_path, progress=progress, export_root=export_root)
        return _to_ui_result(result, exercise_id, filename)
    except AnalysisFailure as failure:
        logger.info("Analysis stopped: %s - %s", failure.code.value, failure.message)
        return _failure_result(failure, exercise_id, filename)
    finally:
        # the upload isn't needed any more, the annotated clip is saved separately
        video_path.unlink(missing_ok=True)


def _to_ui_result(result, exercise_id: str, filename: str) -> AnalysisResult:
    """Convert an ExerciseAnalysisResult into the UI's AnalysisResult."""
    from analysis.models import RuleStatus
    from exercises import FEEDBACK
    from exercises.common.feedback import format_timestamp
    from exercises.squat import feedback as squat_feedback  # default wording

    feedback_module = FEEDBACK.get(exercise_id, squat_feedback)

    rows = [
        RuleResult(
            label="Pose detection",
            score=round(100 * float(result.validation.metrics.get("pose_frame_ratio", 0.0))),
            status="successful",
            badge="Successful",
            rule_id="pose_detection",
        )
    ]
    for rule in result.rule_results:
        evaluable = [o for o in rule.per_rep if o.status is not RuleStatus.NOT_EVALUABLE]
        credits = sum(
            1.0 if o.status is RuleStatus.PASS else 0.5 if o.status is RuleStatus.WARNING else 0.0
            for o in evaluable
        )
        if not evaluable:
            score, status, badge = 0.0, "attention", "Not assessed"
            detail = rule.limitation or "landmarks unreliable"
        else:
            score = round(100 * credits / len(evaluable))
            passed = sum(1 for o in evaluable if o.status is RuleStatus.PASS)
            detail = f"{passed} of {len(evaluable)} reps" if len(evaluable) > 1 else ""
            if rule.status is RuleStatus.PASS:
                status, badge = "good", "Good"
            else:
                status, badge = "attention", "Needs Attention"
        rows.append(
            RuleResult(
                label=rule.title,
                score=float(score),
                status=status,
                badge=badge,
                detail=detail,
                rule_id=rule.rule_id,
                reliability=rule.reliability.label,
            )
        )

    findings = feedback_module.build_findings(result.rule_results)
    improvements = [
        Improvement(
            title=f.title,
            issue=f.issue,
            correction=f.correction,
            severity=f.severity if f.severity in SEVERITY_LABELS else "moderate",
            rule_id=f.rule_id,
            evidence=f.evidence_lines,
            simple_issue=f.simple_issue,
            why=f.why,
            simple_fix=f.simple_fix,
            measures=f.measures,
            measured_from=f.measured_from,
            metric=f.metric,
            phase=f.phase,
            flagged_reps=f.flagged_reps,
            evaluable_reps=f.evaluable_reps,
            reliability=f.reliability.label,
            settings=f.settings,
        )
        for f in findings
    ]

    feedback_text = (
        improvements[0].headline_issue
        if improvements
        else "No technique issues were detected against the configured checks in this set."
    )

    ui = AnalysisResult(
        exercise_id=result.exercise_id or exercise_id,
        exercise_name=result.exercise,
        filename=filename,
        duration=_format_duration(result.video.duration),
        reps=result.summary.complete_reps,
        rows=rows,
        feedback=feedback_text,
        score=result.summary.score,
        positives=list(result.summary.positives),
        improvements=improvements,
        is_demo=False,
        annotated_video=result.annotated_video_path,
        success=True,
        analysis_side=result.analysis_side,
        warnings=list(result.validation.warnings),
        recording_quality=result.validation.quality.value,
        camera_orientation=result.validation.orientation.value,
        side_view_confidence=float(result.validation.side_view_confidence),
        limited_metrics=list(result.validation.limited_metrics),
        score_available=any(
            outcome.status is not RuleStatus.NOT_EVALUABLE
            for rule in result.rule_results
            for outcome in rule.per_rep
        ),
        score_formula=result.summary.score_formula,
        partial_movements=result.summary.partial_movements,
        debug=result.debug,
        rep_results=[
            RepFeedback(
                number=rep.rep_number,
                status=rep.status.value,
                headline=rep.headline,
                passed=list(rep.passed),
                issues=list(rep.issues),
                not_assessed=list(rep.not_assessed),
                reliability=rep.reliability.label,
                timestamp=format_timestamp(rep.bottom_time),
            )
            for rep in result.summary.rep_summaries
        ],
        overview=list(result.summary.overview),
        not_assessed=[
            NotAssessed(title=item.title, reason=item.reason) for item in result.summary.not_assessed
        ],
        reliability=result.summary.reliability.label,
    )
    ui.headline = summary_sentence(ui)
    return ui


def summary_sentence(result: AnalysisResult) -> str:
    """
    The sentence under the verdict, built from the score band, rep count and top
    findings, so it only says what this run actually found.
    """
    name = (result.exercise_name or "movement").lower()
    reps = result.reps
    plural = "" if reps == 1 else "s"

    if not result.score_available:
        return (
            f"FormFix tracked your {name}, but this recording could not support any of its "
            "checks, so nothing was scored."
        )
    titles = [item.title.lower() for item in result.key_improvements[:2]]
    if not titles:
        return (
            f"Your {name} looked good across {reps} rep{plural} - none of the checks FormFix "
            "measured found anything to fix."
        )
    opener = {
        "good": f"Your {name} is mostly solid",
        "moderate": f"Your {name} is close",
        "poor": f"Your {name} needs some work",
    }[result.band]
    if len(titles) == 1:
        return f"{opener}, but {titles[0]} is worth working on."
    return f"{opener} - the two things to work on are {titles[0]} and {titles[1]}."


def _failure_result(failure, exercise_id: str, filename: str) -> AnalysisResult:
    """Turn an AnalysisFailure into something the results page can show."""
    from exercises import DISPLAY_NAMES

    return AnalysisResult(
        exercise_id=exercise_id,
        exercise_name=DISPLAY_NAMES.get(exercise_id, exercise_id.title()),
        filename=filename,
        duration="",
        reps=0,
        success=False,
        error_title="We couldn't analyse this recording",
        error_message=failure.message,
        error_suggestions=list(failure.suggestions),
        recording_quality=failure.validation.quality.value if failure.validation else "unusable",
        camera_orientation=failure.validation.orientation.value if failure.validation else "",
        debug={"failure_code": failure.code.value}
        | ({"validation": failure.validation.diagnostics()} if failure.validation else {}),
    )


def unexpected_failure(filename: str, exercise_id: str) -> AnalysisResult:
    """Result for an unexpected crash. Without this the page gets stuck on
    "analysing" and repeats the crash on every rerun."""
    from exercises import DISPLAY_NAMES

    return AnalysisResult(
        exercise_id=exercise_id,
        exercise_name=DISPLAY_NAMES.get(exercise_id, exercise_id.title()),
        filename=filename or "your-video.mp4",
        duration="",
        reps=0,
        success=False,
        error_title="Something went wrong while analysing this video",
        error_message=(
            "FormFix stopped before it could finish reading this recording, so nothing " "was scored."
        ),
        error_suggestions=[
            "Press Analyse Form again - a single interrupted run usually succeeds on the "
            "second attempt.",
            "If it happens again, re-export the clip as a standard MP4 (H.264) and upload "
            "that instead.",
        ],
        recording_quality="unusable",
        debug={"failure_code": "unexpected_error"},
    )


def _format_duration(seconds: float) -> str:
    if not seconds or seconds != seconds:
        return "00:00"
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{int(round(seconds - minutes * 60)):02d}"


def analyse(
    uploaded_file,
    exercise_id: str = "squat",
    progress: ProgressCallback | None = None,
    export_root: Path | None = None,
) -> AnalysisResult:
    """Runs process_uploaded_video, or demo_analysis if that returns None."""
    result = process_uploaded_video(
        uploaded_file, exercise_id=exercise_id, progress=progress, export_root=export_root
    )
    if result is not None:
        return result

    filename = getattr(uploaded_file, "name", "your-video.mp4")
    return demo_analysis(exercise_id, filename=filename)


# --- Sample data, for when the CV backend can't be imported ---


_DEMO_ANALYSES: dict[str, AnalysisResult] = {
    "squat": AnalysisResult(
        exercise_id="squat",
        exercise_name="Squat",
        filename="your-video.mp4",
        duration="00:12",
        reps=6,
        score=82,
        rows=[
            RuleResult("Pose detection", 100, "successful", "Successful", rule_id="pose_detection"),
            RuleResult(
                "Squat depth",
                67,
                "attention",
                "Needs Attention",
                detail="4 of 6 reps",
                rule_id="squat_depth",
            ),
            RuleResult(
                "Forward torso lean", 92, "good", "Good", detail="6 of 6 reps", rule_id="torso_lean"
            ),
            RuleResult(
                "Heel stability", 100, "good", "Good", detail="6 of 6 reps", rule_id="heel_lift"
            ),
            RuleResult(
                "Return to standing",
                100,
                "good",
                "Good",
                detail="6 of 6 reps",
                rule_id="return_to_standing",
            ),
            RuleResult(
                "Descent control",
                83,
                "good",
                "Good",
                detail="5 of 6 reps",
                rule_id="descent_control",
            ),
        ],
        positives=[
            "6 full reps completed",
            "Chest stayed upright",
            "Heels stayed planted",
        ],
        improvements=[
            Improvement(
                title="Squat depth",
                issue="2 of your 6 repetitions did not reach the configured target depth.",
                correction="Lower yourself under control until your hips reach the target "
                "depth, keeping your whole foot planted.",
                severity="moderate",
                rule_id="squat_depth",
                evidence=["Rep 5 - minimum knee angle 118 deg - 00:09.2"],
                simple_issue="You stop a little high on 2 of your 6 reps.",
                why="Squatting lower works your legs through their full range.",
                simple_fix="Lower until your hips reach about knee height, keeping your "
                "whole foot planted.",
                measures="how far your knees bend at the lowest point of each rep",
                measured_from="your hip, knee and ankle landmarks",
                metric="squat_depth",
                phase="bottom",
                flagged_reps=2,
                evaluable_reps=6,
                reliability="High",
            ),
        ],
        headline="Your squat is mostly solid, but squat depth is worth working on.",
        feedback=(
            "Two of your repetitions stopped above the configured depth target. "
            "Lowering a little further under control will even the set out."
        ),
    ),
    "press": AnalysisResult(
        exercise_id="press",
        exercise_name="Shoulder Press",
        filename="your-video.mp4",
        duration="00:15",
        reps=5,
        score=73,
        rows=[
            RuleResult("Pose detection", 100, "successful", "Successful", rule_id="pose_detection"),
            RuleResult(
                "Arm symmetry",
                60,
                "attention",
                "Needs Attention",
                detail="3 of 5 reps",
                rule_id="press_symmetry",
            ),
            RuleResult(
                "Elbow and wrist alignment",
                90,
                "good",
                "Good",
                detail="5 of 5 reps",
                rule_id="press_alignment",
            ),
            RuleResult(
                "Range of motion",
                80,
                "good",
                "Good",
                detail="4 of 5 reps",
                rule_id="press_rom",
            ),
        ],
        positives=[
            "5 full press reps completed",
            "Wrists stayed stacked over your elbows",
        ],
        improvements=[
            Improvement(
                title="Uneven arms",
                issue="Your right arm reached the top earlier and stayed higher than your left "
                "arm during 3 of your 5 repetitions.",
                correction="Press both dumbbells at the same controlled pace and aim for both "
                "arms to travel together.",
                severity="moderate",
                rule_id="press_symmetry",
                evidence=["Rep 2 - left/right elbow difference 19 deg - 00:05.1"],
                simple_issue="One arm leads the other on 3 of your 5 reps.",
                why="Pressing evenly shares the work between both shoulders.",
                simple_fix="Press both weights at the same speed and aim to finish together.",
                measures="the difference between your left and right elbow angle and wrist height",
                measured_from="both shoulder, elbow and wrist chains",
                metric="press_symmetry",
                phase="press",
                flagged_reps=3,
                evaluable_reps=5,
                reliability="High",
            ),
        ],
        headline="Your shoulder press is close, but uneven arms is worth working on.",
        feedback=(
            "Your right arm leads the press. Aiming for both arms to travel together will "
            "share the work more evenly."
        ),
    ),
    "pulldown": AnalysisResult(
        exercise_id="pulldown",
        exercise_name="Lat Pulldown",
        filename="your-video.mp4",
        duration="00:14",
        reps=6,
        score=75,
        rows=[
            RuleResult("Pose detection", 100, "successful", "Successful", rule_id="pose_detection"),
            RuleResult(
                "Range of motion",
                67,
                "attention",
                "Needs Attention",
                detail="4 of 6 reps",
                rule_id="pulldown_rom",
            ),
            RuleResult(
                "Torso movement",
                83,
                "good",
                "Good",
                detail="5 of 6 reps",
                rule_id="pulldown_torso",
            ),
        ],
        positives=[
            "6 full reps completed",
            "Steady torso through the pull",
        ],
        improvements=[
            Improvement(
                title="Movement range",
                issue="Your arms did not return to a sufficiently extended position in 2 of "
                "your 6 repetitions.",
                correction="Let your arms extend more fully between repetitions and control "
                "the bar on the way back up.",
                severity="moderate",
                rule_id="pulldown_rom",
                evidence=["Rep 4 - elbow extended to 141 deg - 00:08.4"],
                simple_issue="Your arms do not travel through the full range on 2 of your 6 reps.",
                why="A fuller range works your back through more of the movement.",
                simple_fix="Pull until your elbows are down by your sides, then let your arms "
                "straighten again before the next rep.",
                measures="how far your elbow opens and closes between the start and the pull",
                measured_from="your shoulder, elbow and wrist landmarks",
                metric="pulldown_rom",
                phase="return",
                flagged_reps=2,
                evaluable_reps=6,
                reliability="High",
            ),
        ],
        headline="Your lat pulldown is close, but movement range is worth working on.",
        feedback=(
            "Your arms stop short of full extension between repetitions. Letting them "
            "straighten gives the movement its full range."
        ),
    ),
}


def demo_analysis(exercise_id: str = "squat", filename: str = "your-video.mp4") -> AnalysisResult:
    """
    Sample analysis, only used when OpenCV or MediaPipe is missing (labelled
    "Sample output"). tests/test_exercise_registry.py checks the rule ids still
    match the real ones.
    """
    from dataclasses import replace

    template = _DEMO_ANALYSES.get(exercise_id, _DEMO_ANALYSES["squat"])
    return replace(
        template,
        filename=filename,
        is_demo=True,
        rows=list(template.rows),
        positives=list(template.positives),
        improvements=list(template.improvements),
    )


def demo_result() -> AnalysisResult:
    """The example on the homepage - made up, but with the real squat rule ids."""
    return AnalysisResult(
        exercise_id="squat",
        exercise_name="Squat",
        filename="squat_session_004.mp4",
        duration="00:12",
        reps=6,
        rows=[
            RuleResult("Pose detection", 100, "successful", "Successful", rule_id="pose_detection"),
            RuleResult(
                "Squat depth",
                67,
                "attention",
                "Needs Attention",
                detail="4 of 6 reps",
                rule_id="squat_depth",
            ),
            RuleResult(
                "Forward torso lean", 92, "good", "Good", detail="6 of 6 reps", rule_id="torso_lean"
            ),
            RuleResult(
                "Heel stability", 100, "good", "Good", detail="6 of 6 reps", rule_id="heel_lift"
            ),
            RuleResult(
                "Return to standing",
                100,
                "good",
                "Good",
                detail="6 of 6 reps",
                rule_id="return_to_standing",
            ),
            RuleResult(
                "Descent control",
                88,
                "good",
                "Good",
                detail="6 of 6 reps",
                rule_id="descent_control",
            ),
        ],
        feedback=(
            "Two of your repetitions stopped above the configured depth target. "
            "Lowering a little further under control will even the set out."
        ),
    )
