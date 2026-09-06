"""
The typed structures the pipeline stages pass around - objects rather than
loose dictionaries, so each stage's inputs and outputs are explicit and can be
tested on their own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# MediaPipe landmark indices. Only the ones we read get a name.
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

NUM_LANDMARKS = 33

# What a side-view squat analysis depends on, per side.
SIDE_LANDMARKS: dict[str, tuple[int, ...]] = {
    "left": (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX),
    "right": (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX),
}

# same for the upper body: what has to be visible for this side to count
UPPER_BODY_SIDE_LANDMARKS: dict[str, tuple[int, ...]] = {
    "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP),
    "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP),
}

# Simplified skeleton for the annotated video - no face mesh, no fingers.
BODY_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
)


# --- Video ---


@dataclass(frozen=True)
class VideoMetadata:
    """What cv2.VideoCapture reported about the file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    fourcc: str = ""

    @property
    def readable(self) -> bool:
        return self.width > 0 and self.height > 0 and self.frame_count > 0 and self.fps > 0


# --- Pose data ---


@dataclass
class FramePoseData:
    """
    The whole video's pose track, as (frames x landmarks) arrays. Raw and cleaned
    coordinates are both kept so the debug view can see what each stage changed.

        xy_raw      (F, 33, 2) normalised coordinates, NaN when missing
        xy          (F, 33, 2) same, after interpolation and smoothing
        visibility  (F, 33)    0 where no pose was detected
        valid       (F, 33)    True where the value is trustworthy
        pose_found  (F,)       a pose was detected on this frame
        n_poses     (F,)       how many people the detector reported
        timestamps  (F,)       seconds from the start

    identity_switches counts how often the tracked person was re-seeded from
    height rather than followed.
    """

    xy_raw: np.ndarray
    xy: np.ndarray
    visibility: np.ndarray
    valid: np.ndarray
    pose_found: np.ndarray
    n_poses: np.ndarray
    timestamps: np.ndarray
    interpolated_frames: int = 0
    identity_switches: int = 0

    @property
    def frame_count(self) -> int:
        return int(self.xy_raw.shape[0])

    def pixel_xy(self, frame: int, landmark: int, width: int, height: int) -> tuple[float, float]:
        """One landmark in pixel coordinates. MediaPipe normalises x and y separately,
        so never take an angle straight off the normalised values."""
        x, y = self.xy[frame, landmark]
        return float(x) * width, float(y) * height


# --- Validation ---


class RejectionCode(str, Enum):
    """Why a recording can't be analysed. A bad camera angle belongs here, a
    shallow squat doesn't. Codes so tests and exports match on causes, not text."""

    VIDEO_READ_ERROR = "video_read_error"
    VIDEO_TOO_SHORT = "video_too_short"
    VIDEO_TOO_LONG = "video_too_long"
    LOW_RESOLUTION = "low_resolution"
    NO_POSE_DETECTED = "no_pose_detected"
    INSUFFICIENT_POSE_COVERAGE = "insufficient_pose_coverage"
    IMPORTANT_LANDMARKS_MISSING = "important_landmarks_missing"
    INSUFFICIENT_VALID_FRAMES = "insufficient_valid_frames"
    BODY_OUT_OF_FRAME = "body_out_of_frame"
    MULTIPLE_PEOPLE = "multiple_people"
    UNSUPPORTED_CAMERA_ANGLE = "unsupported_camera_angle"
    NO_SQUAT_MOVEMENT = "no_squat_movement"
    # Reps were found, but the movement contradicts the selected exercise.
    EXERCISE_MISMATCH = "exercise_mismatch"
    # Generic version of NO_SQUAT_MOVEMENT for the pulldown and press.
    NO_COMPLETE_REPETITION = "no_complete_repetition"


class RecordingQuality(str, Enum):
    """
    The three outcomes of recording validation.

    GOOD      analyse normally
    LIMITED   analyse, and say which measurements are affected
    UNUSABLE  don't analyse, there isn't enough reliable information
    """

    GOOD = "good"
    LIMITED = "limited"
    UNUSABLE = "unusable"


class CameraOrientation(str, Enum):
    """Rough camera position. A heuristic, so it always travels with a confidence."""

    SIDE = "side"
    DIAGONAL_SIDE = "diagonal_side"
    FRONTAL = "frontal"
    UNKNOWN = "unknown"


# metric names for ValidationResult.limited_metrics, matching rule ids
METRIC_DEPTH = "squat_depth"
METRIC_TORSO_LEAN = "torso_lean"
METRIC_HEEL_LIFT = "heel_lift"
METRIC_EXTENSION = "return_to_standing"
METRIC_KNEE_SYMMETRY = "knee_symmetry"
METRIC_DESCENT_CONTROL = "descent_control"

METRIC_PULLDOWN_TORSO = "pulldown_torso"
METRIC_PULLDOWN_ROM = "pulldown_rom"

METRIC_PRESS_SYMMETRY = "press_symmetry"
METRIC_PRESS_ALIGNMENT = "press_elbow_alignment"
METRIC_PRESS_ROM = "press_rom"

# range-of-motion verdicts shared by the two upper-body analysers
ROM_COMPLETE = "complete_rom"
ROM_LIMITED_TOP = "limited_top_extension"
ROM_LIMITED_BOTTOM = "limited_bottom_range"
ROM_LIMITED_OVERALL = "limited_overall_rom"
ROM_NOT_ASSESSABLE = "not_assessable"


class Reliability(str, Enum):
    """
    How much evidence sits behind one measurement. Four bands, not a probability -
    nothing here is learned. See exercises/common/confidence.py.

        HIGH           several reps, visible landmarks, suitable view
        MEDIUM         measurable, on thinner evidence
        LOW            measurable, but weak
        CANNOT_ASSESS  nothing trustworthy, so the rule reports nothing
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CANNOT_ASSESS = "cannot_assess"

    @property
    def label(self) -> str:
        return {"high": "High", "medium": "Medium", "low": "Low"}.get(self.value, "Cannot assess")

    @property
    def rank(self) -> int:
        """For sorting - higher means better evidence."""
        return {"cannot_assess": 0, "low": 1, "medium": 2, "high": 3}[self.value]


@dataclass
class ValidationResult:
    """A structured suitability verdict, not a bare bool."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    reason_codes: list[RejectionCode] = field(default_factory=list)
    # measurements this recording can't support; they become "not assessed"
    limited_metrics: list[str] = field(default_factory=list)
    # camera-angle estimate and how far to trust it (0-1)
    orientation: CameraOrientation = CameraOrientation.UNKNOWN
    side_view_confidence: float = 0.0
    selected_side: str = ""

    def add_error(self, message: str, code: RejectionCode | None = None) -> None:
        self.errors.append(message)
        if code is not None:
            self.reason_codes.append(code)
        self.valid = False

    def add_warning(self, message: str, limited_metric: str | None = None) -> None:
        self.warnings.append(message)
        if limited_metric and limited_metric not in self.limited_metrics:
            self.limited_metrics.append(limited_metric)

    @property
    def quality(self) -> RecordingQuality:
        """Good, limited or unusable - the three states the UI renders."""
        if not self.valid:
            return RecordingQuality.UNUSABLE
        if self.warnings or self.limited_metrics:
            return RecordingQuality.LIMITED
        return RecordingQuality.GOOD

    def diagnostics(self) -> dict[str, Any]:
        """Flat view of why this recording passed or failed, for the dev UI."""
        return {
            "quality": self.quality.value,
            "orientation": self.orientation.value,
            "side_view_confidence": round(float(self.side_view_confidence), 3),
            "selected_side": self.selected_side,
            "reason_codes": [code.value for code in self.reason_codes],
            "limited_metrics": list(self.limited_metrics),
            "warnings": list(self.warnings),
            **self.metrics,
        }


# --- Measurements ---


@dataclass
class FrameMetrics:
    """One frame's measurements: the analysed side's sagittal values, the left/right
    pairs behind them, and the normalised ones. Unmeasurable is NaN, never zero."""

    frame_index: int
    timestamp: float
    valid: bool
    # HIP-KNEE-ANKLE angle, degrees. ~180 is a straight leg. NaN if invalid.
    knee_angle: float = float("nan")
    # SHOULDER-HIP-KNEE angle, degrees. Smaller means more hip flexion.
    hip_angle: float = float("nan")
    # Torso (hip->shoulder) off vertical, degrees. 0 is upright.
    torso_lean: float = float("nan")
    # Shin (ankle->knee) off vertical, degrees. 0 is a vertical shin.
    shin_inclination: float = float("nan")
    # (knee_y - hip_y) / lower-leg length, y growing down: positive = hip above
    # knee, ~0 = parallel, negative = below
    hip_above_knee: float = float("nan")
    # heel height against the toe of the same foot, over lower-leg length
    heel_toe_offset: float = float("nan")
    # The same minus its standing baseline; NaN before the baseline exists.
    heel_lift: float = float("nan")
    landmark_confidence: float = 0.0

    # --- both sides, for symmetry and the debug export ---
    left_knee_angle: float = float("nan")
    right_knee_angle: float = float("nan")
    left_hip_angle: float = float("nan")
    right_hip_angle: float = float("nan")
    left_shin_inclination: float = float("nan")
    right_shin_inclination: float = float("nan")
    # |left - right| knee flexion, degrees. NaN if either side is missing.
    knee_asymmetry: float = float("nan")
    hip_asymmetry: float = float("nan")
    both_sides_valid: bool = False

    # --- normalised quantities ---
    # Body-scale reference in pixels: torso, else hip width, else lower leg.
    body_scale: float = float("nan")
    # Ankle-to-ankle horizontal separation / body scale. Frontal plane only.
    stance_width: float = float("nan")
    # Mean |knee_x - ankle_x| / body scale, frontal plane. Descriptive only.
    knee_over_ankle_offset: float = float("nan")


class Phase(str, Enum):
    """Movement phase reported by the squat state machine."""

    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"
    UNKNOWN = "unknown"


@dataclass
class SquatRep:
    """One completed rep: phase boundaries and what was measured. The boundaries
    are what make the rules phase-aware, and each is stored as a frame index and a
    duration so nothing downstream assumes a frame rate."""

    number: int
    start_frame: int
    bottom_frame: int
    end_frame: int
    start_time: float
    bottom_time: float
    end_time: float
    duration: float
    min_knee_angle: float
    hip_angle_at_bottom: float
    torso_lean_at_bottom: float
    max_torso_lean: float
    max_torso_lean_frame: int
    hip_above_knee_at_bottom: float
    max_heel_lift: float
    # Frame of the peak reliable heel-lift value, -1 when unreliable.
    max_heel_lift_frame: int
    heel_reliable: bool
    end_knee_angle: float
    end_hip_angle: float

    # --- phase segmentation ---
    # first frame of the descent, same as start_frame in practice
    descent_start_frame: int = -1
    # the bottom window - a short band around bottom_frame, not one frame
    bottom_start_frame: int = -1
    bottom_end_frame: int = -1
    ascent_start_frame: int = -1
    descent_duration: float = float("nan")
    bottom_duration: float = float("nan")
    ascent_duration: float = float("nan")

    # --- other measurements ---
    shin_inclination_at_bottom: float = float("nan")
    min_left_knee_angle: float = float("nan")
    min_right_knee_angle: float = float("nan")
    # Largest |left - right| knee flexion that persisted, degrees.
    max_knee_asymmetry: float = float("nan")
    max_knee_asymmetry_frame: int = -1
    symmetry_reliable: bool = False
    # Median normalised stance width over the rep, frontal plane.
    stance_width: float = float("nan")

    # --- evidence quality ---
    valid_frame_ratio: float = 0.0
    mean_landmark_visibility: float = 0.0


# --- Rules ---


class RuleStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NOT_EVALUABLE = "not_evaluable"


@dataclass
class RepRuleOutcome:
    """One rule evaluated on one rep."""

    rep_number: int
    status: RuleStatus
    # what the outcome rests on: angle, ratio, timestamp and so on
    evidence: dict[str, Any] = field(default_factory=dict)
    evidence_frame: int = -1
    evidence_time: float = float("nan")
    # How much of the phase violated the rule, so one noisy frame can't fire.
    violating_frames: int = 0
    phase_frames: int = 0
    violation_ratio: float = 0.0


@dataclass
class RuleResult:
    """One rule over the whole set. The traceability fields record which
    measurement it read, in which phase, and which template worded it, so a
    finding can be traced from landmark to sentence."""

    rule_id: str
    title: str
    status: RuleStatus
    explanation: str
    # how to improve, only set for warning / fail
    correction: str = ""
    per_rep: list[RepRuleOutcome] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    # why it couldn't be evaluated, when status is NOT_EVALUABLE
    limitation: str = ""

    # --- traceability ---
    # The quantity read, matching a FrameMetrics/SquatRep field name.
    metric: str = ""
    phase: str = ""
    # camera views this rule means anything under
    supported_views: tuple[str, ...] = ()
    # Feedback template key behind the wording, see feedback.py.
    feedback_key: str = ""
    reliability: Reliability = Reliability.CANNOT_ASSESS
    highlight_landmarks: tuple[int, ...] = ()

    @property
    def evaluable(self) -> bool:
        return self.status is not RuleStatus.NOT_EVALUABLE


# --- Failures ---


class FailureCode(str, Enum):
    """Why a run stopped before producing any feedback."""

    INVALID_VIDEO = "invalid_video"
    NO_POSE = "no_pose"
    INSUFFICIENT_VISIBILITY = "insufficient_visibility"
    UNSUITABLE_CAMERA_VIEW = "unsuitable_camera_view"
    BODY_OUT_OF_FRAME = "body_out_of_frame"
    MULTIPLE_PEOPLE = "multiple_people"
    NO_COMPLETE_SQUAT = "no_complete_squat"
    # Generic version of NO_COMPLETE_SQUAT.
    NO_COMPLETE_REPETITION = "no_complete_repetition"
    # The movement completed reps, but not of the exercise that was selected.
    EXERCISE_MISMATCH = "exercise_mismatch"
    ANALYSIS_ERROR = "analysis_error"
    VIDEO_RENDER_ERROR = "video_render_error"


class AnalysisFailure(Exception):
    """Raised when a stage can't continue. Carries a code and user-facing text so
    the UI shows an explanation instead of a traceback."""

    def __init__(
        self,
        code: FailureCode,
        message: str,
        suggestions: list[str] | None = None,
        validation: ValidationResult | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestions = suggestions or []
        self.validation = validation


# --- Final result ---


@dataclass
class RepSummary:
    """One rep's own verdict ("Rep 2: insufficient depth"), as data, so nothing has
    to re-derive it from the rules."""

    rep_number: int
    # pass | warning | fail | not_evaluable across this rep's checks.
    status: RuleStatus
    # one-line verdict, e.g. "Good repetition"
    headline: str
    passed: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    not_assessed: list[str] = field(default_factory=list)
    reliability: Reliability = Reliability.CANNOT_ASSESS
    start_time: float = float("nan")
    bottom_time: float = float("nan")
    end_time: float = float("nan")


@dataclass
class NotAssessedItem:
    """A measurement we chose not to judge, and why."""

    metric: str
    title: str
    reason: str


@dataclass
class SessionSummary:
    """The beginner-facing outcome of the whole set."""

    complete_reps: int
    partial_movements: int
    positives: list[str] = field(default_factory=list)
    # score plus the formula behind it, shown in the technical details
    score: int = 0
    score_formula: str = ""
    rep_summaries: list[RepSummary] = field(default_factory=list)
    # One line per evaluated rule, e.g. "Depth: 3 of 4 reps acceptable".
    overview: list[str] = field(default_factory=list)
    not_assessed: list[NotAssessedItem] = field(default_factory=list)
    # weakest link across the checks that did give a verdict
    reliability: Reliability = Reliability.CANNOT_ASSESS


@dataclass
class ExerciseAnalysisResult:
    """Everything the Streamlit layer needs to render one video. reps holds
    whichever per-rep record the analyser produced, so one results view serves all
    three exercises."""

    success: bool
    exercise: str
    analysis_side: str
    video: VideoMetadata
    validation: ValidationResult
    reps: list[Any]
    rule_results: list[RuleResult]
    summary: SessionSummary
    annotated_video_path: Path | None
    # Pose coverage, interpolation counts, thresholds, per-rep values.
    debug: dict[str, Any] = field(default_factory=dict)
    # per-frame series for the calibration tool and the CSV export
    frame_metrics: list[Any] = field(default_factory=list)
    # "squat" / "pulldown" / "press", matching the analyser registry.
    exercise_id: str = ""


# the squat came first and the export layer, tests and scripts still use
# this name
SquatAnalysisResult = ExerciseAnalysisResult


# progress callback(stage_key, fraction, message). The analyser never
# imports Streamlit - the UI maps stage keys onto its own components.
# analyser never imports Streamlit - the UI maps stage keys onto its own
ProgressCallback = Callable[[str, float, str], None]


# --- Upper-body phases and rep records ---
# Each exercise names its own phases, but they are all string enums of the
# same shape, so one renderer serves all three. They run on the same state
# machine over a 1-D signal that is high at rest and falls into the rep:
#     squat        knee angle     (high standing, low at depth)
#     pulldown     elbow angle    (high extended, low contracted)
#     press        elbow flexion  (high at the shoulders, low at lockout)


class PulldownPhase(str, Enum):
    """Lat-pulldown movement phase."""

    TOP = "top"
    PULLING = "pulling"
    # the contracted position: a short window, not a single frame
    BOTTOM = "bottom"
    RETURNING = "returning"
    UNKNOWN = "unknown"


class PressPhase(str, Enum):
    """Dumbbell shoulder-press movement phase."""

    READY = "ready"
    PRESSING = "pressing"
    # overhead, a short window around peak extension
    TOP = "top"
    LOWERING = "lowering"
    UNKNOWN = "unknown"


@dataclass
class RepetitionSegmentation:
    """Frame boundaries for an upper-body rep, shared by PulldownRep and PressRep.
    Named after the shape of the movement: leave rest, travel, hold, return."""

    start_frame: int = -1
    towards_start_frame: int = -1
    # The extreme window, a short band around the turning point.
    extreme_start_frame: int = -1
    extreme_end_frame: int = -1
    return_start_frame: int = -1
    end_frame: int = -1
    towards_duration: float = float("nan")
    extreme_duration: float = float("nan")
    return_duration: float = float("nan")


@dataclass
class PulldownRep:
    """
    One analysed lat-pulldown rep. Every field is a measurement or a frame index
    pointing at one; no verdicts. The thresholds live in
    exercises/pulldown/config.py.
    """

    number: int
    segmentation: RepetitionSegmentation
    start_time: float
    extreme_time: float
    end_time: float
    duration: float

    # --- range of motion, degrees of elbow flexion ---
    # Sustained maximum elbow angle at the extended top position.
    top_elbow_angle: float = float("nan")
    # Median elbow angle across the contracted bottom window.
    bottom_elbow_angle: float = float("nan")
    # top_elbow_angle - bottom_elbow_angle, the angular excursion.
    rom_degrees: float = float("nan")
    left_top_elbow_angle: float = float("nan")
    right_top_elbow_angle: float = float("nan")
    left_bottom_elbow_angle: float = float("nan")
    right_bottom_elbow_angle: float = float("nan")

    # --- vertical travel, normalised by body scale ---
    # Wrist height above the shoulder line at the top of the movement.
    wrist_rise_at_top: float = float("nan")
    wrist_rise_at_bottom: float = float("nan")
    wrist_travel: float = float("nan")
    # The same for the elbows: did they actually drive down?
    elbow_travel: float = float("nan")

    # --- torso ---
    torso_at_top: float = float("nan")
    torso_at_bottom: float = float("nan")
    # largest sustained torso movement away from this rep's own top posture;
    # torso_mode says in which direction
    max_torso_excursion: float = float("nan")
    max_torso_excursion_frame: int = -1
    # "posterior" when the facing direction was estimable, else "unsigned"
    torso_mode: str = "unsigned"
    # peak torso speed during the pull, deg/s. Exported, no rule reads it.
    peak_torso_velocity: float = float("nan")

    # --- evidence quality ---
    valid_frame_ratio: float = 0.0
    mean_landmark_visibility: float = 0.0
    # true when the analysed arm was usable often enough. Not both arms - ROM is
    # a single-arm angle, and side-on the far arm is behind the near one.
    arms_reliable: bool = False
    # share of the rep where both arms were usable. Export only.
    both_arms_ratio: float = 0.0
    torso_reliable: bool = False

    # --- convenience ---
    @property
    def start_frame(self) -> int:
        return self.segmentation.start_frame

    @property
    def end_frame(self) -> int:
        return self.segmentation.end_frame

    @property
    def extreme_frame(self) -> int:
        seg = self.segmentation
        return (seg.extreme_start_frame + seg.extreme_end_frame) // 2

    @property
    def bottom_time(self) -> float:
        """Alias the shared feedback layer uses for per-rep timestamps."""
        return self.extreme_time


@dataclass
class PressRep:
    """One analysed dumbbell shoulder-press rep."""

    number: int
    segmentation: RepetitionSegmentation
    start_time: float
    extreme_time: float
    end_time: float
    duration: float

    # --- range of motion, per arm (degrees) ---
    left_top_elbow_angle: float = float("nan")
    right_top_elbow_angle: float = float("nan")
    left_bottom_elbow_angle: float = float("nan")
    right_bottom_elbow_angle: float = float("nan")
    # Mean over whichever arms were measurable, for the overall verdict.
    top_elbow_angle: float = float("nan")
    bottom_elbow_angle: float = float("nan")
    left_rom_degrees: float = float("nan")
    right_rom_degrees: float = float("nan")
    rom_degrees: float = float("nan")

    # --- symmetry ---
    # Largest sustained |left - right| elbow angle during the rep.
    max_elbow_angle_difference: float = float("nan")
    max_elbow_angle_difference_frame: int = -1
    # largest sustained left/right wrist height difference over body scale
    # unsigned - higher_side carries the direction.
    max_wrist_height_difference: float = float("nan")
    max_wrist_height_difference_frame: int = -1
    rom_difference: float = float("nan")
    # Seconds between the two arms reaching their own highest position.
    top_timing_difference: float = float("nan")
    # "left", "right" or "" - which arm sat higher during the rep.
    higher_side: str = ""

    # --- frontal-plane alignment ---
    # Largest sustained |wrist_x - elbow_x| / shoulder width, per side.
    max_left_alignment_offset: float = float("nan")
    max_right_alignment_offset: float = float("nan")
    max_left_alignment_frame: int = -1
    max_right_alignment_frame: int = -1

    # --- evidence quality ---
    valid_frame_ratio: float = 0.0
    mean_landmark_visibility: float = 0.0
    both_arms_ratio: float = 0.0
    symmetry_reliable: bool = False
    alignment_reliable: bool = False

    @property
    def start_frame(self) -> int:
        return self.segmentation.start_frame

    @property
    def end_frame(self) -> int:
        return self.segmentation.end_frame

    @property
    def extreme_frame(self) -> int:
        seg = self.segmentation
        return (seg.extreme_start_frame + seg.extreme_end_frame) // 2

    @property
    def bottom_time(self) -> float:
        """Alias the shared feedback layer uses for per-rep timestamps."""
        return self.extreme_time
