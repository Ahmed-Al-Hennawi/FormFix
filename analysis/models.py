"""
Data classes passed between the pipeline stages. I used these instead of loose
dicts so each stage's inputs and outputs are clear and testable on their own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# MediaPipe landmark indices (only the ones I use)
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

# what a side-view squat needs visible, per side
SIDE_LANDMARKS: dict[str, tuple[int, ...]] = {
    "left": (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX),
    "right": (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX),
}

# same for the upper body: what has to be visible for this side to count
UPPER_BODY_SIDE_LANDMARKS: dict[str, tuple[int, ...]] = {
    "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP),
    "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP),
}

# simplified skeleton for the annotated video - no face, no fingers
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
    The whole video's pose track as (frames x landmarks) arrays. Raw and cleaned
    coordinates are both kept so the debug view shows what each stage changed.

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
        """One landmark in pixels. MediaPipe normalises x and y separately, so angles
        must not be taken from the normalised values."""
        x, y = self.xy[frame, landmark]
        return float(x) * width, float(y) * height


# --- Validation ---


class RejectionCode(str, Enum):
    """Why a recording can't be analysed (a bad camera angle, not a shallow squat).
    Codes so tests and exports match on the cause, not the text."""

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
    # reps found, but they don't look like the selected exercise
    EXERCISE_MISMATCH = "exercise_mismatch"
    # NO_SQUAT_MOVEMENT for the pulldown and press
    NO_COMPLETE_REPETITION = "no_complete_repetition"


class RecordingQuality(str, Enum):
    """
    GOOD      analyse normally
    LIMITED   analyse, and say which measurements are affected
    UNUSABLE  don't analyse, there isn't enough reliable information
    """

    GOOD = "good"
    LIMITED = "limited"
    UNUSABLE = "unusable"


class CameraOrientation(str, Enum):
    """Rough camera position. It's a heuristic, so it always comes with a confidence."""

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
    How much evidence is behind one measurement. Four bands, not a probability
    (nothing is learned). See exercises/common/confidence.py.

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
        """Higher means better evidence."""
        return {"cannot_assess": 0, "low": 1, "medium": 2, "high": 3}[self.value]


@dataclass
class ValidationResult:
    """Whether the recording is usable, and why."""

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
        """Good, limited or unusable - the three states the UI shows."""
        if not self.valid:
            return RecordingQuality.UNUSABLE
        if self.warnings or self.limited_metrics:
            return RecordingQuality.LIMITED
        return RecordingQuality.GOOD

    def diagnostics(self) -> dict[str, Any]:
        """Why this recording passed or failed, flattened for the dev UI."""
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
    """One frame's measurements. Anything that couldn't be measured is NaN, never 0."""

    frame_index: int
    timestamp: float
    valid: bool
    # HIP-KNEE-ANKLE, degrees, ~180 = straight leg
    knee_angle: float = float("nan")
    # SHOULDER-HIP-KNEE, degrees, smaller = more hip flexion
    hip_angle: float = float("nan")
    # hip->shoulder off vertical, degrees, 0 = upright
    torso_lean: float = float("nan")
    # ankle->knee off vertical, degrees
    shin_inclination: float = float("nan")
    # (knee_y - hip_y) / lower-leg length, y growing down: positive = hip above
    # knee, ~0 = parallel, negative = below
    hip_above_knee: float = float("nan")
    # heel height against the toe of the same foot, over lower-leg length
    heel_toe_offset: float = float("nan")
    # same in pixels, per foot, NaN when that foot is not tracked well enough. The
    # heel rule divides these by the standing lower-leg length rather than the
    # frame's own, which shortens as the shin tilts
    left_heel_rise_px: float = float("nan")
    right_heel_rise_px: float = float("nan")
    # heel rise above the standing baseline, over the standing lower leg
    heel_lift: float = float("nan")
    landmark_confidence: float = 0.0

    # --- both sides, for symmetry and the debug export ---
    left_knee_angle: float = float("nan")
    right_knee_angle: float = float("nan")
    left_hip_angle: float = float("nan")
    right_hip_angle: float = float("nan")
    left_shin_inclination: float = float("nan")
    right_shin_inclination: float = float("nan")
    # |left - right| knee flexion, degrees
    knee_asymmetry: float = float("nan")
    hip_asymmetry: float = float("nan")
    both_sides_valid: bool = False

    # --- normalised quantities ---
    # body scale in pixels: torso, else hip width, else lower leg
    body_scale: float = float("nan")
    # ankle-to-ankle gap / body scale, front view only
    stance_width: float = float("nan")
    # mean |knee_x - ankle_x| / body scale, front view, descriptive only
    knee_over_ankle_offset: float = float("nan")


class Phase(str, Enum):
    """Squat movement phase."""

    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"
    UNKNOWN = "unknown"


@dataclass
class SquatRep:
    """One completed rep: its phase boundaries and measurements. Boundaries are
    stored as frame index and duration so nothing downstream assumes a frame rate."""

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
    # frame of the peak reliable heel lift, -1 if unreliable
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
    # largest |left - right| knee flexion that persisted, degrees
    max_knee_asymmetry: float = float("nan")
    max_knee_asymmetry_frame: int = -1
    symmetry_reliable: bool = False
    # median stance width over the rep, front view
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
    # frames that broke the rule, so one noisy frame can't trigger it
    violating_frames: int = 0
    phase_frames: int = 0
    violation_ratio: float = 0.0


@dataclass
class RuleResult:
    """One rule over the whole set. The traceability fields let a finding be traced
    back from the sentence to the measurement and phase it came from."""

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
    # matches a FrameMetrics/SquatRep field name
    metric: str = ""
    phase: str = ""
    # camera views this rule is valid for
    supported_views: tuple[str, ...] = ()
    # template key in feedback.py
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
    # NO_COMPLETE_SQUAT for the other exercises
    NO_COMPLETE_REPETITION = "no_complete_repetition"
    # reps completed, but not of the selected exercise
    EXERCISE_MISMATCH = "exercise_mismatch"
    ANALYSIS_ERROR = "analysis_error"
    VIDEO_RENDER_ERROR = "video_render_error"


class AnalysisFailure(Exception):
    """Raised when a stage can't continue. Carries a code and a user-facing message
    so the UI shows an explanation instead of a traceback."""

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
    """One rep's verdict, e.g. "Rep 2: insufficient depth"."""

    rep_number: int
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
    """A measurement I chose not to judge, and why."""

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
    # one line per rule, e.g. "Depth: 3 of 4 reps acceptable"
    overview: list[str] = field(default_factory=list)
    not_assessed: list[NotAssessedItem] = field(default_factory=list)
    # weakest link across the checks that did give a verdict
    reliability: Reliability = Reliability.CANNOT_ASSESS


@dataclass
class ExerciseAnalysisResult:
    """Everything the UI needs to show one video. reps holds whichever rep type the
    analyser made, so one results view works for all three exercises."""

    success: bool
    exercise: str
    analysis_side: str
    video: VideoMetadata
    validation: ValidationResult
    reps: list[Any]
    rule_results: list[RuleResult]
    summary: SessionSummary
    annotated_video_path: Path | None
    # pose coverage, interpolation counts, thresholds, per-rep values
    debug: dict[str, Any] = field(default_factory=dict)
    # per-frame series for the calibration tool and the CSV export
    frame_metrics: list[Any] = field(default_factory=list)
    exercise_id: str = ""


# old name from when there was only the squat, still used in tests and scripts
SquatAnalysisResult = ExerciseAnalysisResult


# progress callback(stage_key, fraction, message). The analyser never imports
# Streamlit, the UI maps the stage keys itself.
ProgressCallback = Callable[[str, float, str], None]


# --- Upper-body phases and rep records ---
# All three exercises use the same state machine on a 1-D signal that is high
# at rest and drops into the rep:
#     squat        knee angle     (high standing, low at depth)
#     pulldown     elbow angle    (high extended, low contracted)
#     press        elbow flexion  (high at the shoulders, low at lockout)


class PulldownPhase(str, Enum):
    """Lat pulldown movement phase."""

    TOP = "top"
    PULLING = "pulling"
    # the contracted position: a short window, not a single frame
    BOTTOM = "bottom"
    RETURNING = "returning"
    UNKNOWN = "unknown"


class PressPhase(str, Enum):
    """Dumbbell shoulder press movement phase."""

    READY = "ready"
    PRESSING = "pressing"
    # overhead, a short window around peak extension
    TOP = "top"
    LOWERING = "lowering"
    UNKNOWN = "unknown"


@dataclass
class RepetitionSegmentation:
    """Frame boundaries for an upper-body rep (used by PulldownRep and PressRep):
    leave rest, travel, hold, return."""

    start_frame: int = -1
    towards_start_frame: int = -1
    # short window around the turning point
    extreme_start_frame: int = -1
    extreme_end_frame: int = -1
    return_start_frame: int = -1
    end_frame: int = -1
    towards_duration: float = float("nan")
    extreme_duration: float = float("nan")
    return_duration: float = float("nan")


@dataclass
class PulldownRep:
    """One analysed lat pulldown rep - measurements only, no verdicts."""

    number: int
    segmentation: RepetitionSegmentation
    start_time: float
    extreme_time: float
    end_time: float
    duration: float

    # --- range of motion, degrees of elbow flexion ---
    # sustained max elbow angle at the top (arms extended)
    top_elbow_angle: float = float("nan")
    # median elbow angle over the bottom window
    bottom_elbow_angle: float = float("nan")
    rom_degrees: float = float("nan")
    left_top_elbow_angle: float = float("nan")
    right_top_elbow_angle: float = float("nan")
    left_bottom_elbow_angle: float = float("nan")
    right_bottom_elbow_angle: float = float("nan")

    # --- vertical travel, normalised by body scale ---
    # wrist height above the shoulder line at the top
    wrist_rise_at_top: float = float("nan")
    wrist_rise_at_bottom: float = float("nan")
    wrist_travel: float = float("nan")
    # same for the elbows - did they actually drive down?
    elbow_travel: float = float("nan")

    # --- torso ---
    torso_at_top: float = float("nan")
    torso_at_bottom: float = float("nan")
    # largest sustained torso movement away from this rep's top posture
    max_torso_excursion: float = float("nan")
    max_torso_excursion_frame: int = -1
    # "posterior" when the facing direction was estimable, else "unsigned"
    torso_mode: str = "unsigned"
    # peak torso speed during the pull, deg/s. Exported, no rule reads it.
    peak_torso_velocity: float = float("nan")

    # --- evidence quality ---
    valid_frame_ratio: float = 0.0
    mean_landmark_visibility: float = 0.0
    # analysed arm usable often enough. Only one arm, since side-on the far arm
    # is hidden behind the near one
    arms_reliable: bool = False
    # share of the rep where both arms were usable. Export only.
    both_arms_ratio: float = 0.0
    torso_reliable: bool = False

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
    # mean of whichever arms were measurable
    top_elbow_angle: float = float("nan")
    bottom_elbow_angle: float = float("nan")
    left_rom_degrees: float = float("nan")
    right_rom_degrees: float = float("nan")
    rom_degrees: float = float("nan")

    # --- symmetry ---
    # largest sustained |left - right| elbow angle
    max_elbow_angle_difference: float = float("nan")
    max_elbow_angle_difference_frame: int = -1
    # largest sustained left/right wrist height gap / body scale (unsigned,
    # higher_side gives the direction)
    max_wrist_height_difference: float = float("nan")
    max_wrist_height_difference_frame: int = -1
    rom_difference: float = float("nan")
    # seconds between each arm reaching its highest point
    top_timing_difference: float = float("nan")
    # "left", "right" or ""
    higher_side: str = ""

    # --- frontal-plane alignment ---
    # largest sustained |wrist_x - elbow_x| / shoulder width, per side
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
