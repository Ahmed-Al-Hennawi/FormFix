"""
Every tunable value for squat analysis, in one place.

The section headings mark each block as one of two kinds. ENGINEERING values
control whether the system is stable and say nothing about how a squat should
be performed. TECHNIQUE values do make claims about performance, and they are
provisional calibration values rather than biomechanical constants.

Angle conventions (see analysis/geometry.py):

    knee angle       HIP-KNEE-ANKLE, ~175-180 standing, smaller = more flexion
    hip angle        SHOULDER-HIP-KNEE, smaller = more hip flexion
    torso lean       hip->shoulder against vertical, 0 = upright
    hip_above_knee   (knee_y - hip_y) / lower-leg length, positive = hip higher
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from analysis.validation import RETRY_TIPS
from exercises.common.spec import (
    ANY_VIEW,
    FRONTAL_VIEWS,
    SAGITTAL_VIEWS,
    VIEW_DIAGONAL,
    VIEW_FRONTAL,
    VIEW_SIDE,
    VIEW_UNKNOWN,
    RuleSpec,
)

__all__ = [
    "ANY_VIEW",
    "RECOMMENDED_VIEW",
    "RECORDING_TIPS",
    "QUICK_TIPS",
    "DEFAULT_CONFIG",
    "DEFAULT_RULES",
    "FRONTAL_VIEWS",
    "SAGITTAL_VIEWS",
    "SIDE_VIEW_RULES",
    "VIEW_DIAGONAL",
    "VIEW_FRONTAL",
    "VIEW_INDEPENDENT_RULES",
    "VIEW_SIDE",
    "VIEW_UNKNOWN",
    "SquatConfig",
    "SquatRule",
    "rule_specs",
    "side_view_rules",
    "view_independent_rules",
]


@dataclass(frozen=True)
class SquatConfig:
    # --- ENGINEERING - MediaPipe detector confidences ---
    POSE_DETECTION_CONFIDENCE: float = 0.5
    POSE_PRESENCE_CONFIDENCE: float = 0.5
    TRACKING_CONFIDENCE: float = 0.5

    # --- ENGINEERING - landmark reliability and gap handling ---
    # relaxed after testing threw out gym recordings that were perfectly usable
    # per-frame visibility floor. 0.5 was too strict - MediaPipe reports 0.3-0.6
    # for a limb partly behind the body, which is what a side view looks like.
    MIN_KEY_LANDMARK_VISIBILITY: float = 0.4
    # same floor under the name the measurement code uses
    MIN_REQUIRED_LANDMARK_VISIBILITY: float = 0.4
    # Stage C - share of frames where hip, knee and ankle are all usable
    MIN_USABLE_FRAME_RATIO: float = 0.5
    # share of frames the knee-angle series must be finite on
    MIN_VALID_FRAME_RATIO: float = 0.45
    # Stage A - a person must be detected on this share of frames
    MIN_POSE_FRAME_RATIO: float = 0.5
    # gaps up to this long get interpolated, longer ones stay missing
    MAX_SHORT_GAP_FRAMES: int = 5

    # --- ENGINEERING - anatomical plausibility gates ---
    # MediaPipe returns a confident position even when it is extrapolating a limb
    # it can't see. These bands are wide - they catch tracking failures, not
    # technique - and anything outside them is NaN, never a fault.
    # a knee doesn't flex past about 25 degrees of interior angle
    PLAUSIBLE_KNEE_ANGLE_MIN: float = 25.0
    PLAUSIBLE_KNEE_ANGLE_MAX: float = 190.0
    # a heel rising more than half a lower leg is tracking failure, not a fault
    HEEL_LIFT_MAX_PLAUSIBLE: float = 0.5
    # a left/right gap this big is one leg being extrapolated behind the other
    KNEE_SYMMETRY_MAX_PLAUSIBLE: float = 45.0

    # --- ENGINEERING - smoothing ---
    # EMA weight for the landmark coordinates. Higher follows the raw signal.
    EMA_ALPHA: float = 0.45
    # EMA weight for the derived knee-angle series
    ANGLE_EMA_ALPHA: float = 0.5

    # --- ENGINEERING - phase state machine / hysteresis (degrees) ---
    # at or above this, the person counts as standing
    STANDING_KNEE_ANGLE: float = 160.0
    # descent has to pass below this before a rep attempt starts
    REP_START_KNEE_ANGLE: float = 150.0
    # and below this for a bottom candidate to exist
    BOTTOM_CANDIDATE_ANGLE: float = 130.0
    # ascent has to get back above this to complete the rep
    REP_END_KNEE_ANGLE: float = 155.0
    # frames a phase transition must hold before it commits
    PHASE_MIN_FRAMES: int = 3
    # knee-angle change (deg/s) counted as movement rather than noise
    MOVEMENT_VELOCITY_THRESHOLD: float = 15.0
    # standing baseline minus deepest angle, below which it isn't a squat attempt
    MIN_RANGE_OF_MOTION: float = 35.0
    # frames either side of the raw minimum, so the bottom is a window median
    BOTTOM_WINDOW_FRAMES: int = 3
    # how far the angle must rise above the running minimum before the bottom
    # commits, so a one-frame wobble can't end the descent early
    BOTTOM_REVERSAL_DELTA: float = 5.0
    # untracked frames tolerated mid-rep before the attempt is abandoned
    MAX_TRACKING_LOSS_FRAMES: int = 10

    # --- ENGINEERING - rep duration bounds, in seconds ---
    MIN_REP_DURATION: float = 1.0
    MAX_REP_DURATION: float = 12.0

    # --- ENGINEERING - video suitability ---
    VIDEO_MIN_DURATION: float = 3.0
    VIDEO_MAX_DURATION: float = 120.0
    # Multi-person ratios: warn above the first, reject above the second.
    MULTI_PERSON_WARN_RATIO: float = 0.10
    MULTI_PERSON_FAIL_RATIO: float = 0.50
    # frontality bands (shoulder-or-hip separation / torso length): small side-on,
    # towards 1 front or rear on. Provisional heuristics.
    #   <= GOOD          clean side view, full confidence
    #   GOOD..FRONTAL    diagonal, analysed with a warning
    #   >= FRONTAL       front-on, sagittal angles not assessed
    SIDE_VIEW_GOOD_RATIO: float = 0.45
    SIDE_VIEW_FRONTAL_RATIO: float = 1.00
    # spread above which the camera angle counts as changing mid-clip
    VIEW_STABILITY_SPREAD: float = 0.35
    # edge margin counted as clipped, then the share of frames allowed inside it
    # before we warn and before we stop
    FRAMING_MARGIN: float = 0.02
    FRAMING_WARN_TOLERANCE: float = 0.25
    FRAMING_FAIL_TOLERANCE: float = 0.70

    # --- ENGINEERING - standing baseline ---
    # stable standing frames needed for a baseline; we take the median
    BASELINE_MIN_FRAMES: int = 5

    # --- TECHNIQUE (PROVISIONAL) - squat depth ---
    # calibration values, still waiting on literature and testing
    # minimum knee angle at or below which a rep passes on depth
    DEPTH_KNEE_ANGLE_PASS: float = 100.0
    # between PASS and this it warns, above it the rep fails
    DEPTH_KNEE_ANGLE_WARN: float = 115.0
    # alternative way to pass: hip reaches knee level within this tolerance
    DEPTH_HIP_KNEE_TOLERANCE: float = 0.12

    # share of the bottom window that must break the condition before we call it
    # shallow - a second guard on top of the window median
    DEPTH_MIN_VIOLATION_RATIO: float = 0.5

    # --- TECHNIQUE (PROVISIONAL) - torso lean ---
    # peak torso lean from vertical that flags a rep. Provisional: build and
    # squat style move the right number around.
    TORSO_LEAN_WARN: float = 45.0
    TORSO_LEAN_FAIL: float = 60.0
    # also flag lean growing this far past the person's own standing baseline
    TORSO_LEAN_DELTA_FAIL: float = 55.0
    # persistence: the lean must hold this many frames and cover this share of
    # the rep, so one jittery landmark can't produce a finding
    TORSO_LEAN_MIN_FRAMES: int = 4
    TORSO_LEAN_MIN_VIOLATION_RATIO: float = 0.15

    # --- TECHNIQUE (PROVISIONAL) - heel lift ---
    # heel rise above the standing baseline, over lower-leg length. Provisional.
    HEEL_LIFT_THRESHOLD: float = 0.06
    # consecutive frames the lift must hold
    HEEL_LIFT_MIN_FRAMES: int = 3
    # below this visibility the heel rule says "cannot assess" instead of guessing
    HEEL_MIN_VISIBILITY: float = 0.5
    # Share of the rep's measurable frames the lift must cover.
    HEEL_LIFT_MIN_VIOLATION_RATIO: float = 0.10

    # --- TECHNIQUE (PROVISIONAL) - return to standing / extension ---
    # degrees of the person's own standing baseline a rep must finish within.
    # Not 180, so nobody is asked to hyperextend. Provisional.
    FULL_EXTENSION_TOLERANCE: float = 12.0

    # --- MEASUREMENT ONLY - left/right symmetry (frontal plane) ---
    # still measured and exported but no longer graded - one camera can't separate
    # a real asymmetry from the far leg's projection error
    SYMMETRY_MIN_FRAMES: int = 4
    # both legs must be usable on this share of the rep, and the far leg must
    # clear this visibility floor, before they get compared
    SYMMETRY_MIN_BOTH_SIDES_RATIO: float = 0.6
    SYMMETRY_MIN_VISIBILITY: float = 0.5

    # --- TECHNIQUE (PROVISIONAL) - movement control / tempo ---
    # a descent taking a fraction of a second is a drop, not a controlled rep.
    # Timing heuristics, and only ever a gentle warning.
    DESCENT_MIN_DURATION: float = 0.45
    # Shorter than this and we say so more firmly.
    DESCENT_FAST_DURATION: float = 0.25

    # --- ENGINEERING - reliability banding (see confidence.py) ---
    # when a verdict is reported as HIGH / MEDIUM / LOW evidence. Nothing here is
    # about technique, only about how well the recording supported it.
    RELIABILITY_HIGH_VISIBILITY: float = 0.75
    RELIABILITY_HIGH_MEASURABLE_RATIO: float = 0.85
    RELIABILITY_HIGH_VIEW_SUPPORT: float = 0.85
    RELIABILITY_HIGH_MIN_REPS: int = 2
    RELIABILITY_MEDIUM_VISIBILITY: float = 0.55
    RELIABILITY_MEDIUM_MEASURABLE_RATIO: float = 0.6
    RELIABILITY_MEDIUM_VIEW_SUPPORT: float = 0.5
    # subject size as median torso length in pixels. Someone filmed from far away
    # carries more error per landmark than visibility shows. Provisional.
    RELIABILITY_MIN_SUBJECT_PIXELS: float = 140.0
    RELIABILITY_POOR_SUBJECT_PIXELS: float = 70.0

    # --- ENGINEERING - which filter smooths the movement signal ---
    # "ema" (default), "butterworth" (Dill et al. 2024), "savgol" or
    # "moving_average". docs/filter_selection.md has the measured trade-off - no
    # filter won on every rep shape, which is why this is still a setting.
    ANGLE_FILTER: str = "ema"

    # --- ENGINEERING - measurement-uncertainty policy ---
    # False (default): a finding is annotated with the published error of the
    # quantity behind it and marked "indicative" if its margin falls inside that
    # error. True also downgrades it a step. See exercises/common/uncertainty.py.
    UNCERTAINTY_STRICT: bool = False

    # Free-form notes shown in the debug view.
    notes: dict[str, str] = field(
        default_factory=lambda: {
            "technique_thresholds": (
                "DEPTH_*, TORSO_LEAN_*, HEEL_LIFT_*, DESCENT_*, "
                "FULL_EXTENSION_TOLERANCE are provisional calibration values pending "
                "empirical/literature validation."
            ),
            "persistence": (
                "*_MIN_FRAMES and *_MIN_VIOLATION_RATIO control how long a violation "
                "must persist before it becomes a finding; they are stability settings, "
                "not technique claims."
            ),
        }
    )

    def as_dict(self) -> dict:
        return asdict(self)


# --- Rule specifications. ---
# Each rule declares what it reads, when that reading is meaningful and how
# much evidence it needs; rules.py evaluates it. SquatRule is RuleSpec renamed.

# The view vocabulary and the rule shape are shared by all three exercises
SquatRule = RuleSpec


def side_view_rules(config: SquatConfig) -> tuple[SquatRule, ...]:
    """Checks that need a side-on view of the movement."""
    return (
        SquatRule(
            name="Squat depth",
            rule_id="squat_depth",
            title="Squat depth",
            metric="squat_depth",
            phase="bottom",
            acceptable_min=None,
            acceptable_max=config.DEPTH_KNEE_ANGLE_PASS,
            tolerance=config.DEPTH_KNEE_ANGLE_WARN - config.DEPTH_KNEE_ANGLE_PASS,
            minimum_persistence_frames=1,
            min_violation_ratio=config.DEPTH_MIN_VIOLATION_RATIO,
            minimum_visibility=config.MIN_KEY_LANDMARK_VISIBILITY,
            supported_views=SAGITTAL_VIEWS,
            feedback_key="depth",
        ),
        SquatRule(
            name="Forward torso lean",
            rule_id="torso_lean",
            title="Forward torso lean",
            metric="torso_lean",
            phase="descent_to_ascent",
            acceptable_min=None,
            acceptable_max=config.TORSO_LEAN_WARN,
            tolerance=config.TORSO_LEAN_FAIL - config.TORSO_LEAN_WARN,
            minimum_persistence_frames=config.TORSO_LEAN_MIN_FRAMES,
            min_violation_ratio=config.TORSO_LEAN_MIN_VIOLATION_RATIO,
            minimum_visibility=config.MIN_KEY_LANDMARK_VISIBILITY,
            supported_views=SAGITTAL_VIEWS,
            feedback_key="torso_lean",
        ),
        SquatRule(
            name="Return to standing",
            rule_id="return_to_standing",
            title="Return to standing",
            metric="return_to_standing",
            phase="completion",
            acceptable_min=None,
            acceptable_max=config.FULL_EXTENSION_TOLERANCE,
            tolerance=config.FULL_EXTENSION_TOLERANCE,
            minimum_persistence_frames=1,
            min_violation_ratio=0.0,
            minimum_visibility=config.MIN_KEY_LANDMARK_VISIBILITY,
            supported_views=SAGITTAL_VIEWS,
            feedback_key="extension",
        ),
        SquatRule(
            name="Descent control",
            rule_id="descent_control",
            title="Descent control",
            metric="descent_control",
            phase="descent",
            acceptable_min=config.DESCENT_MIN_DURATION,
            acceptable_max=None,
            tolerance=config.DESCENT_MIN_DURATION - config.DESCENT_FAST_DURATION,
            minimum_persistence_frames=1,
            min_violation_ratio=0.0,
            minimum_visibility=config.MIN_KEY_LANDMARK_VISIBILITY,
            supported_views=SAGITTAL_VIEWS,
            feedback_key="descent_control",
            threshold_source=(
                "prototype timing heuristic - an observation about movement control, "
                "not a biomechanical claim"
            ),
        ),
    )


def view_independent_rules(config: SquatConfig) -> tuple[SquatRule, ...]:
    """Checks one landmark's vertical movement can support from any angle."""
    return (
        SquatRule(
            name="Heel stability",
            rule_id="heel_lift",
            title="Heel stability",
            metric="heel_lift",
            phase="descent_to_ascent",
            acceptable_min=None,
            acceptable_max=config.HEEL_LIFT_THRESHOLD,
            tolerance=config.HEEL_LIFT_THRESHOLD,
            minimum_persistence_frames=config.HEEL_LIFT_MIN_FRAMES,
            min_violation_ratio=config.HEEL_LIFT_MIN_VIOLATION_RATIO,
            minimum_visibility=config.HEEL_MIN_VISIBILITY,
            supported_views=ANY_VIEW,
            feedback_key="heel_lift",
        ),
    )


def rule_specs(config: SquatConfig) -> tuple[SquatRule, ...]:
    """Every squat rule, in the order the results page shows them."""
    side = {rule.rule_id: rule for rule in side_view_rules(config)}
    any_view = {rule.rule_id: rule for rule in view_independent_rules(config)}
    order = (
        "squat_depth",
        "torso_lean",
        "heel_lift",
        "return_to_standing",
        "descent_control",
    )
    combined = {**side, **any_view}
    return tuple(combined[rule_id] for rule_id in order)


# --- Recording guidance ---

# The camera view this exercise is analysed from.
RECOMMENDED_VIEW = "side-on"

# shown before upload and attached to a rejection, so the advice matches
RECORDING_TIPS: tuple[str, ...] = (
    *RETRY_TIPS,
    "Start standing, perform several full squats, and finish standing.",
)

# The three lines shown beside the camera diagram before upload. People skim
# this panel, so it says only what changes whether the analysis can run; the
# fuller RECORDING_TIPS above are kept for the retry advice after a rejection.
QUICK_TIPS: tuple[str, ...] = (
    "Film from the side, level with your hips.",
    "Fit your whole body in frame, head to feet.",
    "Keep the camera still. A few full squats is enough.",
)


# What the application actually runs with.
DEFAULT_CONFIG = SquatConfig()

# Default rule sets, exposed for the docs, the tests and the debug view.
SIDE_VIEW_RULES = side_view_rules(DEFAULT_CONFIG)
VIEW_INDEPENDENT_RULES = view_independent_rules(DEFAULT_CONFIG)
DEFAULT_RULES = rule_specs(DEFAULT_CONFIG)
