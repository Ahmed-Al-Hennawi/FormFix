"""
All tunable values for the squat analysis in one place.

ENGINEERING values just keep the system stable and say nothing about how to
squat. TECHNIQUE values do judge the squat, and they are my provisional
calibration values, not biomechanical constants.

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
    # per-frame visibility floor. I lowered it from 0.5 after it rejected usable gym
    # videos - MediaPipe gives 0.3-0.6 for a limb partly behind the body side-on
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
    # MediaPipe can be confident about a limb it's guessing. These bands are wide
    # and only catch tracking failures - anything outside becomes NaN, not a fault.
    PLAUSIBLE_KNEE_ANGLE_MIN: float = 25.0
    PLAUSIBLE_KNEE_ANGLE_MAX: float = 190.0
    # a heel rising more than half a lower leg is tracking failure, not a fault
    HEEL_LIFT_MAX_PLAUSIBLE: float = 0.5
    # a left/right gap this big is one leg being extrapolated behind the other
    KNEE_SYMMETRY_MAX_PLAUSIBLE: float = 45.0

    # --- ENGINEERING - smoothing ---
    # EMA weight for the landmark coordinates, higher follows the raw signal more
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
    # multi-person ratios: warn above the first, reject above the second
    MULTI_PERSON_WARN_RATIO: float = 0.10
    MULTI_PERSON_FAIL_RATIO: float = 0.50
    # frontality bands (shoulder or hip gap / torso length), small side-on and
    # close to 1 front-on
    #   <= GOOD          clean side view, full confidence
    #   GOOD..FRONTAL    diagonal, analysed with a warning
    #   >= FRONTAL       front-on, sagittal angles not assessed
    SIDE_VIEW_GOOD_RATIO: float = 0.45
    SIDE_VIEW_FRONTAL_RATIO: float = 1.00
    # spread above which the camera angle counts as changing mid-clip
    VIEW_STABILITY_SPREAD: float = 0.35
    # edge margin that counts as clipped, then the share of frames allowed in it
    # before warning / stopping
    FRAMING_MARGIN: float = 0.02
    FRAMING_WARN_TOLERANCE: float = 0.25
    FRAMING_FAIL_TOLERANCE: float = 0.70

    # --- ENGINEERING - standing baseline ---
    # stable standing frames needed for a baseline (median is used)
    BASELINE_MIN_FRAMES: int = 5

    # --- TECHNIQUE (PROVISIONAL) - squat depth ---
    # knee angle at or below which a rep passes on depth
    DEPTH_KNEE_ANGLE_PASS: float = 100.0
    # between PASS and this it warns, above it the rep fails
    DEPTH_KNEE_ANGLE_WARN: float = 115.0
    # alternative way to pass: hip reaches knee level within this tolerance
    DEPTH_HIP_KNEE_TOLERANCE: float = 0.12

    # share of the bottom window that must be too shallow before flagging it
    DEPTH_MIN_VIOLATION_RATIO: float = 0.5

    # --- TECHNIQUE (PROVISIONAL) - torso lean ---
    # peak torso lean from vertical that flags a rep. Build and squat style change
    # the right number, so this is a rough value
    TORSO_LEAN_WARN: float = 45.0
    TORSO_LEAN_FAIL: float = 60.0
    # also flag lean this far past the person's own standing baseline
    TORSO_LEAN_DELTA_FAIL: float = 55.0
    # the lean has to hold this many frames and cover this share of the rep,
    # so one jittery landmark can't cause a finding
    TORSO_LEAN_MIN_FRAMES: int = 4
    TORSO_LEAN_MIN_VIOLATION_RATIO: float = 0.15

    # --- TECHNIQUE (PROVISIONAL) - heel lift ---
    # heel rise above the standing baseline / lower-leg length
    HEEL_LIFT_THRESHOLD: float = 0.06
    HEEL_LIFT_MIN_FRAMES: int = 3
    # below this the heel rule says "cannot assess" instead of guessing
    HEEL_MIN_VISIBILITY: float = 0.5
    # share of the rep's measurable frames the lift must cover
    HEEL_LIFT_MIN_VIOLATION_RATIO: float = 0.10

    # --- TECHNIQUE (PROVISIONAL) - return to standing / extension ---
    # a rep has to finish within this many degrees of the person's own standing
    # angle - not 180, so nobody is told to hyperextend
    FULL_EXTENSION_TOLERANCE: float = 12.0

    # --- MEASUREMENT ONLY - left/right symmetry (frontal plane) ---
    # measured and exported but not graded - one camera can't tell a real
    # asymmetry apart from the far leg's projection error
    SYMMETRY_MIN_FRAMES: int = 4
    # both legs must be usable on this share of the rep, and the far leg above
    # this visibility, before they're compared
    SYMMETRY_MIN_BOTH_SIDES_RATIO: float = 0.6
    SYMMETRY_MIN_VISIBILITY: float = 0.5

    # --- TECHNIQUE (PROVISIONAL) - movement control / tempo ---
    # a descent in a fraction of a second is a drop, not a controlled rep. Only
    # ever a gentle warning
    DESCENT_MIN_DURATION: float = 0.45
    # shorter than this and the warning is firmer
    DESCENT_FAST_DURATION: float = 0.25

    # --- ENGINEERING - reliability banding (see confidence.py) ---
    # when a verdict counts as HIGH / MEDIUM / LOW evidence (about the recording,
    # not the technique)
    RELIABILITY_HIGH_VISIBILITY: float = 0.75
    RELIABILITY_HIGH_MEASURABLE_RATIO: float = 0.85
    RELIABILITY_HIGH_VIEW_SUPPORT: float = 0.85
    RELIABILITY_HIGH_MIN_REPS: int = 2
    RELIABILITY_MEDIUM_VISIBILITY: float = 0.55
    RELIABILITY_MEDIUM_MEASURABLE_RATIO: float = 0.6
    RELIABILITY_MEDIUM_VIEW_SUPPORT: float = 0.5
    # median torso length in pixels. Someone filmed from far away has more error
    # per landmark than visibility shows
    RELIABILITY_MIN_SUBJECT_PIXELS: float = 140.0
    RELIABILITY_POOR_SUBJECT_PIXELS: float = 70.0

    # --- ENGINEERING - which filter smooths the movement signal ---
    # "ema" (default), "butterworth" (Dill et al. 2024), "savgol" or
    # "moving_average". No filter won on every rep shape, so it's a setting
    # (see docs/filter_selection.md)
    ANGLE_FILTER: str = "ema"

    # --- ENGINEERING - measurement-uncertainty policy ---
    # False: findings inside the published error are marked "indicative".
    # True: they are also downgraded a step. See exercises/common/uncertainty.py
    UNCERTAINTY_STRICT: bool = False

    # notes shown in the debug view
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


# --- Rule specifications (evaluated in rules.py) ---

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
    """Checks that work from any camera angle."""
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
    """All squat rules, in the order the results page shows them."""
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

RECOMMENDED_VIEW = "side-on"

# full tips, shown with the retry advice after a rejection
RECORDING_TIPS: tuple[str, ...] = (
    *RETRY_TIPS,
    "Start standing, perform several full squats, and finish standing.",
)

# the three short lines next to the camera diagram before upload (testers
# didn't read the longer version)
QUICK_TIPS: tuple[str, ...] = (
    "Film from the side, level with your hips.",
    "Fit your whole body in frame, head to feet.",
    "Keep the camera still. A few full squats is enough.",
)


DEFAULT_CONFIG = SquatConfig()

# default rule sets, used by the docs, tests and debug view
SIDE_VIEW_RULES = side_view_rules(DEFAULT_CONFIG)
VIEW_INDEPENDENT_RULES = view_independent_rules(DEFAULT_CONFIG)
DEFAULT_RULES = rule_specs(DEFAULT_CONFIG)
