"""
All tunable values for the dumbbell shoulder press in one place.

ENGINEERING values just keep the system stable. TECHNIQUE (OPERATIONAL) values
do judge the press - they're my prototype values from the exercise definition,
the reference clip and testing, not biomechanical constants.

Conventions:

    elbow angle    SHOULDER-ELBOW-WRIST, ~165-175 overhead, 70-100 racked
    elbow flexion  180 - elbow angle. The movement signal: high at the
                   shoulders, near zero at lockout.
    normalised     a distance over shoulder width in the same frame, so camera
                   distance, resolution and body size all cancel out
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from analysis.models import (
    METRIC_PRESS_ALIGNMENT,
    METRIC_PRESS_ROM,
    METRIC_PRESS_SYMMETRY,
)
from exercises.common.spec import (
    ANY_VIEW,
    FRONTAL_VIEWS,
    PROTOTYPE_THRESHOLD,
    RuleSpec,
)


@dataclass(frozen=True)
class PressConfig:
    # --- ENGINEERING - MediaPipe detector confidences ---
    POSE_DETECTION_CONFIDENCE: float = 0.5
    POSE_PRESENCE_CONFIDENCE: float = 0.5
    TRACKING_CONFIDENCE: float = 0.5

    # --- ENGINEERING - landmark reliability and gap handling ---
    MIN_KEY_LANDMARK_VISIBILITY: float = 0.4
    MIN_REQUIRED_LANDMARK_VISIBILITY: float = 0.4
    MIN_USABLE_FRAME_RATIO: float = 0.5
    MIN_VALID_FRAME_RATIO: float = 0.45
    MIN_POSE_FRAME_RATIO: float = 0.5
    MAX_SHORT_GAP_FRAMES: int = 5

    # --- ENGINEERING - anatomical plausibility gates ---
    # a dumbbell in front of the wrist or an arm crossing the head makes MediaPipe
    # guess confidently. Outside these wide bands is NaN, not a fault
    PLAUSIBLE_ELBOW_ANGLE_MIN: float = 15.0
    PLAUSIBLE_ELBOW_ANGLE_MAX: float = 190.0
    # a left/right elbow gap this big means one arm is being guessed
    ELBOW_DIFFERENCE_MAX_PLAUSIBLE: float = 70.0
    # a wrist-height gap this big is tracking failure, not an uneven press
    WRIST_HEIGHT_DIFFERENCE_MAX_PLAUSIBLE: float = 1.0
    ALIGNMENT_OFFSET_MAX_PLAUSIBLE: float = 1.5

    # --- ENGINEERING - smoothing ---
    EMA_ALPHA: float = 0.45
    ANGLE_EMA_ALPHA: float = 0.5

    # --- ENGINEERING - phase state machine / hysteresis ---
    # adaptive like the pulldown: with a fixed "extended" threshold, someone who
    # never locks out would never count a rep, so the ROM rule would never run.
    #     SEGMENTATION  was this a rep?              adaptive
    #     JUDGEMENT     did it cover enough range?   fixed, the ROM_* values
    # signal is elbow FLEXION (180 - elbow angle), high at the shoulders
    REST_REFERENCE_PERCENTILE: float = 90.0
    REST_REFERENCE_MIN: float = 45.0
    REST_REFERENCE_MAX: float = 140.0
    # degrees below the reference where the arms stop counting as rested
    REST_MARGIN: float = 6.0
    # ...where a press attempt starts
    PRESS_START_MARGIN: float = 14.0
    # ...needed for a real top position
    TOP_MARGIN: float = 30.0
    # ...back above which the rep is complete
    REP_END_MARGIN: float = 10.0
    # fallback resting flexion if the series is unusable
    READY_ELBOW_FLEXION: float = 95.0

    PHASE_MIN_FRAMES: int = 3
    # how far flexion must rise past its running minimum before the top commits
    TOP_REVERSAL_DELTA: float = 5.0
    # minimum range to count as a press attempt (not a technique check)
    MIN_RANGE_OF_MOTION: float = 25.0
    # frames either side of the raw minimum used for a stable top
    TOP_WINDOW_FRAMES: int = 3
    MAX_TRACKING_LOSS_FRAMES: int = 10

    # --- ENGINEERING - rep duration bounds, in seconds ---
    MIN_REP_DURATION: float = 0.8
    MAX_REP_DURATION: float = 12.0

    # --- ENGINEERING - video suitability ---
    VIDEO_MIN_DURATION: float = 3.0
    VIDEO_MAX_DURATION: float = 120.0
    MULTI_PERSON_WARN_RATIO: float = 0.10
    MULTI_PERSON_FAIL_RATIO: float = 0.50
    # frontality bands. Here a HIGH ratio is good because the press is filmed
    # from the front. Shoulder width is about 0.8-0.9 of trunk length, so a
    # straight front view is near 0.8 - asking for 1.0 called good clips diagonal
    SIDE_VIEW_GOOD_RATIO: float = 0.35
    SIDE_VIEW_FRONTAL_RATIO: float = 0.70
    VIEW_STABILITY_SPREAD: float = 0.35
    FRAMING_MARGIN: float = 0.02
    FRAMING_WARN_TOLERANCE: float = 0.25
    FRAMING_FAIL_TOLERANCE: float = 0.70

    # --- ENGINEERING - evidence requirements ---
    BASELINE_MIN_FRAMES: int = 5
    # share of a rep both arms must be usable before they're compared, otherwise
    # symmetry says "cannot assess"
    SYMMETRY_MIN_BOTH_SIDES_RATIO: float = 0.6
    # ...and the far arm has to be above this visibility
    SYMMETRY_MIN_VISIBILITY: float = 0.5
    # alignment is per arm, so a hidden arm doesn't switch it off for the other
    ALIGNMENT_MIN_VISIBILITY: float = 0.5

    # --- TECHNIQUE (OPERATIONAL) - arm symmetry ---
    # three signals instead of one score, because "your left arm stayed lower"
    # is useful to a beginner and "symmetry index 0.72" isn't. Any one of them
    # persisting flags the rep. Bars are well above zero since a few degrees is
    # normal and inside MediaPipe's error.
    # |left - right| elbow angle, degrees
    SYMMETRY_ANGLE_WARN: float = 15.0
    SYMMETRY_ANGLE_FAIL: float = 25.0
    # |left - right| wrist height, in shoulder widths
    SYMMETRY_HEIGHT_WARN: float = 0.12
    SYMMETRY_HEIGHT_FAIL: float = 0.20
    # |left ROM - right ROM| over the rep, degrees
    SYMMETRY_ROM_WARN: float = 15.0
    SYMMETRY_ROM_FAIL: float = 25.0
    # seconds between the arms reaching the top (evidence only)
    SYMMETRY_TIMING_NOTE: float = 0.30
    # an asymmetry has to hold this many frames and cover this share of the press
    SYMMETRY_MIN_FRAMES: int = 5
    SYMMETRY_MIN_VIOLATION_RATIO: float = 0.25

    # --- TECHNIQUE (OPERATIONAL) - elbow / wrist alignment ---
    # from the front the wrist should stay roughly over the elbow, measured as
    # |wrist_x - elbow_x| / shoulder width (0.38 is about 15 cm). I raised it
    # from 0.30/0.45 after testing (docs/threshold_tuning.md) because a rep
    # stopping short of overhead was flagged for both alignment and ROM
    ALIGNMENT_OFFSET_WARN: float = 0.38
    ALIGNMENT_OFFSET_FAIL: float = 0.52
    ALIGNMENT_MIN_FRAMES: int = 5
    ALIGNMENT_MIN_VIOLATION_RATIO: float = 0.25

    # --- TECHNIQUE (OPERATIONAL) - range of motion ---
    # checked at both ends, and says which end fell short. The top isn't 180 -
    # a fully locked elbow under load isn't the goal
    # top: how far the elbow must open overhead
    ROM_TOP_EXTENSION_PASS: float = 155.0
    ROM_TOP_EXTENSION_WARN: float = 143.0
    # bottom: how far the elbow must close at the shoulders
    ROM_BOTTOM_FLEXION_PASS: float = 100.0
    ROM_BOTTOM_FLEXION_WARN: float = 115.0
    # total elbow range within the rep
    ROM_MIN_EXCURSION: float = 50.0

    # notes shown in the technical details panel
    notes: dict[str, str] = field(
        default_factory=lambda: {
            "variant": (
                "Written for a seated dumbbell shoulder press with the back supported, a "
                "pronated grip and both arms pressing together, matching the FormFix "
                "reference clip. Arnold, neutral-grip, single-arm, standing/push and "
                "behind-the-neck variations are not described by these rules."
            ),
            "technique_thresholds": (
                "SYMMETRY_*, ALIGNMENT_* and ROM_* are operational prototype values "
                "selected from the exercise definition, the reference demonstration and "
                "iterative testing. They are not biomechanical constants and are expected "
                "to be tuned."
            ),
            "normalisation": (
                "Positional differences are divided by the athlete's shoulder width in the "
                "same frame, so they are dimensionless and independent of camera distance, "
                "video resolution and body size."
            ),
            "not_implemented": (
                "Sagittal trunk lean (the 'arching back' fault) is deliberately not "
                "assessed: it happens in the plane a front-on camera cannot see, and this "
                "exercise needs a front-on camera for its other three checks."
            ),
        }
    )

    # --- ENGINEERING - which filter smooths the movement signal ---
    # "ema" (default), "butterworth" (Dill et al. 2024), "savgol" or
    # "moving_average" - see docs/filter_selection.md
    ANGLE_FILTER: str = "ema"

    # --- ENGINEERING - measurement-uncertainty policy ---
    # False: findings inside the published error are marked "indicative".
    # True: they are also downgraded a step. See exercises/common/uncertainty.py
    UNCERTAINTY_STRICT: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# --- Recording guidance ---

RECOMMENDED_VIEW = "front-on"

# full tips, shown with the retry advice after a rejection. Filmed from the
# front because two of the three checks compare the arms
RECORDING_TIPS: tuple[str, ...] = (
    "Film from the front, roughly level with your chest, with the camera centred on you.",
    "Keep both shoulders, both elbows and both wrists visible for the whole set.",
    "Leave headroom so the dumbbells stay inside the frame at the top of every press.",
    "Try not to let the dumbbells hide your wrists from the camera.",
    "Keep the camera still, and record several controlled repetitions.",
)

# the three short lines next to the camera diagram before upload
QUICK_TIPS: tuple[str, ...] = (
    "Film from the front, level with your chest.",
    "Keep both arms and the dumbbells in frame at the top.",
    "Keep the camera still. A few controlled reps are enough.",
)


# --- Rule specifications ---

# phase keys from exercises/common/metrics.py, renamed for the press
PHASE_READY = "start"
PHASE_PRESS = "towards"
PHASE_TOP = "extreme"
PHASE_LOWER = "return"
PHASE_WORKING = "working"
PHASE_MOVEMENT = "movement"

PHASE_LABELS: dict[str, str] = {
    PHASE_READY: "the starting position at your shoulders",
    PHASE_PRESS: "the pressing phase",
    PHASE_TOP: "the overhead position",
    PHASE_LOWER: "the lowering phase",
    PHASE_WORKING: "the press and the overhead position",
    PHASE_MOVEMENT: "the whole repetition",
}


def rule_specs(config: PressConfig) -> tuple[RuleSpec, ...]:
    """All press rules, in the order the results page shows them."""
    return (
        RuleSpec(
            name="Arm symmetry",
            rule_id="press_symmetry",
            title="Arm symmetry",
            metric=METRIC_PRESS_SYMMETRY,
            phase=PHASE_WORKING,
            acceptable_min=None,
            acceptable_max=config.SYMMETRY_ANGLE_WARN,
            tolerance=config.SYMMETRY_ANGLE_FAIL - config.SYMMETRY_ANGLE_WARN,
            minimum_persistence_frames=config.SYMMETRY_MIN_FRAMES,
            min_violation_ratio=config.SYMMETRY_MIN_VIOLATION_RATIO,
            minimum_visibility=config.SYMMETRY_MIN_VISIBILITY,
            # comparing the arms needs the camera across the body
            supported_views=FRONTAL_VIEWS,
            feedback_key="press_symmetry",
            threshold_source=PROTOTYPE_THRESHOLD,
        ),
        RuleSpec(
            name="Elbow and wrist alignment",
            rule_id="press_alignment",
            title="Elbow and wrist alignment",
            metric=METRIC_PRESS_ALIGNMENT,
            phase=PHASE_WORKING,
            acceptable_min=None,
            acceptable_max=config.ALIGNMENT_OFFSET_WARN,
            tolerance=config.ALIGNMENT_OFFSET_FAIL - config.ALIGNMENT_OFFSET_WARN,
            minimum_persistence_frames=config.ALIGNMENT_MIN_FRAMES,
            min_violation_ratio=config.ALIGNMENT_MIN_VIOLATION_RATIO,
            minimum_visibility=config.ALIGNMENT_MIN_VISIBILITY,
            # from the side this would be depth, which one camera can't measure
            supported_views=FRONTAL_VIEWS,
            feedback_key="press_alignment",
            threshold_source=PROTOTYPE_THRESHOLD,
        ),
        RuleSpec(
            name="Range of motion",
            rule_id="press_rom",
            title="Range of motion",
            metric=METRIC_PRESS_ROM,
            phase=PHASE_MOVEMENT,
            acceptable_min=config.ROM_MIN_EXCURSION,
            acceptable_max=None,
            tolerance=config.ROM_TOP_EXTENSION_PASS - config.ROM_TOP_EXTENSION_WARN,
            minimum_persistence_frames=1,
            min_violation_ratio=0.0,
            minimum_visibility=config.MIN_KEY_LANDMARK_VISIBILITY,
            # elbow flexion is visible from any reasonable angle
            supported_views=ANY_VIEW,
            feedback_key="press_rom",
            threshold_source=PROTOTYPE_THRESHOLD,
        ),
    )


DEFAULT_CONFIG = PressConfig()
DEFAULT_RULES = rule_specs(DEFAULT_CONFIG)
