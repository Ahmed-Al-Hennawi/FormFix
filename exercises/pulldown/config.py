"""
Every tunable value for the lat pulldown, in one place.

Two kinds, marked by the section headings. ENGINEERING values control
stability - confidences, gap handling, hysteresis, persistence. TECHNIQUE
(OPERATIONAL) values make claims about the movement: prototype values from the
exercise definition, the reference clip and testing, not biomechanical
constants. Each rule carries its threshold_source into the technical details.

Angle conventions:

    elbow angle      SHOULDER-ELBOW-WRIST, ~170-180 extended, 60-90 contracted
    torso angle      mid-hip -> mid-shoulder against image vertical
    torso excursion  torso angle minus this rep's own top-position baseline, so
                     a reclined bench or a tilted camera aren't faults
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from analysis.models import (
    METRIC_PULLDOWN_ROM,
    METRIC_PULLDOWN_TORSO,
)
from exercises.common.spec import (
    ANY_VIEW,
    PROTOTYPE_THRESHOLD,
    SAGITTAL_VIEWS,
    RuleSpec,
)


@dataclass(frozen=True)
class PulldownConfig:
    # --- ENGINEERING - MediaPipe detector confidences ---
    POSE_DETECTION_CONFIDENCE: float = 0.5
    POSE_PRESENCE_CONFIDENCE: float = 0.5
    TRACKING_CONFIDENCE: float = 0.5

    # --- ENGINEERING - landmark reliability and gap handling ---
    # a pulldown machine puts a frame, a cable and a weight stack between the
    # camera and the lifter, so these floors are set for a real gym
    # per-frame visibility floor. Below it the landmark is not measured.
    MIN_KEY_LANDMARK_VISIBILITY: float = 0.4
    MIN_REQUIRED_LANDMARK_VISIBILITY: float = 0.4
    # share of frames one arm's shoulder, elbow, wrist and hip must be usable on
    MIN_USABLE_FRAME_RATIO: float = 0.5
    MIN_VALID_FRAME_RATIO: float = 0.45
    MIN_POSE_FRAME_RATIO: float = 0.5
    # gaps up to this long get interpolated; equipment occlusion is usually brief
    MAX_SHORT_GAP_FRAMES: int = 5

    # --- ENGINEERING - anatomical plausibility gates ---
    # MediaPipe reports a confident position even while extrapolating an arm
    # hidden behind the machine. Wide bands - outside them is NaN, never a fault.
    PLAUSIBLE_ELBOW_ANGLE_MIN: float = 15.0
    PLAUSIBLE_ELBOW_ANGLE_MAX: float = 190.0
    # a trunk moving more than this in one rep is tracking failure, not a pulldown
    PLAUSIBLE_TORSO_EXCURSION_MAX: float = 60.0
    # A left/right elbow gap this big is one arm being extrapolated behind the other.
    ARM_DIFFERENCE_MAX_PLAUSIBLE: float = 60.0

    # --- ENGINEERING - smoothing ---
    EMA_ALPHA: float = 0.45
    # EMA weight for the derived elbow-angle series - a light second pass.
    ANGLE_EMA_ALPHA: float = 0.5

    # --- ENGINEERING - phase state machine / hysteresis ---
    # these four thresholds are adaptive, taken from the lifter's own resting
    # extension. A fixed "extended" level goes circular: someone who never
    # straightens their arms would never cross it, so no rep would be detected,
    # so the range-of-motion rule meant to catch that habit would never run.
    #     SEGMENTATION  was this a rep?              adaptive
    #     JUDGEMENT     did it cover enough range?   fixed, the ROM_* values
    # the reference is a high percentile of the smoothed elbow angle, clamped.
    # The margins below are subtracted from it and give the hysteresis.
    # percentile taken as this person's resting extension
    REST_REFERENCE_PERCENTILE: float = 90.0
    # the reference is clamped into this band
    REST_REFERENCE_MIN: float = 110.0
    REST_REFERENCE_MAX: float = 180.0
    # Degrees below the reference at which the arms stop counting as rested.
    REST_MARGIN: float = 6.0
    # ...at which a rep attempt begins.
    PULL_START_MARGIN: float = 14.0
    # ...needed for a real contracted position to exist.
    CONTRACTED_MARGIN: float = 30.0
    # ...back above which the rep completes.
    REP_END_MARGIN: float = 10.0
    TOP_ELBOW_ANGLE: float = 150.0
    # Frames a phase transition must hold before it commits.
    PHASE_MIN_FRAMES: int = 3
    # how far the elbow must open past the running minimum before the contracted
    # position commits
    CONTRACTION_REVERSAL_DELTA: float = 5.0
    # minimum elbow excursion to count as a pulldown attempt. A segmentation
    # guard against a wobble, not a technique criterion.
    MIN_RANGE_OF_MOTION: float = 25.0
    # frames either side of the raw minimum, so the bottom is a window median
    CONTRACTED_WINDOW_FRAMES: int = 3
    # Untracked frames tolerated mid-rep before the attempt is abandoned.
    MAX_TRACKING_LOSS_FRAMES: int = 10

    # --- ENGINEERING - rep duration bounds, in seconds ---
    MIN_REP_DURATION: float = 0.8
    MAX_REP_DURATION: float = 12.0

    # --- ENGINEERING - video suitability ---
    VIDEO_MIN_DURATION: float = 3.0
    VIDEO_MAX_DURATION: float = 120.0
    MULTI_PERSON_WARN_RATIO: float = 0.10
    MULTI_PERSON_FAIL_RATIO: float = 0.50
    # frontality bands: small for a side or three-quarter view, towards 1 front
    # or rear on. They set the camera-angle confidence. Provisional.
    SIDE_VIEW_GOOD_RATIO: float = 0.45
    SIDE_VIEW_FRONTAL_RATIO: float = 1.00
    VIEW_STABILITY_SPREAD: float = 0.35
    FRAMING_MARGIN: float = 0.02
    FRAMING_WARN_TOLERANCE: float = 0.25
    FRAMING_FAIL_TOLERANCE: float = 0.70

    # --- ENGINEERING - top-position baseline ---
    # stable top-position frames needed for a trunk baseline. People are usually
    # still settling into the seat at frame 0.
    BASELINE_MIN_FRAMES: int = 5
    # Below this mean visibility the trunk rule says "cannot assess".
    TORSO_MIN_VISIBILITY: float = 0.5
    # below this on the head landmarks the facing direction is unknown, so the
    # trunk excursion is reported unsigned
    FACING_MIN_VISIBILITY: float = 0.5
    # share of a rep the ANALYSED arm must be usable on. One arm deliberately -
    # ROM is a joint angle, and side-on the far arm is hidden for much of the
    # pull, so requiring both reported "not assessed" on good side views.
    # One arm, deliberately: range of motion is a joint angle rather than a
    ARM_MIN_USABLE_RATIO: float = 0.5
    # share of a rep both arms must be usable on. Exported only, no rule uses it.
    ARMS_MIN_BOTH_SIDES_RATIO: float = 0.5

    # --- TECHNIQUE (OPERATIONAL) - excessive torso movement ---
    # not aiming for a vertical trunk - a seated pulldown has a small deliberate
    # lean, and the bench often sets one. What these catch is the body swing used
    # to move a weight the lats can't. Picked off the reference demonstration plus
    # a margin for MediaPipe's error. Provisional.
    # Picked off the reference demonstration (about 10-20 deg of trunk inclination
    # degrees of trunk movement from the top baseline that warn...
    TORSO_EXCURSION_WARN: float = 15.0
    # ...and failed.
    TORSO_EXCURSION_FAIL: float = 25.0
    # absolute backstop: this far from vertical during the pull, whatever the
    # baseline was, it isn't really the exercise any more
    TORSO_ABSOLUTE_FAIL: float = 45.0
    # persistence: the movement must hold this many frames and cover this share
    # of the pull, so jitter can't produce a finding
    TORSO_MIN_FRAMES: int = 4
    TORSO_MIN_VIOLATION_RATIO: float = 0.20

    # --- TECHNIQUE (OPERATIONAL) - range of motion ---
    # judged at both ends and reported as which end fell short, since "limited
    # ROM" alone tells a beginner nothing. Not based on the bar reaching the
    # chest - MediaPipe tracks the body, not the equipment. All provisional.
    # Not based on the bar reaching the chest - MediaPipe tracks the body, not the
    # top: how far the elbow must open between reps
    ROM_TOP_EXTENSION_PASS: float = 150.0
    ROM_TOP_EXTENSION_WARN: float = 138.0
    # Bottom: how far the elbow must close during the pull.
    ROM_BOTTOM_FLEXION_PASS: float = 100.0
    ROM_BOTTOM_FLEXION_WARN: float = 115.0
    # total excursion in the rep, so something shallow at both ends without
    # failing either single criterion still gets reported
    ROM_MIN_EXCURSION: float = 45.0
    # supporting evidence only, never a pass condition: vertical wrist travel
    # over trunk length, so a reader can check the angles against real movement
    ROM_MIN_WRIST_TRAVEL: float = 0.35

    # Free-form notes shown in the technical-details panel.
    notes: dict[str, str] = field(
        default_factory=lambda: {
            "variant": (
                "Written for a seated, bilateral, pronated-grip lat pulldown with the bar "
                "travelling in front of the head, matching the FormFix reference clip. "
                "Behind-the-neck, close-grip, single-arm and standing variations are not "
                "described by these rules."
            ),
            "technique_thresholds": (
                "TORSO_* and ROM_* are operational prototype values selected from the "
                "exercise definition, the reference demonstration and iterative testing. "
                "They are not biomechanical constants and are expected to be tuned."
            ),
            "persistence": (
                "*_MIN_FRAMES and *_MIN_VIOLATION_RATIO control how long a violation must "
                "persist before it becomes a finding; they are stability settings, not "
                "technique claims."
            ),
            "not_implemented": (
                "Torso swinging BETWEEN repetitions is measured (peak_torso_velocity) and "
                "exported, but no rule reads it: it was not validated against labelled "
                "recordings, and shipping an untested check would weaken the two that were."
            ),
        }
    )

    # --- ENGINEERING - which filter smooths the movement signal ---
    # "ema" (default), "butterworth" (Dill et al. 2024), "savgol" or
    # "moving_average". docs/filter_selection.md has the measured trade-off.
    ANGLE_FILTER: str = "ema"

    # --- ENGINEERING - measurement-uncertainty policy ---
    # False (default): a finding is annotated with the published error of the
    # quantity behind it and marked "indicative" if its margin falls inside it.
    # True also downgrades it a step. See exercises/common/uncertainty.py.
    UNCERTAINTY_STRICT: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# --- Recording guidance ---

# the camera view this exercise is analysed from
RECOMMENDED_VIEW = "side or three-quarter"

# shown before upload and on a rejection. Trunk movement is sagittal, so it
# isn't visible from directly in front or behind.
RECORDING_TIPS: tuple[str, ...] = (
    "Film from the side, or at a three-quarter angle, roughly level with your chest.",
    "Keep your hips, shoulders, elbows and wrists visible for the whole set.",
    "Stand far enough back that your arms stay in frame at the top of every rep.",
    "Try to keep the machine's frame from hiding your arms.",
    "Keep the camera still, and record several controlled repetitions.",
)

# The three lines shown beside the camera diagram before upload. People skim
# this panel, so it says only what changes whether the analysis can run; the
# fuller RECORDING_TIPS above are kept for the retry advice after a rejection.
QUICK_TIPS: tuple[str, ...] = (
    "Film from the side, level with your chest.",
    "Keep hips, shoulders and both arms in frame.",
    "Keep the camera still. A few controlled reps is enough.",
)


# --- Rule specifications. ---
# Each rule declares what it reads, when that question is meaningful, which
# views support it and how much evidence it needs; rules.py evaluates it.

# Phase keys from exercises/common/metrics.py, renamed for the pulldown.
PHASE_TOP = "start"
PHASE_PULL = "towards"
PHASE_CONTRACTED = "extreme"
PHASE_RETURN = "return"
PHASE_WORKING = "working"
PHASE_MOVEMENT = "movement"

PHASE_LABELS: dict[str, str] = {
    PHASE_TOP: "the extended top position",
    PHASE_PULL: "the pulling phase",
    PHASE_CONTRACTED: "the contracted position",
    PHASE_RETURN: "the return",
    PHASE_WORKING: "the pull and the contracted position",
    PHASE_MOVEMENT: "the whole repetition",
}


def rule_specs(config: PulldownConfig) -> tuple[RuleSpec, ...]:
    """Every pulldown rule, in the order the results page shows them."""
    return (
        RuleSpec(
            name="Range of motion",
            rule_id="pulldown_rom",
            title="Range of motion",
            metric=METRIC_PULLDOWN_ROM,
            phase=PHASE_MOVEMENT,
            acceptable_min=config.ROM_MIN_EXCURSION,
            acceptable_max=None,
            tolerance=config.ROM_TOP_EXTENSION_PASS - config.ROM_TOP_EXTENSION_WARN,
            minimum_persistence_frames=1,
            min_violation_ratio=0.0,
            minimum_visibility=config.MIN_KEY_LANDMARK_VISIBILITY,
            # Elbow flexion is visible from side, diagonal or front; the confidence
            # layer still marks down a view it is unsure about.
            supported_views=ANY_VIEW,
            feedback_key="pulldown_rom",
            threshold_source=PROTOTYPE_THRESHOLD,
        ),
        RuleSpec(
            name="Torso movement",
            rule_id="pulldown_torso",
            title="Torso movement",
            metric=METRIC_PULLDOWN_TORSO,
            phase=PHASE_WORKING,
            acceptable_min=None,
            acceptable_max=config.TORSO_EXCURSION_WARN,
            tolerance=config.TORSO_EXCURSION_FAIL - config.TORSO_EXCURSION_WARN,
            minimum_persistence_frames=config.TORSO_MIN_FRAMES,
            min_violation_ratio=config.TORSO_MIN_VIOLATION_RATIO,
            minimum_visibility=config.TORSO_MIN_VISIBILITY,
            # Trunk lean is sagittal. From directly in front or behind a backward
            # lean is invisible, so the rule reports "not assessed".
            supported_views=SAGITTAL_VIEWS,
            feedback_key="pulldown_torso",
            threshold_source=PROTOTYPE_THRESHOLD,
        ),
    )


# What the application actually runs with.
DEFAULT_CONFIG = PulldownConfig()

# Exposed for the docs, the tests and the technical-details panel.
DEFAULT_RULES = rule_specs(DEFAULT_CONFIG)
