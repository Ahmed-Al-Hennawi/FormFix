"""
All tunable values for the lat pulldown in one place.

ENGINEERING values just keep the system stable. TECHNIQUE (OPERATIONAL) values
judge the movement - my prototype values from the exercise definition, the
reference clip and testing, not biomechanical constants.

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
    # the machine, cable and weight stack get in the way of the camera, so these
    # are set for a real gym
    # per-frame visibility floor
    MIN_KEY_LANDMARK_VISIBILITY: float = 0.4
    MIN_REQUIRED_LANDMARK_VISIBILITY: float = 0.4
    # share of frames one arm's shoulder, elbow, wrist and hip must be usable on
    MIN_USABLE_FRAME_RATIO: float = 0.5
    MIN_VALID_FRAME_RATIO: float = 0.45
    MIN_POSE_FRAME_RATIO: float = 0.5
    # gaps up to this long get interpolated (equipment blocking is usually brief)
    MAX_SHORT_GAP_FRAMES: int = 5

    # --- ENGINEERING - outlier rejection (analysis/stabilise.py) ---
    # frames compared either side when testing one landmark for a tracking error
    OUTLIER_WINDOW_FRAMES: int = 2
    # smallest jump that can be called an error, in torso lengths
    OUTLIER_MIN_JUMP_TORSOS: float = 0.12

    # --- ENGINEERING - anatomical plausibility gates ---
    # MediaPipe can be confident about an arm hidden behind the machine. Wide
    # bands - outside them is NaN, not a fault
    PLAUSIBLE_ELBOW_ANGLE_MIN: float = 15.0
    PLAUSIBLE_ELBOW_ANGLE_MAX: float = 190.0
    # a trunk moving more than this in one rep is tracking failure, not a pulldown
    PLAUSIBLE_TORSO_EXCURSION_MAX: float = 60.0
    # a left/right elbow gap this big means one arm is being guessed
    ARM_DIFFERENCE_MAX_PLAUSIBLE: float = 60.0

    # --- ENGINEERING - smoothing ---
    EMA_ALPHA: float = 0.45
    # EMA weight for the elbow-angle series (light second pass)
    ANGLE_EMA_ALPHA: float = 0.5

    # --- ENGINEERING - phase state machine / hysteresis ---
    # these four are adaptive, based on the person's own resting extension. With
    # a fixed "extended" level, someone who never straightens their arms would
    # never count a rep, so the ROM rule meant to catch that would never run.
    #     SEGMENTATION  was this a rep?              adaptive
    #     JUDGEMENT     did it cover enough range?   fixed, the ROM_* values
    # percentile of the elbow angle used as the resting extension
    REST_REFERENCE_PERCENTILE: float = 90.0
    # clamped to this band
    REST_REFERENCE_MIN: float = 110.0
    REST_REFERENCE_MAX: float = 180.0
    # degrees below the reference where the arms stop counting as rested
    REST_MARGIN: float = 6.0
    # ...where a rep attempt starts
    PULL_START_MARGIN: float = 14.0
    # ...needed for a real contracted position
    CONTRACTED_MARGIN: float = 30.0
    # ...back above which the rep is complete
    REP_END_MARGIN: float = 10.0
    TOP_ELBOW_ANGLE: float = 150.0
    # frames a phase change must hold before it commits
    PHASE_MIN_FRAMES: int = 3
    # how far the elbow must open past the running minimum before the contracted
    # position commits
    CONTRACTION_REVERSAL_DELTA: float = 5.0
    # minimum elbow range to count as an attempt (stops a wobble counting, not a
    # technique check)
    MIN_RANGE_OF_MOTION: float = 25.0
    # frames either side of the raw minimum, so the bottom is a window median
    CONTRACTED_WINDOW_FRAMES: int = 3
    # untracked frames allowed mid-rep before the attempt is abandoned
    MAX_TRACKING_LOSS_FRAMES: int = 10

    # --- ENGINEERING - rep duration bounds, in seconds ---
    MIN_REP_DURATION: float = 0.8
    MAX_REP_DURATION: float = 12.0

    # --- ENGINEERING - video suitability ---
    VIDEO_MIN_DURATION: float = 3.0
    VIDEO_MAX_DURATION: float = 120.0
    MULTI_PERSON_WARN_RATIO: float = 0.10
    MULTI_PERSON_FAIL_RATIO: float = 0.50
    # frontality bands: small for a side or three-quarter view, close to 1 front
    # or back on
    SIDE_VIEW_GOOD_RATIO: float = 0.45
    SIDE_VIEW_FRONTAL_RATIO: float = 1.00
    VIEW_STABILITY_SPREAD: float = 0.35
    FRAMING_MARGIN: float = 0.02
    FRAMING_WARN_TOLERANCE: float = 0.25
    FRAMING_FAIL_TOLERANCE: float = 0.70

    # --- ENGINEERING - top-position baseline ---
    # stable top-position frames needed for a trunk baseline (people are still
    # settling into the seat at the start)
    BASELINE_MIN_FRAMES: int = 5
    # the baseline only uses top frames where the arms are still - reaching up for
    # the bar also labels as "top", and it used to set the trunk reference
    BASELINE_MAX_VELOCITY: float = 20.0
    # and only the last couple of seconds before the first rep
    BASELINE_WINDOW_SECONDS: float = 2.0
    # below this the trunk rule says "cannot assess"
    TORSO_MIN_VISIBILITY: float = 0.5
    # below this on the head landmarks the facing direction is unknown, so the
    # lean has no direction
    FACING_MIN_VISIBILITY: float = 0.5
    # share of a rep the ANALYSED arm must be usable. Only one arm, because
    # side-on the far arm is hidden and requiring both rejected good side views
    ARM_MIN_USABLE_RATIO: float = 0.5
    # share of a rep both arms must be usable (export only, no rule uses it)
    ARMS_MIN_BOTH_SIDES_RATIO: float = 0.5

    # --- TECHNIQUE (OPERATIONAL) - excessive torso movement ---
    # not asking for a vertical trunk - a small lean is normal. This catches
    # swinging the body to move a weight the lats can't. Based on the reference
    # demo plus a margin for MediaPipe's error.
    # degrees of trunk movement from the top baseline that warn...
    TORSO_EXCURSION_WARN: float = 15.0
    # ...and fail
    TORSO_EXCURSION_FAIL: float = 25.0
    # absolute limit: this far from vertical it isn't really a pulldown any more
    TORSO_ABSOLUTE_FAIL: float = 45.0
    # has to hold this many frames and cover this share of the pull, so jitter
    # can't cause a finding
    TORSO_MIN_FRAMES: int = 4
    TORSO_MIN_VIOLATION_RATIO: float = 0.20

    # --- TECHNIQUE (OPERATIONAL) - range of motion ---
    # checked at both ends and says which end fell short ("limited ROM" alone
    # doesn't help a beginner). Not based on the bar reaching the chest, since
    # MediaPipe tracks the body, not the equipment.
    # top: how far the elbow must open between reps
    ROM_TOP_EXTENSION_PASS: float = 150.0
    ROM_TOP_EXTENSION_WARN: float = 138.0
    # bottom: how far the elbow must close during the pull
    ROM_BOTTOM_FLEXION_PASS: float = 100.0
    ROM_BOTTOM_FLEXION_WARN: float = 115.0
    # total range, so a rep a bit short at both ends still gets reported
    ROM_MIN_EXCURSION: float = 45.0
    # evidence only, not a pass condition: vertical wrist travel / trunk length
    ROM_MIN_WRIST_TRAVEL: float = 0.35

    # notes shown in the technical details panel
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
    # "moving_average" - see docs/filter_selection.md
    ANGLE_FILTER: str = "ema"

    # --- ENGINEERING - measurement-uncertainty policy ---
    # False: findings inside the published error are marked "indicative".
    # True: they are also downgraded a step. See exercises/common/uncertainty.py
    UNCERTAINTY_STRICT: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# --- Recording guidance ---

RECOMMENDED_VIEW = "side or three-quarter"

# full tips, shown with the retry advice after a rejection. Filmed from the
# side because trunk lean can't be seen from the front or back
RECORDING_TIPS: tuple[str, ...] = (
    "Film from the side, or at a three-quarter angle, roughly level with your chest.",
    "Keep your hips, shoulders, elbows and wrists visible for the whole set.",
    "Stand far enough back that your arms stay in frame at the top of every rep.",
    "Try to keep the machine's frame from hiding your arms.",
    "Keep the camera still, and record several controlled repetitions.",
)

# the three short lines next to the camera diagram before upload
QUICK_TIPS: tuple[str, ...] = (
    "Film from the side, level with your chest.",
    "Keep hips, shoulders and both arms in frame.",
    "Keep the camera still. A few controlled reps are enough.",
)


# --- Rule specifications (evaluated in rules.py) ---

# phase keys from exercises/common/metrics.py, renamed for the pulldown
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
    """All pulldown rules, in the order the results page shows them."""
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
            # elbow flexion can be seen from any angle, reliability still drops
            # for a view it's unsure about
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
            # a backward lean can't be seen from the front or back
            supported_views=SAGITTAL_VIEWS,
            feedback_key="pulldown_torso",
            threshold_source=PROTOTYPE_THRESHOLD,
        ),
    )


DEFAULT_CONFIG = PulldownConfig()

# used by the docs, tests and technical details panel
DEFAULT_RULES = rule_specs(DEFAULT_CONFIG)
