"""
Everything the exercise explorer displays: a port of the EXERCISES array from
the prototype's script.js plus the pose geometry that used to be inline SVG.
Keeping the geometry as coordinates lets utils/pose.py render all three with
one renderer. Coordinates use the photographs' 0 0 1122 1402 viewBox.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exercises.press import config as press_config
from exercises.pulldown import config as pulldown_config
from exercises.squat import config as squat_config

from . import assets

# The photographs share this intrinsic coordinate space.
POSE_VIEWBOX = "0 0 1122 1402"

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class Pose:
    """Landmark geometry for one exercise figure."""

    spine: tuple[Point, ...]
    links: tuple[Segment, ...]
    focus: tuple[Segment, ...]
    arc: str
    gradient: tuple[float, float, float, float]
    nodes: tuple[Point, ...]
    key_nodes: tuple[Point, ...]


@dataclass(frozen=True)
class Exercise:
    """One of the three movements."""

    id: str
    name: str
    description: str
    areas: tuple[str, ...]
    metric_label: str
    metric_value: str
    image: str
    image_alt: str
    atmosphere_a: str
    atmosphere_b: str
    pose: Pose
    # rule ids this exercise's analyser produces, so the explorer can't advertise
    # a check we don't actually run
    rules: tuple[str, ...] = field(default=())
    # camera position the analysis needs, in plain words
    camera_view: str = ""
    # recording instructions, imported from the exercise's config rather than
    # retyped, so the advice matches what the analyser enforces
    recording_tips: tuple[str, ...] = field(default=())
    # clip length the analyser accepts, also read from the exercise config
    min_duration: float = 0.0
    max_duration: float = 0.0

    @property
    def duration_label(self) -> str:
        """Accepted clip length, e.g. '3-120 seconds'. Empty when unset."""
        if not self.max_duration:
            return ""
        return f"{self.min_duration:g}-{self.max_duration:g} seconds"

    @property
    def metric_html(self) -> str:
        return f"{self.metric_label}&nbsp;<strong>{self.metric_value}</strong>"


SQUAT = Exercise(
    id="squat",
    name="Squat",
    description=(
        "Analysed from the side, FormFix tracks how deep your hips descend, how far "
        "your chest leans forward, whether your heels stay planted, and whether you "
        "stand tall between repetitions."
    ),
    areas=("Hip depth", "Torso angle", "Heel contact", "Return to standing"),
    metric_label="KNEE ANGLE",
    metric_value="92&deg;",
    image=assets.IMAGE_SQUAT,
    image_alt="Athlete performing a barbell squat",
    atmosphere_a="rgba(0, 240, 255, 0.10)",
    atmosphere_b="rgba(57, 255, 20, 0.05)",
    rules=(
        "squat_depth",
        "torso_lean",
        "heel_lift",
        "return_to_standing",
        "descent_control",
    ),
    camera_view="Side-on",
    recording_tips=squat_config.RECORDING_TIPS,
    min_duration=squat_config.DEFAULT_CONFIG.VIDEO_MIN_DURATION,
    max_duration=squat_config.DEFAULT_CONFIG.VIDEO_MAX_DURATION,
    pose=Pose(
        spine=((452, 306), (520, 528), (487, 787)),
        links=(
            ((348, 528), (692, 528)),
            ((374, 792), (600, 782)),
            ((348, 528), (374, 792)),
            ((692, 528), (600, 782)),
            ((348, 528), (340, 675)),
            ((340, 675), (300, 472)),
            ((692, 528), (765, 675)),
            ((765, 675), (792, 450)),
        ),
        focus=(
            ((374, 792), (292, 880)),
            ((292, 880), (317, 1072)),
            ((600, 782), (688, 872)),
            ((688, 872), (662, 1072)),
        ),
        arc="M 637.7 820.5 A 72 72 0 0 0 678.7 943.4",
        gradient=(760, 760, 760, 1010),
        nodes=(
            (452, 306),
            (348, 528),
            (692, 528),
            (374, 792),
            (600, 782),
            (300, 472),
            (792, 450),
            (317, 1072),
            (662, 1072),
            (340, 675),
            (765, 675),
        ),
        key_nodes=((292, 880), (688, 872)),
    ),
)

PRESS = Exercise(
    id="press",
    name="Shoulder Press",
    description=(
        "Filmed from the front, FormFix compares your two arms through the press, "
        "checks that each wrist stays stacked over its elbow, and measures how far "
        "each repetition travels at the shoulders and overhead."
    ),
    areas=(
        "Arm symmetry",
        "Elbow and wrist alignment",
        "Range of motion",
    ),
    metric_label="ELBOW ANGLE",
    metric_value="84&deg;",
    image=assets.IMAGE_PRESS,
    image_alt="Athlete performing a seated dumbbell shoulder press",
    atmosphere_a="rgba(57, 255, 20, 0.09)",
    atmosphere_b="rgba(0, 240, 255, 0.06)",
    rules=("press_symmetry", "press_alignment", "press_rom"),
    camera_view="Front-on",
    recording_tips=press_config.RECORDING_TIPS,
    min_duration=press_config.DEFAULT_CONFIG.VIDEO_MIN_DURATION,
    max_duration=press_config.DEFAULT_CONFIG.VIDEO_MAX_DURATION,
    pose=Pose(
        spine=((552, 250), (554, 397), (531, 749)),
        links=(
            ((430, 398), (678, 396)),
            ((432, 748), (630, 750)),
            ((430, 398), (432, 748)),
            ((678, 396), (630, 750)),
            ((432, 748), (300, 836)),
            ((300, 836), (244, 1152)),
            ((630, 750), (590, 838)),
            ((590, 838), (612, 1152)),
        ),
        focus=(
            ((430, 398), (324, 524)),
            ((324, 524), (324, 360)),
            ((678, 396), (790, 516)),
            ((790, 516), (796, 356)),
        ),
        arc="M 739.5 461.9 A 74 74 0 0 1 792.8 442.1",
        gradient=(790, 380, 790, 545),
        nodes=(
            (552, 250),
            (430, 398),
            (678, 396),
            (432, 748),
            (630, 750),
            (324, 360),
            (796, 356),
            (244, 1152),
            (612, 1152),
            (300, 836),
            (590, 838),
        ),
        key_nodes=((324, 524), (790, 516)),
    ),
)

PULLDOWN = Exercise(
    id="pulldown",
    name="Lat Pulldown",
    description=(
        "Filmed from the side, FormFix measures how much your torso moves away from "
        "its starting position while you pull, and how far your arms travel at both "
        "ends of each repetition."
    ),
    areas=(
        "Torso movement",
        "Range of motion",
    ),
    metric_label="TORSO MOVEMENT",
    metric_value="&plusmn;15&deg;",
    image=assets.IMAGE_PULLDOWN,
    image_alt="Athlete performing a lat pulldown, seen from behind",
    atmosphere_a="rgba(0, 200, 255, 0.11)",
    atmosphere_b="rgba(0, 240, 255, 0.04)",
    rules=("pulldown_rom", "pulldown_torso"),
    camera_view="Side or three-quarter",
    recording_tips=pulldown_config.RECORDING_TIPS,
    min_duration=pulldown_config.DEFAULT_CONFIG.VIDEO_MIN_DURATION,
    max_duration=pulldown_config.DEFAULT_CONFIG.VIDEO_MAX_DURATION,
    pose=Pose(
        spine=((392, 508), (411, 667), (403, 942)),
        links=(
            ((282, 668), (540, 666)),
            ((332, 948), (474, 936)),
            ((282, 668), (332, 948)),
            ((540, 666), (474, 936)),
        ),
        focus=(
            ((282, 668), (208, 778)),
            ((208, 778), (222, 704)),
            ((540, 666), (609, 770)),
            ((609, 770), (676, 685)),
        ),
        arc="M 579.1 725 A 54 54 0 0 1 642.4 727.6",
        gradient=(600, 655, 600, 790),
        nodes=(
            (392, 508),
            (282, 668),
            (540, 666),
            (332, 948),
            (474, 936),
            (222, 704),
            (676, 685),
        ),
        key_nodes=((208, 778), (609, 770)),
    ),
)

# order matters: it drives the tabs, the "01 / 03" counter and the preview
EXERCISES: tuple[Exercise, ...] = (SQUAT, PRESS, PULLDOWN)

BY_ID: dict[str, Exercise] = {exercise.id: exercise for exercise in EXERCISES}


def get(exercise_id: str) -> Exercise:
    """Look up an exercise by id, defaulting to the squat."""
    return BY_ID.get(exercise_id, SQUAT)
