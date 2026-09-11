"""
Which of MediaPipe's 33 landmarks the squat uses. Face, fingers and arms add
nothing to depth or trunk lean, and counting them would make the reliability
figures meaningless. METRIC_REQUIREMENTS is what per-measurement reliability
is based on.
"""

from __future__ import annotations

from dataclasses import dataclass

from analysis.models import (
    LEFT_ANKLE,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    METRIC_DEPTH,
    METRIC_DESCENT_CONTROL,
    METRIC_EXTENSION,
    METRIC_HEEL_LIFT,
    METRIC_KNEE_SYMMETRY,
    METRIC_TORSO_LEAN,
    RIGHT_ANKLE,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
)

# the ones I use, with their MediaPipe Pose ids
SQUAT_LANDMARKS: dict[str, int] = {
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
    "left_knee": LEFT_KNEE,
    "right_knee": RIGHT_KNEE,
    "left_ankle": LEFT_ANKLE,
    "right_ankle": RIGHT_ANKLE,
    "left_heel": LEFT_HEEL,
    "right_heel": RIGHT_HEEL,
    "left_foot_index": LEFT_FOOT_INDEX,
    "right_foot_index": RIGHT_FOOT_INDEX,
}


@dataclass(frozen=True)
class SideChain:
    """One side's chain, in the same order as analysis.models.SIDE_LANDMARKS."""

    name: str
    shoulder: int
    hip: int
    knee: int
    ankle: int
    heel: int
    foot_index: int

    @property
    def core(self) -> tuple[int, int, int, int]:
        """Shoulder/hip/knee/ankle, needed by every sagittal measurement."""
        return (self.shoulder, self.hip, self.knee, self.ankle)

    @property
    def foot(self) -> tuple[int, int]:
        """Heel and toe, only needed by the heel-contact measurement."""
        return (self.heel, self.foot_index)

    @property
    def all_ids(self) -> tuple[int, ...]:
        return (*self.core, *self.foot)


SIDE_CHAINS: dict[str, SideChain] = {
    "left": SideChain(
        "left",
        shoulder=LEFT_SHOULDER,
        hip=LEFT_HIP,
        knee=LEFT_KNEE,
        ankle=LEFT_ANKLE,
        heel=LEFT_HEEL,
        foot_index=LEFT_FOOT_INDEX,
    ),
    "right": SideChain(
        "right",
        shoulder=RIGHT_SHOULDER,
        hip=RIGHT_HIP,
        knee=RIGHT_KNEE,
        ankle=RIGHT_ANKLE,
        heel=RIGHT_HEEL,
        foot_index=RIGHT_FOOT_INDEX,
    ),
}

OPPOSITE_SIDE = {"left": "right", "right": "left"}

# without these a squat can't be measured at all
CORE_LANDMARK_NAMES: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# what each measurement depends on - reliability uses exactly these points
METRIC_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    METRIC_DEPTH: ("hip", "knee", "ankle"),
    METRIC_TORSO_LEAN: ("shoulder", "hip"),
    METRIC_HEEL_LIFT: ("heel", "ankle", "knee"),
    METRIC_EXTENSION: ("hip", "knee", "ankle"),
    METRIC_DESCENT_CONTROL: ("hip", "knee", "ankle"),
    # the only one that needs both legs, so it drops out first
    METRIC_KNEE_SYMMETRY: ("hip", "knee", "ankle"),
}

# metrics comparing both sides, so both need to be visible
BILATERAL_METRICS: frozenset[str] = frozenset({METRIC_KNEE_SYMMETRY})

# metrics measured ACROSS the body. These use 1 - side_view_confidence, since
# a side view is the wrong angle for a left/right comparison
FRONTAL_PLANE_METRICS: frozenset[str] = frozenset({METRIC_KNEE_SYMMETRY})


def is_frontal_plane(metric: str) -> bool:
    return metric in FRONTAL_PLANE_METRICS


def landmark_ids(side: str, roles: tuple[str, ...]) -> tuple[int, ...]:
    """
    ("hip", "knee") on a side -> MediaPipe indices. Raises KeyError on a typo
    on purpose, so it shows up in the tests.
    """
    chain = SIDE_CHAINS[side]
    return tuple(getattr(chain, role) for role in roles)


def required_ids(metric: str, side: str) -> tuple[int, ...]:
    """Landmark indices a metric needs on a side (both sides if bilateral)."""
    roles = METRIC_REQUIREMENTS.get(metric, ())
    if not roles:
        return ()
    if metric in BILATERAL_METRICS:
        return landmark_ids("left", roles) + landmark_ids("right", roles)
    return landmark_ids(side, roles)
