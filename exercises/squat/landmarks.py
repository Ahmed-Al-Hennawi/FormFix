"""
Which of MediaPipe's 33 landmarks the squat reads. The face, fingers and most
of the upper limb contribute nothing to depth or trunk lean, and counting them
as evidence would make the reliability figures meaningless.

This is the only place landmark numbers become squat vocabulary.
METRIC_REQUIREMENTS is what makes per-measurement reliability possible.
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

# The subset we use, with official MediaPipe Pose ids (33-landmark topology).
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
    """
    One body side's chain, in anatomical order. The tuple ordering matches
    analysis.models.SIDE_LANDMARKS.
    """

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

# The other side, for symmetry comparisons.
OPPOSITE_SIDE = {"left": "right", "right": "left"}

# Without these a squat cannot be measured at all, on any camera view.
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

# what each measurement depends on. Reliability comes from exactly these
# points, not the average over all 33.
METRIC_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    METRIC_DEPTH: ("hip", "knee", "ankle"),
    METRIC_TORSO_LEAN: ("shoulder", "hip"),
    METRIC_HEEL_LIFT: ("heel", "ankle", "knee"),
    METRIC_EXTENSION: ("hip", "knee", "ankle"),
    METRIC_DESCENT_CONTROL: ("hip", "knee", "ankle"),
    # Symmetry is the only one needing both legs, so it drops out first.
    METRIC_KNEE_SYMMETRY: ("hip", "knee", "ankle"),
}

# Metrics that compare the two sides and therefore need both chains visible.
BILATERAL_METRICS: frozenset[str] = frozenset({METRIC_KNEE_SYMMETRY})

# metrics measured ACROSS the body rather than along it. side_view_confidence
# measures how side-on the camera is, which is backwards for a left/right
# comparison, so these take the complement.
# side_view_confidence measures how side-on the camera is, which is right
FRONTAL_PLANE_METRICS: frozenset[str] = frozenset({METRIC_KNEE_SYMMETRY})


def is_frontal_plane(metric: str) -> bool:
    """True when a metric is measured across the body rather than along it."""
    return metric in FRONTAL_PLANE_METRICS


def landmark_ids(side: str, roles: tuple[str, ...]) -> tuple[int, ...]:
    """
    Turn ("hip", "knee") on a side into MediaPipe indices. KeyError on an
    unknown side or role, on purpose - a typo should blow up at test time.
    """
    chain = SIDE_CHAINS[side]
    return tuple(getattr(chain, role) for role in roles)


def required_ids(metric: str, side: str) -> tuple[int, ...]:
    """The landmark indices metric needs on side (both sides for
    bilateral metrics)."""
    roles = METRIC_REQUIREMENTS.get(metric, ())
    if not roles:
        return ()
    if metric in BILATERAL_METRICS:
        return landmark_ids("left", roles) + landmark_ids("right", roles)
    return landmark_ids(side, roles)
