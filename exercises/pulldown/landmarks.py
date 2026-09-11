"""
Which MediaPipe landmarks the pulldown uses. All landmark numbers live here, so
a typo is an import error instead of a silently wrong joint. MediaPipe doesn't
track the bar, so there's no "bar to chest" rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from analysis.models import (
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    METRIC_PULLDOWN_ROM,
    METRIC_PULLDOWN_TORSO,
    NOSE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

# the ones I use, with their MediaPipe Pose ids
PULLDOWN_LANDMARKS: dict[str, int] = {
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
}

# head landmarks, only used to work out which way the person faces, so a
# backward lean can be told apart from a forward one
FACING_LANDMARKS: tuple[int, ...] = (NOSE, LEFT_EAR, RIGHT_EAR)


@dataclass(frozen=True)
class ArmChain:
    """One arm's chain, in anatomical order."""

    name: str
    shoulder: int
    elbow: int
    wrist: int
    hip: int

    @property
    def arm(self) -> tuple[int, int, int]:
        return (self.shoulder, self.elbow, self.wrist)

    @property
    def all_ids(self) -> tuple[int, ...]:
        return (self.shoulder, self.elbow, self.wrist, self.hip)


ARM_CHAINS: dict[str, ArmChain] = {
    "left": ArmChain("left", shoulder=LEFT_SHOULDER, elbow=LEFT_ELBOW, wrist=LEFT_WRIST, hip=LEFT_HIP),
    "right": ArmChain(
        "right", shoulder=RIGHT_SHOULDER, elbow=RIGHT_ELBOW, wrist=RIGHT_WRIST, hip=RIGHT_HIP
    ),
}

OPPOSITE_SIDE = {"left": "right", "right": "left"}

# without these the pulldown can't be measured at all
CORE_LANDMARK_NAMES: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

# trunk reference - mid-shoulder to mid-hip
TORSO_LANDMARKS: tuple[int, ...] = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

# what each measurement depends on - reliability uses exactly these points
METRIC_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    METRIC_PULLDOWN_ROM: ("shoulder", "elbow", "wrist"),
    METRIC_PULLDOWN_TORSO: ("shoulder", "hip"),
}

# trunk metrics need both sides of the body
TORSO_METRICS: frozenset[str] = frozenset({METRIC_PULLDOWN_TORSO})


def landmark_ids(side: str, roles: tuple[str, ...]) -> tuple[int, ...]:
    """
    ("shoulder", "elbow") on a side -> MediaPipe indices. Raises KeyError on a
    typo on purpose, so it shows up in the tests.
    """
    chain = ARM_CHAINS[side]
    return tuple(getattr(chain, role) for role in roles)


def required_ids(metric: str, side: str) -> tuple[int, ...]:
    """Landmark indices a metric needs on a side."""
    roles = METRIC_REQUIREMENTS.get(metric, ())
    if not roles:
        return ()
    if metric in TORSO_METRICS:
        return tuple(sorted(set(landmark_ids("left", roles) + landmark_ids("right", roles))))
    return landmark_ids(side, roles)
