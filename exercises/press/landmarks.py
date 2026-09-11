"""
Which MediaPipe landmarks the press uses. All landmark numbers live here, so a
typo is an import error instead of a silently wrong joint. MediaPipe doesn't
track the dumbbells, so one hiding a wrist gives "not assessed".
"""

from __future__ import annotations

from dataclasses import dataclass

from analysis.models import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    METRIC_PRESS_ALIGNMENT,
    METRIC_PRESS_ROM,
    METRIC_PRESS_SYMMETRY,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

# the ones I use, with their MediaPipe Pose ids
PRESS_LANDMARKS: dict[str, int] = {
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
}


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

# both arms - two of the three checks compare them
BOTH_ARM_LANDMARKS: tuple[int, ...] = (
    LEFT_SHOULDER,
    LEFT_ELBOW,
    LEFT_WRIST,
    RIGHT_SHOULDER,
    RIGHT_ELBOW,
    RIGHT_WRIST,
)

# scale reference for the front view. Shoulder width, not trunk length, because
# the trunk looks shorter when the person leans
SHOULDER_LANDMARKS: tuple[int, int] = (LEFT_SHOULDER, RIGHT_SHOULDER)

# without these the press can't be measured at all
CORE_LANDMARK_NAMES: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

# what each measurement depends on
METRIC_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    METRIC_PRESS_ROM: ("shoulder", "elbow", "wrist"),
    METRIC_PRESS_SYMMETRY: ("shoulder", "elbow", "wrist"),
    METRIC_PRESS_ALIGNMENT: ("shoulder", "elbow", "wrist"),
}

# measurements that compare the arms, so both need to be visible
BILATERAL_METRICS: frozenset[str] = frozenset({METRIC_PRESS_SYMMETRY})


def landmark_ids(side: str, roles: tuple[str, ...]) -> tuple[int, ...]:
    chain = ARM_CHAINS[side]
    return tuple(getattr(chain, role) for role in roles)


def required_ids(metric: str, side: str) -> tuple[int, ...]:
    """Landmark indices a metric needs (both arms if bilateral)."""
    roles = METRIC_REQUIREMENTS.get(metric, ())
    if not roles:
        return ()
    if metric in BILATERAL_METRICS:
        return tuple(sorted(set(landmark_ids("left", roles) + landmark_ids("right", roles))))
    return landmark_ids(side, roles)
