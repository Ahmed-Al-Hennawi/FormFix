"""
Which of MediaPipe's 33 landmarks the pulldown reads. Nothing else in the
package has a bare landmark number, so a typo is an import error rather than a
silently wrong joint.

Note what isn't here: the bar. MediaPipe tracks a body, not equipment, so a
rule waiting for the bar to touch the chest couldn't be implemented honestly.
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

# The subset we use, with official MediaPipe Pose ids (33-landmark topology).
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

# head landmarks, used only to work out which way the person faces, so a
# backward trunk lean can be told from a forward one
# facing, so a backward trunk lean can be told from a forward one. When they
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
        """Shoulder/elbow/wrist, which is all the elbow angle needs."""
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

# Without these a lat pulldown cannot be measured at all, on any camera view.
CORE_LANDMARK_NAMES: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

# The trunk reference - mid-shoulder against mid-hip.
TORSO_LANDMARKS: tuple[int, ...] = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

# what each measurement depends on. Reliability comes from exactly these
# points, not the mean over all 33.
METRIC_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    METRIC_PULLDOWN_ROM: ("shoulder", "elbow", "wrist"),
    METRIC_PULLDOWN_TORSO: ("shoulder", "hip"),
}

# Measured from the trunk, so they need both sides of the body.
TORSO_METRICS: frozenset[str] = frozenset({METRIC_PULLDOWN_TORSO})


def landmark_ids(side: str, roles: tuple[str, ...]) -> tuple[int, ...]:
    """
    Turn ("shoulder", "elbow") on a side into MediaPipe indices. KeyError
    on an unknown side or role, on purpose - a typo should blow up at test time.
    """
    chain = ARM_CHAINS[side]
    return tuple(getattr(chain, role) for role in roles)


def required_ids(metric: str, side: str) -> tuple[int, ...]:
    """The landmark indices metric needs on side."""
    roles = METRIC_REQUIREMENTS.get(metric, ())
    if not roles:
        return ()
    if metric in TORSO_METRICS:
        return tuple(sorted(set(landmark_ids("left", roles) + landmark_ids("right", roles))))
    return landmark_ids(side, roles)
