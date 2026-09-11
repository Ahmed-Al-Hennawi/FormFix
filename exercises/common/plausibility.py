"""
Does this movement look like the exercise the user picked?

Without this, someone could upload a bench press, pick "shoulder press" as the
closest option, and get detailed feedback on a movement they never did.

It runs after rep detection on the existing measurements. It's not a
classifier - it only looks for movement the chosen exercise can't produce, so
anything unusual or ambiguous still passes (a shallow press passes, a bench
press fails because the arms never go overhead).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# --- squat ---

# a squat has to bend the knees - this catches an upper-body exercise
# uploaded as a squat
SQUAT_MAX_KNEE_ANGLE = 155.0

# how far the wrists can travel (torso lengths) before a "squat" looks like
# an arm exercise. Generous on purpose
SQUAT_MAX_WRIST_TRAVEL_TORSOS = 2.5

# --- press ---

# wrists above the shoulders at the top, in SHOULDER WIDTHS. A real press
# peaks around 1.5 and a bench press around 0
PRESS_MIN_WRIST_ABOVE_SHOULDER = 0.40

# the elbow has to open on the way up, else it is a shrug or a hold
PRESS_MIN_ROM_DEGREES = 25.0

# --- pulldown ---

# a pulldown starts with the wrists above the shoulders (trunk lengths). Low
# on purpose, it only has to tell it apart from a row or a curl
PULLDOWN_MIN_WRIST_ABOVE_SHOULDER_AT_START = 0.05

# the elbow has to close through the pull
PULLDOWN_MIN_ROM_DEGREES = 20.0

# share of reps that must contradict the exercise before rejecting, so one
# odd rep can't reject a recording
MIN_CONTRADICTING_REP_RATIO = 0.6

# measurable reps needed before this check says anything
MIN_REPS_TO_JUDGE = 2


@dataclass(frozen=True)
class PlausibilityResult:
    """plausible is only False when the evidence contradicts the exercise. Too few
    reps gives True with a reason."""

    plausible: bool
    reason: str = ""
    message: str = ""
    checked_reps: int = 0
    contradicting_reps: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "plausible": self.plausible,
            "reason": self.reason,
            "checked_reps": self.checked_reps,
            "contradicting_reps": self.contradicting_reps,
        }


def _ok(reason: str, checked: int = 0) -> PlausibilityResult:
    return PlausibilityResult(True, reason=reason, checked_reps=checked)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _verdict(
    checked: int,
    contradicting: int,
    message: str,
    reason: str,
) -> PlausibilityResult:
    """Apply the ratio rule to a tally of contradicting reps."""
    if checked < MIN_REPS_TO_JUDGE:
        return _ok("too few measurable reps to judge", checked)
    ratio = contradicting / checked
    if ratio < MIN_CONTRADICTING_REP_RATIO:
        return _ok(f"{contradicting}/{checked} reps inconsistent - within tolerance", checked)
    logger.info("Exercise mismatch: %s (%d/%d reps)", reason, contradicting, checked)
    return PlausibilityResult(
        plausible=False,
        reason=reason,
        message=message,
        checked_reps=checked,
        contradicting_reps=contradicting,
    )


# --- Per-exercise checks ---


def check_squat(reps: Sequence[Any], wrist_travel_torsos: float | None = None) -> PlausibilityResult:
    """
    Does this look like a squat? Fails when the knees never bend, or when the
    arms travel far enough to make it an upper-body exercise.
    """
    if wrist_travel_torsos is not None and wrist_travel_torsos > SQUAT_MAX_WRIST_TRAVEL_TORSOS:
        return PlausibilityResult(
            plausible=False,
            reason=f"wrist travel {wrist_travel_torsos:.1f} torsos exceeds a squat's",
            message=(
                "This looks like an arm exercise rather than a squat. Check you picked "
                "the right exercise before uploading."
            ),
            checked_reps=len(reps),
            contradicting_reps=len(reps),
        )

    checked = contradicting = 0
    for rep in reps:
        angle = _finite(getattr(rep, "min_knee_angle", None))
        if angle is None:
            continue
        checked += 1
        if angle > SQUAT_MAX_KNEE_ANGLE:
            contradicting += 1

    return _verdict(
        checked,
        contradicting,
        message=(
            "We couldn't see a squat in this video - the knees stay almost straight "
            "throughout. Check you picked the right exercise, and film from the side."
        ),
        reason="knees never bend enough for a squat",
    )


def check_press(
    reps: Sequence[Any], wrist_above_shoulder_at_top: float | None = None
) -> PlausibilityResult:
    """
    Does this look like an overhead press? Fails when the wrists never get above
    the shoulders (bench press, curl, front raise) or the elbows barely open.
    wrist_above_shoulder_at_top is in shoulder widths.
    """
    if (
        wrist_above_shoulder_at_top is not None
        and wrist_above_shoulder_at_top < PRESS_MIN_WRIST_ABOVE_SHOULDER
    ):
        return PlausibilityResult(
            plausible=False,
            reason=(
                f"wrists peak {wrist_above_shoulder_at_top:.2f} torsos above the "
                "shoulders - not an overhead movement"
            ),
            message=(
                "This doesn't look like an overhead press - the arms never travel above "
                "the shoulders. Check you picked the right exercise before uploading."
            ),
            checked_reps=len(reps),
            contradicting_reps=len(reps),
        )

    checked = contradicting = 0
    for rep in reps:
        rom = _finite(getattr(rep, "rom_degrees", None))
        if rom is None:
            continue
        checked += 1
        if rom < PRESS_MIN_ROM_DEGREES:
            contradicting += 1

    return _verdict(
        checked,
        contradicting,
        message=(
            "We couldn't see a press in this video - the arms barely move through their "
            "range. Check you picked the right exercise before uploading."
        ),
        reason="elbow range too small for a press",
    )


def check_pulldown(
    reps: Sequence[Any], wrist_above_shoulder_at_start: float | None = None
) -> PlausibilityResult:
    """Does this look like a lat pulldown? Fails when the arms never start above the
    shoulders (a row or a curl) or the elbows barely close."""
    if (
        wrist_above_shoulder_at_start is not None
        and wrist_above_shoulder_at_start < PULLDOWN_MIN_WRIST_ABOVE_SHOULDER_AT_START
    ):
        return PlausibilityResult(
            plausible=False,
            reason=(
                f"wrists start {wrist_above_shoulder_at_start:.2f} torsos above the "
                "shoulders - the pull doesn't come from overhead"
            ),
            message=(
                "This doesn't look like a lat pulldown - the arms never reach above the "
                "shoulders. Check you picked the right exercise before uploading."
            ),
            checked_reps=len(reps),
            contradicting_reps=len(reps),
        )

    checked = contradicting = 0
    for rep in reps:
        rom = _finite(getattr(rep, "rom_degrees", None))
        if rom is None:
            continue
        checked += 1
        if rom < PULLDOWN_MIN_ROM_DEGREES:
            contradicting += 1

    return _verdict(
        checked,
        contradicting,
        message=(
            "We couldn't see a pulldown in this video - the arms barely move through "
            "their range. Check you picked the right exercise before uploading."
        ),
        reason="elbow range too small for a pulldown",
    )


# retry tips for this case - lighting or framing advice doesn't help if the
# wrong exercise was picked
MISMATCH_TIPS = [
    "Check the exercise you selected matches the one in the video.",
    "FormFix analyses the squat, the shoulder press and the lat pulldown.",
    "Film the whole movement from start to finish, from the side.",
]
