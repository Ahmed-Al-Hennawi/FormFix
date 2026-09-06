"""
Does this movement look like the exercise the user chose?

The upload page asks people to pick their exercise, and nothing downstream
questions that choice - each analyser measures whatever it was pointed at and
writes confident, specific feedback about it. So somebody uploads a bench
press, picks "shoulder press" because it is the closest option, and gets a
detailed critique of a movement they never performed.

This runs after rep detection, on the measurements the analysers already
produce, and asks whether the movement is consistent with the label. It is not
a classifier - it never works out what the exercise actually was.

The bias is one-directional: every check looks for a contradiction, movement
the chosen exercise cannot produce. Anything ambiguous or merely unusual
passes, so a shallow or sideways-filmed press still passes while a bench press
fails on the arms never travelling overhead.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# --- squat ---

# a squat has to bend the knees. The rep detector already rejects anything
# with no dip, so this only catches an upper-body exercise mislabelled.
SQUAT_MAX_KNEE_ANGLE = 155.0

# how far the wrists may travel, in torso lengths, before a "squat" looks
# like an arm exercise. Generous - it catches a press labelled as a squat.
SQUAT_MAX_WRIST_TRAVEL_TORSOS = 2.5

# --- press ---

# an overhead press finishes with the wrists well above the shoulders, in
# SHOULDER WIDTHS. A real press peaks near 1.5 of these; a bench press
# finishes near 0, so the threshold sits well between them.
PRESS_MIN_WRIST_ABOVE_SHOULDER = 0.40

# the elbow has to open on the way up, else it is a shrug or a hold
PRESS_MIN_ROM_DEGREES = 25.0

# --- pulldown ---

# a pulldown starts with the arms overhead, so at the top of the rep the
# wrists are above the shoulders. In trunk lengths, and low on purpose - it
# only has to separate a pulldown from a row or a curl.
PULLDOWN_MIN_WRIST_ABOVE_SHOULDER_AT_START = 0.05

# The elbow has to close through the pull.
PULLDOWN_MIN_ROM_DEGREES = 20.0

# share of measurable reps that must contradict the exercise before we stop.
# Over half, so one odd rep can never reject a recording.
MIN_CONTRADICTING_REP_RATIO = 0.6

# reps that must be measurable before the gate says anything at all
MIN_REPS_TO_JUDGE = 2


@dataclass(frozen=True)
class PlausibilityResult:
    """Whether the movement matches the selected exercise. plausible is False only
    when the evidence contradicts the label; too few measurable reps is True with a
    reason saying so."""

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
    Does this look like an overhead press? Fails when the wrists never finish above
    the shoulders, or when the elbows barely open. wrist_above_shoulder_at_top is
    the highest the wrists get above the shoulder line across the set, in torso
    lengths - a bench press, a curl and a front raise all fail on it.
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
    shoulders - which is what separates it from a row or a curl - or when the
    elbows barely close."""
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


# what the retry text says here - nothing about lighting or framing helps
# when the wrong exercise was selected
MISMATCH_TIPS = [
    "Check the exercise you selected matches the one in the video.",
    "FormFix analyses the squat, the shoulder press and the lat pulldown.",
    "Film the whole movement from start to finish, from the side.",
]
