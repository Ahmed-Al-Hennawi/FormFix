"""
A second squat configuration, built from published biomechanics.

FormFix's own thresholds are operational values. Rather than overwrite them
with numbers from a paper, the literature values sit here alongside the
defaults and scripts/evaluate_videos.py --preset literature runs the evaluation
under them, so the two can be compared on the same recordings.

Kotiuk et al. (2022), via Rao et al. (2025), for a parallel squat:

    knee flexion    113 +/- 7 degrees
    hip flexion     128 +/- 9 degrees
    ankle           23 +/- 6 degrees

Watch the convention: they report flexion, we measure the interior hip-knee-
ankle angle, so for the knee interior = 180 - flexion. That makes 113 of
flexion 67 of interior angle, far deeper than our 100 degree pass bar - they
measured a supervised parallel squat, and 67 would fail nearly every real
recording. So the preset takes the upper end of the published range as its pass
bar and mean + 1 SD as its warning bar. That reading is my judgement, and it is
why this preset is here for comparison rather than adopted.

Dill et al. (2024) define a correct squat much as these rules do, and give two
usable numbers: correct reps reached a mean peak knee angle of 137.4 degrees,
about 20 above the faulty variants. That 20-degree gap is the scale the
warning band is built from.

Simoes et al. (2024) set acceptable angular deviation at 10 degrees for an
experienced user and 20 for a beginner, which is why the bands below are wide.
"""

from __future__ import annotations

from dataclasses import replace

from .config import DEFAULT_CONFIG, SquatConfig

__all__ = ["LITERATURE_CONFIG", "PROVENANCE", "literature_config"]


# Where each changed value came from and what I did to it. The evaluation
# harness prints this, so a run can't be reported without the provenance of
# the numbers behind it.
PROVENANCE: dict[str, str] = {
    "DEPTH_KNEE_ANGLE_PASS": (
        "Kotiuk et al. (2022) via Rao et al. (2025): knee flexion 113 +/- 7 deg for a "
        "horizontal squat, i.e. an interior hip-knee-ankle angle of 67 +/- 7 deg. The "
        "preset takes the most permissive traceable reading - the mean plus two SD of "
        "the interior angle, 81 deg - rather than the mean, because the source measured "
        "a supervised parallel squat and FormFix analyses unsupervised recordings."
    ),
    "DEPTH_KNEE_ANGLE_WARN": (
        "Dill et al. (2024): correct squats reached a mean peak knee angle about 20 deg "
        "away from the faulty variants in the same protocol. The warning band is set one "
        "such separation above the pass bar, so the band between 'good' and 'flagged' is "
        "the size of a difference that was empirically detectable."
    ),
    "TORSO_LEAN_WARN": (
        "Dill et al. (2024) define a correct squat as one in which 'the spine remains "
        "straight throughout' and make excessive forward bending their first fault "
        "condition (E2), but publish no trunk angle. The default 45 deg is retained; "
        "only the failure bar is tightened to the point at which their E2 condition is "
        "unambiguously present."
    ),
    "HEEL_LIFT_THRESHOLD": (
        "Dill et al. (2024): 'heels remain on the floor throughout the exercise' is a "
        "criterion of correct execution, so any detectable lift is a fault. The default "
        "0.06 lower-leg lengths is retained as the intent, but the derived measurement "
        "band for a normalised length is 0.15, so the preset raises the bar to the "
        "point where a lift is distinguishable from normalisation noise."
    ),
    "FULL_EXTENSION_TOLERANCE": (
        "Simoes et al. (2024) use a 20 deg allowance for a beginner and 10 deg for an "
        "experienced user. The preset adopts the beginner value, since FormFix's stated "
        "audience is people training without a coach."
    ),
}


def literature_config(base: SquatConfig | None = None) -> SquatConfig:
    """The default config with its technique thresholds swapped for the
    literature-derived ones. Engineering thresholds stay as they are - no paper has
    an opinion on EMA weights."""
    base = base or DEFAULT_CONFIG
    return replace(
        base,
        # Kotiuk et al. (2022), converted to interior angle, upper bound.
        DEPTH_KNEE_ANGLE_PASS=81.0,
        # Dill et al. (2024): 20 deg correct/incorrect separation.
        DEPTH_KNEE_ANGLE_WARN=101.0,
        # Dill et al. (2024) E2. Fail bar tightened, warn bar left alone.
        TORSO_LEAN_WARN=45.0,
        TORSO_LEAN_FAIL=55.0,
        # Above the derived +/-0.15 band for a normalised length.
        HEEL_LIFT_THRESHOLD=0.16,
        # Simoes et al. (2024) beginner allowance.
        FULL_EXTENSION_TOLERANCE=20.0,
        notes={
            **base.notes,
            "preset": (
                "LITERATURE preset - technique thresholds derived from Kotiuk et al. "
                "(2022), Dill et al. (2024) and Simoes et al. (2024). See "
                "exercises/squat/literature_config.py:PROVENANCE for what each value "
                "was derived from and what judgement was applied. NOT validated "
                "against labelled recordings, exactly like the defaults."
            ),
        },
    )


# The preset, ready to hand straight to the analyser.
LITERATURE_CONFIG = literature_config()
