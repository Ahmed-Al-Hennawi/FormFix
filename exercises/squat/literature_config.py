"""
A second squat config based on published biomechanics.

Instead of overwriting my own thresholds, the literature values sit here next
to them, and scripts/evaluate_videos.py --preset literature runs the evaluation
with them so both can be compared on the same videos.

Kotiuk et al. (2022), via Rao et al. (2025), for a parallel squat:

    knee flexion    113 +/- 7 degrees
    hip flexion     128 +/- 9 degrees
    ankle           23 +/- 6 degrees

They report flexion but I measure the interior angle (interior = 180 -
flexion), so 113 flexion is 67 interior - much deeper than my 100 degree pass
bar, and it would fail nearly every real video. So the pass bar uses the upper
end of their range and the warning bar mean + 1 SD. That's my own judgement,
which is why this preset is for comparison and not the default.

Dill et al. (2024): correct reps reached a mean peak knee angle of 137.4
degrees, about 20 above the faulty ones. That 20 degree gap sets the warning
band.

Simoes et al. (2024) allow 10 degrees of deviation for experienced users and
20 for beginners, which is why the bands are wide.
"""

from __future__ import annotations

from dataclasses import replace

from .config import DEFAULT_CONFIG, SquatConfig

__all__ = ["LITERATURE_CONFIG", "PROVENANCE", "literature_config"]


# where each changed value came from and what I did to it. The evaluation
# harness prints this with every run
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
    """The default config with the technique thresholds swapped for the literature
    ones. Engineering values stay the same."""
    base = base or DEFAULT_CONFIG
    return replace(
        base,
        # Kotiuk et al. (2022), converted to interior angle, upper bound
        DEPTH_KNEE_ANGLE_PASS=81.0,
        # Dill et al. (2024): 20 deg gap between correct and incorrect
        DEPTH_KNEE_ANGLE_WARN=101.0,
        # Dill et al. (2024) E2 - fail bar tightened, warn bar unchanged
        TORSO_LEAN_WARN=45.0,
        TORSO_LEAN_FAIL=55.0,
        # above the derived +/-0.15 band for a normalised length
        HEEL_LIFT_THRESHOLD=0.16,
        # Simoes et al. (2024) beginner allowance
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


LITERATURE_CONFIG = literature_config()
