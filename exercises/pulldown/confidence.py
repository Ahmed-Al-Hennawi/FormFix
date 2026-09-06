"""Where the pulldown's reliability bands sit. exercises/common/confidence.py
explains the levels; this only supplies the boundaries."""

from __future__ import annotations

from exercises.common.confidence import ReliabilityBands

from .config import PulldownConfig

# Same banding as the squat: how much evidence stands behind a measurement
# doesn't depend on the exercise.
DEFAULT_BANDS = ReliabilityBands()


def bands_for(config: PulldownConfig) -> ReliabilityBands:
    """The banding used for pulldown measurements."""
    del config  # nothing pulldown-specific yet; the argument is here for tuning
    return DEFAULT_BANDS
