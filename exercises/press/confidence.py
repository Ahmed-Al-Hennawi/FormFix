"""Where the press's reliability bands sit. exercises/common/confidence.py
explains the levels; this only supplies the boundaries."""

from __future__ import annotations

from analysis.models import METRIC_PRESS_ALIGNMENT, METRIC_PRESS_SYMMETRY
from exercises.common.confidence import ReliabilityBands

from .config import PressConfig

# Same banding as the other two: how much evidence stands behind a measurement
# doesn't depend on the exercise.
DEFAULT_BANDS = ReliabilityBands()

# The frontal-plane measurements. For these a nearly side-on diagonal camera
# is the bad case, so their view support is the complement of the side-view
# confidence (see common.confidence.view_support).
FRONTAL_PLANE_METRICS: frozenset[str] = frozenset({METRIC_PRESS_SYMMETRY, METRIC_PRESS_ALIGNMENT})


def bands_for(config: PressConfig) -> ReliabilityBands:
    """The banding used for press measurements."""
    del config  # nothing press-specific yet; the argument is here for tuning
    return DEFAULT_BANDS


def is_frontal_plane(metric: str) -> bool:
    """True for metrics measured across the body rather than along it."""
    return metric in FRONTAL_PLANE_METRICS
