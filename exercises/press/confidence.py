"""Reliability bands for the press (levels are explained in common/confidence.py)."""

from __future__ import annotations

from analysis.models import METRIC_PRESS_ALIGNMENT, METRIC_PRESS_SYMMETRY
from exercises.common.confidence import ReliabilityBands

from .config import PressConfig

# same bands as the other exercises - evidence doesn't depend on the exercise
DEFAULT_BANDS = ReliabilityBands()

# front-on measurements, so a nearly side-on camera is the bad case here
# (see common.confidence.view_support)
FRONTAL_PLANE_METRICS: frozenset[str] = frozenset({METRIC_PRESS_SYMMETRY, METRIC_PRESS_ALIGNMENT})


def bands_for(config: PressConfig) -> ReliabilityBands:
    del config  # nothing press-specific yet; the argument is here for tuning
    return DEFAULT_BANDS


def is_frontal_plane(metric: str) -> bool:
    return metric in FRONTAL_PLANE_METRICS
