"""Reliability bands for the pulldown (levels are explained in common/confidence.py)."""

from __future__ import annotations

from exercises.common.confidence import ReliabilityBands

from .config import PulldownConfig

# same bands as the squat - evidence doesn't depend on the exercise
DEFAULT_BANDS = ReliabilityBands()


def bands_for(config: PulldownConfig) -> ReliabilityBands:
    del config  # nothing pulldown-specific yet; the argument is here for tuning
    return DEFAULT_BANDS
