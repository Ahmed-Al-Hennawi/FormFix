"""
Did a rule violation last long enough to be real? One jumpy landmark can push
a torso angle five degrees past a threshold while the lifter stands still.

Both conditions have to be met: longest_run ignores one or two frame spikes,
and violation_ratio catches a fault that flickers through most of the phase.
Unmeasurable frames don't count either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PersistenceEvidence:
    """How much of a phase actually violated a rule."""

    measurable_frames: int
    violating_frames: int
    longest_run: int
    # violating_frames / measurable_frames
    ratio: float
    # frame (in the full series) with the worst value inside the longest run
    peak_index: int
    peak_value: float

    @property
    def has_evidence(self) -> bool:
        return self.measurable_frames > 0

    def triggers(self, min_frames: int, min_ratio: float) -> bool:
        """Both conditions have to pass."""
        if not self.has_evidence:
            return False
        return self.longest_run >= max(min_frames, 1) and self.ratio >= min_ratio


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end_exclusive), ...] for each unbroken True run."""
    runs: list[tuple[int, int]] = []
    start = -1
    for i, flag in enumerate(mask):
        if flag and start < 0:
            start = i
        elif not flag and start >= 0:
            runs.append((start, i))
            start = -1
    if start >= 0:
        runs.append((start, len(mask)))
    return runs


def assess(
    values: np.ndarray,
    violates: np.ndarray,
    offset: int = 0,
    prefer_max: bool = True,
) -> PersistenceEvidence:
    """
    How persistently the rule was broken over a phase.

        values      measured series for the phase, NaN where unmeasurable
        violates    boolean of the same length, True where the rule is broken
        offset      index of values[0] in the full series, so peak_index is real
        prefer_max  True when the largest value is the worst one
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    violating = np.asarray(violates, dtype=bool) & finite

    measurable = int(finite.sum())
    violating_count = int(violating.sum())
    ratio = float(violating_count / measurable) if measurable else 0.0

    runs = _runs(violating)
    longest = max((end - start for start, end in runs), default=0)

    peak_index, peak_value = -1, float("nan")
    if runs:
        start, end = max(runs, key=lambda r: r[1] - r[0])
        chunk = values[start:end]
        local = int(np.nanargmax(chunk) if prefer_max else np.nanargmin(chunk))
        peak_index = offset + start + local
        peak_value = float(chunk[local])

    return PersistenceEvidence(
        measurable_frames=measurable,
        violating_frames=violating_count,
        longest_run=longest,
        ratio=ratio,
        peak_index=peak_index,
        peak_value=peak_value,
    )


def sustained_extreme(values: np.ndarray, window: int, prefer_max: bool = True) -> tuple[float, int]:
    """
    Most extreme value held for `window` frames in a row, and where it peaked.
    This way the number shown is one the lifter actually held.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if window <= 0 or n < window:
        return float("nan"), -1

    best = float("nan")
    best_at = -1
    for i in range(n - window + 1):
        chunk = values[i : i + window]
        if not np.all(np.isfinite(chunk)):
            continue
        held = float(np.min(chunk) if prefer_max else np.max(chunk))
        better = not np.isfinite(best) or (held > best if prefer_max else held < best)
        if better:
            best = held
            best_at = i + int(np.argmax(chunk) if prefer_max else np.argmin(chunk))
    return best, best_at
