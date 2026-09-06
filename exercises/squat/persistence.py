"""Persistence filtering for the squat. The implementation is in
exercises/common/persistence.py - this just re-exports it."""

from __future__ import annotations

from exercises.common.persistence import (
    PersistenceEvidence,
    assess,
    sustained_extreme,
)

__all__ = ["PersistenceEvidence", "assess", "sustained_extreme"]
