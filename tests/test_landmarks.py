"""
The squat's landmark subset.

A wrong index here doesn't break anything visibly - the pipeline carries on and
produces plausible angles measured on the wrong joint.
"""

from __future__ import annotations

import pytest

from analysis.models import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    METRIC_KNEE_SYMMETRY,
    METRIC_TORSO_LEAN,
    NUM_LANDMARKS,
    SIDE_LANDMARKS,
)
from exercises.squat.landmarks import (
    BILATERAL_METRICS,
    CORE_LANDMARK_NAMES,
    METRIC_REQUIREMENTS,
    OPPOSITE_SIDE,
    SIDE_CHAINS,
    SQUAT_LANDMARKS,
    landmark_ids,
    required_ids,
)


class TestSubset:
    def test_official_mediapipe_indices(self):
        # Documented ids from MediaPipe's 33-landmark topology.
        assert SQUAT_LANDMARKS["left_shoulder"] == 11
        assert SQUAT_LANDMARKS["right_shoulder"] == 12
        assert SQUAT_LANDMARKS["left_hip"] == 23
        assert SQUAT_LANDMARKS["right_hip"] == 24
        assert SQUAT_LANDMARKS["left_knee"] == 25
        assert SQUAT_LANDMARKS["right_knee"] == 26
        assert SQUAT_LANDMARKS["left_ankle"] == 27
        assert SQUAT_LANDMARKS["right_ankle"] == 28
        assert SQUAT_LANDMARKS["left_heel"] == 29
        assert SQUAT_LANDMARKS["right_heel"] == 30
        assert SQUAT_LANDMARKS["left_foot_index"] == 31
        assert SQUAT_LANDMARKS["right_foot_index"] == 32

    def test_subset_is_a_subset(self):
        assert len(SQUAT_LANDMARKS) == 12
        assert all(0 <= i < NUM_LANDMARKS for i in SQUAT_LANDMARKS.values())
        assert len(set(SQUAT_LANDMARKS.values())) == 12

    def test_core_names_resolve(self):
        assert all(name in SQUAT_LANDMARKS for name in CORE_LANDMARK_NAMES)

    def test_chains_match_the_shared_side_tuples(self):
        for side, chain in SIDE_CHAINS.items():
            assert chain.all_ids == SIDE_LANDMARKS[side]
        assert OPPOSITE_SIDE == {"left": "right", "right": "left"}

    def test_chain_groups(self):
        left = SIDE_CHAINS["left"]
        assert left.core == (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        assert left.foot == (29, 31)


class TestMetricRequirements:
    def test_each_metric_declares_what_it_needs(self):
        assert METRIC_REQUIREMENTS[METRIC_TORSO_LEAN] == ("shoulder", "hip")
        assert "knee" in METRIC_REQUIREMENTS[METRIC_KNEE_SYMMETRY]

    def test_bilateral_metric_requires_both_sides(self):
        ids = required_ids(METRIC_KNEE_SYMMETRY, "left")
        assert METRIC_KNEE_SYMMETRY in BILATERAL_METRICS
        assert set(ids) >= {LEFT_KNEE, 26}  # both knees

    def test_unilateral_metric_uses_the_analysed_side_only(self):
        ids = required_ids(METRIC_TORSO_LEAN, "right")
        assert set(ids) == {12, 24}

    def test_unknown_role_fails_loudly(self):
        with pytest.raises((KeyError, AttributeError)):
            landmark_ids("left", ("elbwo",))

    def test_unknown_side_fails_loudly(self):
        with pytest.raises(KeyError):
            landmark_ids("middle", ("hip",))
