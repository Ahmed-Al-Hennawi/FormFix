"""
Tests for far-side occlusion in side-view recordings. MediaPipe gave confident
but drifting positions for limbs it couldn't see, which made the squat overlay
look unstable. Instead of trusting the detector, the fix checks whether the
limb is lined up behind its twin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analysis.annotation import (
    FAR_OCCLUSION_TORSO_FRACTION,
    FULL_DRAW_VISIBILITY,
    MIN_DRAW_VISIBILITY,
    MIN_FAR_SIDE_VISIBILITY,
    _DrawPolicy,
)
from analysis.models import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_KNEE,
)

WIDTH = 1080.0
TORSO_PX = 300.0
OCCLUSION_PX = TORSO_PX * FAR_OCCLUSION_TORSO_FRACTION  # 36 px


def policy(**kw) -> _DrawPolicy:
    base = dict(
        trust_visibility=True,
        far_limbs=frozenset({LEFT_KNEE, LEFT_ANKLE, LEFT_ELBOW}),
        max_jump_px=0.0,
        occlusion_px=OCCLUSION_PX,
    )
    base.update(kw)
    return _DrawPolicy(**base)


def track(pairs: dict[int, float], n: int = 33) -> tuple[np.ndarray, np.ndarray]:
    """Build (xy, valid) with the given normalised x positions."""
    xy = np.full((n, 2), np.nan)
    valid = np.zeros(n, dtype=bool)
    for landmark, x in pairs.items():
        xy[landmark] = (x, 0.5)
        valid[landmark] = True
    return xy, valid


# --- The case from the recording ---


def test_far_knee_behind_near_knee_is_occluded():
    # legs together side-on, so the knees are within a few pixels
    xy, valid = track({LEFT_KNEE: 0.500, RIGHT_KNEE: 0.505})
    assert policy().is_occluded(LEFT_KNEE, xy, valid, WIDTH)


def test_far_knee_driven_forward_is_visible():
    # bottom of a squat with legs apart - both knees really visible
    xy, valid = track({LEFT_KNEE: 0.40, RIGHT_KNEE: 0.55})
    assert not policy().is_occluded(LEFT_KNEE, xy, valid, WIDTH)


def test_threshold_is_horizontal_only():
    """Vertical separation is real movement, so only the horizontal gap counts."""
    xy, valid = track({LEFT_KNEE: 0.500, RIGHT_KNEE: 0.505})
    xy[LEFT_KNEE][1] = 0.20  # far above its twin
    xy[RIGHT_KNEE][1] = 0.80
    assert policy().is_occluded(LEFT_KNEE, xy, valid, WIDTH)


@pytest.mark.parametrize(
    "gap_px,expected",
    [(0.0, True), (10.0, True), (35.0, True), (37.0, False), (60.0, False), (200.0, False)],
)
def test_threshold_boundary(gap_px, expected):
    # 36 px is the threshold itself, float rounding makes the exact boundary
    # pointless to test
    xy, valid = track({LEFT_KNEE: 0.5, RIGHT_KNEE: 0.5 + gap_px / WIDTH})
    assert policy().is_occluded(LEFT_KNEE, xy, valid, WIDTH) is expected


# --- Failing safe - none of these should hide a limb on a guess ---


def test_disabled_when_no_occlusion_distance():
    xy, valid = track({LEFT_KNEE: 0.500, RIGHT_KNEE: 0.505})
    assert not policy(occlusion_px=0.0).is_occluded(LEFT_KNEE, xy, valid, WIDTH)


def test_landmark_without_a_twin_is_never_occluded():
    # shoulders and hips give the torso its width, never hidden
    xy, valid = track({LEFT_SHOULDER: 0.5, LEFT_HIP: 0.5})
    assert not policy().is_occluded(LEFT_SHOULDER, xy, valid, WIDTH)
    assert not policy().is_occluded(LEFT_HIP, xy, valid, WIDTH)


def test_untracked_twin_is_not_occlusion():
    xy, valid = track({LEFT_KNEE: 0.5})  # right knee absent
    assert not policy().is_occluded(LEFT_KNEE, xy, valid, WIDTH)


def test_non_finite_coordinates_are_not_occlusion():
    xy, valid = track({LEFT_KNEE: 0.5, RIGHT_KNEE: 0.5})
    xy[RIGHT_KNEE][0] = np.nan
    valid[RIGHT_KNEE] = True
    assert not policy().is_occluded(LEFT_KNEE, xy, valid, WIDTH)


def test_short_valid_array_does_not_raise():
    xy, valid = track({LEFT_KNEE: 0.5, RIGHT_KNEE: 0.5}, n=33)
    assert policy().is_occluded(LEFT_KNEE, xy, valid[:20], WIDTH) is False


# --- The constants have to be coherent ---


def test_ramp_is_not_dead_code_for_far_limbs():
    """Far-side floor has to stay below FULL_DRAW_VISIBILITY."""
    assert FULL_DRAW_VISIBILITY > MIN_FAR_SIDE_VISIBILITY


def test_far_floor_is_stricter_than_the_general_floor():
    assert MIN_FAR_SIDE_VISIBILITY > MIN_DRAW_VISIBILITY


# --- Sources without confidence values ---


def test_occlusion_is_skipped_when_confidence_is_uninformative():
    """
    Synthetic data has no confidence and often puts both sides at the same x,
    which looks like occlusion - so the check is skipped instead of hiding half
    the skeleton.
    """
    import numpy as np

    from analysis.annotation import _drawable_points
    from analysis.models import FramePoseData, VideoMetadata

    n = 33
    xy = np.full((1, n, 2), 0.5)
    visibility = np.zeros((1, n))  # nothing reported
    valid = np.ones((1, n), dtype=bool)
    pose = FramePoseData(
        xy_raw=xy.copy(),
        xy=xy,
        visibility=visibility,
        valid=valid,
        pose_found=np.ones(1, dtype=bool),
        n_poses=np.ones(1, dtype=int),
        timestamps=np.zeros(1),
    )
    video = VideoMetadata(
        path=Path("fixture.mp4"), width=1080, height=1920, fps=30.0, frame_count=1, duration=1 / 30
    )
    pol = policy(trust_visibility=False)
    points, _ = _drawable_points(pose, 0, video, pol)
    # both knees at x=0.5, but nothing is hidden
    assert LEFT_KNEE in points
    assert RIGHT_KNEE in points
