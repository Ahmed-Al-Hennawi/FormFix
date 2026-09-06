"""
Body-relative normalisation. Scales and shifts a synthetic body around the
frame and checks the normalised numbers stay put, which is the whole point of
normalising in the first place.
"""

from __future__ import annotations

import math

from analysis.normalisation import body_scale, hip_centred, normalise, torso_reference


def body(scale: float = 1.0, offset=(0.0, 0.0)):
    """A rough upright body in pixels, at a given size and frame position."""
    ox, oy = offset
    return {
        "left_shoulder": (ox + 100 * scale, oy + 100 * scale),
        "right_shoulder": (ox + 140 * scale, oy + 100 * scale),
        "left_hip": (ox + 105 * scale, oy + 300 * scale),
        "right_hip": (ox + 135 * scale, oy + 300 * scale),
        "knee": (ox + 120 * scale, oy + 420 * scale),
        "ankle": (ox + 120 * scale, oy + 540 * scale),
    }


class TestBodyScale:
    def test_prefers_torso_length(self):
        b = body()
        mid_shoulder, mid_hip = torso_reference(
            b["left_shoulder"], b["right_shoulder"], b["left_hip"], b["right_hip"]
        )
        assert math.isclose(body_scale(mid_shoulder, mid_hip), 200.0, rel_tol=1e-6)

    def test_falls_back_to_hip_width_then_lower_leg(self):
        b = body()
        hips_only = body_scale(None, None, left_hip=b["left_hip"], right_hip=b["right_hip"])
        assert math.isclose(hips_only, 30.0, rel_tol=1e-6)

        leg_only = body_scale(None, None, knee=b["knee"], ankle=b["ankle"])
        assert math.isclose(leg_only, 120.0, rel_tol=1e-6)

    def test_no_reference_is_nan_not_zero(self):
        assert math.isnan(body_scale(None, None))

    def test_degenerate_segment_is_rejected(self):
        assert math.isnan(body_scale((10.0, 10.0), (10.0, 10.0)))


class TestScaleInvariance:
    def test_normalised_distance_is_independent_of_camera_distance(self):
        near, far = body(scale=2.0), body(scale=0.5)
        results = []
        for b in (near, far):
            mid_shoulder, mid_hip = torso_reference(
                b["left_shoulder"], b["right_shoulder"], b["left_hip"], b["right_hip"]
            )
            scale = body_scale(mid_shoulder, mid_hip)
            stance = abs(b["left_hip"][0] - b["right_hip"][0])
            results.append(normalise(stance, scale))
        assert math.isclose(results[0], results[1], rel_tol=1e-9)

    def test_hip_centring_removes_position_in_frame(self):
        left, right = body(offset=(0, 0)), body(offset=(700, 40))
        centred = []
        for b in (left, right):
            mid_shoulder, mid_hip = torso_reference(
                b["left_shoulder"], b["right_shoulder"], b["left_hip"], b["right_hip"]
            )
            scale = body_scale(mid_shoulder, mid_hip)
            centred.append(hip_centred(b["knee"], mid_hip, scale))
        assert centred[0] is not None
        assert all(
            math.isclose(a, b, abs_tol=1e-9) for a, b in zip(centred[0], centred[1], strict=True)
        )


class TestGuards:
    def test_normalise_rejects_a_zero_scale(self):
        assert math.isnan(normalise(10.0, 0.0))

    def test_normalise_propagates_missing_values(self):
        assert math.isnan(normalise(float("nan"), 100.0))

    def test_hip_centred_returns_none_when_unusable(self):
        assert hip_centred(None, (0.0, 0.0), 100.0) is None
        assert hip_centred((1.0, 1.0), None, 100.0) is None
        assert hip_centred((1.0, 1.0), (0.0, 0.0), float("nan")) is None
