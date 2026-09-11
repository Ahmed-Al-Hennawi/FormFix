"""
Fake front-view shoulder press for testing: a 33-landmark pose track from a
simple 2D stick model, replacing only the detector's output. Faults you can set:

    top_elbow / bottom_elbow      the range each arm actually covers
    right_lag                     right arm trailing the left, as a fraction
                                  of the movement (arm asymmetry)
    left_drift / right_drift      wrist drifting out from the elbow, in
                                  shoulder widths (alignment fault)
    noise                         landmark jitter, normalised units
    visibility                    per-group MediaPipe visibility scores
    view_compression              how side-on the camera is

Horizontal distances are multiplied by ASPECT because MediaPipe normalises x
and y separately - without it a 90 degree elbow wouldn't measure 90 in pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from analysis.models import NUM_LANDMARKS, FramePoseData

WIDTH, HEIGHT = 720, 1280
FPS = 30.0
ASPECT = HEIGHT / WIDTH

# body sizes, normalised to frame height
SHOULDER_HALF_WIDTH = 0.085  # half the shoulder width, before aspect scaling
TORSO = 0.24
UPPER_ARM = 0.13
FOREARM = 0.12
HEAD = 0.06

# upper arm angle from vertical at the bottom and top of the press
UPPER_ARM_BOTTOM_DEG = 78.0
UPPER_ARM_TOP_DEG = 12.0


@dataclass
class PressSpec:
    """One synthetic shoulder-press recording's shape."""

    ready_seconds: float = 1.0
    rep_seconds: float = 2.4
    reps: int = 3
    # elbow angle at the shoulders and overhead
    bottom_elbow: float = 85.0
    top_elbow: float = 168.0
    # per-arm overrides (None = use the value above)
    right_bottom_elbow: float | None = None
    right_top_elbow: float | None = None
    # right arm lag as a fraction of the movement (0.25 is clearly uneven)
    right_lag: float = 0.0
    # outward wrist drift at the top, in shoulder widths
    left_drift: float = 0.0
    right_drift: float = 0.0
    noise: float = 0.0
    visibility: float = 0.95
    wrist_visibility: float = 0.95
    # 1.0 = straight front view, 0.0 = fully side-on
    view_compression: float = 1.0


def press_profile(spec: PressSpec) -> np.ndarray:
    """Per-frame press height d in [0, 1]: 0 at the shoulders, 1 overhead."""
    chunks = [np.zeros(int(spec.ready_seconds * FPS))]
    for i in range(spec.reps):
        half = int(spec.rep_seconds * FPS / 2)
        chunks.append(np.linspace(0.0, 1.0, half))
        chunks.append(np.linspace(1.0, 0.0, half))
        if i < spec.reps - 1:
            chunks.append(np.zeros(int(0.4 * FPS)))
    chunks.append(np.zeros(int(spec.ready_seconds * FPS)))
    return np.concatenate(chunks)


def _arm(
    shoulder: tuple[float, float],
    lateral: int,
    d: float,
    bottom_elbow: float,
    top_elbow: float,
    drift: float,
    half_width: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    One arm's elbow and wrist at press height d. lateral is +1 for the arm on the
    +x side. drift adds outward wrist movement towards the top (alignment fault).
    """
    theta = math.radians(UPPER_ARM_BOTTOM_DEG + (UPPER_ARM_TOP_DEG - UPPER_ARM_BOTTOM_DEG) * d)
    elbow_angle = bottom_elbow + (top_elbow - bottom_elbow) * d
    bend = math.radians(180.0 - elbow_angle)
    phi = theta - bend

    elbow = (
        shoulder[0] + lateral * ASPECT * UPPER_ARM * math.sin(theta),
        shoulder[1] - UPPER_ARM * math.cos(theta),
    )
    wrist = (
        elbow[0]
        + lateral * ASPECT * FOREARM * math.sin(phi)
        + lateral * ASPECT * drift * 2 * half_width * d,
        elbow[1] - FOREARM * math.cos(phi),
    )
    return elbow, wrist


def skeleton_at(d: float, spec: PressSpec) -> dict[int, tuple[float, float]]:
    """Landmark positions (normalised x, y) at press height d."""
    centre_x = 0.5
    shoulder_y = 0.42
    half = SHOULDER_HALF_WIDTH * spec.view_compression

    # MediaPipe's "left" is the person's left, so on the image's right when
    # they face the camera
    left_shoulder = (centre_x + ASPECT * half, shoulder_y)
    right_shoulder = (centre_x - ASPECT * half, shoulder_y)

    # right arm can lag behind (asymmetry fault)
    d_left = d
    d_right = max(0.0, d - spec.right_lag)

    left_elbow, left_wrist = _arm(
        left_shoulder, +1, d_left, spec.bottom_elbow, spec.top_elbow, spec.left_drift, half
    )
    right_elbow, right_wrist = _arm(
        right_shoulder,
        -1,
        d_right,
        spec.right_bottom_elbow if spec.right_bottom_elbow is not None else spec.bottom_elbow,
        spec.right_top_elbow if spec.right_top_elbow is not None else spec.top_elbow,
        spec.right_drift,
        half,
    )

    left_hip = (centre_x + ASPECT * half * 0.8, shoulder_y + TORSO)
    right_hip = (centre_x - ASPECT * half * 0.8, shoulder_y + TORSO)
    nose = (centre_x, shoulder_y - HEAD)
    left_ear = (centre_x + ASPECT * 0.02, shoulder_y - HEAD - 0.005)
    right_ear = (centre_x - ASPECT * 0.02, shoulder_y - HEAD - 0.005)
    left_knee = (left_hip[0], left_hip[1] + 0.16)
    right_knee = (right_hip[0], right_hip[1] + 0.16)
    left_ankle = (left_hip[0], left_hip[1] + 0.30)
    right_ankle = (right_hip[0], right_hip[1] + 0.30)

    return {
        0: nose,
        7: left_ear,
        8: right_ear,
        11: left_shoulder,
        12: right_shoulder,
        13: left_elbow,
        14: right_elbow,
        15: left_wrist,
        16: right_wrist,
        23: left_hip,
        24: right_hip,
        25: left_knee,
        26: right_knee,
        27: left_ankle,
        28: right_ankle,
    }


WRIST_LANDMARKS = (15, 16)
ARM_LANDMARKS = (13, 14, 15, 16)


def make_pose_data(spec: PressSpec, seed: int = 0) -> FramePoseData:
    """What the detector would have returned for this spec."""
    profile = press_profile(spec)
    n = len(profile)
    rng = np.random.default_rng(seed)

    xy = np.full((n, NUM_LANDMARKS, 2), np.nan)
    visibility = np.zeros((n, NUM_LANDMARKS))
    pose_found = np.ones(n, dtype=bool)
    n_poses = np.ones(n, dtype=np.int16)

    for f, d in enumerate(profile):
        for lm, (x, y) in skeleton_at(float(d), spec).items():
            jitter = rng.normal(0, spec.noise, 2) if spec.noise else (0.0, 0.0)
            xy[f, lm] = (x + jitter[0], y + jitter[1])
            visibility[f, lm] = spec.wrist_visibility if lm in WRIST_LANDMARKS else spec.visibility

    return FramePoseData(
        xy_raw=xy,
        xy=xy.copy(),
        visibility=visibility,
        valid=np.isfinite(xy[:, :, 0]),
        pose_found=pose_found,
        n_poses=n_poses,
        timestamps=np.arange(n) / FPS,
    )


def write_plain_video(path, n_frames: int) -> None:
    """A grey video matching the synthetic track, for annotation tests."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    frame = np.full((HEIGHT, WIDTH, 3), 34, dtype=np.uint8)
    try:
        for _ in range(n_frames):
            writer.write(frame)
    finally:
        writer.release()
