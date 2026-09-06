"""
Synthetic side-view squat.

A 33-landmark pose track from a 2D linkage: fixed ankle, tilting shank,
rotating thigh, leaning torso. Only the detector's output is replaced. Faults
you can dial in:

    depth        how deep each rep goes (1.0 = full depth)
    torso_gain   degrees of forward torso lean at full depth
    heel_lift    normalised heel rise around the bottom
    end_depth    residual flexion at the end of the last rep (extension fault)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from analysis.models import NUM_LANDMARKS, FramePoseData

WIDTH, HEIGHT = 720, 1280
FPS = 30.0

# Body segment lengths, normalised to frame height.
SHANK = 0.16
THIGH = 0.16
TORSO = 0.24
HEAD = 0.06
FOOT = 0.055


@dataclass
class SyntheticSpec:
    """One synthetic recording's shape."""

    standing_seconds: float = 1.2
    rep_seconds: float = 2.4
    reps: int = 3
    depth: float = 1.0
    torso_gain_deg: float = 35.0
    heel_lift: float = 0.0
    end_depth: float = 0.0
    facing: int = 1  # +1 faces right, -1 faces left
    noise: float = 0.0
    visibility: float = 0.95
    heel_visibility: float = 0.95
    # Horizontal offset between the near and far sides of the body, in normalised
    # x, which is what the camera-angle heuristic measures. On this 720x1280 frame
    # an offset of s gives a frontality ratio of about s * 2.34: ~0.26
    # reads as diagonal, ~0.45 as front-on.
    # normalised x, which is what the camera-angle heuristic measures. Small
    side_offset: float = 0.012


def depth_profile(spec: SyntheticSpec) -> np.ndarray:
    """Per-frame squat depth d in [0, 1]."""
    stand = np.zeros(int(spec.standing_seconds * FPS))
    chunks = [stand]
    for i in range(spec.reps):
        half = int(spec.rep_seconds * FPS / 2)
        down = np.linspace(0.0, spec.depth, half)
        up_target = spec.end_depth if i == spec.reps - 1 else 0.0
        up = np.linspace(spec.depth, up_target, half)
        chunks.extend([down, up])
        if i < spec.reps - 1:
            chunks.append(np.zeros(int(0.5 * FPS)))
    chunks.append(np.full(int(spec.standing_seconds * FPS), spec.end_depth))
    return np.concatenate(chunks)


def skeleton_at(d: float, spec: SyntheticSpec) -> dict[int, tuple[float, float]]:
    """Landmark positions (normalised x, y) for squat depth d."""
    s = spec.facing
    ankle = (0.5, 0.86)

    shank_tilt = math.radians(25.0 * d)
    knee = (ankle[0] + s * SHANK * math.sin(shank_tilt), ankle[1] - SHANK * math.cos(shank_tilt))

    thigh_tilt = math.radians(100.0 * d)
    hip = (knee[0] - s * THIGH * math.sin(thigh_tilt), knee[1] - THIGH * math.cos(thigh_tilt))

    torso_tilt = math.radians(spec.torso_gain_deg * d)
    shoulder = (hip[0] + s * TORSO * math.sin(torso_tilt), hip[1] - TORSO * math.cos(torso_tilt))

    nose = (shoulder[0] + s * 0.02, shoulder[1] - HEAD)
    heel_y = ankle[1] + 0.015 - spec.heel_lift * SHANK * _bottom_weight(d)
    heel = (ankle[0] - s * 0.02, heel_y)
    foot = (ankle[0] + s * FOOT, ankle[1] + 0.02)
    elbow = (shoulder[0] + s * 0.05, shoulder[1] + 0.10)
    wrist = (elbow[0] + s * 0.05, elbow[1] + 0.08)

    # Side-on, the far side nearly overlaps the near side. A bigger offset
    # stands in for a diagonal or front-on camera.
    off = spec.side_offset

    def far(p):
        return (p[0] - s * off, p[1] + 0.004)

    return {
        0: nose,
        11: shoulder,
        12: far(shoulder),
        13: elbow,
        14: far(elbow),
        15: wrist,
        16: far(wrist),
        23: hip,
        24: far(hip),
        25: knee,
        26: far(knee),
        27: ankle,
        28: far(ankle),
        29: heel,
        30: far(heel),
        31: foot,
        32: far(foot),
    }


def _bottom_weight(d: float) -> float:
    """Heel lift only appears near the bottom of the movement."""
    return max(0.0, (d - 0.6) / 0.4)


def make_pose_data(spec: SyntheticSpec, seed: int = 0) -> FramePoseData:
    """The FramePoseData the detector would have returned for this spec."""
    profile = depth_profile(spec)
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
            vis = spec.heel_visibility if lm in (29, 30, 31, 32) else spec.visibility
            visibility[f, lm] = vis

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
