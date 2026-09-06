"""
Synthetic seated lat pulldown.

A 33-landmark pose track from a 2D linkage: fixed seat, a trunk that can lean
back, an upper arm rotating from overhead to the side, a forearm hinged at a
controlled elbow angle. Only the detector's output is replaced, so every
number the rules see came through the real code paths. Faults you can dial in:

    top_elbow / bottom_elbow  the range each repetition covers
    torso_gain_deg            backward trunk lean during the pull
    noise                     landmark jitter, normalised units
    visibility                per-group MediaPipe visibility scores
    side_offset               how far across the body the camera looks
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from analysis.models import NUM_LANDMARKS, FramePoseData

WIDTH, HEIGHT = 720, 1280
FPS = 30.0

# Body segment lengths, normalised to frame height. MediaPipe normalises x to
# the frame width and y to the frame height separately, so every horizontal
# displacement below is multiplied by ASPECT to undo that - without it a
# 90-degree elbow doesn't measure 90 once the pipeline converts back to pixels.
# MediaPipe normalises x to the frame width and y to the frame height
# separately, so equal normalised dx and dy is not 45 degrees in pixels, it is
ASPECT = HEIGHT / WIDTH

TORSO = 0.24
UPPER_ARM = 0.13
FOREARM = 0.12
HEAD = 0.06

# Upper-arm rotation away from straight overhead, degrees, at each end of the
# movement: nearly overhead at the top, beside the trunk when contracted.
UPPER_ARM_TOP_DEG = 15.0
UPPER_ARM_BOTTOM_DEG = 160.0


@dataclass
class PulldownSpec:
    """One synthetic lat-pulldown recording's shape."""

    top_seconds: float = 1.0
    rep_seconds: float = 2.4
    reps: int = 3
    # Elbow angle at the extended and the contracted position, degrees.
    top_elbow: float = 172.0
    bottom_elbow: float = 80.0
    # The seated trunk's own recline at the top, degrees from vertical. This
    # is normal posture, not a fault.
    torso_base_deg: float = 12.0
    # Extra backward lean added at the contracted position, degrees. This is
    # the fault the torso rule looks for.
    torso_gain_deg: float = 4.0
    # +1 faces right (+x), -1 faces left.
    facing: int = 1
    noise: float = 0.0
    visibility: float = 0.95
    # Shoulders and hips. Lowering it hides the trunk reference while the
    # arms stay measurable, which is the common seated case.
    torso_visibility: float = 0.95
    # Hips alone, which a bench or a thigh routinely occludes.
    hip_visibility: float | None = None
    head_visibility: float = 0.9
    # Horizontal offset between the near and far sides of the body, in normalised
    # x. On this 720x1280 frame an offset of s gives a frontality ratio of
    # roughly s * 2.34: ~0.012 side-on, ~0.20 diagonal, ~0.45 front-on.
    # normalised x. Small is a side view, large means the camera is looking
    side_offset: float = 0.012


def pull_profile(spec: PulldownSpec) -> np.ndarray:
    """Per-frame pull depth d in [0, 1]: 0 extended, 1 contracted."""
    chunks = [np.zeros(int(spec.top_seconds * FPS))]
    for i in range(spec.reps):
        half = int(spec.rep_seconds * FPS / 2)
        chunks.append(np.linspace(0.0, 1.0, half))
        chunks.append(np.linspace(1.0, 0.0, half))
        if i < spec.reps - 1:
            chunks.append(np.zeros(int(0.4 * FPS)))
    chunks.append(np.zeros(int(spec.top_seconds * FPS)))
    return np.concatenate(chunks)


def skeleton_at(d: float, spec: PulldownSpec) -> dict[int, tuple[float, float]]:
    """Landmark positions (normalised x, y) at pull depth d."""
    fx = spec.facing
    hip = (0.5, 0.72)

    # A backward lean moves the shoulders away from the facing direction,
    # which is what the analysis reads as posterior lean.
    lean = math.radians(spec.torso_base_deg + spec.torso_gain_deg * d)
    shoulder = (
        hip[0] - fx * ASPECT * TORSO * math.sin(lean),
        hip[1] - TORSO * math.cos(lean),
    )

    # Upper arm rotates from overhead down to the side, forearm hinges at the
    # requested elbow angle.
    theta = math.radians(UPPER_ARM_TOP_DEG + (UPPER_ARM_BOTTOM_DEG - UPPER_ARM_TOP_DEG) * d)
    elbow_angle = spec.top_elbow + (spec.bottom_elbow - spec.top_elbow) * d
    bend = math.radians(180.0 - elbow_angle)
    phi = theta - bend

    elbow = (
        shoulder[0] + fx * ASPECT * UPPER_ARM * math.sin(theta),
        shoulder[1] - UPPER_ARM * math.cos(theta),
    )
    wrist = (
        elbow[0] + fx * ASPECT * FOREARM * math.sin(phi),
        elbow[1] - FOREARM * math.cos(phi),
    )

    nose = (shoulder[0] + fx * ASPECT * 0.025, shoulder[1] - HEAD)
    ear = (shoulder[0] + fx * ASPECT * 0.016, shoulder[1] - HEAD - 0.005)
    knee = (hip[0] + fx * ASPECT * 0.07, hip[1] + 0.02)
    ankle = (knee[0] + fx * ASPECT * 0.012, knee[1] + 0.14)

    # Both sides sit symmetrically about the mid-sagittal plane, so a larger
    # offset rotates the subject instead of sliding them out of frame.
    half = spec.side_offset / 2.0

    def near(p):
        return (p[0] + fx * half, p[1])

    def far(p):
        return (p[0] - fx * half, p[1] + 0.004)

    return {
        0: nose,
        7: near(ear),
        8: far(ear),
        11: near(shoulder),
        12: far(shoulder),
        13: near(elbow),
        14: far(elbow),
        15: near(wrist),
        16: far(wrist),
        23: near(hip),
        24: far(hip),
        25: near(knee),
        26: far(knee),
        27: near(ankle),
        28: far(ankle),
    }


# Landmark groups, so a test can hide exactly one of them.
HEAD_LANDMARKS = (0, 7, 8)
TORSO_LANDMARKS = (11, 12, 23, 24)
HIP_LANDMARKS = (23, 24)
ARM_LANDMARKS = (13, 14, 15, 16)


def make_pose_data(spec: PulldownSpec, seed: int = 0) -> FramePoseData:
    """The FramePoseData the detector would have returned for this spec."""
    profile = pull_profile(spec)
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
            if lm in HEAD_LANDMARKS:
                vis = spec.head_visibility
            elif lm in HIP_LANDMARKS and spec.hip_visibility is not None:
                vis = spec.hip_visibility
            elif lm in TORSO_LANDMARKS:
                vis = spec.torso_visibility
            else:
                vis = spec.visibility
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
