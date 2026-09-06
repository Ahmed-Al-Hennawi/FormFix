"""
Landmarks in, numbers out. Measurement only - no thresholds, no verdicts.

    landmarks -> per-frame metrics -> per-rep facts

Per arm: elbow flexion angle, wrist height above the shoulder line, elbow
height, and the wrist-over-elbow offset. All the normalised ones use shoulder
width rather than trunk length, because they live in the plane facing the
camera and trunk length foreshortens the moment the lifter leans.

Conventions: image y grows downward, a positive wrist_height_difference means
the LEFT wrist is higher, and anything unmeasurable is NaN rather than 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from analysis.geometry import calculate_angle, distance
from analysis.models import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    FramePoseData,
    PressRep,
    RepetitionSegmentation,
    VideoMetadata,
)
from analysis.normalisation import normalise
from exercises.common.metrics import (
    effective_fps,
    frames_for,
    plausible,
    safe_max,
    safe_min,
    series,
    span_seconds,
)
from exercises.common.persistence import sustained_extreme
from exercises.common.phases import MovementPhase, RawRep

from .config import PressConfig
from .landmarks import ARM_CHAINS, SHOULDER_LANDMARKS

logger = logging.getLogger(__name__)


@dataclass
class PressFrameMetrics:
    """Everything measured on one frame."""

    frame_index: int
    timestamp: float
    valid: bool

    left_elbow_angle: float = float("nan")
    right_elbow_angle: float = float("nan")
    elbow_angle: float = float("nan")
    elbow_flexion: float = float("nan")
    # |left - right| elbow angle. NaN unless both arms are usable.
    elbow_angle_difference: float = float("nan")

    left_wrist_height: float = float("nan")
    right_wrist_height: float = float("nan")
    # left - right wrist height. POSITIVE means the LEFT wrist is higher.
    wrist_height_difference: float = float("nan")
    left_elbow_height: float = float("nan")
    right_elbow_height: float = float("nan")
    elbow_height_difference: float = float("nan")

    # |wrist_x - elbow_x| / shoulder width, per arm. 0 is a stacked forearm.
    left_alignment_offset: float = float("nan")
    right_alignment_offset: float = float("nan")

    shoulder_width: float = float("nan")
    landmark_confidence: float = 0.0
    both_arms_valid: bool = False


def compute_frame_metrics(
    video: VideoMetadata,
    pose: FramePoseData,
    side: str,
    config: PressConfig,
) -> list[PressFrameMetrics]:
    """Measure every frame, both arms, in pixel space. MediaPipe normalises x and y
    separately, so an angle - or a horizontal offset compared against a vertical
    one - is skewed on any non-square frame."""
    chain = ARM_CHAINS[side]
    metrics: list[PressFrameMetrics] = []

    for f in range(pose.frame_count):
        timestamp = float(pose.timestamps[f])

        def usable(lm: int, frame: int = f) -> bool:
            return bool(
                pose.valid[frame, lm]
                and pose.visibility[frame, lm] >= config.MIN_REQUIRED_LANDMARK_VISIBILITY
            )

        def px(lm: int, frame: int = f):
            if not pose.valid[frame, lm]:
                return None
            return pose.pixel_xy(frame, lm, video.width, video.height)

        arm_valid = all(usable(lm) for lm in chain.arm)
        if not pose.pose_found[f] and not arm_valid:
            metrics.append(PressFrameMetrics(frame_index=f, timestamp=timestamp, valid=False))
            continue

        left_shoulder, right_shoulder = px(LEFT_SHOULDER), px(RIGHT_SHOULDER)
        shoulder_width = distance(left_shoulder, right_shoulder)
        shoulder_line_y = _shoulder_line(left_shoulder, right_shoulder)
        # turning towards side-on collapses the apparent shoulder separation and
        # every normalised value explodes, so the trunk estimate acts as a floor
        trunk_estimate = _trunk_fallback(px)
        if not np.isfinite(shoulder_width) or shoulder_width < 1e-6:
            shoulder_width = trunk_estimate
        elif np.isfinite(trunk_estimate) and shoulder_width < 0.4 * trunk_estimate:
            shoulder_width = 0.4 * trunk_estimate

        angles: dict[str, float] = {}
        heights: dict[str, float] = {}
        elbow_heights: dict[str, float] = {}
        offsets: dict[str, float] = {}
        arm_ok: dict[str, bool] = {}

        for name, other in ARM_CHAINS.items():
            shoulder, elbow, wrist = px(other.shoulder), px(other.elbow), px(other.wrist)
            arm_ok[name] = all(usable(lm) for lm in other.arm)
            angles[name] = plausible(
                calculate_angle(shoulder, elbow, wrist),
                config.PLAUSIBLE_ELBOW_ANGLE_MIN,
                config.PLAUSIBLE_ELBOW_ANGLE_MAX,
            )
            heights[name] = (
                normalise(shoulder_line_y - wrist[1], shoulder_width)
                if wrist is not None and np.isfinite(shoulder_line_y)
                else float("nan")
            )
            elbow_heights[name] = (
                normalise(shoulder_line_y - elbow[1], shoulder_width)
                if elbow is not None and np.isfinite(shoulder_line_y)
                else float("nan")
            )
            offsets[name] = (
                plausible(
                    normalise(abs(wrist[0] - elbow[0]), shoulder_width),
                    0.0,
                    config.ALIGNMENT_OFFSET_MAX_PLAUSIBLE,
                )
                if wrist is not None and elbow is not None
                else float("nan")
            )

        both_arms = arm_ok["left"] and arm_ok["right"]
        usable_angles = [angles[n] for n in ARM_CHAINS if arm_ok[n] and np.isfinite(angles[n])]
        mean_angle = float(np.mean(usable_angles)) if usable_angles else float("nan")
        flexion = 180.0 - mean_angle if np.isfinite(mean_angle) else float("nan")

        angle_difference = (
            plausible(
                abs(angles["left"] - angles["right"]),
                0.0,
                config.ELBOW_DIFFERENCE_MAX_PLAUSIBLE,
            )
            if both_arms and np.isfinite(angles["left"]) and np.isfinite(angles["right"])
            else float("nan")
        )
        height_difference = (
            plausible(
                heights["left"] - heights["right"],
                -config.WRIST_HEIGHT_DIFFERENCE_MAX_PLAUSIBLE,
                config.WRIST_HEIGHT_DIFFERENCE_MAX_PLAUSIBLE,
            )
            if both_arms and np.isfinite(heights["left"]) and np.isfinite(heights["right"])
            else float("nan")
        )
        elbow_height_difference = (
            elbow_heights["left"] - elbow_heights["right"]
            if both_arms and np.isfinite(elbow_heights["left"]) and np.isfinite(elbow_heights["right"])
            else float("nan")
        )

        metrics.append(
            PressFrameMetrics(
                frame_index=f,
                timestamp=timestamp,
                valid=bool(usable_angles) and np.isfinite(mean_angle),
                left_elbow_angle=angles["left"],
                right_elbow_angle=angles["right"],
                elbow_angle=mean_angle,
                elbow_flexion=flexion,
                elbow_angle_difference=angle_difference,
                left_wrist_height=heights["left"],
                right_wrist_height=heights["right"],
                wrist_height_difference=height_difference,
                left_elbow_height=elbow_heights["left"],
                right_elbow_height=elbow_heights["right"],
                elbow_height_difference=elbow_height_difference,
                left_alignment_offset=offsets["left"] if arm_ok["left"] else float("nan"),
                right_alignment_offset=offsets["right"] if arm_ok["right"] else float("nan"),
                shoulder_width=shoulder_width,
                landmark_confidence=float(np.mean([pose.visibility[f, lm] for lm in chain.arm])),
                both_arms_valid=both_arms,
            )
        )
    return metrics


def _shoulder_line(left, right) -> float:
    """Mid-shoulder height in pixels; one shoulder alone will do."""
    points = [p for p in (left, right) if p is not None]
    if not points:
        return float("nan")
    return float(np.mean([p[1] for p in points]))


def _trunk_fallback(px) -> float:
    """Trunk length as a scale reference when the shoulders overlap. Only reached
    near side-on, where the frontal-plane rules are already off, so this just keeps
    the exported numbers finite."""
    shoulders = [p for p in (px(LEFT_SHOULDER), px(RIGHT_SHOULDER)) if p is not None]
    hips = [p for p in (px(LEFT_HIP), px(RIGHT_HIP)) if p is not None]
    if not shoulders or not hips:
        return float("nan")
    mid_shoulder = (
        float(np.mean([p[0] for p in shoulders])),
        float(np.mean([p[1] for p in shoulders])),
    )
    mid_hip = (float(np.mean([p[0] for p in hips])), float(np.mean([p[1] for p in hips])))
    length = distance(mid_shoulder, mid_hip)
    # a trunk is roughly 1.5 shoulder widths, so scale it into the same units
    return length / 1.5 if np.isfinite(length) else float("nan")


def movement_signal(metrics: list[PressFrameMetrics]) -> np.ndarray:
    """The signal the rep state machine runs on: elbow flexion (180 - elbow angle),
    high at the shoulders and near zero overhead. Using flexion is what lets the
    press reuse the same machine as the other two."""
    return series(metrics, "elbow_flexion", range(len(metrics)))


def phase_frames(rep: PressRep, phase: str) -> range:
    """Frame range a rule should read for phase on rep."""
    return frames_for(rep.segmentation, phase)


# --- Per-repetition facts ---


def bottom_windows(raw_reps: list[RawRep], index: int, fps: float, frame_count: int) -> list[range]:
    """
    Frames where the dumbbells are genuinely back at the shoulders. Not the rep's
    own first frame - the state machine only commits once the press has started, so
    reading the bottom angle there over-reports depth and hides a shallow press.
    """
    raw = raw_reps[index]
    span = max(int(round(1.0 * fps)), 1)
    pre_limit = raw_reps[index - 1].end_frame if index > 0 else 0
    pre_start = max(pre_limit, raw.start_frame - span, 0)
    post_limit = raw_reps[index + 1].start_frame if index + 1 < len(raw_reps) else frame_count - 1
    post_end = min(post_limit, raw.end_frame + span, frame_count - 1)
    return [
        range(pre_start, min(raw.start_frame, frame_count - 1) + 1),
        range(min(raw.end_frame, frame_count - 1), post_end + 1),
    ]


def _sustained_bottom(
    metrics: list[PressFrameMetrics], windows: list[range], attribute: str, hold: int
) -> float:
    """
    Smallest elbow angle held for hold consecutive frames in any window - the
    deepest position actually stayed at, not one jittering frame's apparent depth.
    """
    best = float("nan")
    for window in windows:
        values = series(metrics, attribute, window)
        held, _ = sustained_extreme(values, hold, prefer_max=False)
        if not np.isfinite(held):
            held = safe_min(values)
        if np.isfinite(held) and (not np.isfinite(best) or held < best):
            best = held
    return best


def _sustained_top(
    metrics: list[PressFrameMetrics], windows: list[range], attribute: str, hold: int
) -> float:
    """
    Largest value of attribute held for hold consecutive frames in any
    window - the extension actually reached, not one frame's apparent lockout.
    """
    best = float("nan")
    for window in windows:
        values = series(metrics, attribute, window)
        held, _ = sustained_extreme(values, hold, prefer_max=True)
        if not np.isfinite(held):
            held = safe_max(values)
        if np.isfinite(held) and (not np.isfinite(best) or held > best):
            best = held
    return best


def build_reps(
    raw_reps: list[RawRep],
    metrics: list[PressFrameMetrics],
    pose: FramePoseData,
    config: PressConfig,
) -> list[PressRep]:
    """Turn detected rep boundaries into measured reps, taking each measurement from
    the frames of the phase it belongs to. Peaks are sustained extremes."""
    timestamps = pose.timestamps
    fps = effective_fps(timestamps)
    reps: list[PressRep] = []

    for index, raw in enumerate(raw_reps):
        number = index + 1
        lo = max(raw.extreme_frame - config.TOP_WINDOW_FRAMES, raw.start_frame)
        hi = min(raw.extreme_frame + config.TOP_WINDOW_FRAMES, raw.end_frame)
        rep_frames = range(raw.start_frame, raw.end_frame + 1)
        lower_start = min(hi + 1, raw.end_frame)
        rest_windows = bottom_windows(raw_reps, index, fps, len(metrics))

        segmentation = RepetitionSegmentation(
            start_frame=raw.start_frame,
            towards_start_frame=raw.start_frame,
            extreme_start_frame=lo,
            extreme_end_frame=hi,
            return_start_frame=lower_start,
            end_frame=raw.end_frame,
            towards_duration=span_seconds(timestamps, raw.start_frame, lo),
            extreme_duration=span_seconds(timestamps, lo, hi),
            return_duration=span_seconds(timestamps, lower_start, raw.end_frame),
        )
        working = frames_for(segmentation, "working")

        # range of motion, per arm: each arm's best sustained value anywhere in the
        # rep, not what both held in a shared window. Otherwise an arm arriving late
        # gets a limited-range finding for what is really a timing fault.
        left_top = _sustained_top(metrics, [rep_frames], "left_elbow_angle", 3)
        right_top = _sustained_top(metrics, [rep_frames], "right_elbow_angle", 3)
        left_bottom = _sustained_bottom(metrics, rest_windows, "left_elbow_angle", 3)
        right_bottom = _sustained_bottom(metrics, rest_windows, "right_elbow_angle", 3)
        measured_tops = [v for v in (left_top, right_top) if np.isfinite(v)]
        top_angle = float(np.mean(measured_tops)) if measured_tops else float("nan")
        bottom_angle = _sustained_bottom(metrics, rest_windows, "elbow_angle", 3)

        left_rom = _difference(left_top, left_bottom)
        right_rom = _difference(right_top, right_bottom)
        rom = _difference(top_angle, bottom_angle)
        rom_difference = (
            abs(left_rom - right_rom)
            if np.isfinite(left_rom) and np.isfinite(right_rom)
            else float("nan")
        )

        # Symmetry: sustained peaks, never a single spike.
        angle_gap, angle_offset = sustained_extreme(
            series(metrics, "elbow_angle_difference", working), config.SYMMETRY_MIN_FRAMES
        )
        height_signed = series(metrics, "wrist_height_difference", working)
        height_gap, height_offset = sustained_extreme(np.abs(height_signed), config.SYMMETRY_MIN_FRAMES)
        higher_side = _higher_side(height_signed)

        # Alignment: worst sustained offset per arm.
        left_offset, left_offset_at = sustained_extreme(
            series(metrics, "left_alignment_offset", working), config.ALIGNMENT_MIN_FRAMES
        )
        right_offset, right_offset_at = sustained_extreme(
            series(metrics, "right_alignment_offset", working), config.ALIGNMENT_MIN_FRAMES
        )

        valid_flags = np.asarray([metrics[i].valid for i in rep_frames], dtype=bool)
        visibility = np.asarray([metrics[i].landmark_confidence for i in rep_frames], dtype=np.float64)
        both_arms = np.asarray([metrics[i].both_arms_valid for i in rep_frames], dtype=bool)
        both_ratio = float(both_arms.mean()) if both_arms.size else 0.0
        arm_visibility = pose.visibility[
            raw.start_frame : raw.end_frame + 1,
            [c for chain in ARM_CHAINS.values() for c in chain.arm],
        ]
        # alignment is judged per side with the worse one reported, so it only
        # needs one arm's elbow/wrist pair
        side_visibility = [
            float(
                np.mean(
                    pose.visibility[raw.start_frame : raw.end_frame + 1, [chain.elbow, chain.wrist]]
                )
            )
            for chain in ARM_CHAINS.values()
        ]

        reps.append(
            PressRep(
                number=number,
                segmentation=segmentation,
                start_time=float(timestamps[raw.start_frame]),
                extreme_time=float(timestamps[raw.extreme_frame]),
                end_time=float(timestamps[raw.end_frame]),
                duration=float(timestamps[raw.end_frame] - timestamps[raw.start_frame]),
                left_top_elbow_angle=left_top,
                right_top_elbow_angle=right_top,
                left_bottom_elbow_angle=left_bottom,
                right_bottom_elbow_angle=right_bottom,
                top_elbow_angle=top_angle,
                bottom_elbow_angle=bottom_angle,
                left_rom_degrees=left_rom,
                right_rom_degrees=right_rom,
                rom_degrees=rom,
                max_elbow_angle_difference=angle_gap,
                max_elbow_angle_difference_frame=(
                    working.start + angle_offset if angle_offset >= 0 else -1
                ),
                max_wrist_height_difference=height_gap,
                max_wrist_height_difference_frame=(
                    working.start + height_offset if height_offset >= 0 else -1
                ),
                rom_difference=rom_difference,
                top_timing_difference=_top_timing_difference(metrics, working, timestamps),
                higher_side=higher_side,
                max_left_alignment_offset=left_offset,
                max_right_alignment_offset=right_offset,
                max_left_alignment_frame=(
                    working.start + left_offset_at if left_offset_at >= 0 else -1
                ),
                max_right_alignment_frame=(
                    working.start + right_offset_at if right_offset_at >= 0 else -1
                ),
                valid_frame_ratio=float(valid_flags.mean()) if valid_flags.size else 0.0,
                mean_landmark_visibility=float(np.mean(visibility)) if visibility.size else 0.0,
                both_arms_ratio=both_ratio,
                symmetry_reliable=bool(
                    both_ratio >= config.SYMMETRY_MIN_BOTH_SIDES_RATIO
                    and arm_visibility.size
                    and float(np.mean(arm_visibility)) >= config.SYMMETRY_MIN_VISIBILITY
                ),
                alignment_reliable=bool(
                    side_visibility and max(side_visibility) >= config.ALIGNMENT_MIN_VISIBILITY
                ),
            )
        )
    return reps


def _difference(a: float, b: float) -> float:
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")


def _higher_side(signed_difference: np.ndarray) -> str:
    """Which arm sat higher through the working phase. The median of the signed
    difference, so one frame where the arms crossed can't name the wrong side."""
    finite = signed_difference[np.isfinite(signed_difference)]
    if finite.size == 0:
        return ""
    value = float(np.median(finite))
    if abs(value) < 1e-6:
        return ""
    return "left" if value > 0 else "right"


def _top_timing_difference(
    metrics: list[PressFrameMetrics], frames: range, timestamps: np.ndarray
) -> float:
    """Seconds between the arms reaching their own highest position. Supporting
    evidence only - it comes from two independently noisy peak frames - but "your
    right arm arrives first" is more actionable than a bare angle gap."""
    left = series(metrics, "left_wrist_height", frames)
    right = series(metrics, "right_wrist_height", frames)
    if not (np.isfinite(left).any() and np.isfinite(right).any()):
        return float("nan")
    left_at = frames.start + int(np.nanargmax(left))
    right_at = frames.start + int(np.nanargmax(right))
    if not (0 <= left_at < len(timestamps) and 0 <= right_at < len(timestamps)):
        return float("nan")
    return abs(float(timestamps[left_at]) - float(timestamps[right_at]))


FPS_FROM_TIMESTAMPS = effective_fps
REST_PHASE = MovementPhase.REST
SAFE_MAX = safe_max
SHOULDER_IDS = SHOULDER_LANDMARKS
