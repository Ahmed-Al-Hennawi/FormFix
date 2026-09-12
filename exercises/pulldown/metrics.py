"""
Landmarks in, numbers out. Only measuring here.

    landmarks -> per-frame metrics -> top-position baseline -> trunk excursion
              -> per-rep facts

The elbow angle alone isn't enough - you get the same angle by pulling the bar
down or by leaning back under a bar that barely moved. So I also measure trunk
angle, trunk movement from the person's own top posture, trunk velocity
(export only) and wrist/elbow rise above the shoulder line.

Image y grows downward, angles are interior angles in degrees, and anything
unmeasurable is NaN, not 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from analysis.geometry import (
    calculate_angle,
    inclination_from_vertical,
    signed_inclination_from_vertical,
)
from analysis.models import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    FramePoseData,
    PulldownRep,
    RepetitionSegmentation,
    VideoMetadata,
)
from analysis.normalisation import body_scale, normalise, torso_reference
from exercises.common.metrics import (
    PHASE_MOVEMENT,
    effective_fps,
    frames_for,
    median,
    plausible,
    raw_series,
    safe_max,
    series,
    settled_frames,
    span_seconds,
)
from exercises.common.persistence import sustained_extreme
from exercises.common.phases import MovementPhase, RawRep

from .config import PulldownConfig
from .landmarks import ARM_CHAINS, FACING_LANDMARKS, TORSO_LANDMARKS

logger = logging.getLogger(__name__)


@dataclass
class PulldownFrameMetrics:
    """Measurements for one frame (NaN if unmeasurable, never 0)."""

    frame_index: int
    timestamp: float
    valid: bool

    # SHOULDER-ELBOW-WRIST, degrees, ~170-180 extended
    elbow_angle: float = float("nan")
    left_elbow_angle: float = float("nan")
    right_elbow_angle: float = float("nan")
    elbow_angle_difference: float = float("nan")

    # mid-hip -> mid-shoulder from vertical, degrees, unsigned
    torso_angle: float = float("nan")
    # same segment with its direction kept: positive = shoulders to the +x side
    torso_angle_signed: float = float("nan")
    # same but positive = leaning backwards
    torso_posterior: float = float("nan")
    # trunk movement away from the person's own top posture, degrees
    torso_excursion: float = float("nan")
    # trunk speed, deg/s - exported for a possible swinging check, not used yet
    torso_velocity: float = float("nan")
    torso_reliable: bool = False

    # wrist height above the shoulder line / trunk length; positive is above
    wrist_rise: float = float("nan")
    elbow_rise: float = float("nan")

    body_scale: float = float("nan")
    landmark_confidence: float = 0.0
    analysed_arm_valid: bool = False
    both_arms_valid: bool = False


# --- Per-frame measurements ---


def compute_frame_metrics(
    video: VideoMetadata,
    pose: FramePoseData,
    side: str,
    config: PulldownConfig,
) -> list[PulldownFrameMetrics]:
    """Measure every frame, both arms, in pixels (normalised coordinates skew angles
    on a phone clip)."""
    chain = ARM_CHAINS[side]
    metrics: list[PulldownFrameMetrics] = []

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
            metrics.append(PulldownFrameMetrics(frame_index=f, timestamp=timestamp, valid=False))
            continue

        per_arm: dict[str, float] = {}
        arm_ok: dict[str, bool] = {}
        for name, other in ARM_CHAINS.items():
            per_arm[name] = plausible(
                calculate_angle(px(other.shoulder), px(other.elbow), px(other.wrist)),
                config.PLAUSIBLE_ELBOW_ANGLE_MIN,
                config.PLAUSIBLE_ELBOW_ANGLE_MAX,
            )
            arm_ok[name] = all(usable(lm) for lm in other.arm)

        both_arms = arm_ok["left"] and arm_ok["right"]
        left, right = per_arm["left"], per_arm["right"]
        difference = (
            plausible(abs(left - right), 0.0, config.ARM_DIFFERENCE_MAX_PLAUSIBLE)
            if both_arms and np.isfinite(left) and np.isfinite(right)
            else float("nan")
        )

        # movement signal: mean of the usable arms. Two arms halve the noise, and
        # falling back to one keeps side views working
        usable_angles = [per_arm[n] for n in ARM_CHAINS if arm_ok[n] and np.isfinite(per_arm[n])]
        elbow_angle = float(np.mean(usable_angles)) if usable_angles else float("nan")

        mid_shoulder, mid_hip = torso_reference(
            px(LEFT_SHOULDER), px(RIGHT_SHOULDER), px(LEFT_HIP), px(RIGHT_HIP)
        )
        torso_reliable = all(usable(lm) for lm in TORSO_LANDMARKS)
        torso_angle = inclination_from_vertical(mid_hip, mid_shoulder)
        torso_signed = signed_inclination_from_vertical(mid_hip, mid_shoulder)

        scale = body_scale(
            mid_shoulder,
            mid_hip,
            left_hip=px(LEFT_HIP),
            right_hip=px(RIGHT_HIP),
        )
        wrist_rise = _mean_rise(px, ARM_CHAINS, "wrist", mid_shoulder, scale)
        elbow_rise = _mean_rise(px, ARM_CHAINS, "elbow", mid_shoulder, scale)

        confidence = float(np.mean([pose.visibility[f, lm] for lm in chain.arm]))

        metrics.append(
            PulldownFrameMetrics(
                frame_index=f,
                timestamp=timestamp,
                valid=bool(usable_angles) and np.isfinite(elbow_angle),
                elbow_angle=elbow_angle,
                left_elbow_angle=left,
                right_elbow_angle=right,
                elbow_angle_difference=difference,
                torso_angle=torso_angle if torso_reliable else float("nan"),
                torso_angle_signed=torso_signed if torso_reliable else float("nan"),
                torso_reliable=torso_reliable,
                wrist_rise=wrist_rise,
                elbow_rise=elbow_rise,
                body_scale=scale,
                landmark_confidence=confidence,
                analysed_arm_valid=arm_ok[side],
                both_arms_valid=both_arms,
            )
        )

    _apply_torso_velocity(metrics, pose.timestamps)
    return metrics


def _mean_rise(px, chains, role: str, mid_shoulder, scale: float) -> float:
    """Mean height of a landmark pair above the shoulder line, normalised. Relative
    to the shoulders so shifting in the seat doesn't count as travel."""
    if mid_shoulder is None:
        return float("nan")
    values = []
    for chain in chains.values():
        point = px(getattr(chain, role))
        if point is None:
            continue
        values.append(normalise(mid_shoulder[1] - point[1], scale))
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _apply_torso_velocity(metrics: list[PulldownFrameMetrics], timestamps: np.ndarray) -> None:
    """Trunk angular speed in deg/s. Exported only, for a future swinging check."""
    for i in range(1, len(metrics)):
        a, b = metrics[i - 1].torso_angle, metrics[i].torso_angle
        dt = float(timestamps[i]) - float(timestamps[i - 1])
        if dt <= 1e-9 or not (np.isfinite(a) and np.isfinite(b)):
            continue
        metrics[i].torso_velocity = (b - a) / dt


# --- Facing direction and the top-position baseline ---


def estimate_facing(
    video: VideoMetadata,
    pose: FramePoseData,
    config: PulldownConfig,
) -> tuple[int, float]:
    """
    Which way the person faces: +1 towards +x, -1 towards -x, 0 unknown. Needed
    to tell leaning back from leaning forward. Uses the median offset of the head
    landmarks from the shoulders. (0, 0.0) if the head isn't visible enough.
    """
    offsets: list[float] = []
    visibilities: list[float] = []
    for f in range(pose.frame_count):
        if not pose.pose_found[f]:
            continue
        shoulders = [
            pose.pixel_xy(f, lm, video.width, video.height)
            for lm in (LEFT_SHOULDER, RIGHT_SHOULDER)
            if pose.valid[f, lm]
        ]
        if not shoulders:
            continue
        mid_x = float(np.mean([p[0] for p in shoulders]))
        head = [
            pose.pixel_xy(f, lm, video.width, video.height)
            for lm in FACING_LANDMARKS
            if pose.valid[f, lm] and pose.visibility[f, lm] >= config.FACING_MIN_VISIBILITY
        ]
        if not head:
            continue
        offsets.append(float(np.mean([p[0] for p in head])) - mid_x)
        visibilities.append(float(np.mean([pose.visibility[f, lm] for lm in FACING_LANDMARKS])))

    if len(offsets) < 5:
        return 0, 0.0
    offset = float(np.median(offsets))
    confidence = float(np.mean(visibilities)) if visibilities else 0.0
    scale_guard = abs(offset) > 1e-3 * max(video.width, 1)
    if confidence < config.FACING_MIN_VISIBILITY or not scale_guard:
        return 0, confidence
    return (1 if offset > 0 else -1), confidence


def apply_posterior_lean(metrics: list[PulldownFrameMetrics], facing: int) -> None:
    """Signed trunk angle with positive = leaning back. Someone facing +x leans back
    towards -x, hence the -facing. NaN if facing is unknown."""
    if facing == 0:
        return
    for m in metrics:
        if np.isfinite(m.torso_angle_signed):
            m.torso_posterior = -facing * m.torso_angle_signed


def top_baseline(
    metrics: list[PulldownFrameMetrics],
    phases: list[MovementPhase],
    first_rep_start: int | None,
    config: PulldownConfig,
    elbow_signal: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> dict[str, float]:
    """The person's own posture at the top, so a reclined seat or tilted camera isn't
    a fault. Median over the top frames where the arms had stopped moving, and
    preferably the last couple of seconds before the first rep - reaching up for
    the bar labels as "top" too, and it used to set the trunk reference."""
    top = [
        i
        for i, phase in enumerate(phases)
        if phase is MovementPhase.REST and i < len(metrics) and metrics[i].valid
    ]
    if elbow_signal is not None and timestamps is not None:
        still = settled_frames(top, elbow_signal, timestamps, config.BASELINE_MAX_VELOCITY)
        if len(still) >= config.BASELINE_MIN_FRAMES:
            top = still
    if first_rep_start is not None:
        before_first = [i for i in top if i < first_rep_start]
        if len(before_first) >= config.BASELINE_MIN_FRAMES:
            window = config.BASELINE_MIN_FRAMES
            if timestamps is not None:
                window = max(
                    window,
                    int(round(config.BASELINE_WINDOW_SECONDS * effective_fps(timestamps))),
                )
            top = before_first[-window:]
    if len(top) < config.BASELINE_MIN_FRAMES:
        logger.warning("Few settled top-position frames (%d) for the pulldown baseline", len(top))

    return {
        "torso_angle": median([metrics[i].torso_angle for i in top]),
        "torso_posterior": median([metrics[i].torso_posterior for i in top]),
        "elbow_angle": median([metrics[i].elbow_angle for i in top]),
        "wrist_rise": median([metrics[i].wrist_rise for i in top]),
        "elbow_rise": median([metrics[i].elbow_rise for i in top]),
        "top_frames": float(len(top)),
    }


def apply_torso_excursion(
    metrics: list[PulldownFrameMetrics],
    baseline: dict[str, float],
    config: PulldownConfig,
) -> str:
    """
    Fill in torso_excursion and return the mode used. "posterior" is the backward
    lean minus its baseline (needs the facing direction). "unsigned" is just the
    size of the change, so the wording says "moved" instead of "leaned back".
    """
    posterior_ref = baseline.get("torso_posterior", float("nan"))
    unsigned_ref = baseline.get("torso_angle", float("nan"))

    if np.isfinite(posterior_ref) and any(np.isfinite(m.torso_posterior) for m in metrics):
        mode = "posterior"
        for m in metrics:
            if np.isfinite(m.torso_posterior):
                m.torso_excursion = plausible(
                    m.torso_posterior - posterior_ref,
                    -config.PLAUSIBLE_TORSO_EXCURSION_MAX,
                    config.PLAUSIBLE_TORSO_EXCURSION_MAX,
                )
        return mode

    if np.isfinite(unsigned_ref):
        for m in metrics:
            if np.isfinite(m.torso_angle):
                m.torso_excursion = plausible(
                    abs(m.torso_angle - unsigned_ref),
                    0.0,
                    config.PLAUSIBLE_TORSO_EXCURSION_MAX,
                )
    return "unsigned"


# --- Per-repetition facts ---


def phase_frames(rep: PulldownRep, phase: str) -> range:
    """Frame range a rule should read for this phase."""
    return frames_for(rep.segmentation, phase)


def top_windows(
    raw_reps: list[RawRep],
    index: int,
    fps: float,
    frame_count: int,
) -> list[range]:
    """
    Frames where the arms are really at the top. Not the rep's first frame - the
    rep only starts once the pull is moving, so reading there gave false "limited
    top extension" findings. The windows either side are kept separate.
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


def _sustained_top(
    metrics: list[PulldownFrameMetrics], windows: list[range], attribute: str, hold: int
) -> float:
    """Largest value held for `hold` frames in a row, so one jittery wrist doesn't count."""
    best = float("nan")
    for window in windows:
        values = series(metrics, attribute, window)
        held, _ = sustained_extreme(values, hold)
        if not np.isfinite(held):
            held = safe_max(values)
        if np.isfinite(held) and (not np.isfinite(best) or held > best):
            best = held
    return best


def build_reps(
    raw_reps: list[RawRep],
    metrics: list[PulldownFrameMetrics],
    pose: FramePoseData,
    side: str,
    torso_mode: str,
    config: PulldownConfig,
) -> list[PulldownRep]:
    """Turn rep boundaries into measured reps, each value taken from its own phase.
    Peaks are sustained extremes."""
    timestamps = pose.timestamps
    fps = effective_fps(timestamps)
    reps: list[PulldownRep] = []

    for index, raw in enumerate(raw_reps):
        number = index + 1
        rest_windows = top_windows(raw_reps, index, fps, len(metrics))
        lo = max(raw.extreme_frame - config.CONTRACTED_WINDOW_FRAMES, raw.start_frame)
        hi = min(raw.extreme_frame + config.CONTRACTED_WINDOW_FRAMES, raw.end_frame)
        contracted = range(lo, hi + 1)
        rep_frames = range(raw.start_frame, raw.end_frame + 1)
        return_start = min(hi + 1, raw.end_frame)

        segmentation = RepetitionSegmentation(
            start_frame=raw.start_frame,
            towards_start_frame=raw.start_frame,
            extreme_start_frame=lo,
            extreme_end_frame=hi,
            return_start_frame=return_start,
            end_frame=raw.end_frame,
            towards_duration=span_seconds(timestamps, raw.start_frame, lo),
            extreme_duration=span_seconds(timestamps, lo, hi),
            return_duration=span_seconds(timestamps, return_start, raw.end_frame),
        )

        top_elbow = _sustained_top(metrics, rest_windows, "elbow_angle", 3)
        bottom_elbow = median(list(series(metrics, "elbow_angle", contracted)))
        rom = (
            float(top_elbow - bottom_elbow)
            if np.isfinite(top_elbow) and np.isfinite(bottom_elbow)
            else float("nan")
        )

        excursion = series(metrics, "torso_excursion", rep_frames)
        peak, offset = sustained_extreme(excursion, config.TORSO_MIN_FRAMES)
        if not np.isfinite(peak):  # never held long enough
            peak = safe_max(excursion)
            offset = int(np.nanargmax(excursion)) if np.isfinite(peak) else -1
        peak_frame = raw.start_frame + offset if offset >= 0 else -1

        velocity = raw_series(metrics, "torso_velocity", frames_for(segmentation, "working"))
        peak_velocity = (
            float(np.nanmax(np.abs(velocity))) if np.isfinite(velocity).any() else float("nan")
        )

        valid_flags = np.asarray([metrics[i].valid for i in rep_frames], dtype=bool)
        visibility = np.asarray([metrics[i].landmark_confidence for i in rep_frames], dtype=np.float64)
        analysed_arm = np.asarray([metrics[i].analysed_arm_valid for i in rep_frames], dtype=bool)
        both_arms = np.asarray([metrics[i].both_arms_valid for i in rep_frames], dtype=bool)
        torso_flags = np.asarray([metrics[i].torso_reliable for i in rep_frames], dtype=bool)
        torso_vis = pose.visibility[raw.start_frame : raw.end_frame + 1, list(TORSO_LANDMARKS)]

        rest_frames = [f for window in rest_windows for f in window]
        wrist_top = median([_at(metrics, f, "wrist_rise") for f in rest_frames])
        wrist_bottom = median(list(raw_series(metrics, "wrist_rise", contracted)))
        elbow_top = median([_at(metrics, f, "elbow_rise") for f in rest_frames])
        elbow_bottom = median(list(raw_series(metrics, "elbow_rise", contracted)))

        reps.append(
            PulldownRep(
                number=number,
                segmentation=segmentation,
                start_time=float(timestamps[raw.start_frame]),
                extreme_time=float(timestamps[raw.extreme_frame]),
                end_time=float(timestamps[raw.end_frame]),
                duration=float(timestamps[raw.end_frame] - timestamps[raw.start_frame]),
                top_elbow_angle=top_elbow,
                bottom_elbow_angle=bottom_elbow,
                rom_degrees=rom,
                left_top_elbow_angle=safe_max(series(metrics, "left_elbow_angle", rep_frames)),
                right_top_elbow_angle=safe_max(series(metrics, "right_elbow_angle", rep_frames)),
                left_bottom_elbow_angle=median(list(series(metrics, "left_elbow_angle", contracted))),
                right_bottom_elbow_angle=median(list(series(metrics, "right_elbow_angle", contracted))),
                wrist_rise_at_top=wrist_top,
                wrist_rise_at_bottom=wrist_bottom,
                wrist_travel=(
                    float(wrist_top - wrist_bottom)
                    if np.isfinite(wrist_top) and np.isfinite(wrist_bottom)
                    else float("nan")
                ),
                elbow_travel=(
                    float(elbow_top - elbow_bottom)
                    if np.isfinite(elbow_top) and np.isfinite(elbow_bottom)
                    else float("nan")
                ),
                torso_at_top=median([_at(metrics, f, "torso_angle") for f in rest_frames]),
                torso_at_bottom=median(list(series(metrics, "torso_angle", contracted))),
                max_torso_excursion=peak,
                max_torso_excursion_frame=peak_frame,
                torso_mode=torso_mode,
                peak_torso_velocity=peak_velocity,
                valid_frame_ratio=float(valid_flags.mean()) if valid_flags.size else 0.0,
                mean_landmark_visibility=float(np.mean(visibility)) if visibility.size else 0.0,
                arms_reliable=bool(
                    analysed_arm.size and float(analysed_arm.mean()) >= config.ARM_MIN_USABLE_RATIO
                ),
                both_arms_ratio=float(both_arms.mean()) if both_arms.size else 0.0,
                torso_reliable=bool(
                    torso_flags.size
                    and float(torso_flags.mean()) >= 0.5
                    and torso_vis.size
                    and float(np.mean(torso_vis)) >= config.TORSO_MIN_VISIBILITY
                ),
            )
        )
    return reps


def _at(metrics: list[PulldownFrameMetrics], frame: int, attribute: str) -> float:
    """NaN if the frame is out of range."""
    if 0 <= frame < len(metrics):
        return float(getattr(metrics[frame], attribute, float("nan")))
    return float("nan")


def movement_signal(metrics: list[PulldownFrameMetrics]) -> np.ndarray:
    """Signal for the rep state machine: mean elbow angle, high with arms extended."""
    return series(metrics, "elbow_angle", range(len(metrics)))


PHASE_WHOLE_REPETITION = PHASE_MOVEMENT
FPS_FROM_TIMESTAMPS = effective_fps
