"""
Landmarks in, numbers out. Measurement only - nothing here decides whether a
squat was any good.

    landmarks -> per-frame metrics -> standing baseline -> heel displacement
              -> per-rep facts

A knee angle alone can't describe a squat: two people can hit the same knee
angle with very different hip, torso and shin positions. So we measure knee and
hip flexion per leg, trunk and shin inclination, the depth relation, heel
displacement, left/right symmetry, stance width and the phase durations.

Conventions: image y grows downward, angles are interior angles in degrees,
normalised distances are divided by a body dimension from the same frame, and
anything unmeasurable is NaN rather than 0.
"""

from __future__ import annotations

import logging

import numpy as np

from analysis.geometry import (
    calculate_angle,
    distance,
    horizontal_distance,
    inclination_from_vertical,
    vertical_offset,
)
from analysis.models import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    FrameMetrics,
    FramePoseData,
    Phase,
    SquatRep,
    VideoMetadata,
)
from analysis.normalisation import body_scale, normalise, torso_reference

from .config import SquatConfig
from .landmarks import OPPOSITE_SIDE, SIDE_CHAINS
from .persistence import sustained_extreme

logger = logging.getLogger(__name__)

# --- Movement phases a rule can be scoped to. ---
# rule phases, not the state machine's per-frame Phase enum - a rule asks
# which part of the rep its check is about
PHASE_STANDING = "standing"
PHASE_DESCENT = "descent"
PHASE_BOTTOM = "bottom"
PHASE_ASCENT = "ascent"
PHASE_MOVEMENT = "descent_to_ascent"
PHASE_COMPLETION = "completion"

PHASE_LABELS: dict[str, str] = {
    PHASE_STANDING: "starting position",
    PHASE_DESCENT: "the descent",
    PHASE_BOTTOM: "the bottom position",
    PHASE_ASCENT: "the ascent",
    PHASE_MOVEMENT: "the whole movement",
    PHASE_COMPLETION: "the finish of the repetition",
}


def phase_frames(rep: SquatRep, phase: str) -> range:
    """Frame range a rule should read for phase on rep. A torso angle at the bottom
    is meant to differ from a standing one, so checking everywhere invents findings."""
    if phase == PHASE_DESCENT:
        return range(rep.descent_start_frame, max(rep.bottom_start_frame, rep.descent_start_frame))
    if phase == PHASE_BOTTOM:
        return range(rep.bottom_start_frame, rep.bottom_end_frame + 1)
    if phase == PHASE_ASCENT:
        return range(min(rep.ascent_start_frame, rep.end_frame), rep.end_frame + 1)
    if phase == PHASE_COMPLETION:
        return range(rep.end_frame, rep.end_frame + 1)
    # PHASE_MOVEMENT and anything unrecognised: the whole rep.
    return range(rep.start_frame, rep.end_frame + 1)


def series(metrics: list[FrameMetrics], attribute: str, frames: range | None = None) -> np.ndarray:
    """
    One measurement as a NaN-gapped float series over a frame range. Invalid
    frames come back NaN, not 0.
    """
    frames = frames if frames is not None else range(len(metrics))
    values = []
    for i in frames:
        if 0 <= i < len(metrics) and metrics[i].valid:
            values.append(float(getattr(metrics[i], attribute)))
        else:
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _median(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def _plausible(value: float, low: float, high: float) -> float:
    """value if it is anatomically possible, else NaN. The bands are wide because
    they reject tracking failures, not technique."""
    if not np.isfinite(value):
        return float("nan")
    return float(value) if low <= value <= high else float("nan")


# --- Per-frame measurements ---


def compute_frame_metrics(
    video: VideoMetadata,
    pose: FramePoseData,
    side: str,
    config: SquatConfig,
) -> list[FrameMetrics]:
    """Measure every frame, both legs, in pixel space. MediaPipe normalises x and y
    separately, so an angle off the normalised values is skewed on any non-square
    frame."""
    chain = SIDE_CHAINS[side]
    metrics: list[FrameMetrics] = []

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

        core_valid = all(usable(lm) for lm in chain.core)
        if not pose.pose_found[f] and not core_valid:
            metrics.append(FrameMetrics(frame_index=f, timestamp=timestamp, valid=False))
            continue

        per_side: dict[str, dict[str, float]] = {}
        side_valid: dict[str, bool] = {}
        for name, other in SIDE_CHAINS.items():
            p_sh, p_hip = px(other.shoulder), px(other.hip)
            p_knee, p_ankle = px(other.knee), px(other.ankle)
            per_side[name] = {
                # plausibility gate: MediaPipe reports a confident position even for
                # a limb it is extrapolating behind the body
                "knee": _plausible(
                    calculate_angle(p_hip, p_knee, p_ankle),
                    config.PLAUSIBLE_KNEE_ANGLE_MIN,
                    config.PLAUSIBLE_KNEE_ANGLE_MAX,
                ),
                "hip": calculate_angle(p_sh, p_hip, p_knee),
                # ankle -> knee against vertical; larger means the knee is further forward
                "shin": inclination_from_vertical(p_ankle, p_knee),
            }
            side_valid[name] = all(usable(lm) for lm in other.core)

        mid_shoulder, mid_hip = torso_reference(
            px(LEFT_SHOULDER), px(RIGHT_SHOULDER), px(LEFT_HIP), px(RIGHT_HIP)
        )
        torso = inclination_from_vertical(mid_hip, mid_shoulder)

        p_hip, p_knee, p_ankle = px(chain.hip), px(chain.knee), px(chain.ankle)
        lower_leg = distance(p_knee, p_ankle)
        hip_above_knee = normalise(
            (p_knee[1] - p_hip[1]) if (p_hip is not None and p_knee is not None) else float("nan"),
            lower_leg,
        )

        scale = body_scale(
            mid_shoulder,
            mid_hip,
            left_hip=px(LEFT_HIP),
            right_hip=px(RIGHT_HIP),
            knee=p_knee,
            ankle=p_ankle,
        )
        stance = normalise(horizontal_distance(px(LEFT_ANKLE), px(RIGHT_ANKLE)), scale)
        knee_offsets = [
            normalise(horizontal_distance(px(c.knee), px(c.ankle)), scale) for c in SIDE_CHAINS.values()
        ]
        knee_offset = _median(knee_offsets)

        # heel against the toe of the same foot, so camera drift can't look like a
        # lift. Positive means heel above toe.
        p_heel, p_toe = px(chain.heel), px(chain.foot_index)
        foot_reliable = (
            pose.visibility[f, chain.heel] >= config.HEEL_MIN_VISIBILITY
            and pose.visibility[f, chain.foot_index] >= config.HEEL_MIN_VISIBILITY
        )
        heel_toe_offset = (
            normalise(vertical_offset(p_toe, p_heel) * -1.0, lower_leg)
            if foot_reliable
            else float("nan")
        )

        left, right = per_side["left"], per_side["right"]
        both_valid = side_valid["left"] and side_valid["right"]
        knee_asym = (
            _plausible(
                abs(left["knee"] - right["knee"]),
                0.0,
                config.KNEE_SYMMETRY_MAX_PLAUSIBLE,
            )
            if both_valid and np.isfinite(left["knee"]) and np.isfinite(right["knee"])
            else float("nan")
        )
        hip_asym = (
            abs(left["hip"] - right["hip"])
            if both_valid and np.isfinite(left["hip"]) and np.isfinite(right["hip"])
            else float("nan")
        )

        selected = per_side[side]
        confidence = float(np.mean([pose.visibility[f, lm] for lm in chain.core]))

        metrics.append(
            FrameMetrics(
                frame_index=f,
                timestamp=timestamp,
                valid=core_valid and np.isfinite(selected["knee"]),
                knee_angle=selected["knee"],
                hip_angle=selected["hip"],
                torso_lean=torso,
                shin_inclination=selected["shin"],
                hip_above_knee=hip_above_knee,
                heel_toe_offset=heel_toe_offset,
                landmark_confidence=confidence,
                left_knee_angle=left["knee"],
                right_knee_angle=right["knee"],
                left_hip_angle=left["hip"],
                right_hip_angle=right["hip"],
                left_shin_inclination=left["shin"],
                right_shin_inclination=right["shin"],
                knee_asymmetry=knee_asym,
                hip_asymmetry=hip_asym,
                both_sides_valid=both_valid,
                body_scale=scale,
                stance_width=stance,
                knee_over_ankle_offset=knee_offset,
            )
        )
    return metrics


# --- Standing baseline ---


def standing_baseline(
    pose: FramePoseData,
    metrics: list[FrameMetrics],
    phases: list[Phase],
    first_rep_start: int | None,
    video: VideoMetadata,
    side: str,
    config: SquatConfig,
) -> dict[str, float]:
    """
    The lifter's own standing posture, measured rather than assumed - comparing a
    finish to 180 degrees would penalise anyone whose stance isn't a locked knee.
    A median over real standing frames, since people are still settling at frame 0.
    """
    chain = SIDE_CHAINS[side]
    standing = [i for i, p in enumerate(phases) if p is Phase.STANDING and metrics[i].valid]
    if first_rep_start is not None:
        before_first = [i for i in standing if i < first_rep_start]
        if len(before_first) >= config.BASELINE_MIN_FRAMES:
            standing = before_first
    if len(standing) < config.BASELINE_MIN_FRAMES:
        logger.warning("Few standing frames (%d) for the baseline", len(standing))

    heel_ys: list[float] = []
    lower_legs: list[float] = []
    for i in standing:
        if pose.valid[i, chain.heel]:
            heel_ys.append(pose.pixel_xy(i, chain.heel, video.width, video.height)[1])
        knee = (
            pose.pixel_xy(i, chain.knee, video.width, video.height)
            if pose.valid[i, chain.knee]
            else None
        )
        ankle = (
            pose.pixel_xy(i, chain.ankle, video.width, video.height)
            if pose.valid[i, chain.ankle]
            else None
        )
        leg = distance(knee, ankle)
        if np.isfinite(leg):
            lower_legs.append(leg)

    return {
        "knee_angle": _median([metrics[i].knee_angle for i in standing]),
        "torso_lean": _median([metrics[i].torso_lean for i in standing]),
        "hip_angle": _median([metrics[i].hip_angle for i in standing]),
        "shin_inclination": _median([metrics[i].shin_inclination for i in standing]),
        "stance_width": _median([metrics[i].stance_width for i in standing]),
        "heel_toe_offset": _median([metrics[i].heel_toe_offset for i in standing]),
        "heel_y": _median(heel_ys),
        "lower_leg_px": _median(lower_legs),
        "standing_frames": float(len(standing)),
    }


def apply_heel_metric(
    pose: FramePoseData,
    metrics: list[FrameMetrics],
    video: VideoMetadata,
    side: str,
    baseline: dict[str, float],
    config: SquatConfig,
) -> None:
    """
    Heel rise against this person's own standing foot. Each frame carries the
    heel's height above the toe over lower-leg length; subtracting the standing
    median gives how much higher the heel is than when standing.

    Measured inside the foot rather than against an image position, so camera
    distance and someone drifting across the frame can't look like a heel lift.
    """
    del pose, video, side  # the per-frame offsets already carry the geometry
    reference = baseline.get("heel_toe_offset", float("nan"))
    if not np.isfinite(reference):
        return
    for m in metrics:
        if not np.isfinite(m.heel_toe_offset):
            continue
        m.heel_lift = _plausible(
            m.heel_toe_offset - reference,
            -config.HEEL_LIFT_MAX_PLAUSIBLE,
            config.HEEL_LIFT_MAX_PLAUSIBLE,
        )


# --- Per-repetition facts ---


def build_reps(
    raw_reps,
    metrics: list[FrameMetrics],
    knee_smoothed: np.ndarray,
    pose: FramePoseData,
    side: str,
    config: SquatConfig,
) -> list[SquatRep]:
    """Turn detected rep boundaries into measured reps, taking each measurement
    from the frames of the phase it belongs to. Peaks are sustained maxima."""
    chain = SIDE_CHAINS[side]
    other = SIDE_CHAINS[OPPOSITE_SIDE[side]]
    timestamps = pose.timestamps
    fps = _effective_fps(timestamps)
    reps: list[SquatRep] = []

    for number, raw in enumerate(raw_reps, start=1):
        bottom_lo = max(raw.bottom_frame - config.BOTTOM_WINDOW_FRAMES, raw.start_frame)
        bottom_hi = min(raw.bottom_frame + config.BOTTOM_WINDOW_FRAMES, raw.end_frame)
        bottom_window = range(bottom_lo, bottom_hi + 1)
        rep_frames = range(raw.start_frame, raw.end_frame + 1)

        def window_median(attribute: str, frames=bottom_window) -> float:
            return _median(list(series(metrics, attribute, frames)))

        torso_series = series(metrics, "torso_lean", rep_frames)
        max_torso, torso_offset = sustained_extreme(torso_series, config.TORSO_LEAN_MIN_FRAMES)
        if not np.isfinite(max_torso):  # never held for long enough to sustain
            max_torso = (
                float(np.nanmax(torso_series)) if np.isfinite(torso_series).any() else float("nan")
            )
            torso_offset = int(np.nanargmax(torso_series)) if np.isfinite(max_torso) else -1
        max_torso_frame = raw.start_frame + torso_offset if torso_offset >= 0 else -1

        heel_series = np.asarray([metrics[i].heel_lift for i in rep_frames], dtype=np.float64)
        heel_vis = pose.visibility[raw.start_frame : raw.end_frame + 1, [chain.heel, chain.foot_index]]
        heel_reliable = bool(heel_vis.size and float(np.mean(heel_vis)) >= config.HEEL_MIN_VISIBILITY)
        sustained_lift, lift_offset = sustained_extreme(heel_series, config.HEEL_LIFT_MIN_FRAMES)
        lift_frame = raw.start_frame + lift_offset if lift_offset >= 0 else -1

        asym_series = series(metrics, "knee_asymmetry", rep_frames)
        both_valid = np.asarray([metrics[i].both_sides_valid for i in rep_frames], dtype=bool)
        both_ratio = float(both_valid.mean()) if both_valid.size else 0.0
        other_vis = pose.visibility[raw.start_frame : raw.end_frame + 1, list(other.core)]
        symmetry_reliable = bool(
            both_ratio >= config.SYMMETRY_MIN_BOTH_SIDES_RATIO
            and other_vis.size
            and float(np.mean(other_vis)) >= config.SYMMETRY_MIN_VISIBILITY
        )
        max_asym, asym_offset = sustained_extreme(asym_series, config.SYMMETRY_MIN_FRAMES)
        asym_frame = raw.start_frame + asym_offset if asym_offset >= 0 else -1

        end_knee, end_hip = _finish_posture(raw, number, raw_reps, metrics, knee_smoothed, fps)

        valid_flags = np.asarray([metrics[i].valid for i in rep_frames], dtype=bool)
        visibility = np.asarray([metrics[i].landmark_confidence for i in rep_frames], dtype=np.float64)

        ascent_start = min(bottom_hi + 1, raw.end_frame)
        reps.append(
            SquatRep(
                number=number,
                start_frame=raw.start_frame,
                bottom_frame=raw.bottom_frame,
                end_frame=raw.end_frame,
                start_time=float(timestamps[raw.start_frame]),
                bottom_time=float(timestamps[raw.bottom_frame]),
                end_time=float(timestamps[raw.end_frame]),
                duration=float(timestamps[raw.end_frame] - timestamps[raw.start_frame]),
                min_knee_angle=window_median("knee_angle"),
                hip_angle_at_bottom=window_median("hip_angle"),
                torso_lean_at_bottom=window_median("torso_lean"),
                max_torso_lean=max_torso,
                max_torso_lean_frame=max_torso_frame,
                hip_above_knee_at_bottom=window_median("hip_above_knee"),
                max_heel_lift=sustained_lift,
                max_heel_lift_frame=lift_frame,
                heel_reliable=heel_reliable,
                end_knee_angle=end_knee,
                end_hip_angle=end_hip,
                descent_start_frame=raw.start_frame,
                bottom_start_frame=bottom_lo,
                bottom_end_frame=bottom_hi,
                ascent_start_frame=ascent_start,
                descent_duration=_span_seconds(timestamps, raw.start_frame, bottom_lo),
                bottom_duration=_span_seconds(timestamps, bottom_lo, bottom_hi),
                ascent_duration=_span_seconds(timestamps, ascent_start, raw.end_frame),
                shin_inclination_at_bottom=window_median("shin_inclination"),
                min_left_knee_angle=_safe_min(series(metrics, "left_knee_angle", bottom_window)),
                min_right_knee_angle=_safe_min(series(metrics, "right_knee_angle", bottom_window)),
                max_knee_asymmetry=max_asym,
                max_knee_asymmetry_frame=asym_frame,
                symmetry_reliable=symmetry_reliable,
                stance_width=window_median("stance_width", rep_frames),
                valid_frame_ratio=float(valid_flags.mean()) if valid_flags.size else 0.0,
                mean_landmark_visibility=(float(np.mean(visibility)) if visibility.size else 0.0),
            )
        )
    return reps


def _effective_fps(timestamps: np.ndarray) -> float:
    """
    Frames per second from the timestamps themselves - nothing assumes 30 fps.
    """
    if len(timestamps) < 2:
        return 30.0
    step = float(np.median(np.diff(timestamps)))
    return 1.0 / max(step, 1e-6)


def _span_seconds(timestamps: np.ndarray, start: int, end: int) -> float:
    if start < 0 or end < 0 or start >= len(timestamps) or end >= len(timestamps):
        return float("nan")
    return float(max(timestamps[end] - timestamps[start], 0.0))


def _safe_min(values: np.ndarray) -> float:
    return float(np.nanmin(values)) if np.isfinite(values).any() else float("nan")


def _finish_posture(
    raw,
    number: int,
    raw_reps,
    metrics: list[FrameMetrics],
    knee_smoothed: np.ndarray,
    fps: float,
) -> tuple[float, float]:
    """
    Peak extension just after a rep commits. The state machine commits as soon as
    the ascent clears its threshold, before the lifter has finished standing up, so
    reading the angle at that frame under-reports every finish.
    """
    window_end = raw.end_frame + int(round(0.8 * fps))
    if number < len(raw_reps):
        window_end = min(window_end, raw_reps[number].start_frame)
    window_end = min(window_end, len(knee_smoothed) - 1)

    knee_values = [
        knee_smoothed[i] for i in range(raw.end_frame, window_end + 1) if np.isfinite(knee_smoothed[i])
    ]
    hip_values = [
        metrics[i].hip_angle
        for i in range(raw.end_frame, window_end + 1)
        if metrics[i].valid and np.isfinite(metrics[i].hip_angle)
    ]
    end_knee = float(np.max(knee_values)) if knee_values else float("nan")
    end_hip = float(np.max(hip_values)) if hip_values else float("nan")
    return end_knee, end_hip
