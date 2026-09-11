"""
Is this recording good enough to measure? Runs after pose detection and before
any technique checks.

Five stages: pose coverage, key landmarks, usable frames, framing, camera
orientation. The camera angle only lowers confidence (gym clips are rarely a
perfect side view), it never rejects a video by itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .models import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    METRIC_DEPTH,
    METRIC_EXTENSION,
    METRIC_HEEL_LIFT,
    METRIC_TORSO_LEAN,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    SIDE_LANDMARKS,
    CameraOrientation,
    FramePoseData,
    RejectionCode,
    ValidationResult,
    VideoMetadata,
)

logger = logging.getLogger(__name__)

# what a squat measurement needs per side (no heel or foot)
CORE_SIDE_LANDMARKS: dict[str, tuple[int, ...]] = {
    "left": (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right": (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
}

# same for the upper body. No hips, because seated they are often hidden
CORE_ARM_LANDMARKS: dict[str, tuple[int, ...]] = {
    "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}


@dataclass(frozen=True)
class FramingRegion:
    """Body region that should stay in frame. If a critical one is out of frame for
    most of the clip the analysis stops, otherwise it just warns."""

    name: str
    landmarks: tuple[int, ...]
    critical: bool = True
    # measurement that depends on this region, marked unavailable when clipped
    limited_metric: str | None = None
    warn_message: str | None = None
    fail_message: str | None = None


@dataclass(frozen=True)
class OrientationNote:
    """What a camera angle costs this exercise. Front-on is useless for squat depth
    but ideal for press symmetry, so each exercise sets its own."""

    orientation: CameraOrientation
    message: str
    # so the warning can name the measurement it disabled
    primary_metric: str | None = None
    # other measurements this view can't support
    limited_metrics: tuple[str, ...] = ()


# default when an exercise supplies none: the side-on squat profile
SAGITTAL_FRAMING_REGIONS: tuple[FramingRegion, ...] = (
    FramingRegion("head/upper torso", (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)),
    FramingRegion("hips", (LEFT_HIP, RIGHT_HIP)),
    FramingRegion(
        "feet",
        (LEFT_ANKLE, RIGHT_ANKLE),
        critical=False,
        limited_metric=METRIC_HEEL_LIFT,
        warn_message=(
            "Your feet leave the camera frame during part of the movement, "
            "which can make the heel check less reliable."
        ),
        fail_message=(
            "Your feet are outside the camera frame for most of the set, "
            "so heel contact could not be checked."
        ),
    ),
)

SAGITTAL_ORIENTATION_NOTES: tuple[OrientationNote, ...] = (
    OrientationNote(
        CameraOrientation.FRONTAL,
        "This recording looks front-on or rear-on. FormFix analysed what it "
        "could still see, but knee and hip angles can only be measured from "
        "a side view, so the depth, torso-lean and lockout checks are not "
        "assessed for this clip.",
        primary_metric=METRIC_DEPTH,
        limited_metrics=(METRIC_TORSO_LEAN, METRIC_EXTENSION),
    ),
    OrientationNote(
        CameraOrientation.DIAGONAL_SIDE,
        "The camera is not fully side-on. FormFix could analyse this squat, "
        "although a clearer side view may improve the accuracy of some "
        "measurements.",
    ),
    OrientationNote(
        CameraOrientation.UNKNOWN,
        "The camera angle could not be estimated from this recording, so the "
        "measurements may be less accurate than usual.",
    ),
)


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds for the checks. All values come from the exercise config."""

    min_duration: float
    max_duration: float
    # Stage A - share of frames a person must be found in
    min_pose_frame_ratio: float
    # Stage B - per-landmark visibility floor
    min_key_landmark_visibility: float
    # Stage C - share of frames where all core landmarks clear that floor
    min_usable_frame_ratio: float
    # share of frames with a second person before a note is added
    multi_person_warn_ratio: float
    # not used any more (see MIN_SWITCHES_TO_REJECT), kept so the configs match
    multi_person_fail_ratio: float
    # Stage E - frontality bands: max(shoulder sep, hip sep) / torso length,
    # near 0 side-on and near 1 facing the camera
    side_view_good_ratio: float
    side_view_frontal_ratio: float
    # Stage D - edge margin that counts as cropped, then the share of frames
    # allowed in it before warning / stopping
    framing_margin: float
    framing_warn_tolerance: float
    framing_fail_tolerance: float
    # Stage E - spread above which the camera angle counts as changing mid-clip
    view_stability_spread: float = 0.35
    # Stage B/C - what this exercise can't be measured without, per side
    core_landmarks: dict[str, tuple[int, ...]] = field(
        default_factory=lambda: dict(CORE_SIDE_LANDMARKS)
    )
    # name used for them in the "we couldn't track..." message
    core_landmarks_label: str = "hips, knees and ankles"
    # Stage B - wider set, only for breaking a tie between the two sides
    side_landmarks: dict[str, tuple[int, ...]] = field(default_factory=lambda: dict(SIDE_LANDMARKS))
    # Stage D - body regions that must stay in frame
    framing_regions: tuple[FramingRegion, ...] = SAGITTAL_FRAMING_REGIONS
    # Stage E - what each orientation costs this exercise
    orientation_notes: tuple[OrientationNote, ...] = SAGITTAL_ORIENTATION_NOTES


# fewer switches than this never reject - one or two is someone walking past
MIN_SWITCHES_TO_REJECT = 4

# switches per tracked second above which the track is unusable (a rate so
# long and short clips are treated the same)
MAX_SWITCH_RATE_PER_SECOND = 0.5

RETRY_TIPS = [
    "Record from the side, roughly level with your hips.",
    "Keep your entire body - head to feet - inside the frame.",
    "Place the camera far enough away and keep it still.",
    "Make sure the area is reasonably well lit.",
]


# --- File-level checks (before detection) ---


def validate_file(video: VideoMetadata, config: ValidationConfig) -> ValidationResult:
    """Checks that only need the file metadata, before detection."""
    result = ValidationResult()
    result.metrics.update(
        width=video.width,
        height=video.height,
        fps=round(video.fps, 2),
        frame_count=video.frame_count,
        duration=round(video.duration, 2),
        codec=video.fourcc,
    )

    if not video.readable:
        result.add_error(
            "We couldn't read this video file. It may be corrupted or in an "
            "unsupported format - try re-exporting it as an MP4.",
            RejectionCode.VIDEO_READ_ERROR,
        )
        return result

    if video.duration and video.duration < config.min_duration:
        result.add_error(
            f"This clip is only {video.duration:.1f} seconds long - too short to "
            "contain a complete squat. Record your full set and upload again.",
            RejectionCode.VIDEO_TOO_SHORT,
        )
    if video.duration and video.duration > config.max_duration:
        result.add_error(
            f"This clip is {video.duration:.0f} seconds long, which is more than "
            f"FormFix currently analyses ({config.max_duration:.0f} seconds). "
            "Trim it to a single set and upload again.",
            RejectionCode.VIDEO_TOO_LONG,
        )
    if video.width < 200 or video.height < 200:
        result.add_error(
            "The video resolution is too low for reliable body tracking. "
            "Record at a higher resolution if you can.",
            RejectionCode.LOW_RESOLUTION,
        )
    return result


# --- Stage A - pose coverage ---


def pose_detection_ratio(pose: FramePoseData) -> float:
    return float(pose.pose_found.mean()) if pose.frame_count else 0.0


# --- Stage B / C - key landmarks and usable frames ---


def usable_frame_ratios(
    pose: FramePoseData,
    min_visibility: float,
    core_landmarks: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, float]:
    """Per side, share of frames where every core landmark is above the visibility
    floor. Over the whole clip, so briefly losing a joint is fine."""
    frames = pose.frame_count
    ratios: dict[str, float] = {}
    for side, ids in (core_landmarks or CORE_SIDE_LANDMARKS).items():
        if not frames:
            ratios[side] = 0.0
            continue
        ok = pose.valid[:, list(ids)] & (pose.visibility[:, list(ids)] >= min_visibility)
        ratios[side] = float(np.mean(np.all(ok, axis=1)))
    return ratios


def mean_side_visibility(
    pose: FramePoseData, side_landmarks: dict[str, tuple[int, ...]] | None = None
) -> dict[str, float]:
    """Mean visibility of each side's exercise landmarks, over found frames."""
    found = pose.pose_found
    scores: dict[str, float] = {}
    for side, ids in (side_landmarks or SIDE_LANDMARKS).items():
        vis = pose.visibility[found][:, list(ids)]
        scores[side] = float(np.mean(vis)) if vis.size else 0.0
    return scores


def select_analysis_side(
    pose: FramePoseData,
    min_visibility: float,
    core_landmarks: dict[str, tuple[int, ...]] | None = None,
    side_landmarks: dict[str, tuple[int, ...]] | None = None,
) -> tuple[str, dict[str, float]]:
    """Pick the side with the best evidence (usable-frame ratio, then visibility).
    For two-arm exercises both arms are still measured."""
    usable = usable_frame_ratios(pose, min_visibility, core_landmarks)
    visibility = mean_side_visibility(pose, side_landmarks)
    best = max(usable, key=lambda side: (round(usable[side], 3), visibility[side]))
    logger.info(
        "Analysis side: %s (usable left %.2f / right %.2f)",
        best,
        usable["left"],
        usable["right"],
    )
    return best, {side: round(value, 3) for side, value in usable.items()}


# --- Stage E - camera orientation ---


def frontality_series(pose: FramePoseData, width: int = 1, height: int = 1) -> np.ndarray:
    """
    Per-frame max(shoulder gap, hip gap) / torso length - small side-on, close to
    1 front-on. Has to be in pixels: with normalised coordinates a phone clip was
    inflated 1.8x and normal side views came out as frontal.
    """
    found = pose.pose_found
    xy = pose.xy_raw[found]
    if xy.shape[0] < 5:
        return np.array([], dtype=np.float64)

    px = np.asarray(xy[:, :, 0], dtype=np.float64) * float(max(width, 1))
    py = np.asarray(xy[:, :, 1], dtype=np.float64) * float(max(height, 1))

    shoulder_sep = np.abs(px[:, LEFT_SHOULDER] - px[:, RIGHT_SHOULDER])
    hip_sep = np.abs(px[:, LEFT_HIP] - px[:, RIGHT_HIP])
    mid_shoulder_x = (px[:, LEFT_SHOULDER] + px[:, RIGHT_SHOULDER]) / 2
    mid_hip_x = (px[:, LEFT_HIP] + px[:, RIGHT_HIP]) / 2
    mid_shoulder_y = (py[:, LEFT_SHOULDER] + py[:, RIGHT_SHOULDER]) / 2
    mid_hip_y = (py[:, LEFT_HIP] + py[:, RIGHT_HIP]) / 2
    torso_len = np.hypot(mid_hip_x - mid_shoulder_x, mid_hip_y - mid_shoulder_y)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(torso_len > 1e-6, np.maximum(shoulder_sep, hip_sep) / torso_len, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    return ratio if ratio.size >= 5 else np.array([], dtype=np.float64)


def frontality_ratio(pose: FramePoseData, width: int = 1, height: int = 1) -> float:
    """Median of frontality_series, NaN if it can't be judged."""
    ratio = frontality_series(pose, width, height)
    return float(np.median(ratio)) if ratio.size else float("nan")


def view_stability(pose: FramePoseData, width: int = 1, height: int = 1) -> float:
    """IQR of the frontality ratio - a big spread means the camera angle changes
    during the clip."""
    ratio = frontality_series(pose, width, height)
    if ratio.size == 0:
        return float("nan")
    return float(np.percentile(ratio, 75) - np.percentile(ratio, 25))


def classify_orientation(ratio: float, config: ValidationConfig) -> tuple[CameraOrientation, float]:
    """
    Frontality ratio -> orientation and a side-view confidence (linear between
    the bands):

        <= good ratio     -> SIDE,          confidence 1.0
        between           -> DIAGONAL_SIDE, confidence 1..0
        >= frontal ratio  -> FRONTAL,       confidence 0.0
    """
    if not np.isfinite(ratio):
        return CameraOrientation.UNKNOWN, 0.0

    good, frontal = config.side_view_good_ratio, config.side_view_frontal_ratio
    span = max(frontal - good, 1e-6)
    confidence = float(np.clip((frontal - ratio) / span, 0.0, 1.0))

    if ratio <= good:
        return CameraOrientation.SIDE, 1.0
    if ratio >= frontal:
        return CameraOrientation.FRONTAL, 0.0
    return CameraOrientation.DIAGONAL_SIDE, confidence


# --- The pose-track validation itself ---


def validate_pose_track(
    video: VideoMetadata,
    pose: FramePoseData,
    config: ValidationConfig,
) -> ValidationResult:
    """Can this pose track be analysed? Only rejects when the evidence is missing."""
    result = ValidationResult()

    # --- Stage A: was a person found often enough? ---
    ratio = pose_detection_ratio(pose)
    result.metrics["pose_frame_ratio"] = round(ratio, 3)

    if ratio == 0.0:
        result.add_error(
            "We couldn't detect a person in this video. Make sure you are "
            "clearly visible and well lit, then try again.",
            RejectionCode.NO_POSE_DETECTED,
        )
        return result

    if ratio < config.min_pose_frame_ratio:
        result.add_error(
            "We could only track your body in "
            f"{ratio:.0%} of the video, which isn't enough for a reliable "
            "analysis. Improve the lighting, keep your whole body in view, and "
            "keep the camera still.",
            RejectionCode.INSUFFICIENT_POSE_COVERAGE,
        )

    # --- more than one person? ---
    # what matters is whether the tracker stayed on one person, not how busy
    # the gym was
    detected = pose.n_poses[pose.pose_found]
    multi_ratio = float((detected > 1).mean()) if detected.size else 0.0
    tracked_frames = int(pose.pose_found.sum())
    switches = int(getattr(pose, "identity_switches", 0))
    switch_rate = switches / max(1.0, tracked_frames / max(video.fps, 1.0))

    result.metrics["multi_person_ratio"] = round(multi_ratio, 3)
    result.metrics["identity_switches"] = switches
    result.metrics["identity_switch_rate"] = round(switch_rate, 3)

    if switches >= MIN_SWITCHES_TO_REJECT and switch_rate >= MAX_SWITCH_RATE_PER_SECOND:
        result.add_error(
            "FormFix kept losing track of which person to analyse in this video. "
            "Record with the person being analysed clearly closest to the camera.",
            RejectionCode.MULTIPLE_PEOPLE,
        )
    elif switches > 0:
        result.add_warning(
            "Other people move through this video and tracking was briefly "
            "interrupted. FormFix followed the person nearest the camera, but a "
            "clearer recording is more reliable."
        )
    elif multi_ratio >= config.multi_person_warn_ratio:
        result.add_warning(
            "Another person appears in part of the video. FormFix analysed the "
            "person nearest the camera and tracked them throughout."
        )

    # --- Stage B + C: key landmarks and usable frames ---
    usable = usable_frame_ratios(pose, config.min_key_landmark_visibility, config.core_landmarks)
    side, side_scores = select_analysis_side(
        pose, config.min_key_landmark_visibility, config.core_landmarks, config.side_landmarks
    )
    best_usable = usable[side]

    result.selected_side = side
    result.metrics["usable_frame_ratio"] = round(best_usable, 3)
    result.metrics["usable_frame_ratio_by_side"] = {k: round(v, 3) for k, v in usable.items()}
    result.metrics["side_visibility"] = {
        k: round(v, 3) for k, v in mean_side_visibility(pose, config.side_landmarks).items()
    }
    result.metrics["side_scores"] = side_scores
    result.metrics["average_landmark_visibility"] = round(
        float(np.mean(pose.visibility[pose.pose_found])) if pose.pose_found.any() else 0.0, 3
    )

    if best_usable < config.min_usable_frame_ratio:
        result.add_error(
            f"We couldn't clearly track your {config.core_landmarks_label} for enough "
            f"of the video (only {best_usable:.0%} of frames were measurable). "
            "Record again with your whole body visible, in good light, and keep "
            "the camera still.",
            RejectionCode.IMPORTANT_LANDMARKS_MISSING,
        )
    elif best_usable < config.min_usable_frame_ratio + 0.15:
        result.add_warning(
            f"Body tracking was intermittent ({best_usable:.0%} of frames were "
            "measurable), so some measurements rest on fewer frames than usual."
        )

    # --- Stage D: framing ---
    _check_framing(pose, config, result)

    # --- Stage E: camera orientation (never rejects on its own) ---
    _assess_camera_angle(pose, config, result, video.width, video.height)

    return result


def _check_framing(pose: FramePoseData, config: ValidationConfig, result: ValidationResult) -> None:
    """
    Check the important body regions stay in frame. Uses the raw coordinates
    because interpolation would hide cropping.
    """
    margin = config.framing_margin
    found = pose.pose_found
    if not found.any():
        return

    out_ratios: dict[str, float] = {}
    for region in config.framing_regions:
        pts = pose.xy_raw[found][:, list(region.landmarks), :]
        with np.errstate(invalid="ignore"):
            outside = (
                (pts[:, :, 0] < margin)
                | (pts[:, :, 0] > 1 - margin)
                | (pts[:, :, 1] < margin)
                | (pts[:, :, 1] > 1 - margin)
            )
        per_frame = np.nanmax(outside.astype(float), axis=1)
        out_ratios[region.name] = float(np.nanmean(per_frame)) if per_frame.size else 0.0

    result.metrics["framing_out_of_frame_ratio"] = {k: round(v, 3) for k, v in out_ratios.items()}

    for region in config.framing_regions:
        ratio = out_ratios.get(region.name, 0.0)
        if ratio <= config.framing_warn_tolerance:
            continue
        severe = ratio > config.framing_fail_tolerance

        if not region.critical:
            message = (region.fail_message if severe else region.warn_message) or (
                f"Your {region.name} leave the camera frame during part of the movement, "
                "which can reduce the accuracy of some measurements."
            )
            result.add_warning(message, region.limited_metric)
            continue

        if severe:
            result.add_error(
                region.fail_message
                or (
                    f"Your {region.name} are outside the camera frame for most of the "
                    "video, so the movement cannot be measured. Move the camera further "
                    "away so your whole body stays visible."
                ),
                RejectionCode.BODY_OUT_OF_FRAME,
            )
        else:
            result.add_warning(
                region.warn_message
                or (
                    f"Your {region.name} leave the camera frame during part of the "
                    "movement, which can reduce the accuracy of some measurements."
                )
            )


def _assess_camera_angle(
    pose: FramePoseData,
    config: ValidationConfig,
    result: ValidationResult,
    width: int = 1,
    height: int = 1,
) -> None:
    """Estimate the camera angle and note what it costs the exercise. Never stops a run."""
    ratio = frontality_ratio(pose, width, height)
    orientation, confidence = classify_orientation(ratio, config)
    spread = view_stability(pose, width, height)

    result.orientation = orientation
    result.side_view_confidence = confidence
    result.metrics["frontality_ratio"] = None if not np.isfinite(ratio) else round(ratio, 3)
    result.metrics["frontality_spread"] = None if not np.isfinite(spread) else round(spread, 3)
    result.metrics["camera_orientation"] = orientation.value
    result.metrics["side_view_confidence"] = round(confidence, 3)

    if np.isfinite(spread) and spread > config.view_stability_spread:
        # camera angle changes partway through
        result.add_warning(
            "The camera angle appears to change during this recording. FormFix "
            "estimated one viewing angle for the whole clip, so some measurements "
            "may be less accurate than usual - a single continuous shot is more "
            "reliable."
        )

    note = next((n for n in config.orientation_notes if n.orientation is orientation), None)
    if note is not None:
        result.add_warning(note.message, note.primary_metric)
        result.limited_metrics.extend(
            metric for metric in note.limited_metrics if metric not in result.limited_metrics
        )


def merge(*results: ValidationResult) -> ValidationResult:
    merged = ValidationResult()
    for partial in results:
        merged.errors.extend(partial.errors)
        merged.warnings.extend(partial.warnings)
        merged.metrics.update(partial.metrics)
        merged.reason_codes.extend(partial.reason_codes)
        merged.limited_metrics.extend(
            m for m in partial.limited_metrics if m not in merged.limited_metrics
        )
        if partial.orientation is not CameraOrientation.UNKNOWN:
            merged.orientation = partial.orientation
            merged.side_view_confidence = partial.side_view_confidence
        if partial.selected_side:
            merged.selected_side = partial.selected_side
        merged.valid = merged.valid and partial.valid
    return merged
