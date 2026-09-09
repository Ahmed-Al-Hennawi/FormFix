"""
The lat-pulldown pipeline, in order:

    probe video -> validate file -> detect pose -> validate recording ->
    clean + smooth -> select side -> measure per frame -> detect reps ->
    top-position baseline -> trunk excursion -> evaluate rules -> reliability
    -> feedback -> render annotated video -> export

This module owns none of that logic, only the order. Measurement is in
metrics.py, judgement in rules.py, wording in feedback.py, thresholds in
config.py.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np

from analysis import filters, pose_detector, smoothing, trimming, validation
from analysis.annotation import (
    UPPER_BODY_DRAWN,
    FrameState,
    HudLine,
    JointAngle,
    OverlayEvent,
    render_annotated_video,
)
from analysis.export import export_result
from analysis.models import (
    LEFT_HIP,
    LEFT_SHOULDER,
    METRIC_PULLDOWN_ROM,
    METRIC_PULLDOWN_TORSO,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    UPPER_BODY_SIDE_LANDMARKS,
    AnalysisFailure,
    CameraOrientation,
    ExerciseAnalysisResult,
    FailureCode,
    ProgressCallback,
    PulldownPhase,
    PulldownRep,
    RejectionCode,
    RuleStatus,
    ValidationResult,
    VideoMetadata,
)
from analysis.video_processor import new_session_dir, probe_video
from exercises.common import confidence as common_confidence
from exercises.common import plausibility, uncertainty
from exercises.common.failures import failure_code_for
from exercises.common.metrics import (
    PHASE_EXTREME,
    PHASE_RETURN,
    PHASE_TOWARDS,
    segment_at,
)

from . import feedback as feedback_mod
from . import metrics as metrics_mod
from . import phases as phases_mod
from .config import (
    DEFAULT_CONFIG,
    RECORDING_TIPS,
    PulldownConfig,
    rule_specs,
)
from .landmarks import ARM_CHAINS, landmark_ids
from .phases import detect_reps, named_phases
from .rules import HIGHLIGHT_ROLES, evaluate_all

logger = logging.getLogger(__name__)

# seconds either side of the evidence frame that a banner stays up
EVENT_CONTEXT_SECONDS = 0.5

# Generic rep segment -> the caption the pulldown shows for it.
SEGMENT_PHASES = {
    PHASE_TOWARDS: PulldownPhase.PULLING,
    PHASE_EXTREME: PulldownPhase.BOTTOM,
    PHASE_RETURN: PulldownPhase.RETURNING,
}

# what the overlay says about a flagged rep. The part before the dash is the
# small label by the joints; the whole line goes in the status pill.
BANNER_TEXT = {
    "pulldown_rom": "Pull range - stopping short",
    "pulldown_torso": "Torso movement - swinging",
}

# what we say on a rule the recording couldn't support
LIMITATION_TEXT = {
    METRIC_PULLDOWN_TORSO: (
        "Torso movement is only visible from a side or three-quarter view, so it was "
        "not assessed for this recording."
    ),
    METRIC_PULLDOWN_ROM: (
        "Your arms were not visible clearly enough to measure how far the movement " "travelled."
    ),
}


def smooth_movement_signal(series, config, fps: float):
    """
    Smooth the signal that drives rep segmentation and the turning-point
    measurements. Which filter runs is a config choice (ANGLE_FILTER): an EMA lags
    exactly where the range-of-motion angles are read, but a 2 Hz Butterworth eats
    real signal on a sharp turnaround. Default is "ema".
    """
    name = getattr(config, "ANGLE_FILTER", "ema")
    if name == "ema":
        return smoothing.smooth_series(series, config.ANGLE_EMA_ALPHA)
    return filters.apply_named_filter(name, series, fps)


def analyse_lat_pulldown(
    video_path: Path,
    progress: ProgressCallback | None = None,
    config: PulldownConfig = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    export_root: Path | None = None,
) -> ExerciseAnalysisResult:
    """Analyse one lat-pulldown video end to end. Raises AnalysisFailure if the
    recording can't be analysed."""
    started = time.monotonic()
    report = progress or (lambda stage, fraction, message: None)
    try:
        return _run(video_path, report, config, output_dir, export_root, started)
    except AnalysisFailure:
        raise
    except Exception as exc:
        logger.exception("Unexpected lat-pulldown analysis error")
        raise AnalysisFailure(
            FailureCode.ANALYSIS_ERROR,
            "Something went wrong while analysing this video.",
            suggestions=["Try uploading the clip again, or try a different recording."],
        ) from exc


def _run(
    video_path: Path,
    report: ProgressCallback,
    config: PulldownConfig,
    output_dir: Path | None,
    export_root: Path | None,
    started: float,
) -> ExerciseAnalysisResult:
    # --- 1. video ---
    report("prepare", 0.0, "Preparing video")
    video = probe_video(video_path)
    validation_config = validation_config_for(config)

    report("validate", 0.1, "Checking recording")
    file_check = validation.validate_file(video, validation_config)
    if not file_check.valid:
        raise AnalysisFailure(
            failure_code_for(file_check),
            file_check.errors[0],
            suggestions=list(RECORDING_TIPS),
            validation=file_check,
        )

    # --- 2. pose detection ---
    report("landmarks", 0.0, "Detecting body landmarks")
    pose = pose_detector.detect_poses(
        video,
        detection_confidence=config.POSE_DETECTION_CONFIDENCE,
        presence_confidence=config.POSE_PRESENCE_CONFIDENCE,
        tracking_confidence=config.TRACKING_CONFIDENCE,
        on_frame=lambda i, n: report("landmarks", (i + 1) / max(n, 1), "Detecting body landmarks"),
    )

    # --- 3. recording validation ---
    report("validate", 0.6, "Checking recording quality")
    track_check = validation.validate_pose_track(video, pose, validation_config)
    checks = validation.merge(file_check, track_check)
    logger.info("Recording quality: %s | %s", checks.quality.value, checks.diagnostics())
    if not checks.valid:
        raise AnalysisFailure(
            failure_code_for(checks),
            checks.errors[0],
            suggestions=checks.errors[1:] + list(RECORDING_TIPS),
            validation=checks,
        )

    # --- 4. clean + smooth ---
    report("measure", 0.05, "Measuring movement")
    interpolated = smoothing.interpolate_short_gaps(pose, config.MAX_SHORT_GAP_FRAMES)
    smoothing.ema_smooth(pose, config.EMA_ALPHA)

    # --- 5. side selection ---
    # both arms get measured; this only picks whose numbers get quoted
    side, side_scores = validation.select_analysis_side(
        pose,
        config.MIN_KEY_LANDMARK_VISIBILITY,
        validation_config.core_landmarks,
        validation_config.side_landmarks,
    )
    checks.selected_side = side
    checks.metrics["analysis_side"] = side
    checks.metrics["side_scores"] = side_scores

    # --- 6. per-frame measurements ---
    report("measure", 0.4, "Measuring joint angles")
    metrics = metrics_mod.compute_frame_metrics(video, pose, side, config)
    facing, facing_confidence = metrics_mod.estimate_facing(video, pose, config)
    metrics_mod.apply_posterior_lean(metrics, facing)

    elbow_series = metrics_mod.movement_signal(metrics)
    elbow_smoothed = smooth_movement_signal(elbow_series, config, video.fps)

    valid_ratio = float(np.isfinite(elbow_smoothed).mean()) if len(elbow_smoothed) else 0.0
    checks.metrics["valid_elbow_angle_ratio"] = round(valid_ratio, 3)
    if valid_ratio < config.MIN_VALID_FRAME_RATIO:
        checks.add_error(
            "We couldn't measure your arm movement in enough of the video.",
            RejectionCode.INSUFFICIENT_VALID_FRAMES,
        )
        raise AnalysisFailure(
            FailureCode.INSUFFICIENT_VISIBILITY,
            "We couldn't measure your arm movement in enough of the video.",
            suggestions=list(RECORDING_TIPS),
            validation=checks,
        )

    # --- 7. phases + reps ---
    report("reps", 0.0, "Detecting repetitions")
    resting_reference = phases_mod.resting_extension(elbow_smoothed, config)
    detection = detect_reps(elbow_smoothed, pose.timestamps, config, resting_reference)
    if not detection.reps:
        checks.add_error(
            "No complete lat-pulldown repetition was detected.",
            RejectionCode.NO_COMPLETE_REPETITION,
        )
        raise AnalysisFailure(
            FailureCode.NO_COMPLETE_REPETITION,
            "We couldn't detect a complete lat pulldown clearly in this video."
            + (
                f" {detection.partial_movements} partial movement(s) were seen but did not "
                "qualify as full repetitions."
                if detection.partial_movements
                else ""
            ),
            suggestions=[
                "Start with your arms extended, pull the bar to your upper chest, and let "
                "your arms return fully between repetitions.",
                *RECORDING_TIPS,
            ],
            validation=checks,
        )

    # --- 8. baseline + per-rep measurements ---
    baseline = metrics_mod.top_baseline(
        metrics,
        detection.phases,
        detection.reps[0].start_frame if detection.reps else None,
        config,
    )
    torso_mode = metrics_mod.apply_torso_excursion(metrics, baseline, config)
    reps = metrics_mod.build_reps(detection.reps, metrics, pose, side, torso_mode, config)

    # --- 8b. is this actually a pulldown? ---
    # a row and a curl also produce clean elbow reps. What separates a pulldown
    # is where the movement starts: hands above the shoulders.
    peak_wrist_rise = _peak_wrist_rise(metrics)
    plausible = plausibility.check_pulldown(reps, peak_wrist_rise)
    if not plausible.plausible:
        checks.add_error(plausible.message, RejectionCode.EXERCISE_MISMATCH)
        raise AnalysisFailure(
            FailureCode.EXERCISE_MISMATCH,
            plausible.message,
            suggestions=plausibility.MISMATCH_TIPS,
            validation=checks,
        )

    # --- 9. rules + feedback ---
    report("rules", 0.0, "Assessing technique")
    rule_results = evaluate_all(
        reps,
        config,
        metrics=metrics,
        pose=pose,
        orientation=checks.orientation,
        side_view_confidence=float(checks.side_view_confidence),
        side=side,
        recording_quality=checks.quality.value,
    )
    _apply_recording_limitations(rule_results, checks)

    # attach the published measurement error to every finding
    _specs = {spec.rule_id: spec for spec in rule_specs(config)}
    uncertainty_notes = uncertainty.annotate_uncertainty(
        rule_results,
        _specs,
        view_support={
            rule_id: common_confidence.view_support(
                spec.supported_views,
                checks.orientation,
                float(checks.side_view_confidence),
                frontal_plane=False,
            )
            for rule_id, spec in _specs.items()
        },
        strict=bool(getattr(config, "UNCERTAINTY_STRICT", False)),
    )
    checks.metrics["measurement_uncertainty"] = [note.as_dict() for note in uncertainty_notes]

    report("feedback", 0.0, "Building feedback")
    summary = feedback_mod.build_summary(reps, detection.partial_movements, rule_results)

    # --- 10. annotated video ---
    report("render", 0.0, "Rendering analysed video")
    # cv2.VideoWriter doesn't create the folder and doesn't error if it can't
    # open the file - it silently drops every frame
    session_dir = output_dir or new_session_dir()
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Output directory %s is unusable; using a scratch directory", session_dir)
        session_dir = new_session_dir()
    annotated_path = session_dir / "annotated.mp4"
    # which frames to render. Frame numbers stay original, so the timestamps in
    # the feedback still refer to the upload.
    render_window = trimming.compute_render_window(reps, pose.frame_count, video.fps)
    frame_states = _frame_states(detection, reps, metrics, pose.frame_count)
    events = _overlay_events(rule_results, video, pose.frame_count, side)
    chain = ARM_CHAINS[side]
    try:
        final_path = render_annotated_video(
            video,
            pose,
            metrics,
            frame_states,
            events,
            side,
            annotated_path,
            # side-on, the overlay holds occluded limbs to a higher confidence bar
            camera_orientation=checks.orientation.value,
            on_frame=lambda i, n: report("render", (i + 1) / max(n, 1), "Rendering analysed video"),
            emphasis_landmarks=chain.all_ids,
            marker_landmarks=(chain.elbow,),
            marker_label="CONTRACTED",
            # the ROM rule reads the elbow angle, so draw it on the elbow
            angle_joints=(JointAngle("Elbow", chain.shoulder, chain.elbow, chain.wrist),),
            # upper body only. Nothing below the hip is measured, and the legs
            # are half under the seat pad where tracking is at its worst
            drawn_landmarks=UPPER_BODY_DRAWN,
            frame_range=(render_window.start_frame, render_window.end_frame),
        )
    except Exception:
        logger.exception("Annotated-video rendering failed")
        final_path = None
        checks.add_warning("The annotated video could not be rendered for this run.")

    # --- 11. result + export ---
    debug = _debug_block(
        config,
        checks,
        baseline,
        detection,
        reps,
        rule_results,
        side,
        interpolated,
        torso_mode,
        facing,
        facing_confidence,
        resting_reference,
        started,
    )
    # what the renderer actually wrote, for the technical panel
    debug["render_window"] = render_window.as_dict()

    result = ExerciseAnalysisResult(
        success=True,
        exercise="Lat Pulldown",
        analysis_side=side,
        video=video,
        validation=checks,
        reps=reps,
        rule_results=rule_results,
        summary=summary,
        annotated_video_path=final_path,
        debug=debug,
        frame_metrics=metrics,
        exercise_id="pulldown",
    )

    if export_root is not None:
        export_dir = export_result(result, export_root, _run_id(video_path))
        if export_dir:
            debug["export_dir"] = str(export_dir)

    report("complete", 1.0, "Complete")
    return result


# --- Helpers ---


def validation_config_for(config: PulldownConfig) -> validation.ValidationConfig:
    """
    Pulldown recording requirements in the generic validation vocabulary. Three
    things differ from the squat: the core landmarks are the arm chain, the framing
    regions are the arms and trunk (a pulldown's hands go above the head, which is
    where clips get cropped), and a front-on recording can't show trunk lean.
    """
    return validation.ValidationConfig(
        min_duration=config.VIDEO_MIN_DURATION,
        max_duration=config.VIDEO_MAX_DURATION,
        min_pose_frame_ratio=config.MIN_POSE_FRAME_RATIO,
        min_key_landmark_visibility=config.MIN_KEY_LANDMARK_VISIBILITY,
        min_usable_frame_ratio=config.MIN_USABLE_FRAME_RATIO,
        multi_person_warn_ratio=config.MULTI_PERSON_WARN_RATIO,
        multi_person_fail_ratio=config.MULTI_PERSON_FAIL_RATIO,
        side_view_good_ratio=config.SIDE_VIEW_GOOD_RATIO,
        side_view_frontal_ratio=config.SIDE_VIEW_FRONTAL_RATIO,
        framing_margin=config.FRAMING_MARGIN,
        framing_warn_tolerance=config.FRAMING_WARN_TOLERANCE,
        framing_fail_tolerance=config.FRAMING_FAIL_TOLERANCE,
        view_stability_spread=config.VIEW_STABILITY_SPREAD,
        core_landmarks=dict(validation.CORE_ARM_LANDMARKS),
        core_landmarks_label="shoulders, elbows and wrists",
        side_landmarks=dict(UPPER_BODY_SIDE_LANDMARKS),
        framing_regions=(
            validation.FramingRegion(
                "shoulders and hips", (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
            ),
            validation.FramingRegion(
                "arms",
                landmark_ids("left", ("elbow", "wrist")) + landmark_ids("right", ("elbow", "wrist")),
                critical=False,
                limited_metric=METRIC_PULLDOWN_ROM,
                warn_message=(
                    "Your hands leave the top of the frame during part of the movement, "
                    "which can make the range-of-motion check less reliable."
                ),
                fail_message=(
                    "Your arms are outside the camera frame for most of the set, so the "
                    "range of motion could not be measured. Step back, or tilt the camera "
                    "up, so your hands stay visible at the top of every repetition."
                ),
            ),
        ),
        orientation_notes=(
            validation.OrientationNote(
                CameraOrientation.FRONTAL,
                "This recording looks front-on or rear-on. FormFix analysed your arm "
                "movement, but leaning backwards happens towards or away from the camera "
                "in this view, so the torso check is not assessed for this clip. Film "
                "from the side or at a three-quarter angle to have it checked.",
                primary_metric=METRIC_PULLDOWN_TORSO,
            ),
            validation.OrientationNote(
                CameraOrientation.DIAGONAL_SIDE,
                "The camera is not fully side-on. FormFix could analyse this pulldown, "
                "although a clearer side or three-quarter view may improve the accuracy "
                "of the torso measurement.",
            ),
            validation.OrientationNote(
                CameraOrientation.UNKNOWN,
                "The camera angle could not be estimated from this recording, so the "
                "measurements may be less accurate than usual.",
            ),
        ),
    )


def _apply_recording_limitations(rule_results, checks: ValidationResult) -> None:
    """Switch off the rules this recording can't support, and only those. The
    affected rule becomes NOT_EVALUABLE with a reason."""
    if not checks.limited_metrics:
        return
    limited = set(checks.limited_metrics)
    for rule in rule_results:
        if rule.rule_id not in limited and rule.metric not in limited:
            continue
        rule.status = RuleStatus.NOT_EVALUABLE
        rule.limitation = LIMITATION_TEXT.get(
            rule.metric, "This check is not reliable for the camera angle in this recording."
        )
        rule.explanation = rule.limitation
        rule.correction = ""
        for outcome in rule.per_rep:
            outcome.status = RuleStatus.NOT_EVALUABLE


def _frame_states(detection, reps: list[PulldownRep], metrics, frame_count: int):
    """Per-frame HUD state: phase, rep counter, contracted-window flag."""
    phases = named_phases(detection.phases)
    states: list[FrameState] = []
    total = len(reps)
    for f in range(frame_count):
        phase = phases[f] if f < len(phases) else PulldownPhase.UNKNOWN
        rep_number = 0
        in_window = False
        for rep in reps:
            if rep.start_frame <= f <= rep.end_frame:
                rep_number = rep.number
                seg = rep.segmentation
                in_window = seg.extreme_start_frame <= f <= seg.extreme_end_frame
                # inside a committed rep the caption comes from that rep's own segmentation
                segment = segment_at(seg, f)
                if segment is not None:
                    phase = SEGMENT_PHASES[segment]
                break
            if f > rep.end_frame:
                rep_number = rep.number
        # the elbow angle is drawn on the elbow; only the trunk gets a HUD chip
        extra: list[HudLine] = []
        metric = metrics[f] if f < len(metrics) else None
        if metric is not None and metric.valid and np.isfinite(metric.torso_angle):
            extra.append(HudLine(f"TORSO {metric.torso_angle:.0f}"))
        states.append(
            FrameState(
                phase=phase,
                rep_number=rep_number,
                total_reps=total,
                is_bottom_window=in_window,
                extra=tuple(extra),
            )
        )
    return states


def _overlay_events(rule_results, video: VideoMetadata, frame_count: int, side: str):
    """One banner per flagged rep, around its evidence frame, with the joints worth
    highlighting."""
    context = max(int(EVENT_CONTEXT_SECONDS * video.fps), 3)
    events: list[OverlayEvent] = []
    for rule in rule_results:
        text = BANNER_TEXT.get(rule.rule_id, rule.title)
        highlights = _highlight_ids(rule.rule_id, side)
        for outcome in rule.per_rep:
            if outcome.status not in (RuleStatus.FAIL, RuleStatus.WARNING):
                continue
            if outcome.evidence_frame < 0:
                continue
            events.append(
                OverlayEvent(
                    start_frame=max(outcome.evidence_frame - context, 0),
                    end_frame=min(outcome.evidence_frame + context, frame_count - 1),
                    text=text,
                    level=outcome.status.value,
                    highlight_landmarks=highlights,
                )
            )
    return events


def _highlight_ids(rule_id: str, side: str) -> tuple[int, ...]:
    """A rule's highlight roles -> MediaPipe landmark indices."""
    roles = HIGHLIGHT_ROLES.get(rule_id, ())
    if not roles or side not in ARM_CHAINS:
        return ()
    if rule_id == "pulldown_torso":
        # A trunk finding is about the whole trunk, so light up both sides.
        return tuple(sorted({*landmark_ids("left", roles), *landmark_ids("right", roles)}))
    return landmark_ids(side, roles)


def _debug_block(
    config,
    checks,
    baseline,
    detection,
    reps,
    rule_results,
    side,
    interpolated,
    torso_mode,
    facing,
    facing_confidence,
    resting_reference,
    started,
) -> dict:
    """Developer metrics for the evaluation write-up, not for normal users."""
    return {
        "config": config.as_dict(),
        "rule_specs": [spec.as_dict() for spec in rule_specs(config)],
        "recording_quality": checks.quality.value,
        "validation": checks.diagnostics(),
        "pose_frame_ratio": checks.metrics.get("pose_frame_ratio"),
        "usable_frame_ratio": checks.metrics.get("usable_frame_ratio"),
        "camera_orientation": checks.orientation.value,
        "side_view_confidence": round(float(checks.side_view_confidence), 3),
        "limited_metrics": list(checks.limited_metrics),
        "interpolated_cells": interpolated,
        "analysis_side": side,
        "side_scores": checks.metrics.get("side_scores"),
        "torso_measurement_mode": torso_mode,
        "resting_extension_reference_deg": round(float(resting_reference), 1),
        "rep_detection_thresholds": {
            "rest_level": round(resting_reference - config.REST_MARGIN, 1),
            "start_level": round(resting_reference - config.PULL_START_MARGIN, 1),
            "extreme_level": round(resting_reference - config.CONTRACTED_MARGIN, 1),
            "end_level": round(resting_reference - config.REP_END_MARGIN, 1),
        },
        "facing_direction": {1: "+x", -1: "-x", 0: "unknown"}[facing],
        "facing_confidence": round(float(facing_confidence), 3),
        "top_baseline": {k: (round(v, 2) if np.isfinite(v) else None) for k, v in baseline.items()},
        "partial_movements": detection.partial_movements,
        "partial_reasons": detection.partial_reasons,
        "rep_boundaries": [
            {
                "rep": rep.number,
                "start": rep.start_time,
                "contracted": rep.extreme_time,
                "end": rep.end_time,
                "pull_s": rep.segmentation.towards_duration,
                "contracted_s": rep.segmentation.extreme_duration,
                "return_s": rep.segmentation.return_duration,
                "top_elbow_angle": rep.top_elbow_angle,
                "bottom_elbow_angle": rep.bottom_elbow_angle,
                "rom_degrees": rep.rom_degrees,
                "max_torso_excursion": rep.max_torso_excursion,
                "peak_torso_velocity": rep.peak_torso_velocity,
                "wrist_travel": rep.wrist_travel,
                "valid_frame_ratio": rep.valid_frame_ratio,
                "arms_reliable": rep.arms_reliable,
                "both_arms_ratio": rep.both_arms_ratio,
                "torso_reliable": rep.torso_reliable,
            }
            for rep in reps
        ],
        "rule_reliability": {rule.rule_id: rule.reliability.value for rule in rule_results},
        "rule_persistence": {
            rule.rule_id: [
                {
                    "rep": outcome.rep_number,
                    "status": outcome.status.value,
                    "violating_frames": outcome.violating_frames,
                    "phase_frames": outcome.phase_frames,
                    "violation_ratio": outcome.violation_ratio,
                }
                for outcome in rule.per_rep
            ]
            for rule in rule_results
        },
        "analysis_seconds": round(time.monotonic() - started, 1),
    }


def _run_id(video_path: Path) -> str:
    """Short, non-identifying id for the export bundle."""
    digest = hashlib.sha1(video_path.name.encode("utf-8")).hexdigest()[:8]
    return f"{time.strftime('%Y%m%d-%H%M%S')}-pulldown-{digest}"


def _peak_wrist_rise(metrics) -> float | None:
    """Highest the wrists get above the shoulder line, in trunk lengths. A pulldown
    reaches overhead every rep; a row or a curl never does."""
    best = None
    for frame in metrics:
        value = getattr(frame, "wrist_rise", None)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(number):
            continue
        if best is None or number > best:
            best = number
    return best
