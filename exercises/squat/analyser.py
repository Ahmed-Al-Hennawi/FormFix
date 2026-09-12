"""
The squat pipeline, in order:

    probe video -> validate file -> detect pose -> validate recording ->
    clean + smooth -> select side -> measure per frame -> detect reps ->
    standing baseline -> evaluate rules -> reliability -> feedback ->
    render annotated video -> export

This file only runs the steps in order. Measuring is in metrics.py, judging in
rules.py, wording in feedback.py and thresholds in config.py.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np

from analysis import filters, pose_detector, smoothing, stabilise, trimming, validation
from analysis.annotation import (
    FrameState,
    JointAngle,
    OverlayEvent,
    render_annotated_video,
)
from analysis.export import export_result
from analysis.models import (
    METRIC_DEPTH,
    METRIC_EXTENSION,
    METRIC_HEEL_LIFT,
    METRIC_TORSO_LEAN,
    AnalysisFailure,
    FailureCode,
    Phase,
    ProgressCallback,
    RejectionCode,
    RuleStatus,
    SquatAnalysisResult,
    SquatRep,
    ValidationResult,
    VideoMetadata,
)
from analysis.video_processor import new_session_dir, probe_video
from exercises.common import confidence as common_confidence
from exercises.common import plausibility, uncertainty

from . import feedback as feedback_mod
from . import metrics as metrics_mod
from .config import DEFAULT_CONFIG, SquatConfig, rule_specs
from .landmarks import FRONTAL_PLANE_METRICS as _FRONTAL_PLANE_METRICS
from .landmarks import SIDE_CHAINS, landmark_ids
from .phases import PhaseDetectionResult, detect_reps
from .rules import HIGHLIGHT_ROLES, evaluate_all

logger = logging.getLogger(__name__)

# seconds either side of the evidence frame that a banner stays up
EVENT_CONTEXT_SECONDS = 0.5

# overlay text for a flagged rep - the part before the dash is the short label
BANNER_TEXT = {
    "squat_depth": "Depth - stopping high",
    "torso_lean": "Chest - leaning forward",
    "heel_lift": "Heels - lifting",
    "return_to_standing": "Finish - not standing tall",
    "descent_control": "Tempo - dropping quickly",
}


def smooth_movement_signal(series, config, fps: float):
    """
    Smooth the signal used for rep detection and depth/lockout. The filter is set
    by ANGLE_FILTER: an EMA lags at the turning points, but a 2 Hz Butterworth
    cuts real signal on a sharp turnaround (see docs/filter_selection.md).
    """
    name = getattr(config, "ANGLE_FILTER", "ema")
    if name == "ema":
        return smoothing.smooth_series(series, config.ANGLE_EMA_ALPHA)
    return filters.apply_named_filter(name, series, fps)


def analyse_squat(
    video_path: Path,
    progress: ProgressCallback | None = None,
    config: SquatConfig = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    export_root: Path | None = None,
) -> SquatAnalysisResult:
    """Analyse one squat video end to end. Raises AnalysisFailure if it can't."""
    started = time.monotonic()
    report = progress or (lambda stage, fraction, message: None)

    try:
        return _run(video_path, report, config, output_dir, export_root, started)
    except AnalysisFailure:
        raise
    except Exception as exc:
        logger.exception("Unexpected analysis error")
        raise AnalysisFailure(
            FailureCode.ANALYSIS_ERROR,
            "Something went wrong while analysing this video.",
            suggestions=["Try uploading the clip again, or try a different recording."],
        ) from exc


def _run(
    video_path: Path,
    report: ProgressCallback,
    config: SquatConfig,
    output_dir: Path | None,
    export_root: Path | None,
    started: float,
) -> SquatAnalysisResult:
    # --- 1. video ---
    report("prepare", 0.0, "Preparing video")
    video = probe_video(video_path)

    validation_config = validation_config_for(config)

    report("validate", 0.1, "Checking recording")
    file_check = validation.validate_file(video, validation_config)
    if not file_check.valid:
        raise AnalysisFailure(
            _failure_code_for(file_check),
            file_check.errors[0],
            suggestions=validation.RETRY_TIPS,
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
            _failure_code_for(checks),
            checks.errors[0],
            suggestions=checks.errors[1:] + validation.RETRY_TIPS,
            validation=checks,
        )

    # --- 4. clean + smooth ---
    report("measure", 0.05, "Measuring movement")
    # drop impossible landmark jumps before anything is filled in or smoothed -
    # a low-pass filter would spread a spike instead of removing it
    stabiliser = stabilise.reject_outliers(
        pose,
        video,
        window=config.OUTLIER_WINDOW_FRAMES,
        min_jump_torsos=config.OUTLIER_MIN_JUMP_TORSOS,
    )
    interpolated = smoothing.interpolate_short_gaps(pose, config.MAX_SHORT_GAP_FRAMES)
    smoothing.ema_smooth(pose, config.EMA_ALPHA)

    # --- 5. side selection ---
    side, side_scores = validation.select_analysis_side(pose, config.MIN_KEY_LANDMARK_VISIBILITY)
    checks.selected_side = side
    checks.metrics["analysis_side"] = side
    checks.metrics["side_scores"] = side_scores

    # --- 6. per-frame measurements ---
    report("measure", 0.4, "Measuring joint angles")
    metrics = metrics_mod.compute_frame_metrics(video, pose, side, config)
    knee_series = np.array([m.knee_angle for m in metrics], dtype=np.float64)
    knee_smoothed = smooth_movement_signal(knee_series, config, video.fps)

    valid_ratio = float(np.isfinite(knee_smoothed).mean()) if len(knee_smoothed) else 0.0
    checks.metrics["valid_knee_angle_ratio"] = round(valid_ratio, 3)
    if valid_ratio < config.MIN_VALID_FRAME_RATIO:
        checks.add_error(
            "We couldn't measure your knee movement in enough of the video.",
            RejectionCode.INSUFFICIENT_VALID_FRAMES,
        )
        raise AnalysisFailure(
            FailureCode.INSUFFICIENT_VISIBILITY,
            "We couldn't measure your knee movement in enough of the video.",
            suggestions=validation.RETRY_TIPS,
            validation=checks,
        )

    # --- 7. phases + reps ---
    report("reps", 0.0, "Detecting repetitions")
    detection = detect_reps(knee_smoothed, pose.timestamps, config)
    if not detection.reps:
        checks.add_error("No complete squat repetition was detected.", RejectionCode.NO_SQUAT_MOVEMENT)
        raise AnalysisFailure(
            FailureCode.NO_COMPLETE_SQUAT,
            "We couldn't detect a complete squat clearly in this video."
            + (
                f" {detection.partial_movements} partial movement(s) were seen but did not "
                "qualify as full repetitions."
                if detection.partial_movements
                else ""
            ),
            suggestions=[
                "Start standing, perform several full squats, and finish standing.",
                *validation.RETRY_TIPS,
            ],
            validation=checks,
        )

    # --- 8. baselines + per-rep measurements ---
    baseline = metrics_mod.standing_baseline(
        pose,
        metrics,
        detection.phases,
        detection.reps[0].start_frame if detection.reps else None,
        video,
        side,
        config,
        knee_signal=knee_smoothed,
    )
    heel_foot = metrics_mod.apply_heel_metric(pose, metrics, video, side, baseline, config)
    reps = metrics_mod.build_reps(
        detection.reps, metrics, knee_smoothed, pose, side, config, heel_side=heel_foot
    )

    # --- 8b. is this actually a squat? ---
    # check the user picked the right exercise before giving squat feedback
    plausible = plausibility.check_squat(reps)
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
        baseline["knee_angle"],
        baseline["torso_lean"],
        config,
        metrics=metrics,
        pose=pose,
        orientation=checks.orientation,
        side_view_confidence=float(checks.side_view_confidence),
        side=side,
        recording_quality=checks.quality.value,
    )
    # anything the recording can't support comes back as "not assessed"
    _apply_recording_limitations(rule_results, checks)

    # attach the published measurement error to each finding
    _specs = {spec.rule_id: spec for spec in rule_specs(config)}
    uncertainty_notes = uncertainty.annotate_uncertainty(
        rule_results,
        _specs,
        view_support={
            rule_id: common_confidence.view_support(
                spec.supported_views,
                checks.orientation,
                float(checks.side_view_confidence),
                frontal_plane=spec.metric in _FRONTAL_PLANE_METRICS,
            )
            for rule_id, spec in _specs.items()
        },
        # the filter's own bias counts as systematic error
        systematic={
            "squat_depth": (
                filters.depth_bias_for(getattr(config, "ANGLE_FILTER", "ema")),
                (
                    f"worst-case depth bias of the {getattr(config, 'ANGLE_FILTER', 'ema')} "
                    "smoothing filter, measured by scripts/compare_filters.py"
                ),
            )
        },
        strict=bool(getattr(config, "UNCERTAINTY_STRICT", False)),
    )
    checks.metrics["measurement_uncertainty"] = [note.as_dict() for note in uncertainty_notes]

    report("feedback", 0.0, "Building feedback")
    summary = feedback_mod.build_summary(reps, detection.partial_movements, rule_results)

    # --- 10. annotated video ---
    report("render", 0.0, "Rendering analysed video")
    session_dir = output_dir or new_session_dir()
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Output directory %s is unusable; using a scratch directory", session_dir)
        session_dir = new_session_dir()
    annotated_path = session_dir / "annotated.mp4"
    # which frames to render (frame numbers stay the original ones)
    render_window = trimming.compute_render_window(reps, pose.frame_count, video.fps)
    frame_states = _frame_states(detection, reps, pose.frame_count)
    events = _overlay_events(rule_results, video, pose.frame_count, side)
    try:
        final_path = render_annotated_video(
            video,
            pose,
            metrics,
            frame_states,
            events,
            side,
            annotated_path,
            # side-on, hidden limbs need a higher confidence to be drawn
            camera_orientation=checks.orientation.value,
            on_frame=lambda i, n: report("render", (i + 1) / max(n, 1), "Rendering analysed video"),
            angle_joints=_angle_joints(side),
            frame_range=(render_window.start_frame, render_window.end_frame),
        )
    except Exception as exc:
        logger.exception("Annotated-video rendering failed")
        final_path = None
        checks.add_warning("The annotated video could not be rendered for this run.")
        del exc

    # --- 11. result + export ---
    debug = _debug_block(
        config, checks, baseline, detection, reps, rule_results, side, interpolated, started
    )
    # what the renderer actually wrote, for the technical panel
    debug["render_window"] = render_window.as_dict()
    debug["outlier_rejection"] = stabiliser.as_dict()
    debug["heel_measurement_foot"] = heel_foot or "none"

    result = SquatAnalysisResult(
        success=True,
        exercise="Squat",
        analysis_side=side,
        video=video,
        validation=checks,
        reps=reps,
        rule_results=rule_results,
        summary=summary,
        annotated_video_path=final_path,
        debug=debug,
        frame_metrics=metrics,
    )

    if export_root is not None:
        run_id = _run_id(video_path)
        export_dir = export_result(result, export_root, run_id)
        if export_dir:
            debug["export_dir"] = str(export_dir)

    report("complete", 1.0, "Complete")
    return result


# --- Helpers ---


def validation_config_for(config: SquatConfig) -> validation.ValidationConfig:
    """Squat recording requirements in the format the generic validation expects."""
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
    )


# rejection code -> UI failure code
_FAILURE_CODES: dict[RejectionCode, FailureCode] = {
    RejectionCode.VIDEO_READ_ERROR: FailureCode.INVALID_VIDEO,
    RejectionCode.VIDEO_TOO_SHORT: FailureCode.INVALID_VIDEO,
    RejectionCode.VIDEO_TOO_LONG: FailureCode.INVALID_VIDEO,
    RejectionCode.LOW_RESOLUTION: FailureCode.INVALID_VIDEO,
    RejectionCode.NO_POSE_DETECTED: FailureCode.NO_POSE,
    RejectionCode.INSUFFICIENT_POSE_COVERAGE: FailureCode.INSUFFICIENT_VISIBILITY,
    RejectionCode.IMPORTANT_LANDMARKS_MISSING: FailureCode.INSUFFICIENT_VISIBILITY,
    RejectionCode.INSUFFICIENT_VALID_FRAMES: FailureCode.INSUFFICIENT_VISIBILITY,
    RejectionCode.BODY_OUT_OF_FRAME: FailureCode.BODY_OUT_OF_FRAME,
    RejectionCode.MULTIPLE_PEOPLE: FailureCode.MULTIPLE_PEOPLE,
    RejectionCode.UNSUPPORTED_CAMERA_ANGLE: FailureCode.UNSUITABLE_CAMERA_VIEW,
    RejectionCode.NO_SQUAT_MOVEMENT: FailureCode.NO_COMPLETE_SQUAT,
}

# message for a rule the recording couldn't support
_LIMITATION_TEXT = {
    METRIC_DEPTH: "Depth needs a side view: knee and hip angles cannot be measured "
    "reliably from this camera position.",
    METRIC_TORSO_LEAN: "Forward torso lean is only visible from the side, so it was "
    "not assessed for this recording.",
    METRIC_EXTENSION: "Returning to standing is judged from the knee angle, which "
    "this camera position does not show reliably.",
    METRIC_HEEL_LIFT: "Your heels were not visible enough in this recording, so heel "
    "contact was not assessed.",
}


def _failure_code_for(checks: ValidationResult) -> FailureCode:
    """First known rejection code wins, unknown ones fall back."""
    for code in checks.reason_codes:
        mapped = _FAILURE_CODES.get(code)
        if mapped is not None:
            return mapped
    return FailureCode.INVALID_VIDEO


def _apply_recording_limitations(rule_results, checks: ValidationResult) -> None:
    """Mark the rules this recording can't support as NOT_EVALUABLE, with a reason."""
    if not checks.limited_metrics:
        return
    limited = set(checks.limited_metrics)
    for rule in rule_results:
        if rule.rule_id not in limited:
            continue
        rule.status = RuleStatus.NOT_EVALUABLE
        rule.limitation = _LIMITATION_TEXT.get(
            rule.rule_id, "This check is not reliable for the camera angle in this recording."
        )
        rule.explanation = rule.limitation
        rule.correction = ""
        for outcome in rule.per_rep:
            outcome.status = RuleStatus.NOT_EVALUABLE


def _angle_joints(side: str) -> tuple[JointAngle, ...]:
    """The two angles the rules use (trunk lean is from vertical, so it isn't one)."""
    chain = SIDE_CHAINS.get(side)
    if chain is None:
        return ()
    return (
        JointAngle("Knee", chain.hip, chain.knee, chain.ankle),
        JointAngle("Hip", chain.shoulder, chain.hip, chain.knee),
    )


def _frame_states(
    detection: PhaseDetectionResult,
    reps: list[SquatRep],
    frame_count: int,
) -> list[FrameState]:
    """Per-frame HUD state: phase, rep counter, bottom-window flag."""
    states: list[FrameState] = []
    total = len(reps)
    for f in range(frame_count):
        phase = detection.phases[f] if f < len(detection.phases) else Phase.UNKNOWN
        rep_number = 0
        is_bottom = False
        for rep in reps:
            if rep.start_frame <= f <= rep.end_frame:
                rep_number = rep.number
                is_bottom = rep.bottom_start_frame <= f <= rep.bottom_end_frame
                # inside a rep use that rep's own segmentation, the raw state track
                # can still have labels from an abandoned attempt
                if f < rep.bottom_start_frame:
                    phase = Phase.DESCENDING
                elif is_bottom:
                    phase = Phase.BOTTOM
                else:
                    phase = Phase.ASCENDING
                break
            if f > rep.end_frame:
                rep_number = rep.number  # between reps, show the last one done
        states.append(
            FrameState(phase=phase, rep_number=rep_number, total_reps=total, is_bottom_window=is_bottom)
        )
    return states


def _overlay_events(
    rule_results, video: VideoMetadata, frame_count: int, side: str
) -> list[OverlayEvent]:
    """One banner per flagged rep, shown around its evidence frame."""
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
    if not roles:
        return ()
    if side not in SIDE_CHAINS:
        return ()
    return landmark_ids(side, roles)


def _debug_block(
    config: SquatConfig,
    checks: ValidationResult,
    baseline: dict[str, float],
    detection: PhaseDetectionResult,
    reps: list[SquatRep],
    rule_results,
    side: str,
    interpolated: int,
    started: float,
) -> dict:
    """Debug metrics for the evaluation. Shown with FORMFIX_DEBUG=1 and in the export."""
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
        "standing_baseline": {
            k: (round(v, 2) if np.isfinite(v) else None) for k, v in baseline.items()
        },
        "partial_movements": detection.partial_movements,
        "partial_reasons": detection.partial_reasons,
        "rep_boundaries": [
            {
                "rep": rep.number,
                "start": rep.start_time,
                "bottom": rep.bottom_time,
                "end": rep.end_time,
                "descent_s": rep.descent_duration,
                "bottom_s": rep.bottom_duration,
                "ascent_s": rep.ascent_duration,
                "min_knee_angle": rep.min_knee_angle,
                "max_torso_lean": rep.max_torso_lean,
                "shin_at_bottom": rep.shin_inclination_at_bottom,
                "max_knee_asymmetry": rep.max_knee_asymmetry,
                "valid_frame_ratio": rep.valid_frame_ratio,
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
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{digest}"


# kept for the calibration script
compute_frame_metrics = metrics_mod.compute_frame_metrics
