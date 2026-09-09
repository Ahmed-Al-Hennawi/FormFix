"""
The shoulder-press pipeline, in order:

    probe video -> validate file -> detect pose -> validate recording ->
    clean + smooth -> measure per frame -> detect reps -> evaluate rules ->
    reliability -> feedback -> render annotated video -> export

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
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    METRIC_PRESS_ALIGNMENT,
    METRIC_PRESS_ROM,
    METRIC_PRESS_SYMMETRY,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    UPPER_BODY_SIDE_LANDMARKS,
    AnalysisFailure,
    CameraOrientation,
    ExerciseAnalysisResult,
    FailureCode,
    PressPhase,
    PressRep,
    ProgressCallback,
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

from . import confidence as press_confidence
from . import feedback as feedback_mod
from . import metrics as metrics_mod
from . import phases as phases_mod
from .config import DEFAULT_CONFIG, RECORDING_TIPS, PressConfig, rule_specs
from .landmarks import BOTH_ARM_LANDMARKS
from .phases import detect_reps, named_phases
from .rules import evaluate_all, highlight_ids

logger = logging.getLogger(__name__)

EVENT_CONTEXT_SECONDS = 0.5

# Generic rep segment -> the caption the press shows for it.
SEGMENT_PHASES = {
    PHASE_TOWARDS: PressPhase.PRESSING,
    PHASE_EXTREME: PressPhase.TOP,
    PHASE_RETURN: PressPhase.LOWERING,
}

# what the overlay says about a flagged rep. The part before the dash is the
# small label by the joints; the whole line goes in the status pill.
BANNER_TEXT = {
    "press_symmetry": "Uneven arms - one arm leads",
    "press_alignment": "Wrist position - drifting",
    "press_rom": "Press range - stopping short",
}

LIMITATION_TEXT = {
    METRIC_PRESS_SYMMETRY: (
        "Both arms were not visible clearly enough at the same time to compare them, so "
        "arm symmetry was not assessed."
    ),
    METRIC_PRESS_ALIGNMENT: (
        "Your elbows and wrists were not visible clearly enough to check how the "
        "dumbbells were stacked over your forearms."
    ),
    METRIC_PRESS_ROM: (
        "Your arms were not visible clearly enough to measure how far each press " "travelled."
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


def analyse_shoulder_press(
    video_path: Path,
    progress: ProgressCallback | None = None,
    config: PressConfig = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    export_root: Path | None = None,
) -> ExerciseAnalysisResult:
    """Analyse one shoulder-press video end to end. Raises AnalysisFailure if the
    recording can't be analysed."""
    started = time.monotonic()
    report = progress or (lambda stage, fraction, message: None)
    try:
        return _run(video_path, report, config, output_dir, export_root, started)
    except AnalysisFailure:
        raise
    except Exception as exc:
        logger.exception("Unexpected shoulder-press analysis error")
        raise AnalysisFailure(
            FailureCode.ANALYSIS_ERROR,
            "Something went wrong while analysing this video.",
            suggestions=["Try uploading the clip again, or try a different recording."],
        ) from exc


def _run(
    video_path: Path,
    report: ProgressCallback,
    config: PressConfig,
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
    # a press is bilateral, so both arms are always measured. This only decides
    # whose numbers the interface quotes.
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
    flexion_series = metrics_mod.movement_signal(metrics)
    flexion_smoothed = smooth_movement_signal(flexion_series, config, video.fps)

    valid_ratio = float(np.isfinite(flexion_smoothed).mean()) if len(flexion_smoothed) else 0.0
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
    resting_reference = phases_mod.resting_flexion(flexion_smoothed, config)
    detection = detect_reps(flexion_smoothed, pose.timestamps, config, resting_reference)
    if not detection.reps:
        checks.add_error(
            "No complete shoulder-press repetition was detected.",
            RejectionCode.NO_COMPLETE_REPETITION,
        )
        raise AnalysisFailure(
            FailureCode.NO_COMPLETE_REPETITION,
            "We couldn't detect a complete shoulder press clearly in this video."
            + (
                f" {detection.partial_movements} partial movement(s) were seen but did not "
                "qualify as full repetitions."
                if detection.partial_movements
                else ""
            ),
            suggestions=[
                "Start with the dumbbells at shoulder height, press overhead, and lower "
                "them back under control before the next repetition.",
                *RECORDING_TIPS,
            ],
            validation=checks,
        )

    reps = metrics_mod.build_reps(detection.reps, metrics, pose, config)

    # --- 7b. is this actually an overhead press? ---
    # a bench press or a curl reaches this point with good reps, so check the one
    # thing that separates an overhead movement
    peak_wrist_rise = _peak_wrist_rise(metrics)
    plausible = plausibility.check_press(reps, peak_wrist_rise)
    if not plausible.plausible:
        checks.add_error(plausible.message, RejectionCode.EXERCISE_MISMATCH)
        raise AnalysisFailure(
            FailureCode.EXERCISE_MISMATCH,
            plausible.message,
            suggestions=plausibility.MISMATCH_TIPS,
            validation=checks,
        )

    # --- 8. rules + feedback ---
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
                frontal_plane=press_confidence.is_frontal_plane(spec.metric),
            )
            for rule_id, spec in _specs.items()
        },
        strict=bool(getattr(config, "UNCERTAINTY_STRICT", False)),
    )
    checks.metrics["measurement_uncertainty"] = [note.as_dict() for note in uncertainty_notes]

    report("feedback", 0.0, "Building feedback")
    summary = feedback_mod.build_summary(reps, detection.partial_movements, rule_results)

    # --- 9. annotated video ---
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
            emphasis_landmarks=BOTH_ARM_LANDMARKS,
            marker_landmarks=(LEFT_WRIST, RIGHT_WRIST),
            marker_label="TOP",
            # every press rule reads both arms, so draw both elbow angles
            angle_joints=(
                JointAngle("Left elbow", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                JointAngle("Right elbow", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            ),
            # upper body only: seated, the legs sit behind the bench and no
            # press rule reads them
            drawn_landmarks=UPPER_BODY_DRAWN,
            frame_range=(render_window.start_frame, render_window.end_frame),
        )
    except Exception:
        logger.exception("Annotated-video rendering failed")
        final_path = None
        checks.add_warning("The annotated video could not be rendered for this run.")

    # --- 10. result + export ---
    debug = _debug_block(
        config,
        checks,
        detection,
        reps,
        rule_results,
        side,
        interpolated,
        resting_reference,
        started,
    )
    # what the renderer actually wrote, for the technical panel
    debug["render_window"] = render_window.as_dict()

    result = ExerciseAnalysisResult(
        success=True,
        exercise="Shoulder Press",
        analysis_side=side,
        video=video,
        validation=checks,
        reps=reps,
        rule_results=rule_results,
        summary=summary,
        annotated_video_path=final_path,
        debug=debug,
        frame_metrics=metrics,
        exercise_id="press",
    )

    if export_root is not None:
        export_dir = export_result(result, export_root, _run_id(video_path))
        if export_dir:
            debug["export_dir"] = str(export_dir)

    report("complete", 1.0, "Complete")
    return result


# --- Helpers ---


def validation_config_for(config: PressConfig) -> validation.ValidationConfig:
    """
    Press recording requirements in the generic validation vocabulary. The
    camera-view policy is the mirror of the squat's: front-on is the requirement
    here, since two of the three checks compare the arms. A side-on press warns and
    switches those two off while range of motion still runs.
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
                "hands",
                (LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST),
                critical=False,
                limited_metric=METRIC_PRESS_ROM,
                warn_message=(
                    "Your hands leave the top of the frame during part of the press, "
                    "which can make the range-of-motion check less reliable."
                ),
                fail_message=(
                    "Your hands are outside the camera frame for most of the set, so how "
                    "far each press travelled could not be measured. Step back, or tilt "
                    "the camera up, so the dumbbells stay visible at the top."
                ),
            ),
        ),
        orientation_notes=(
            validation.OrientationNote(
                CameraOrientation.SIDE,
                "This recording looks side-on. FormFix measured how far each press "
                "travelled, but comparing your two arms - and checking that each wrist "
                "stays stacked over its elbow - needs a front-on view, because from the "
                "side one arm hides the other. Film from the front to have those "
                "checked.",
                primary_metric=METRIC_PRESS_SYMMETRY,
                limited_metrics=(METRIC_PRESS_ALIGNMENT,),
            ),
            validation.OrientationNote(
                CameraOrientation.DIAGONAL_SIDE,
                "The camera is not fully front-on. FormFix could analyse this press, "
                "although a squarer front view would improve the accuracy of the "
                "left/right comparison.",
            ),
            validation.OrientationNote(
                CameraOrientation.UNKNOWN,
                "The camera angle could not be estimated from this recording, so the "
                "measurements may be less accurate than usual.",
            ),
        ),
    )


def _apply_recording_limitations(rule_results, checks: ValidationResult) -> None:
    """Switch off the rules this *recording* cannot support - and only those."""
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


def _frame_states(detection, reps: list[PressRep], metrics, frame_count: int):
    """Per-frame HUD state: phase, rep counter, top-window flag, live values."""
    phases = named_phases(detection.phases)
    states: list[FrameState] = []
    total = len(reps)
    for f in range(frame_count):
        phase = phases[f] if f < len(phases) else PressPhase.UNKNOWN
        rep_number = 0
        in_window = False
        for rep in reps:
            if rep.start_frame <= f <= rep.end_frame:
                rep_number = rep.number
                seg = rep.segmentation
                in_window = seg.extreme_start_frame <= f <= seg.extreme_end_frame
                # inside a committed rep the caption comes from that rep's own segmentation
                # repetition's own segmentation.
                segment = segment_at(seg, f)
                if segment is not None:
                    phase = SEGMENT_PHASES[segment]
                break
            if f > rep.end_frame:
                rep_number = rep.number
        # both elbow angles are on the elbows, so the HUD only carries rep and phase
        extra: list[HudLine] = []
        metric = metrics[f] if f < len(metrics) else None
        del metric
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
    """A banner per flagged per-rep outcome, around its evidence frame only."""
    context = max(int(EVENT_CONTEXT_SECONDS * video.fps), 3)
    events: list[OverlayEvent] = []
    for rule in rule_results:
        text = BANNER_TEXT.get(rule.rule_id, rule.title)
        for outcome in rule.per_rep:
            if outcome.status not in (RuleStatus.FAIL, RuleStatus.WARNING):
                continue
            if outcome.evidence_frame < 0:
                continue
            # The alignment finding belongs to one arm; highlight that one.
            worst = outcome.evidence.get("worst_side") or side
            events.append(
                OverlayEvent(
                    start_frame=max(outcome.evidence_frame - context, 0),
                    end_frame=min(outcome.evidence_frame + context, frame_count - 1),
                    text=text,
                    level=outcome.status.value,
                    highlight_landmarks=highlight_ids(rule.rule_id, worst),
                )
            )
    return events


def _debug_block(
    config,
    checks,
    detection,
    reps,
    rule_results,
    side,
    interpolated,
    resting_reference,
    started,
) -> dict:
    """Developer metrics for thesis evaluation - never shown to normal users."""
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
        "resting_flexion_reference_deg": round(float(resting_reference), 1),
        "rep_detection_thresholds": {
            "rest_level": round(resting_reference - config.REST_MARGIN, 1),
            "start_level": round(resting_reference - config.PRESS_START_MARGIN, 1),
            "extreme_level": round(resting_reference - config.TOP_MARGIN, 1),
            "end_level": round(resting_reference - config.REP_END_MARGIN, 1),
        },
        "partial_movements": detection.partial_movements,
        "partial_reasons": detection.partial_reasons,
        "rep_boundaries": [
            {
                "rep": rep.number,
                "start": rep.start_time,
                "top": rep.extreme_time,
                "end": rep.end_time,
                "press_s": rep.segmentation.towards_duration,
                "top_s": rep.segmentation.extreme_duration,
                "lower_s": rep.segmentation.return_duration,
                "top_elbow_angle": rep.top_elbow_angle,
                "bottom_elbow_angle": rep.bottom_elbow_angle,
                "rom_degrees": rep.rom_degrees,
                "left_rom": rep.left_rom_degrees,
                "right_rom": rep.right_rom_degrees,
                "max_elbow_angle_difference": rep.max_elbow_angle_difference,
                "max_wrist_height_difference": rep.max_wrist_height_difference,
                "max_left_alignment_offset": rep.max_left_alignment_offset,
                "max_right_alignment_offset": rep.max_right_alignment_offset,
                "top_timing_difference": rep.top_timing_difference,
                "higher_side": rep.higher_side,
                "both_arms_ratio": rep.both_arms_ratio,
                "symmetry_reliable": rep.symmetry_reliable,
                "alignment_reliable": rep.alignment_reliable,
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
    """A short, non-identifying id for the export bundle."""
    digest = hashlib.sha1(video_path.name.encode("utf-8")).hexdigest()[:8]
    return f"{time.strftime('%Y%m%d-%H%M%S')}-press-{digest}"


def _peak_wrist_rise(metrics) -> float | None:
    """
    Highest either wrist gets above the shoulder line, in shoulder widths - the
    signal that separates an overhead press from a bench press, a curl or a front
    raise. The peak across the whole clip, so one clean lockout is enough and a
    badly tracked rep can't argue the exercise was wrong.
    """
    best = None
    for frame in metrics:
        for attribute in ("left_wrist_height", "right_wrist_height"):
            value = getattr(frame, attribute, None)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(number):
                continue
            if best is None or number > best:
                best = number
    return best
