"""
Squat technique rules. Each one checks one measurement in the phase where it
matters, and only if the camera view supports it - otherwise it returns
NOT_EVALUABLE with a reason instead of guessing.

Left out on purpose because 2D pose can't support them: lower-back rounding,
injury risk, joint loading, and "knees past toes" as a blanket error.
"""

from __future__ import annotations

import math

import numpy as np

from analysis.models import (
    CameraOrientation,
    FrameMetrics,
    FramePoseData,
    Reliability,
    RepRuleOutcome,
    RuleResult,
    RuleStatus,
    SquatRep,
)

from .confidence import gather_evidence, reliability_for
from .config import SquatConfig, SquatRule, rule_specs
from .metrics import phase_frames, series
from .persistence import PersistenceEvidence, assess

# joints to highlight when a rule fires, as roles (the analyser turns them into
# ids once it knows the side)
HIGHLIGHT_ROLES: dict[str, tuple[str, ...]] = {
    "squat_depth": ("hip", "knee"),
    "torso_lean": ("shoulder", "hip"),
    "heel_lift": ("heel", "ankle", "foot_index"),
    "return_to_standing": ("hip", "knee", "ankle"),
    "descent_control": ("hip", "knee"),
}


def format_timestamp(seconds: float) -> str:
    """00:04.2 style timestamp."""
    if not math.isfinite(seconds):
        return "-"
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:04.1f}"


def _aggregate(outcomes: list[RepRuleOutcome]) -> RuleStatus:
    """Rule status for the whole set: FAIL if at least half the evaluable reps
    failed, WARNING if any failed or warned, PASS otherwise."""
    evaluable = [o for o in outcomes if o.status is not RuleStatus.NOT_EVALUABLE]
    if not evaluable:
        return RuleStatus.NOT_EVALUABLE
    fails = sum(1 for o in evaluable if o.status is RuleStatus.FAIL)
    warns = sum(1 for o in evaluable if o.status is RuleStatus.WARNING)
    if fails and fails * 2 >= len(evaluable):
        return RuleStatus.FAIL
    if fails or warns:
        return RuleStatus.WARNING
    return RuleStatus.PASS


def _count_phrase(count: int, total: int) -> str:
    return f"{count} of your {total} repetitions" if total > 1 else "your repetition"


_EMPTY_EVIDENCE = PersistenceEvidence(0, 0, 0, 0.0, -1, float("nan"))


def _persistence(
    spec: SquatRule | None,
    metrics: list[FrameMetrics] | None,
    rep: SquatRep,
    attribute: str,
    exceeds: float,
    prefer_max: bool = True,
) -> PersistenceEvidence:
    """How persistently the value broke the limit during the rule's phase. Returns
    empty evidence without a frame series, so tests can use made-up reps."""
    if spec is None or metrics is None:
        return _EMPTY_EVIDENCE
    frames = phase_frames(rep, spec.phase)
    values = series(metrics, attribute, frames)
    with np.errstate(invalid="ignore"):
        violating = values > exceeds if prefer_max else values < exceeds
    return assess(values, violating, offset=frames.start, prefer_max=prefer_max)


def _with_persistence(outcome: RepRuleOutcome, evidence: PersistenceEvidence) -> RepRuleOutcome:
    outcome.violating_frames = evidence.violating_frames
    outcome.phase_frames = evidence.measurable_frames
    outcome.violation_ratio = round(evidence.ratio, 3)
    return outcome


# --- Rule 1 - squat depth ---


def rule_squat_depth(
    reps: list[SquatRep],
    config: SquatConfig,
    spec: SquatRule | None = None,
    metrics: list[FrameMetrics] | None = None,
) -> RuleResult:
    """Depth from the minimum knee angle or the hip reaching knee level - either
    one passing passes the rep."""
    outcomes: list[RepRuleOutcome] = []
    for rep in reps:
        knee_ok = (
            math.isfinite(rep.min_knee_angle) and rep.min_knee_angle <= config.DEPTH_KNEE_ANGLE_PASS
        )
        hip_ok = (
            math.isfinite(rep.hip_above_knee_at_bottom)
            and rep.hip_above_knee_at_bottom <= config.DEPTH_HIP_KNEE_TOLERANCE
        )
        evidence = _persistence(
            spec, metrics, rep, "knee_angle", config.DEPTH_KNEE_ANGLE_PASS, prefer_max=True
        )

        if not math.isfinite(rep.min_knee_angle):
            status = RuleStatus.NOT_EVALUABLE
        elif knee_ok or hip_ok:
            status = RuleStatus.PASS
        elif (
            spec is not None
            and evidence.has_evidence
            and not evidence.triggers(spec.minimum_persistence_frames, spec.min_violation_ratio)
        ):
            # not enough of the bottom window was shallow to call it
            status = RuleStatus.PASS
        elif rep.min_knee_angle <= config.DEPTH_KNEE_ANGLE_WARN:
            status = RuleStatus.WARNING
        else:
            status = RuleStatus.FAIL

        outcomes.append(
            _with_persistence(
                RepRuleOutcome(
                    rep_number=rep.number,
                    status=status,
                    evidence={
                        "min_knee_angle": round(rep.min_knee_angle, 1),
                        "hip_above_knee_at_bottom": (
                            round(rep.hip_above_knee_at_bottom, 3)
                            if math.isfinite(rep.hip_above_knee_at_bottom)
                            else None
                        ),
                        "hip_flexion_at_bottom": (
                            round(rep.hip_angle_at_bottom, 1)
                            if math.isfinite(rep.hip_angle_at_bottom)
                            else None
                        ),
                        "bottom_time": format_timestamp(rep.bottom_time),
                    },
                    evidence_frame=rep.bottom_frame,
                    evidence_time=rep.bottom_time,
                ),
                evidence,
            )
        )

    status = _aggregate(outcomes)
    total = len(reps)
    flagged = [o for o in outcomes if o.status in (RuleStatus.FAIL, RuleStatus.WARNING)]

    if status is RuleStatus.NOT_EVALUABLE:
        explanation = "Knee angles could not be measured reliably enough to assess depth."
    elif not flagged:
        explanation = (
            f"All {total} repetitions reached the configured FormFix depth target."
            if total > 1
            else "Your repetition reached the configured FormFix depth target."
        )
    else:
        shallowest = max(
            (o for o in flagged if o.evidence.get("min_knee_angle") is not None),
            key=lambda o: o.evidence["min_knee_angle"],
            default=flagged[0],
        )
        explanation = (
            f"{_count_phrase(len(flagged), total).capitalize()} did not reach the configured "
            f"target depth. The shallowest (rep {shallowest.rep_number}, "
            f"{shallowest.evidence['bottom_time']}) reached a minimum knee angle of about "
            f"{shallowest.evidence['min_knee_angle']:.0f} deg."
        )

    return _result(
        spec,
        rule_id="squat_depth",
        title="Squat depth",
        status=status,
        explanation=explanation,
        correction=(
            "Lower yourself under control until your hips reach the target depth, "
            "keeping your whole foot planted. If depth is hard to reach, try a "
            "slightly wider stance or reduce the weight."
            if flagged
            else ""
        ),
        outcomes=outcomes,
        evidence={
            "threshold_pass_deg": config.DEPTH_KNEE_ANGLE_PASS,
            "threshold_warn_deg": config.DEPTH_KNEE_ANGLE_WARN,
            "hip_knee_tolerance": config.DEPTH_HIP_KNEE_TOLERANCE,
        },
        feedback_key="depth",
    )


# --- Rule 2 - excessive forward torso lean ---


def rule_torso_lean(
    reps: list[SquatRep],
    standing_torso_lean: float,
    config: SquatConfig,
    spec: SquatRule | None = None,
    metrics: list[FrameMetrics] | None = None,
) -> RuleResult:
    """Peak forward lean, both absolute and compared to the person's own standing
    posture, so a tilted camera isn't a fault."""
    outcomes: list[RepRuleOutcome] = []
    baseline_ok = math.isfinite(standing_torso_lean)
    for rep in reps:
        peak = rep.max_torso_lean
        evidence = _persistence(spec, metrics, rep, "torso_lean", config.TORSO_LEAN_WARN)
        if not math.isfinite(peak):
            status = RuleStatus.NOT_EVALUABLE
            delta = float("nan")
        else:
            delta = peak - standing_torso_lean if baseline_ok else float("nan")
            exceeded_abs = peak >= config.TORSO_LEAN_FAIL
            exceeded_delta = math.isfinite(delta) and delta >= config.TORSO_LEAN_DELTA_FAIL
            persistent = (
                spec is None
                or not evidence.has_evidence
                or evidence.triggers(spec.minimum_persistence_frames, spec.min_violation_ratio)
            )
            if not persistent:
                status = RuleStatus.PASS
            elif exceeded_abs or exceeded_delta:
                status = RuleStatus.FAIL
            elif peak >= config.TORSO_LEAN_WARN:
                status = RuleStatus.WARNING
            else:
                status = RuleStatus.PASS

        outcomes.append(
            _with_persistence(
                RepRuleOutcome(
                    rep_number=rep.number,
                    status=status,
                    evidence={
                        "max_torso_lean": round(peak, 1) if math.isfinite(peak) else None,
                        "standing_baseline": round(standing_torso_lean, 1) if baseline_ok else None,
                        "change_from_baseline": round(delta, 1) if math.isfinite(delta) else None,
                        "at_bottom": (
                            round(rep.torso_lean_at_bottom, 1)
                            if math.isfinite(rep.torso_lean_at_bottom)
                            else None
                        ),
                    },
                    evidence_frame=rep.max_torso_lean_frame,
                    evidence_time=rep.bottom_time,
                ),
                evidence,
            )
        )

    status = _aggregate(outcomes)
    total = len(reps)
    flagged = [o for o in outcomes if o.status in (RuleStatus.FAIL, RuleStatus.WARNING)]

    if status is RuleStatus.NOT_EVALUABLE:
        explanation = "Torso inclination could not be measured reliably in this recording."
    elif not flagged:
        explanation = "Your torso stayed within the configured lean range throughout the set."
    else:
        worst = max(
            (o for o in flagged if o.evidence.get("max_torso_lean") is not None),
            key=lambda o: o.evidence["max_torso_lean"],
            default=flagged[0],
        )
        baseline_part = (
            f" (your standing baseline was about {worst.evidence['standing_baseline']:.0f} deg)"
            if worst.evidence.get("standing_baseline") is not None
            else ""
        )
        explanation = (
            f"Your torso leaned further forward than the configured range in "
            f"{_count_phrase(len(flagged), total)}. The strongest lean was about "
            f"{worst.evidence['max_torso_lean']:.0f} deg from vertical during rep "
            f"{worst.rep_number}{baseline_part}."
        )

    return _result(
        spec,
        rule_id="torso_lean",
        title="Forward torso lean",
        status=status,
        explanation=explanation,
        correction=(
            "Try keeping your chest a little more upright as you descend - a slightly "
            "wider stance or lighter weight can make that easier to hold."
            if flagged
            else ""
        ),
        outcomes=outcomes,
        evidence={
            "warn_deg": config.TORSO_LEAN_WARN,
            "fail_deg": config.TORSO_LEAN_FAIL,
            "delta_fail_deg": config.TORSO_LEAN_DELTA_FAIL,
            "min_persistence_frames": config.TORSO_LEAN_MIN_FRAMES,
            "min_violation_ratio": config.TORSO_LEAN_MIN_VIOLATION_RATIO,
        },
        feedback_key="torso_lean",
    )


# --- Rule 3 - heel lift ---


def rule_heel_lift(
    reps: list[SquatRep],
    config: SquatConfig,
    spec: SquatRule | None = None,
    metrics: list[FrameMetrics] | None = None,
) -> RuleResult:
    """Heel rise above the person's standing heel height / lower-leg length.
    Unreliable heel landmarks give NOT_EVALUABLE."""
    outcomes: list[RepRuleOutcome] = []
    for rep in reps:
        evidence = _persistence(spec, metrics, rep, "heel_lift", config.HEEL_LIFT_THRESHOLD)
        # the lift has to be big enough AND held: a frame or two over the line is
        # foot-tracking noise, which is what used to warn on planted heels. With no
        # frame series at all (unit tests) the size on its own has to do.
        held = (
            evidence.triggers(spec.minimum_persistence_frames, spec.min_violation_ratio)
            if spec is not None and evidence.has_evidence
            else spec is None
        )
        if not rep.heel_reliable or not math.isfinite(rep.max_heel_lift):
            status = RuleStatus.NOT_EVALUABLE
        elif rep.max_heel_lift >= config.HEEL_LIFT_THRESHOLD and held:
            # only ever a warning - from 2D landmarks it's just worth checking
            status = RuleStatus.WARNING
        else:
            status = RuleStatus.PASS

        outcomes.append(
            _with_persistence(
                RepRuleOutcome(
                    rep_number=rep.number,
                    status=status,
                    evidence={
                        "heel_lift_ratio": (
                            round(rep.max_heel_lift, 3) if math.isfinite(rep.max_heel_lift) else None
                        ),
                        "heel_landmarks_reliable": rep.heel_reliable,
                    },
                    evidence_frame=rep.max_heel_lift_frame,
                    evidence_time=rep.bottom_time,
                ),
                evidence,
            )
        )

    status = _aggregate(outcomes)
    total = len(reps)
    flagged = [o for o in outcomes if o.status is RuleStatus.WARNING]
    unassessed = [o for o in outcomes if o.status is RuleStatus.NOT_EVALUABLE]

    if status is RuleStatus.NOT_EVALUABLE:
        explanation = (
            "Heel position could not be assessed reliably because your feet were "
            "partially obscured or tracked with low confidence."
        )
    elif flagged:
        first = flagged[0]
        explanation = (
            f"Your heel appears to rise during {_count_phrase(len(flagged), total)} "
            f"(first noticed around {format_timestamp(first.evidence_time)}). This is "
            "estimated from the visible foot landmarks."
        )
    else:
        explanation = "Your heels appear to stay planted throughout the set."

    return _result(
        spec,
        rule_id="heel_lift",
        title="Heel stability",
        status=status,
        explanation=explanation,
        correction=(
            "Keep your whole foot planted and your weight spread across mid-foot "
            "and heel - it gives you a more stable base to push from."
            if flagged
            else ""
        ),
        outcomes=outcomes,
        evidence={
            "threshold_ratio": config.HEEL_LIFT_THRESHOLD,
            "min_persistence_frames": config.HEEL_LIFT_MIN_FRAMES,
        },
        feedback_key="heel_lift",
        limitation=(
            f"Heel landmarks were unreliable on {len(unassessed)} of {total} repetitions."
            if unassessed and status is not RuleStatus.NOT_EVALUABLE
            else ""
        ),
    )


# --- Rule 4 - return to standing / extension ---


def rule_extension(
    reps: list[SquatRep],
    standing_knee_angle: float,
    config: SquatConfig,
    spec: SquatRule | None = None,
    metrics: list[FrameMetrics] | None = None,
) -> RuleResult:
    """Did each rep come back to the person's own standing posture (not 180)?"""
    del metrics  # one measurement at completion, so no phase series to filter
    outcomes: list[RepRuleOutcome] = []
    baseline_ok = math.isfinite(standing_knee_angle)
    for rep in reps:
        end_angle = rep.end_knee_angle
        if not baseline_ok or not math.isfinite(end_angle):
            status = RuleStatus.NOT_EVALUABLE
            shortfall = float("nan")
        else:
            shortfall = standing_knee_angle - end_angle
            if shortfall <= config.FULL_EXTENSION_TOLERANCE:
                status = RuleStatus.PASS
            elif shortfall <= config.FULL_EXTENSION_TOLERANCE * 2:
                status = RuleStatus.WARNING
            else:
                status = RuleStatus.FAIL
        outcomes.append(
            RepRuleOutcome(
                rep_number=rep.number,
                status=status,
                evidence={
                    "end_knee_angle": round(end_angle, 1) if math.isfinite(end_angle) else None,
                    "end_hip_angle": (
                        round(rep.end_hip_angle, 1) if math.isfinite(rep.end_hip_angle) else None
                    ),
                    "standing_baseline": round(standing_knee_angle, 1) if baseline_ok else None,
                    "shortfall_deg": round(shortfall, 1) if math.isfinite(shortfall) else None,
                    "end_time": format_timestamp(rep.end_time),
                },
                evidence_frame=rep.end_frame,
                evidence_time=rep.end_time,
                phase_frames=1,
                violating_frames=1 if status in (RuleStatus.WARNING, RuleStatus.FAIL) else 0,
                violation_ratio=1.0 if status in (RuleStatus.WARNING, RuleStatus.FAIL) else 0.0,
            )
        )

    status = _aggregate(outcomes)
    total = len(reps)
    flagged = [o for o in outcomes if o.status in (RuleStatus.FAIL, RuleStatus.WARNING)]

    if status is RuleStatus.NOT_EVALUABLE:
        explanation = (
            "A standing baseline could not be established, so the finish position was not assessed."
        )
    elif not flagged:
        explanation = "You returned to your starting position consistently after every repetition."
    else:
        worst = flagged[-1]
        explanation = (
            f"{_count_phrase(len(flagged), total).capitalize()} ended before you fully "
            f"returned to your starting position. Rep {worst.rep_number} finished at a "
            f"knee angle of about {worst.evidence['end_knee_angle']:.0f} deg against your "
            f"standing baseline of about {worst.evidence['standing_baseline']:.0f} deg."
        )

    return _result(
        spec,
        rule_id="return_to_standing",
        title="Return to standing",
        status=status,
        explanation=explanation,
        correction=(
            "Finish each repetition by standing tall - hips and knees comfortably "
            "extended - before starting the next one."
            if flagged
            else ""
        ),
        outcomes=outcomes,
        evidence={"tolerance_deg": config.FULL_EXTENSION_TOLERANCE},
        feedback_key="extension",
    )


# --- Rule 5 - descent control (timing) ---


def rule_descent_control(
    reps: list[SquatRep],
    config: SquatConfig,
    spec: SquatRule | None = None,
    metrics: list[FrameMetrics] | None = None,
) -> RuleResult:
    """How long the descent took, in seconds. About control, never an injury warning."""
    del metrics  # a per-rep duration, so no frame series to filter
    outcomes: list[RepRuleOutcome] = []
    for rep in reps:
        duration = rep.descent_duration
        if not math.isfinite(duration) or duration <= 0:
            status = RuleStatus.NOT_EVALUABLE
        elif duration >= config.DESCENT_MIN_DURATION:
            status = RuleStatus.PASS
        elif duration >= config.DESCENT_FAST_DURATION:
            status = RuleStatus.WARNING
        else:
            status = RuleStatus.FAIL
        outcomes.append(
            RepRuleOutcome(
                rep_number=rep.number,
                status=status,
                evidence={
                    "descent_seconds": round(duration, 2) if math.isfinite(duration) else None,
                    "bottom_seconds": (
                        round(rep.bottom_duration, 2) if math.isfinite(rep.bottom_duration) else None
                    ),
                    "ascent_seconds": (
                        round(rep.ascent_duration, 2) if math.isfinite(rep.ascent_duration) else None
                    ),
                    "rep_seconds": round(rep.duration, 2) if math.isfinite(rep.duration) else None,
                },
                evidence_frame=rep.descent_start_frame,
                evidence_time=rep.start_time,
                phase_frames=1,
                violating_frames=1 if status in (RuleStatus.WARNING, RuleStatus.FAIL) else 0,
                violation_ratio=1.0 if status in (RuleStatus.WARNING, RuleStatus.FAIL) else 0.0,
            )
        )

    status = _aggregate(outcomes)
    total = len(reps)
    flagged = [o for o in outcomes if o.status in (RuleStatus.FAIL, RuleStatus.WARNING)]

    if status is RuleStatus.NOT_EVALUABLE:
        explanation = "Repetition timing could not be measured for this recording."
    elif not flagged:
        explanation = "You lowered at a controlled speed on every repetition."
    else:
        fastest = min(
            (o for o in flagged if o.evidence.get("descent_seconds") is not None),
            key=lambda o: o.evidence["descent_seconds"],
            default=flagged[0],
        )
        explanation = (
            f"You dropped down quickly in {_count_phrase(len(flagged), total)}. The "
            f"fastest descent (rep {fastest.rep_number}) took about "
            f"{fastest.evidence['descent_seconds']:.1f} seconds."
        )

    return _result(
        spec,
        rule_id="descent_control",
        title="Descent control",
        status=status,
        explanation=explanation,
        correction=(
            "Lower yourself a little more slowly - taking about a second on the way "
            "down gives you more control at the bottom."
            if flagged
            else ""
        ),
        outcomes=outcomes,
        evidence={
            "min_seconds": config.DESCENT_MIN_DURATION,
            "fast_seconds": config.DESCENT_FAST_DURATION,
        },
        feedback_key="descent_control",
    )


# --- Assembly ---


def _result(
    spec: SquatRule | None,
    *,
    rule_id: str,
    title: str,
    status: RuleStatus,
    explanation: str,
    correction: str,
    outcomes: list[RepRuleOutcome],
    evidence: dict,
    feedback_key: str,
    limitation: str = "",
) -> RuleResult:
    """Add the spec metadata to a computed rule outcome."""
    return RuleResult(
        rule_id=rule_id,
        title=spec.title if spec else title,
        status=status,
        explanation=explanation,
        correction=correction,
        per_rep=outcomes,
        evidence={
            **evidence,
            **(
                {
                    "phase": spec.phase,
                    "supported_views": list(spec.supported_views),
                    "threshold_source": spec.threshold_source,
                }
                if spec
                else {}
            ),
        },
        limitation=limitation,
        metric=spec.metric if spec else rule_id,
        phase=spec.phase if spec else "",
        supported_views=spec.supported_views if spec else (),
        feedback_key=spec.feedback_key if spec else feedback_key,
    )


# message when a rule is switched off because of the camera view
VIEW_LIMITATION_TEXT: dict[str, str] = {
    "squat_depth": (
        "Depth needs a side view: knee and hip angles cannot be measured reliably "
        "from this camera position."
    ),
    "torso_lean": (
        "Forward torso lean is only visible from the side, so it was not assessed "
        "for this recording."
    ),
    "return_to_standing": (
        "Returning to standing is judged from the knee angle, which this camera "
        "position does not show reliably."
    ),
    "descent_control": (
        "Repetition timing is taken from the knee-angle trace, which needs a "
        "side-on view of the movement."
    ),
    "heel_lift": (
        "Your heels were not visible enough in this recording, so heel contact was " "not assessed."
    ),
}


def not_evaluable(spec: SquatRule, reps: list[SquatRep], reason: str) -> RuleResult:
    """A rule the recording can't support."""
    return RuleResult(
        rule_id=spec.rule_id,
        title=spec.title,
        status=RuleStatus.NOT_EVALUABLE,
        explanation=reason,
        correction="",
        per_rep=[
            RepRuleOutcome(rep_number=rep.number, status=RuleStatus.NOT_EVALUABLE) for rep in reps
        ],
        evidence={"phase": spec.phase, "supported_views": list(spec.supported_views)},
        limitation=reason,
        metric=spec.metric,
        phase=spec.phase,
        supported_views=spec.supported_views,
        feedback_key=spec.feedback_key,
        reliability=Reliability.CANNOT_ASSESS,
    )


_EVALUATORS = {
    "squat_depth": lambda reps, ctx, spec, config: rule_squat_depth(reps, config, spec, ctx["metrics"]),
    "torso_lean": lambda reps, ctx, spec, config: rule_torso_lean(
        reps, ctx["standing_torso_lean"], config, spec, ctx["metrics"]
    ),
    "heel_lift": lambda reps, ctx, spec, config: rule_heel_lift(reps, config, spec, ctx["metrics"]),
    "return_to_standing": lambda reps, ctx, spec, config: rule_extension(
        reps, ctx["standing_knee_angle"], config, spec, ctx["metrics"]
    ),
    "descent_control": lambda reps, ctx, spec, config: rule_descent_control(
        reps, config, spec, ctx["metrics"]
    ),
}


def _subject_scale(metrics: list[FrameMetrics] | None) -> float:
    """Median torso length in pixels, used to cap reliability for far-away people."""
    if not metrics:
        return float("nan")
    values = [m.body_scale for m in metrics if m.valid and np.isfinite(m.body_scale)]
    return float(np.median(values)) if values else float("nan")


def evaluate_all(
    reps: list[SquatRep],
    standing_knee_angle: float,
    standing_torso_lean: float,
    config: SquatConfig,
    metrics: list[FrameMetrics] | None = None,
    pose: FramePoseData | None = None,
    orientation: CameraOrientation = CameraOrientation.SIDE,
    side_view_confidence: float = 1.0,
    side: str = "left",
    recording_quality: str = "good",
) -> list[RuleResult]:
    """
    Run every rule, check the camera view first, then attach reliability. The
    view check comes first so an unsupported measurement gives no finding at all.
    """
    context = {
        "metrics": metrics,
        "standing_knee_angle": standing_knee_angle,
        "standing_torso_lean": standing_torso_lean,
    }
    results: list[RuleResult] = []
    subject_scale = _subject_scale(metrics)

    for spec in rule_specs(config):
        if orientation.value not in spec.supported_views:
            results.append(
                not_evaluable(
                    spec,
                    reps,
                    VIEW_LIMITATION_TEXT.get(
                        spec.rule_id,
                        "This check is not reliable for the camera angle in this recording.",
                    ),
                )
            )
            continue

        result = _EVALUATORS[spec.rule_id](reps, context, spec, config)

        if pose is not None:
            evidence = gather_evidence(
                spec.metric,
                reps,
                result.per_rep,
                pose,
                spec.phase,
                spec.supported_views,
                orientation,
                side_view_confidence,
                side,
                subject_scale_px=subject_scale,
                recording_quality=recording_quality,
            )
            result.reliability = reliability_for(evidence, config)
            result.evidence["reliability_evidence"] = evidence.as_dict()
        elif result.status is not RuleStatus.NOT_EVALUABLE:
            result.reliability = Reliability.MEDIUM

        if result.status is RuleStatus.NOT_EVALUABLE and not result.limitation:
            result.limitation = result.explanation
        results.append(result)

    return results
