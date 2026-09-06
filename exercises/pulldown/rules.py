"""
The lat-pulldown technique rules, two of them. Each asks one narrow question
about one measurement, in the phase where that means something, and only when
the camera view supports it.

Same shape as the squat's: numbers arrive measured from metrics.py, every
threshold crossing goes through persistence.py, and thresholds are ranges with
a tolerance band.

Deliberately not implemented, because one 2D camera can't support them:
scapular movement, lat activation, grip width, whether the bar touched the
chest, and anything about injury or loading.
"""

from __future__ import annotations

import math

import numpy as np

from analysis.models import (
    METRIC_PULLDOWN_ROM,
    METRIC_PULLDOWN_TORSO,
    ROM_COMPLETE,
    ROM_LIMITED_BOTTOM,
    ROM_LIMITED_OVERALL,
    ROM_LIMITED_TOP,
    ROM_NOT_ASSESSABLE,
    CameraOrientation,
    FramePoseData,
    PulldownRep,
    Reliability,
    RepRuleOutcome,
    RuleResult,
    RuleStatus,
)
from exercises.common.confidence import gather_evidence, reliability_for
from exercises.common.feedback import format_timestamp
from exercises.common.metrics import series
from exercises.common.persistence import PersistenceEvidence, assess
from exercises.common.spec import RuleSpec

from .confidence import bands_for
from .config import PulldownConfig, rule_specs
from .landmarks import ARM_CHAINS, required_ids
from .metrics import PulldownFrameMetrics, phase_frames

# joints to highlight in the video when a rule fires, as roles - the analyser
# resolves them to MediaPipe ids once it knows which side is being analysed
HIGHLIGHT_ROLES: dict[str, tuple[str, ...]] = {
    "pulldown_rom": ("shoulder", "elbow", "wrist"),
    "pulldown_torso": ("shoulder", "hip"),
}

_EMPTY_EVIDENCE = PersistenceEvidence(0, 0, 0, 0.0, -1, float("nan"))


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


def _persistence(
    spec: RuleSpec | None,
    metrics: list[PulldownFrameMetrics] | None,
    rep: PulldownRep,
    attribute: str,
    exceeds: float,
    prefer_max: bool = True,
) -> PersistenceEvidence:
    """How persistently attribute broke exceeds during the rule's phase. Empty
    evidence with no frame series, which is what lets tests drive a rule from
    constructed reps."""
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


def _round(value: float, digits: int = 1):
    return round(float(value), digits) if math.isfinite(value) else None


# --- Rule 1 - range of motion ---


def classify_rom(rep: PulldownRep, config: PulldownConfig) -> tuple[str, RuleStatus]:
    """
    Which end of the movement fell short, from the elbow's own excursion: the top
    (do the arms return towards extension), the bottom (does the pull close the
    elbow) and the total excursion. A category, not a score, because
    "limited range of motion" tells a beginner nothing.
    """
    top, bottom = rep.top_elbow_angle, rep.bottom_elbow_angle
    if not (math.isfinite(top) and math.isfinite(bottom)) or not rep.arms_reliable:
        return ROM_NOT_ASSESSABLE, RuleStatus.NOT_EVALUABLE

    top_short = top < config.ROM_TOP_EXTENSION_PASS
    top_bad = top < config.ROM_TOP_EXTENSION_WARN
    bottom_short = bottom > config.ROM_BOTTOM_FLEXION_PASS
    bottom_bad = bottom > config.ROM_BOTTOM_FLEXION_WARN
    excursion_short = math.isfinite(rep.rom_degrees) and rep.rom_degrees < config.ROM_MIN_EXCURSION

    if not top_short and not bottom_short and not excursion_short:
        return ROM_COMPLETE, RuleStatus.PASS

    if top_short and bottom_short:
        status = RuleStatus.FAIL if (top_bad or bottom_bad) else RuleStatus.WARNING
        return ROM_LIMITED_OVERALL, status
    if bottom_short:
        return ROM_LIMITED_BOTTOM, (RuleStatus.FAIL if bottom_bad else RuleStatus.WARNING)
    if top_short:
        return ROM_LIMITED_TOP, (RuleStatus.FAIL if top_bad else RuleStatus.WARNING)
    return ROM_LIMITED_OVERALL, RuleStatus.WARNING


def rule_range_of_motion(
    reps: list[PulldownRep],
    config: PulldownConfig,
    spec: RuleSpec | None = None,
    metrics: list[PulldownFrameMetrics] | None = None,
) -> RuleResult:
    """
    Did each rep use the configured range at both ends? The criterion is the body's
    joint excursion, not the bar's position - MediaPipe tracks landmarks, not
    equipment. Wrist travel is reported as supporting evidence but never passes a
    rep on its own.
    """
    del metrics  # per-rep extremes, so no frame series to filter
    outcomes: list[RepRuleOutcome] = []
    for rep in reps:
        category, status = classify_rom(rep, config)
        outcomes.append(
            RepRuleOutcome(
                rep_number=rep.number,
                status=status,
                evidence={
                    "rom_status": category,
                    "top_elbow_angle": _round(rep.top_elbow_angle),
                    "bottom_elbow_angle": _round(rep.bottom_elbow_angle),
                    "rom_degrees": _round(rep.rom_degrees),
                    "wrist_travel": _round(rep.wrist_travel, 3),
                    "elbow_travel": _round(rep.elbow_travel, 3),
                    "analysed_arm_usable": rep.arms_reliable,
                    "both_arms_visible_ratio": _round(rep.both_arms_ratio, 3),
                    "extreme_time": format_timestamp(rep.extreme_time),
                },
                evidence_frame=rep.extreme_frame,
                evidence_time=rep.extreme_time,
                phase_frames=1,
                violating_frames=1 if status in (RuleStatus.WARNING, RuleStatus.FAIL) else 0,
                violation_ratio=1.0 if status in (RuleStatus.WARNING, RuleStatus.FAIL) else 0.0,
            )
        )

    status = _aggregate(outcomes)
    total = len(reps)
    flagged = [o for o in outcomes if o.status in (RuleStatus.FAIL, RuleStatus.WARNING)]
    categories = [o.evidence["rom_status"] for o in flagged]
    top_issues = sum(1 for c in categories if c in (ROM_LIMITED_TOP, ROM_LIMITED_OVERALL))
    bottom_issues = sum(1 for c in categories if c in (ROM_LIMITED_BOTTOM, ROM_LIMITED_OVERALL))

    if status is RuleStatus.NOT_EVALUABLE:
        explanation = (
            "Your elbow angles could not be measured reliably enough to assess how far "
            "the movement travelled."
        )
        correction = ""
    elif not flagged:
        explanation = (
            f"All {total} repetitions covered the configured movement range, at both the "
            "extended and the contracted position."
            if total > 1
            else "Your repetition covered the configured movement range at both ends."
        )
        correction = ""
    elif bottom_issues >= top_issues:
        deepest = max(
            (o for o in flagged if o.evidence.get("bottom_elbow_angle") is not None),
            key=lambda o: o.evidence["bottom_elbow_angle"],
            default=flagged[0],
        )
        explanation = (
            f"The pull stopped early in {_count_phrase(len(flagged), total)}. At the end of "
            f"the pull your elbows were still at about "
            f"{deepest.evidence['bottom_elbow_angle']:.0f} deg (rep {deepest.rep_number}), "
            f"against the configured pulling criterion of "
            f"{config.ROM_BOTTOM_FLEXION_PASS:.0f} deg or less."
        )
        correction = (
            "Complete the pull through a comfortable, controlled range - think about "
            "driving your elbows down towards your sides rather than pulling with your "
            "hands. Reduce the weight if the last part of the pull is hard to reach."
        )
    else:
        least = min(
            (o for o in flagged if o.evidence.get("top_elbow_angle") is not None),
            key=lambda o: o.evidence["top_elbow_angle"],
            default=flagged[0],
        )
        explanation = (
            f"Your arms did not return to a sufficiently extended position in "
            f"{_count_phrase(len(flagged), total)}. The most restricted (rep "
            f"{least.rep_number}) reached about {least.evidence['top_elbow_angle']:.0f} deg "
            f"of elbow extension, against the configured criterion of "
            f"{config.ROM_TOP_EXTENSION_PASS:.0f} deg."
        )
        correction = (
            "Let your arms extend more fully between repetitions and control the bar on "
            "the way back up, rather than starting the next pull early."
        )

    return _result(
        spec,
        rule_id="pulldown_rom",
        title="Range of motion",
        metric=METRIC_PULLDOWN_ROM,
        status=status,
        explanation=explanation,
        correction=correction,
        outcomes=outcomes,
        evidence={
            "top_extension_pass_deg": config.ROM_TOP_EXTENSION_PASS,
            "top_extension_warn_deg": config.ROM_TOP_EXTENSION_WARN,
            "bottom_flexion_pass_deg": config.ROM_BOTTOM_FLEXION_PASS,
            "bottom_flexion_warn_deg": config.ROM_BOTTOM_FLEXION_WARN,
            "min_excursion_deg": config.ROM_MIN_EXCURSION,
        },
        feedback_key="pulldown_rom",
    )


# --- Rule 2 - excessive torso movement ---


def rule_torso_movement(
    reps: list[PulldownRep],
    config: PulldownConfig,
    spec: RuleSpec | None = None,
    metrics: list[PulldownFrameMetrics] | None = None,
) -> RuleResult:
    """
    Did the trunk stay reasonably still while the arms did the work?

    Not aiming for a vertical trunk - a seated pulldown has a small deliberate lean
    and the bench often sets one. What we measure is the change away from this
    person's own top position during the pull.

    The peak has to be sustained and cover a configured share of the pull. Where
    the facing direction is known the excursion is a signed backward lean;
    otherwise it's an unsigned change and the wording says "moved".
    """
    outcomes: list[RepRuleOutcome] = []
    for rep in reps:
        peak = rep.max_torso_excursion
        evidence = _persistence(spec, metrics, rep, "torso_excursion", config.TORSO_EXCURSION_WARN)

        if not rep.torso_reliable or not math.isfinite(peak):
            status = RuleStatus.NOT_EVALUABLE
        else:
            persistent = (
                spec is None
                or not evidence.has_evidence
                or evidence.triggers(spec.minimum_persistence_frames, spec.min_violation_ratio)
            )
            absolute = rep.torso_at_bottom
            exceeded_absolute = math.isfinite(absolute) and absolute >= config.TORSO_ABSOLUTE_FAIL
            if not persistent:
                status = RuleStatus.PASS
            elif peak >= config.TORSO_EXCURSION_FAIL or exceeded_absolute:
                status = RuleStatus.FAIL
            elif peak >= config.TORSO_EXCURSION_WARN:
                status = RuleStatus.WARNING
            else:
                status = RuleStatus.PASS

        outcomes.append(
            _with_persistence(
                RepRuleOutcome(
                    rep_number=rep.number,
                    status=status,
                    evidence={
                        "max_torso_excursion": _round(peak),
                        "torso_at_top": _round(rep.torso_at_top),
                        "torso_at_bottom": _round(rep.torso_at_bottom),
                        "torso_mode": rep.torso_mode,
                        "peak_torso_velocity": _round(rep.peak_torso_velocity),
                        "extreme_time": format_timestamp(rep.extreme_time),
                    },
                    evidence_frame=rep.max_torso_excursion_frame,
                    evidence_time=rep.extreme_time,
                ),
                evidence,
            )
        )

    status = _aggregate(outcomes)
    total = len(reps)
    flagged = [o for o in outcomes if o.status in (RuleStatus.FAIL, RuleStatus.WARNING)]
    signed = any(o.evidence.get("torso_mode") == "posterior" for o in outcomes)
    direction = "backward" if signed else "away from its starting position"

    if status is RuleStatus.NOT_EVALUABLE:
        explanation = (
            "Your shoulders and hips were not visible clearly enough at the same time "
            "to measure how much your torso moved during the pull."
        )
        correction = ""
    elif not flagged:
        explanation = (
            "Your torso stayed within the configured range of its starting position "
            "throughout the set - the pull looked like it came from your arms and back."
        )
        correction = ""
    else:
        worst = max(
            (o for o in flagged if o.evidence.get("max_torso_excursion") is not None),
            key=lambda o: o.evidence["max_torso_excursion"],
            default=flagged[0],
        )
        explanation = (
            f"Your torso moved {direction} considerably during the pulling phase in "
            f"{_count_phrase(len(flagged), total)}. The largest movement was about "
            f"{worst.evidence['max_torso_excursion']:.0f} deg away from your own top "
            f"position during rep {worst.rep_number}, against the configured tolerance "
            f"of {config.TORSO_EXCURSION_WARN:.0f} deg."
        )
        correction = (
            "Keep your torso more stable and pull the bar by driving your elbows "
            "downwards rather than by swinging your body. A lighter weight makes that "
            "position much easier to hold."
        )

    return _result(
        spec,
        rule_id="pulldown_torso",
        title="Torso movement",
        metric=METRIC_PULLDOWN_TORSO,
        status=status,
        explanation=explanation,
        correction=correction,
        outcomes=outcomes,
        evidence={
            "warn_deg": config.TORSO_EXCURSION_WARN,
            "fail_deg": config.TORSO_EXCURSION_FAIL,
            "absolute_fail_deg": config.TORSO_ABSOLUTE_FAIL,
            "min_persistence_frames": config.TORSO_MIN_FRAMES,
            "min_violation_ratio": config.TORSO_MIN_VIOLATION_RATIO,
            "measurement_mode": "posterior" if signed else "unsigned",
        },
        feedback_key="pulldown_torso",
    )


# --- Assembly ---


def _result(
    spec: RuleSpec | None,
    *,
    rule_id: str,
    title: str,
    metric: str,
    status: RuleStatus,
    explanation: str,
    correction: str,
    outcomes: list[RepRuleOutcome],
    evidence: dict,
    feedback_key: str,
    limitation: str = "",
) -> RuleResult:
    """Bolt the declared spec metadata onto a computed rule outcome."""
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
        metric=spec.metric if spec else metric,
        phase=spec.phase if spec else "",
        supported_views=spec.supported_views if spec else (),
        feedback_key=spec.feedback_key if spec else feedback_key,
    )


# what we say when a rule is switched off because of the camera view
VIEW_LIMITATION_TEXT: dict[str, str] = {
    "pulldown_torso": (
        "Torso movement happens towards and away from the camera in a front-on or "
        "rear-on recording, so it cannot be measured from this angle. Film from the "
        "side or at a three-quarter angle to have this checked."
    ),
    "pulldown_rom": ("Your arm angles could not be measured reliably from this camera position."),
}


def not_evaluable(spec: RuleSpec, reps: list[PulldownRep], reason: str) -> RuleResult:
    """A rule the recording can't support: reported, not guessed at."""
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
    "pulldown_rom": rule_range_of_motion,
    "pulldown_torso": rule_torso_movement,
}


def _subject_scale(metrics: list[PulldownFrameMetrics] | None) -> float:
    """Median trunk length in pixels. Someone filmed from far away carries more
    landmark error than the visibility score admits, so reliability is capped."""
    if not metrics:
        return float("nan")
    values = [m.body_scale for m in metrics if m.valid and np.isfinite(m.body_scale)]
    return float(np.median(values)) if values else float("nan")


def evaluate_all(
    reps: list[PulldownRep],
    config: PulldownConfig,
    metrics: list[PulldownFrameMetrics] | None = None,
    pose: FramePoseData | None = None,
    orientation: CameraOrientation = CameraOrientation.SIDE,
    side_view_confidence: float = 1.0,
    side: str = "left",
    recording_quality: str = "good",
) -> list[RuleResult]:
    """Run every declared rule, gate it by camera view, attach reliability. View
    gating happens before any threshold is compared."""
    results: list[RuleResult] = []
    subject_scale = _subject_scale(metrics)
    if side not in ARM_CHAINS:
        side = "left"

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

        result = _EVALUATORS[spec.rule_id](reps, config, spec, metrics)

        if pose is not None:
            evidence = gather_evidence(
                spec.metric,
                reps,
                result.per_rep,
                pose,
                required_ids(spec.metric, side),
                lambda rep, phase=spec.phase: phase_frames(rep, phase),
                spec.supported_views,
                orientation,
                side_view_confidence,
                subject_scale_px=subject_scale,
                recording_quality=recording_quality,
            )
            result.reliability = reliability_for(evidence, bands_for(config))
            result.evidence["reliability_evidence"] = evidence.as_dict()
        elif result.status is not RuleStatus.NOT_EVALUABLE:
            result.reliability = Reliability.MEDIUM

        if result.status is RuleStatus.NOT_EVALUABLE and not result.limitation:
            result.limitation = result.explanation
        results.append(result)

    return results
