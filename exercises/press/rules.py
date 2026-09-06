"""
The shoulder-press technique rules, three of them. Each asks one narrow
question about one measurement, in the phase where that means something, and
only when the camera view supports it.

Same shape as the other two: numbers arrive measured from metrics.py, every
threshold crossing goes through persistence.py, and thresholds are ranges
rather than exact targets - human movement is naturally a bit asymmetric and no
forearm is perfectly vertical, so anything tighter would measure MediaPipe's
error instead.

Deliberately not implemented: sagittal trunk lean, which happens in the plane a
front-on camera can't see, and any shoulder-mobility or impingement claim.
"""

from __future__ import annotations

import math

import numpy as np

from analysis.models import (
    METRIC_PRESS_ALIGNMENT,
    METRIC_PRESS_ROM,
    METRIC_PRESS_SYMMETRY,
    ROM_COMPLETE,
    ROM_LIMITED_BOTTOM,
    ROM_LIMITED_OVERALL,
    ROM_LIMITED_TOP,
    ROM_NOT_ASSESSABLE,
    CameraOrientation,
    FramePoseData,
    PressRep,
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

from .confidence import bands_for, is_frontal_plane
from .config import PressConfig, rule_specs
from .landmarks import ARM_CHAINS, landmark_ids, required_ids
from .metrics import PressFrameMetrics, phase_frames

# Joints to highlight in the annotated video when a rule fires.
HIGHLIGHT_ROLES: dict[str, tuple[str, ...]] = {
    "press_symmetry": ("shoulder", "elbow", "wrist"),
    "press_alignment": ("elbow", "wrist"),
    "press_rom": ("shoulder", "elbow", "wrist"),
}

# Rules whose findings are about both arms at once.
BILATERAL_RULES: frozenset[str] = frozenset({"press_symmetry", "press_rom"})

_EMPTY_EVIDENCE = PersistenceEvidence(0, 0, 0, 0.0, -1, float("nan"))


def _aggregate(outcomes: list[RepRuleOutcome]) -> RuleStatus:
    """FAIL if at least half the evaluable reps failed, WARNING if any failed
    or warned, PASS otherwise, NOT_EVALUABLE if none had evidence."""
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


def _round(value: float, digits: int = 1):
    return round(float(value), digits) if math.isfinite(value) else None


def _persistence(
    spec: RuleSpec | None,
    metrics: list[PressFrameMetrics] | None,
    rep: PressRep,
    attribute: str,
    exceeds: float,
    absolute: bool = False,
) -> PersistenceEvidence:
    """How persistently attribute broke exceeds during the phase."""
    if spec is None or metrics is None:
        return _EMPTY_EVIDENCE
    frames = phase_frames(rep, spec.phase)
    values = series(metrics, attribute, frames)
    if absolute:
        values = np.abs(values)
    with np.errstate(invalid="ignore"):
        violating = values > exceeds
    return assess(values, violating, offset=frames.start, prefer_max=True)


def _with_persistence(outcome: RepRuleOutcome, evidence: PersistenceEvidence) -> RepRuleOutcome:
    outcome.violating_frames = evidence.violating_frames
    outcome.phase_frames = evidence.measurable_frames
    outcome.violation_ratio = round(evidence.ratio, 3)
    return outcome


# --- Rule 1 - arm symmetry ---


def classify_symmetry(rep: PressRep, config: PressConfig) -> tuple[list[str], RuleStatus]:
    """
    Which symmetry signals fired on this rep, and how firmly. Three signals rather
    than one combined index, because a beginner can act on "your left arm stayed
    lower" and can't act on "symmetry score 0.72":

        angle    |left - right| elbow angle           degrees
        height   |left - right| wrist height          shoulder widths
        rom      |left ROM - right ROM|               degrees

    The first two describe the movement as it happens; the third describes the rep
    as a whole, and catches an arm travelling the same way but not as far.
    """
    if not rep.symmetry_reliable:
        return [], RuleStatus.NOT_EVALUABLE

    measurable = [
        value
        for value in (
            rep.max_elbow_angle_difference,
            rep.max_wrist_height_difference,
            rep.rom_difference,
        )
        if math.isfinite(value)
    ]
    if not measurable:
        return [], RuleStatus.NOT_EVALUABLE

    signals: list[str] = []
    failed = False

    checks = (
        (
            "elbow angle",
            rep.max_elbow_angle_difference,
            config.SYMMETRY_ANGLE_WARN,
            config.SYMMETRY_ANGLE_FAIL,
        ),
        (
            "wrist height",
            rep.max_wrist_height_difference,
            config.SYMMETRY_HEIGHT_WARN,
            config.SYMMETRY_HEIGHT_FAIL,
        ),
        ("range of motion", rep.rom_difference, config.SYMMETRY_ROM_WARN, config.SYMMETRY_ROM_FAIL),
    )
    for name, value, warn, fail in checks:
        if not math.isfinite(value) or value < warn:
            continue
        signals.append(name)
        if value >= fail:
            failed = True

    if not signals:
        return [], RuleStatus.PASS
    return signals, (RuleStatus.FAIL if failed else RuleStatus.WARNING)


def rule_symmetry(
    reps: list[PressRep],
    config: PressConfig,
    spec: RuleSpec | None = None,
    metrics: list[PressFrameMetrics] | None = None,
) -> RuleResult:
    """
    Did both arms do the movement reasonably together? Judged over the whole working
    phase and only when both arms were individually visible for enough of it. One
    arm is never inferred from the other: a recording where one is hidden says
    "cannot assess", which is not the same as "your arms were even".
    """
    outcomes: list[RepRuleOutcome] = []
    for rep in reps:
        signals, status = classify_symmetry(rep, config)
        evidence = _persistence(
            spec, metrics, rep, "elbow_angle_difference", config.SYMMETRY_ANGLE_WARN
        )
        height_evidence = _persistence(
            spec,
            metrics,
            rep,
            "wrist_height_difference",
            config.SYMMETRY_HEIGHT_WARN,
            absolute=True,
        )
        # either in-movement signal can carry the persistence - an arm sitting
        # lower without much angle difference is still asymmetric
        best = max((evidence, height_evidence), key=lambda e: (e.longest_run, e.ratio))
        if (
            status in (RuleStatus.WARNING, RuleStatus.FAIL)
            and spec is not None
            and best.has_evidence
            and not best.triggers(spec.minimum_persistence_frames, spec.min_violation_ratio)
            and not (
                math.isfinite(rep.rom_difference) and rep.rom_difference >= config.SYMMETRY_ROM_WARN
            )
        ):
            # never lasted long enough to be movement rather than noise
            status = RuleStatus.PASS
            signals = []

        outcomes.append(
            _with_persistence(
                RepRuleOutcome(
                    rep_number=rep.number,
                    status=status,
                    evidence={
                        "signals": signals,
                        "max_elbow_angle_difference": _round(rep.max_elbow_angle_difference),
                        "max_wrist_height_difference": _round(rep.max_wrist_height_difference, 3),
                        "rom_difference": _round(rep.rom_difference),
                        "top_timing_difference": _round(rep.top_timing_difference, 2),
                        "higher_side": rep.higher_side,
                        "both_arms_visible_ratio": _round(rep.both_arms_ratio, 3),
                        "extreme_time": format_timestamp(rep.extreme_time),
                    },
                    evidence_frame=(
                        rep.max_wrist_height_difference_frame
                        if "wrist height" in signals
                        else rep.max_elbow_angle_difference_frame
                    ),
                    evidence_time=rep.extreme_time,
                ),
                best,
            )
        )

    status = _aggregate(outcomes)
    total = len(reps)
    flagged = [o for o in outcomes if o.status in (RuleStatus.FAIL, RuleStatus.WARNING)]

    if status is RuleStatus.NOT_EVALUABLE:
        explanation = (
            "Both arms were not visible clearly enough at the same time to compare them, "
            "so arm symmetry was not assessed."
        )
        correction = ""
    elif not flagged:
        explanation = (
            "Your arm movement stayed reasonably symmetrical across the analysed "
            "repetitions - both arms travelled together and reached similar positions."
        )
        correction = ""
    else:
        sides = [o.evidence.get("higher_side") for o in flagged if o.evidence.get("higher_side")]
        side = max(set(sides), key=sides.count) if sides else ""
        other = {"left": "right", "right": "left"}.get(side, "")
        worst = max(
            (o for o in flagged if o.evidence.get("max_elbow_angle_difference") is not None),
            key=lambda o: o.evidence["max_elbow_angle_difference"],
            default=flagged[0],
        )
        where = (
            f"Your {side} arm stayed higher than your {other} arm"
            if side
            else "One arm stayed ahead of the other"
        )
        explanation = (
            f"{where} during {_count_phrase(len(flagged), total)}. FormFix measured a "
            f"left/right elbow-angle difference of up to about "
            f"{worst.evidence['max_elbow_angle_difference']:.0f} deg that held through a "
            f"meaningful part of the press, against the configured tolerance of "
            f"{config.SYMMETRY_ANGLE_WARN:.0f} deg."
        )
        correction = (
            "Press both dumbbells at the same controlled pace and aim for both arms to "
            "travel together. Slowing the press down slightly makes it much easier to "
            "feel which side is leading."
        )

    return _result(
        spec,
        rule_id="press_symmetry",
        title="Arm symmetry",
        metric=METRIC_PRESS_SYMMETRY,
        status=status,
        explanation=explanation,
        correction=correction,
        outcomes=outcomes,
        evidence={
            "angle_warn_deg": config.SYMMETRY_ANGLE_WARN,
            "angle_fail_deg": config.SYMMETRY_ANGLE_FAIL,
            "height_warn_shoulder_widths": config.SYMMETRY_HEIGHT_WARN,
            "height_fail_shoulder_widths": config.SYMMETRY_HEIGHT_FAIL,
            "rom_warn_deg": config.SYMMETRY_ROM_WARN,
            "min_persistence_frames": config.SYMMETRY_MIN_FRAMES,
            "min_violation_ratio": config.SYMMETRY_MIN_VIOLATION_RATIO,
        },
        feedback_key="press_symmetry",
    )


# --- Rule 2 - elbow / wrist alignment ---


def rule_alignment(
    reps: list[PressRep],
    config: PressConfig,
    spec: RuleSpec | None = None,
    metrics: list[PressFrameMetrics] | None = None,
) -> RuleResult:
    """
    Did each wrist stay reasonably stacked over its own elbow? Measured in the
    frontal plane as |wrist_x - elbow_x| / shoulder width, so the same movement
    gives the same value at any camera distance or body size.

    The sides are assessed independently and the worse one reported, since a drift
    usually belongs to one arm and averaging would hide it.
    """
    outcomes: list[RepRuleOutcome] = []
    for rep in reps:
        left_evidence = _persistence(
            spec, metrics, rep, "left_alignment_offset", config.ALIGNMENT_OFFSET_WARN
        )
        right_evidence = _persistence(
            spec, metrics, rep, "right_alignment_offset", config.ALIGNMENT_OFFSET_WARN
        )
        offsets = {
            "left": rep.max_left_alignment_offset,
            "right": rep.max_right_alignment_offset,
        }
        measurable = {k: v for k, v in offsets.items() if math.isfinite(v)}

        if not rep.alignment_reliable or not measurable:
            status = RuleStatus.NOT_EVALUABLE
            worst_side = ""
            worst_value = float("nan")
            best = _EMPTY_EVIDENCE
        else:
            worst_side = max(measurable, key=lambda k: measurable[k])
            worst_value = measurable[worst_side]
            best = left_evidence if worst_side == "left" else right_evidence
            persistent = (
                spec is None
                or not best.has_evidence
                or best.triggers(spec.minimum_persistence_frames, spec.min_violation_ratio)
            )
            if not persistent or worst_value < config.ALIGNMENT_OFFSET_WARN:
                status = RuleStatus.PASS
            elif worst_value >= config.ALIGNMENT_OFFSET_FAIL:
                status = RuleStatus.FAIL
            else:
                status = RuleStatus.WARNING

        frame = rep.max_left_alignment_frame if worst_side == "left" else rep.max_right_alignment_frame
        outcomes.append(
            _with_persistence(
                RepRuleOutcome(
                    rep_number=rep.number,
                    status=status,
                    evidence={
                        "worst_side": worst_side,
                        "max_alignment_offset": _round(worst_value, 3),
                        "left_alignment_offset": _round(rep.max_left_alignment_offset, 3),
                        "right_alignment_offset": _round(rep.max_right_alignment_offset, 3),
                        "extreme_time": format_timestamp(rep.extreme_time),
                    },
                    evidence_frame=frame,
                    evidence_time=rep.extreme_time,
                ),
                best,
            )
        )

    status = _aggregate(outcomes)
    total = len(reps)
    flagged = [o for o in outcomes if o.status in (RuleStatus.FAIL, RuleStatus.WARNING)]

    if status is RuleStatus.NOT_EVALUABLE:
        explanation = (
            "Your elbows and wrists were not visible clearly enough to check how the "
            "dumbbells were stacked over your forearms."
        )
        correction = ""
    elif not flagged:
        explanation = (
            "Your wrists stayed within the configured alignment range of your elbows "
            "throughout the press."
        )
        correction = ""
    else:
        worst = max(
            (o for o in flagged if o.evidence.get("max_alignment_offset") is not None),
            key=lambda o: o.evidence["max_alignment_offset"],
            default=flagged[0],
        )
        which = worst.evidence.get("worst_side") or "one"
        explanation = (
            f"Your {which} wrist drifted noticeably away from being stacked over your "
            f"{which} elbow in {_count_phrase(len(flagged), total)}. The largest "
            f"horizontal wrist-to-elbow offset was about "
            f"{worst.evidence['max_alignment_offset'] * 100:.0f}% of your shoulder width "
            f"(rep {worst.rep_number}), against the configured tolerance of "
            f"{config.ALIGNMENT_OFFSET_WARN * 100:.0f}%."
        )
        correction = (
            "Keep each dumbbell stacked above its own forearm as you press, so your "
            "wrist travels straight up over your elbow rather than drifting in or out."
        )

    return _result(
        spec,
        rule_id="press_alignment",
        title="Elbow and wrist alignment",
        metric=METRIC_PRESS_ALIGNMENT,
        status=status,
        explanation=explanation,
        correction=correction,
        outcomes=outcomes,
        evidence={
            "warn_shoulder_widths": config.ALIGNMENT_OFFSET_WARN,
            "fail_shoulder_widths": config.ALIGNMENT_OFFSET_FAIL,
            "min_persistence_frames": config.ALIGNMENT_MIN_FRAMES,
            "min_violation_ratio": config.ALIGNMENT_MIN_VIOLATION_RATIO,
        },
        feedback_key="press_alignment",
    )


# --- Rule 3 - range of motion ---


def classify_rom(rep: PressRep, config: PressConfig) -> tuple[str, RuleStatus]:
    """Which end of the press fell short. The top criterion is not 180 degrees - a
    locked-out elbow is neither required nor desirable under load."""
    top, bottom = rep.top_elbow_angle, rep.bottom_elbow_angle
    if not (math.isfinite(top) and math.isfinite(bottom)):
        return ROM_NOT_ASSESSABLE, RuleStatus.NOT_EVALUABLE

    top_short = top < config.ROM_TOP_EXTENSION_PASS
    top_bad = top < config.ROM_TOP_EXTENSION_WARN
    bottom_short = bottom > config.ROM_BOTTOM_FLEXION_PASS
    bottom_bad = bottom > config.ROM_BOTTOM_FLEXION_WARN
    excursion_short = math.isfinite(rep.rom_degrees) and rep.rom_degrees < config.ROM_MIN_EXCURSION

    if not top_short and not bottom_short and not excursion_short:
        return ROM_COMPLETE, RuleStatus.PASS
    if top_short and bottom_short:
        return ROM_LIMITED_OVERALL, (RuleStatus.FAIL if (top_bad or bottom_bad) else RuleStatus.WARNING)
    if top_short:
        return ROM_LIMITED_TOP, (RuleStatus.FAIL if top_bad else RuleStatus.WARNING)
    if bottom_short:
        return ROM_LIMITED_BOTTOM, (RuleStatus.FAIL if bottom_bad else RuleStatus.WARNING)
    return ROM_LIMITED_OVERALL, RuleStatus.WARNING


def rule_range_of_motion(
    reps: list[PressRep],
    config: PressConfig,
    spec: RuleSpec | None = None,
    metrics: list[PressFrameMetrics] | None = None,
) -> RuleResult:
    """
    Did each press cover the configured range at the top and the bottom? Reported
    as which end fell short, because "limited range of motion" tells a beginner
    nothing - stopping short of overhead and never lowering the dumbbells back are
    different habits needing different corrections.
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
                    "left_rom_degrees": _round(rep.left_rom_degrees),
                    "right_rom_degrees": _round(rep.right_rom_degrees),
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
            "each press travelled."
        )
        correction = ""
    elif not flagged:
        explanation = (
            f"All {total} repetitions covered the configured movement range, at the "
            "shoulders and overhead."
            if total > 1
            else "Your repetition covered the configured movement range at both ends."
        )
        correction = ""
    elif top_issues >= bottom_issues:
        lowest = min(
            (o for o in flagged if o.evidence.get("top_elbow_angle") is not None),
            key=lambda o: o.evidence["top_elbow_angle"],
            default=flagged[0],
        )
        explanation = (
            f"Your presses stopped before reaching the configured top range in "
            f"{_count_phrase(len(flagged), total)}. The most restricted (rep "
            f"{lowest.rep_number}) reached about "
            f"{lowest.evidence['top_elbow_angle']:.0f} deg of elbow extension, against "
            f"the configured criterion of {config.ROM_TOP_EXTENSION_PASS:.0f} deg."
        )
        correction = (
            "Finish each press through a comfortable, controlled range before starting "
            "to lower the dumbbells. Reduce the weight if the last part of the press is "
            "hard to complete."
        )
    else:
        highest = min(
            (o for o in flagged if o.evidence.get("bottom_elbow_angle") is not None),
            key=lambda o: -o.evidence["bottom_elbow_angle"],
            default=flagged[0],
        )
        explanation = (
            f"The dumbbells did not come back down to the configured starting range in "
            f"{_count_phrase(len(flagged), total)}. At the lowest point your elbows were "
            f"still at about {highest.evidence['bottom_elbow_angle']:.0f} deg (rep "
            f"{highest.rep_number}), against the configured criterion of "
            f"{config.ROM_BOTTOM_FLEXION_PASS:.0f} deg or less."
        )
        correction = (
            "Lower the dumbbells back to about shoulder height under control between "
            "presses, rather than starting the next repetition early."
        )

    return _result(
        spec,
        rule_id="press_rom",
        title="Range of motion",
        metric=METRIC_PRESS_ROM,
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
        feedback_key="press_rom",
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


VIEW_LIMITATION_TEXT: dict[str, str] = {
    "press_symmetry": (
        "Comparing your left and right arm needs a front-on view - filmed from the side, "
        "one arm hides the other, so the difference between them would be measuring the "
        "camera angle rather than your movement."
    ),
    "press_alignment": (
        "Checking that each wrist stays stacked over its elbow needs a front-on view; "
        "from the side that relationship is depth, which a single camera cannot resolve."
    ),
    "press_rom": "Your arm angles could not be measured reliably from this camera position.",
}


def not_evaluable(spec: RuleSpec, reps: list[PressRep], reason: str) -> RuleResult:
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
    "press_symmetry": rule_symmetry,
    "press_alignment": rule_alignment,
    "press_rom": rule_range_of_motion,
}


def _subject_scale(metrics: list[PressFrameMetrics] | None) -> float:
    """Median shoulder width in pixels, scaled to be comparable with the trunk-length
    figure the other two exercises use, so one set of ceilings covers all three."""
    if not metrics:
        return float("nan")
    values = [m.shoulder_width for m in metrics if m.valid and np.isfinite(m.shoulder_width)]
    return float(np.median(values)) * 1.5 if values else float("nan")


def evaluate_all(
    reps: list[PressRep],
    config: PressConfig,
    metrics: list[PressFrameMetrics] | None = None,
    pose: FramePoseData | None = None,
    orientation: CameraOrientation = CameraOrientation.FRONTAL,
    side_view_confidence: float = 0.0,
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
                frontal_plane=is_frontal_plane(spec.metric),
            )
            result.reliability = reliability_for(evidence, bands_for(config))
            result.evidence["reliability_evidence"] = evidence.as_dict()
        elif result.status is not RuleStatus.NOT_EVALUABLE:
            result.reliability = Reliability.MEDIUM

        if result.status is RuleStatus.NOT_EVALUABLE and not result.limitation:
            result.limitation = result.explanation
        results.append(result)

    return results


def highlight_ids(rule_id: str, side: str) -> tuple[int, ...]:
    """A rule's highlight roles -> MediaPipe landmark indices."""
    roles = HIGHLIGHT_ROLES.get(rule_id, ())
    if not roles:
        return ()
    if rule_id in BILATERAL_RULES or side not in ARM_CHAINS:
        return tuple(sorted({*landmark_ids("left", roles), *landmark_ids("right", roles)}))
    return landmark_ids(side, roles)
