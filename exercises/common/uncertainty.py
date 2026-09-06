"""
How big a deviation has to be before it is worth calling a finding.

Every rule compares a measured value against a threshold, and both sides used
to be treated as exact. They aren't: the measurement comes from a monocular
pose estimator whose error is published and is large next to some of these
thresholds. A squat is flagged as shallow above 115 degrees of knee angle, and
Dill et al. (2023) put MediaPipe's knee-angle RMSE during squats at 9.14
degrees with a good camera angle - so a rep measured at 118 is not
distinguishable from one at 108.

This carries each measurement's published error through to the verdict and
marks any finding whose margin sits inside that error as indicative rather than
established. By default it only annotates; strict=True also downgrades those
findings a step, so both policies can be run over the same evaluation set.

Everything in PUBLISHED_BANDS traces to a figure in a paper. Anything derived
rather than quoted is marked derived=True with the working written out.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

from analysis.models import RuleStatus

__all__ = [
    "UncertaintyBand",
    "UncertaintyNote",
    "PUBLISHED_BANDS",
    "GONIOMETRY_REFERENCE",
    "ACCEPTABILITY_THRESHOLDS",
    "ACCEPTABILITY_SOURCE",
    "acceptability",
    "band_for",
    "annotate_uncertainty",
]


# --- The published error figures. ---


@dataclass(frozen=True)
class UncertaintyBand:
    """
    One measurement's error, in that measurement's own units. sigma is an RMSE
   , not a standard deviation, since RMSE is what the source studies report,
    and it is used as a scale, not a probability: a margin of one sigma counts as
    indistinguishable from zero.
    """

    metric: str
    sigma: float
    unit: str
    source: str
    derived: bool = False
    derivation: str = ""
    # Extra term for a known systematic bias, e.g. filter lag.
    systematic: float = 0.0
    systematic_source: str = ""

    @property
    def total(self) -> float:
        """
        Random and systematic error, added in quadrature as usual for independent
        sources, so a small systematic term next to a large random one barely moves it.
        """
        return math.hypot(self.sigma, self.systematic)

    def as_dict(self) -> dict:
        out = asdict(self)
        out["total"] = round(self.total, 3)
        if self.unit == "deg":
            out["acceptability"] = acceptability(self.total)
        return out


# the bars this literature uses. Mercadal-Baudart et al. (2024) evaluated a
# single-camera pose model against VICON: under 12 degrees is "good" (better
# than a physiotherapist by eye), under 6 "very good". FormFix reads about
# +/-10.7 for a knee angle, which is why 3 degrees is not a finding here.
# evaluated a single-camera 3D pose model against VICON: under 12 degrees is
ACCEPTABILITY_THRESHOLDS: dict[str, float] = {
    "good": 12.0,
    "very_good": 6.0,
}
ACCEPTABILITY_SOURCE = (
    "Mercadal-Baudart, Liu, Farrell, Boyne, Gonzalez Escribano, Smolic & Simms (2024), "
    "Heliyon 10:e27596 - 'good' if the metric error is below the ~12 deg accuracy of a "
    "physiotherapist's by-eye assessment in low-speed functional activities, 'very good' "
    "if below half of that"
)


def acceptability(sigma: float) -> str:
    """
    "very good" / "good" / "worse than by-eye assessment", so a band in the debug
    view is never a bare number with nothing to compare it against.
    """
    if sigma < ACCEPTABILITY_THRESHOLDS["very_good"]:
        return "very good"
    if sigma < ACCEPTABILITY_THRESHOLDS["good"]:
        return "good"
    return "worse than by-eye assessment"


# Hancock et al. (2018), via Dill et al. (2024): the smallest knee-angle
# difference clinical goniometry can call significant, on a stationary subject
GONIOMETRY_REFERENCE: dict[str, float] = {
    "digital_inclinometer": 6.0,
    "long_arm_goniometer": 10.0,
    "visual_estimation": 14.0,
    "short_arm_goniometer": 14.0,
}
GONIOMETRY_SOURCE = (
    "Hancock, Hepworth & Wembridge (2018), J Exp Orthop 5:46 - minimum significant "
    "difference by goniometry method, on a stationary subject; quoted in Dill et al. (2024)"
)


# Sagittal joint flexion (knee, hip, elbow) from one camera, limb NEAREST the camera.
_KNEE_NEAR = UncertaintyBand(
    metric="sagittal_joint_angle",
    sigma=10.7,
    unit="deg",
    source=(
        "Dill et al. (2024), Sensors 24(23):7772, Fig. 8 - knee-angle RMSE of monocular "
        "MediaPipe from a lateral camera, 810 squat repetitions over nine subjects, "
        "against marker-based motion capture"
    ),
)

# same for the limb FURTHEST from the camera. The biggest view-dependent
# effect in the literature, and why we pick the more visible side.
# correction - the biggest view-dependent effect in the literature, and why we
_KNEE_FAR = UncertaintyBand(
    metric="sagittal_joint_angle_far_side",
    sigma=25.1,
    unit="deg",
    source=(
        "Dill et al. (2024), Sensors 24(23):7772, Fig. 8b - knee-angle RMSE for the "
        "occluded (far) limb from a lateral camera; 25.1 deg against 10.7 deg for the "
        "near limb in the same recordings"
    ),
)

# squat knee angle with the camera at its best angle. Separate from the one
# above because it is a different study with a different setup.
# figure above because it is a different study with a different setup.
_KNEE_OPTIMAL = UncertaintyBand(
    metric="sagittal_joint_angle_optimal_view",
    sigma=9.14,
    unit="deg",
    source=(
        "Dill et al. (2023), Curr Dir Biomed Eng 9(1):563-566, Tab. 3 - right-knee-angle "
        "RMSE during squats, best camera angle (9.14 deg; 14.48 deg at an unfavourable angle)"
    ),
)

# trunk inclination. Not reported directly anywhere, so it comes from the
# positional error of the landmarks behind it.
_TRUNK = UncertaintyBand(
    metric="trunk_inclination",
    sigma=6.5,
    unit="deg",
    derived=True,
    source=(
        "Derived from Dill et al. (2024), Sensors 24(23):7772 - median landmark RMSE of "
        "56.3 mm for monocular MediaPipe 3D estimation"
    ),
    derivation=(
        "Trunk inclination is the angle of the mid-hip -> mid-shoulder segment. With a "
        "positional error e at each end and an adult hip-to-shoulder length L of roughly "
        "500 mm, the worst-case angular error is atan(2e / L) for anti-correlated errors "
        "and atan(e / L) for independent ones. Taking e = 56.3 mm and the independent "
        "case gives atan(56.3/500) = 6.4 deg; rounded to 6.5. This is an estimate, not a "
        "measurement, and it is the weakest number in this table."
    ),
)

# left-vs-right comparison of the same angle. Two independent measurements,
# so the difference carries sqrt(2) times the error of one.
# measurements, so the difference carries sqrt(2) times the error of one.
_BILATERAL = UncertaintyBand(
    metric="bilateral_angle_difference",
    sigma=15.1,
    unit="deg",
    derived=True,
    source="Derived from Dill et al. (2024) knee-angle RMSE (10.7 deg per limb)",
    derivation=(
        "A left-right difference is the difference of two independent estimates, so its "
        "error is sqrt(2) x the single-limb error: sqrt(2) x 10.7 = 15.1 deg. This assumes "
        "both limbs are equally visible; from a side view the far limb is not, and the "
        "band would be sqrt(10.7^2 + 25.1^2) = 27.3 deg, which is why FormFix restricts "
        "the symmetry check to frontal and diagonal views."
    ),
)

# lengths normalised by another body dimension. The uncertainty is mostly
# the instability of the normalising dimension, which Dill et al. measured.
# length, wrist offset over shoulder width). The uncertainty is mostly the
_NORMALISED_LENGTH = UncertaintyBand(
    metric="normalised_length",
    sigma=0.15,
    unit="normalised",
    derived=True,
    source=(
        "Derived from Dill et al. (2023), Curr Dir Biomed Eng 9(1):563-566, Tab. 2 - "
        "shoulder-width RMSE 5.71-29.78% and hip-width RMSE 4.99-34.37% of the true "
        "normalised width, across subjects and camera angles"
    ),
    derivation=(
        "MediaPipe's estimate of a fixed body dimension varies by roughly 5-34% of its "
        "true value depending on the camera angle, even though anatomy makes it constant. "
        "Any quantity normalised by such a dimension inherits that variability. 0.15 is "
        "the mid-range of the published spread, expressed as a fraction of the "
        "normalising length."
    ),
)

# timing. Frame timestamps are exact to a frame interval, so this is pure
# quantisation; it exists so no rule ends up with no band at all.
# is pure quantisation; it exists so no rule ends up with no band at all.
_DURATION = UncertaintyBand(
    metric="duration",
    sigma=0.067,
    unit="s",
    derived=True,
    source="Frame-rate quantisation, not a pose-estimation error",
    derivation=(
        "Two frame boundaries at 30 fps, i.e. 2/30 s. Phase boundaries are located to "
        "the nearest frame at each end, so a duration carries two quantisation errors."
    ),
)


PUBLISHED_BANDS: dict[str, UncertaintyBand] = {
    band.metric: band
    for band in (
        _KNEE_NEAR,
        _KNEE_FAR,
        _KNEE_OPTIMAL,
        _TRUNK,
        _BILATERAL,
        _NORMALISED_LENGTH,
        _DURATION,
    )
}


# which band applies to which metric. Written out rather than inferred, so a
# new metric with no error figure is a visible gap and not a silent zero.
METRIC_BANDS: dict[str, str] = {
    # squat
    "squat_depth": "sagittal_joint_angle",
    "torso_lean": "trunk_inclination",
    "heel_lift": "normalised_length",
    "return_to_standing": "sagittal_joint_angle",
    "descent_control": "duration",
    # lat pulldown
    "pulldown_rom": "sagittal_joint_angle",
    "pulldown_torso": "trunk_inclination",
    # shoulder press
    "press_rom": "sagittal_joint_angle",
    "press_symmetry": "bilateral_angle_difference",
    "press_alignment": "normalised_length",
}


# which evidence key holds each rule's measured value per rep
MEASURED_KEYS: dict[str, str] = {
    "squat_depth": "min_knee_angle",
    "torso_lean": "max_torso_lean",
    "heel_lift": "heel_lift_ratio",
    "return_to_standing": "shortfall_deg",
    "descent_control": "descent_seconds",
    # lat pulldown
    "pulldown_torso": "max_torso_excursion",
    # shoulder press
    "press_symmetry": "max_elbow_angle_difference",
    "press_alignment": "max_alignment_offset",
    # the two range-of-motion rules are missing on purpose: each combines three
    # criteria in different units, so there is no single value to compare
}


# --- Picking the band that applies to one measurement in one recording. ---


def band_for(
    metric: str,
    *,
    rule_id: str = "",
    view_support: float = 1.0,
    far_side: bool = False,
    systematic: float = 0.0,
    systematic_source: str = "",
) -> UncertaintyBand | None:
    """
    The band for one metric, adjusted for this recording. view_support is the same
    0-1 number the reliability layer computes: Dill et al. saw knee-angle RMSE go
    from 9.14 to 14.48 degrees between a good and a bad camera angle, a factor of
    1.58, so the band is scaled linearly between 1.0 and that as support falls.
    That interpolation is derived, not the paper's, and is marked as such.

    far_side picks the occluded-limb figure, which is over twice the near-limb one.
    """
    key = METRIC_BANDS.get(rule_id) or METRIC_BANDS.get(metric) or metric
    band = PUBLISHED_BANDS.get(key)
    if band is None:
        return None

    if far_side and key == "sagittal_joint_angle":
        band = replace(PUBLISHED_BANDS["sagittal_joint_angle_far_side"], metric=key)

    # 14.48 / 9.14, from Dill et al. (2023) Tab. 3, squat right knee.
    worst_case_factor = 1.584
    support = min(max(float(view_support), 0.0), 1.0)
    if band.unit == "deg" and support < 1.0:
        scale = 1.0 + (worst_case_factor - 1.0) * (1.0 - support)
        band = replace(
            band,
            sigma=round(band.sigma * scale, 2),
            derived=True,
            derivation=((band.derivation + " ") if band.derivation else "")
            + (
                f"Scaled by {scale:.2f} for a camera view whose support for this "
                f"measurement is {support:.2f}. The scale interpolates linearly to the "
                "1.58x degradation Dill et al. (2023) measured between a favourable and "
                "an unfavourable camera angle; the interpolation is FormFix's own."
            ),
        )

    if systematic:
        band = replace(band, systematic=abs(systematic), systematic_source=systematic_source)
    return band


# --- The note attached to a finding ---


@dataclass(frozen=True)
class UncertaintyNote:
    """What the measurement error says about one rule's verdict."""

    rule_id: str
    band: UncertaintyBand
    # how far past the threshold the worst flagged rep was, in the metric's units
    worst_margin: float | None
    margins: dict[int, float] = field(default_factory=dict)
    inconclusive_reps: tuple[int, ...] = ()
    beyond_measurement_error: bool = False
    sentence: str = ""

    def as_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "band": self.band.as_dict(),
            "worst_margin": (round(self.worst_margin, 3) if self.worst_margin is not None else None),
            "margins": {str(k): round(v, 3) for k, v in self.margins.items()},
            "inconclusive_reps": list(self.inconclusive_reps),
            "beyond_measurement_error": self.beyond_measurement_error,
        }


def _thresholds(spec) -> tuple[float | None, float | None, bool]:
    """(pass_bar, fail_bar, higher_is_worse) for one rule spec. A rule with
    acceptable_max is broken by being too high, one with acceptable_min too low."""
    if spec.acceptable_max is not None:
        return spec.acceptable_max, spec.acceptable_max + spec.tolerance, True
    if spec.acceptable_min is not None:
        return spec.acceptable_min, spec.acceptable_min - spec.tolerance, False
    return None, None, True


def _margin(value: float, bar: float, higher_is_worse: bool) -> float:
    """Distance past bar, positive when the value is in violation."""
    return (value - bar) if higher_is_worse else (bar - value)


def annotate_uncertainty(
    rule_results,
    specs: dict,
    *,
    view_support: dict[str, float] | None = None,
    far_side_rules: tuple[str, ...] = (),
    systematic: dict[str, tuple[float, str]] | None = None,
    strict: bool = False,
) -> list[UncertaintyNote]:
    """
    Attach an uncertainty note to every evaluable rule result. Mutates
    rule_results in place and returns the notes for the debug export.
    strict=True also downgrades marginal verdicts a step.
    """
    view_support = view_support or {}
    systematic = systematic or {}
    notes: list[UncertaintyNote] = []

    for result in rule_results:
        spec = specs.get(result.rule_id)
        if spec is None or result.status is RuleStatus.NOT_EVALUABLE:
            continue

        sys_value, sys_source = systematic.get(result.rule_id, (0.0, ""))
        band = band_for(
            result.metric or spec.metric,
            rule_id=result.rule_id,
            view_support=view_support.get(result.rule_id, 1.0),
            far_side=result.rule_id in far_side_rules,
            systematic=sys_value,
            systematic_source=sys_source,
        )
        if band is None:
            continue

        pass_bar, _fail_bar, higher_is_worse = _thresholds(spec)
        measured_key = MEASURED_KEYS.get(result.rule_id, "")
        margins: dict[int, float] = {}
        inconclusive: list[int] = []
        cleared = False

        for outcome in result.per_rep:
            if outcome.status not in (RuleStatus.FAIL, RuleStatus.WARNING):
                continue
            raw = outcome.evidence.get(measured_key) if measured_key else None
            if raw is None or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                continue
            # always the pass bar, never the fail bar - the question is whether there
            # is a finding at all
            if pass_bar is None:
                continue
            margin = _margin(float(raw), float(pass_bar), higher_is_worse)
            margins[outcome.rep_number] = margin
            if margin < band.total:
                inconclusive.append(outcome.rep_number)
            else:
                cleared = True

        worst = max(margins.values()) if margins else None
        sentence = ""
        if margins:
            unit = "deg" if band.unit == "deg" else band.unit
            if cleared:
                sentence = (
                    f"Measurement note: the largest deviation was {worst:.1f} {unit} beyond "
                    f"the configured limit, which clears the +/-{band.total:.1f} {unit} "
                    "uncertainty published for this measurement from a single camera."
                )
            else:
                sentence = (
                    f"Measurement note: the largest deviation was only {worst:.1f} {unit} "
                    f"beyond the configured limit, inside the +/-{band.total:.1f} {unit} "
                    "uncertainty published for this measurement from a single camera. Treat "
                    "this as indicative rather than established."
                )

        note = UncertaintyNote(
            rule_id=result.rule_id,
            band=band,
            worst_margin=worst,
            margins=margins,
            inconclusive_reps=tuple(inconclusive),
            beyond_measurement_error=cleared,
            sentence=sentence,
        )
        notes.append(note)

        result.evidence["measurement_uncertainty"] = note.as_dict()
        if sentence:
            result.explanation = f"{result.explanation} {sentence}".strip()

        if strict and inconclusive:
            _downgrade(result, tuple(inconclusive))

    return notes


def _downgrade(result, inconclusive: tuple[int, ...]) -> None:
    """Soften the reps whose margin stayed inside the band: FAIL -> WARNING,
    WARNING -> PASS. The rule's overall status is recomputed from its reps, so the
    aggregate can't contradict them."""
    step = {RuleStatus.FAIL: RuleStatus.WARNING, RuleStatus.WARNING: RuleStatus.PASS}
    for outcome in result.per_rep:
        if outcome.rep_number in inconclusive and outcome.status in step:
            outcome.status = step[outcome.status]

    evaluable = [o for o in result.per_rep if o.status is not RuleStatus.NOT_EVALUABLE]
    if not evaluable:
        return
    if any(o.status is RuleStatus.FAIL for o in evaluable):
        result.status = RuleStatus.FAIL
    elif any(o.status is RuleStatus.WARNING for o in evaluable):
        result.status = RuleStatus.WARNING
    else:
        result.status = RuleStatus.PASS
        result.correction = ""
