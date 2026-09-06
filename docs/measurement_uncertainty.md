# Measurement uncertainty

## The gap this closes

Until now every rule in FormFix compared a measured value against a threshold
and treated both sides of that comparison as exact. The threshold was honestly
labelled as provisional. The *measurement* was not labelled at all — and it is
the less certain of the two.

MediaPipe's error on the exact quantities FormFix reads has been measured and
published, and it is large relative to some of the thresholds in use:

| Measurement | Published error | FormFix threshold it feeds |
| --- | --- | --- |
| Knee angle, near limb, single lateral camera | RMSE **10.7°** | depth pass bar at 100°, warning at 115° |
| Knee angle, far (occluded) limb, same recording | RMSE **25.1°** | — (which is why the far limb is never used) |
| Knee angle during squats, optimal camera angle | RMSE **9.14°** (14.48° at a poor angle) | as above |
| Shoulder width / hip width, as a fraction of true | RMSE **5.7–29.8%** / **5.0–34.4%** | every normalised length |

Sources: Dill et al. (2024) *Sensors* 24(23):7772; Dill et al. (2023)
*Curr Dir Biomed Eng* 9(1):563–566.

Put plainly: a repetition FormFix measures at 108° cannot be distinguished
from one at 98°. Reporting the first as shallow and the second as fine states
more than a single camera can support.

A useful yardstick sits in the same literature. Hancock et al. (2018), quoted
by Dill et al., report the minimum difference in knee angle that clinical
goniometry can call significant — on a *stationary* patient on an examination
table:

| Method | Minimum significant difference |
| --- | --- |
| Digital inclinometer | 6° |
| Long-arm goniometer | 10° |
| Visual estimation / short-arm goniometer | 14° |

FormFix's 10.7° band sits between a long-arm goniometer and a physiotherapist's
naked eye. That is a reasonable place for a phone camera to be. It is not a
place from which 3° distinctions can be drawn.

Mercadal-Baudart et al. (2024) turn the same benchmark into an explicit
acceptability criterion for exactly this kind of system: a metric whose error
is below **12°** is "good", because it beats by-eye assessment, and below
**6°** is "very good". FormFix labels every angular band against it:

| FormFix measurement | Band | Verdict |
| --- | --- | --- |
| Trunk inclination | ±6.5° | good |
| Knee angle, near limb (depth) | ±10.7°, or ±11.6° including filter bias | good |
| Bilateral knee difference (left/right) | ±15.1° | **worse than by-eye assessment** |
| Knee angle, far limb from a side view | ±25.1° | **worse than by-eye assessment** |

Two of FormFix's measurements clear the field's own bar for usefulness and two
do not. Saying so is more valuable than not measuring it.

## What FormFix does about it

`exercises/common/uncertainty.py` carries the published error of each
measurement through to the point where the verdict is formed.

For every flagged repetition it computes the **margin**: how far past the pass
bar the measurement actually went. It compares that margin to the band, and
labels the finding:

* **beyond measurement error** — the margin cleared the band. The finding
  stands on evidence the instrument can support.
* **inside measurement error** — the margin did not. The finding is reported,
  but described as *indicative rather than established*.

Either way, one sentence is appended to the rule's explanation, and the full
record — band, source, per-repetition margins, which repetitions were
inconclusive — goes into `evidence["measurement_uncertainty"]`, the developer
panel (`FORMFIX_DEBUG`) and the JSON export.

### Two policies, deliberately not one

By default the layer **annotates and does not overrule**. No verdict, no
score and no reliability label changes. Setting `UNCERTAINTY_STRICT = True`
(or `--strict-uncertainty` on the evaluation harness) additionally downgrades
a marginal finding one step: FAIL → WARNING, WARNING → PASS.

Both exist because the choice between them is a real question that a
dissertation should answer with evidence rather than assert. On one synthetic
recording of three 105° repetitions, the three available policies disagree by
17 score points:

| Policy | Depth verdict | Score |
| --- | --- | --- |
| Default thresholds, annotate only | warning | 90 |
| Default thresholds, strict uncertainty | pass | 100 |
| Literature preset, annotate only | fail | 83 |

That spread is not a defect. It is the honest size of the disagreement between
three defensible positions about the same 105° squat, and it is only visible
because the uncertainty is modelled rather than assumed away.

## Where each band comes from

Every figure is either quoted from a paper or derived here, and derived ones
carry their derivation in the code so a reader can disagree with the argument
rather than only with the number.

| Band | Value | Quoted or derived |
| --- | --- | --- |
| Sagittal joint angle, near limb | ±10.7° | Quoted — Dill et al. (2024), Fig. 8 |
| Sagittal joint angle, far limb | ±25.1° | Quoted — Dill et al. (2024), Fig. 8b |
| Squat knee angle, optimal view | ±9.14° | Quoted — Dill et al. (2023), Tab. 3 |
| Trunk inclination | ±6.5° | **Derived** — `atan(56.3 mm / 500 mm)` from the median monocular landmark RMSE and an adult hip-to-shoulder length |
| Bilateral angle difference | ±15.1° | **Derived** — `√2 × 10.7°`, the error of a difference of two independent estimates |
| Normalised length | ±0.15 | **Derived** — mid-range of the published 5–34% instability of MediaPipe's estimate of a fixed body width |
| Duration | ±0.067 s | **Derived** — two frame boundaries at 30 fps; not a pose-estimation error at all |

Two adjustments are applied on top:

**Camera view.** Dill et al. (2023) measured knee-angle RMSE rising from 9.14°
to 14.48° between a favourable and an unfavourable camera angle — a factor of
1.58. The band is scaled linearly between 1.0 and that factor as the existing
`view_support` score falls from 1 to 0. The interpolation is FormFix's, not
the paper's, and any band it touches is marked derived. It is applied only to
angles: borrowing an angle citation for a duration would be dishonest.

**Filter bias.** The smoothing filter has its own measured systematic error on
the depth measurement (see [filter selection](filter_selection.md)). It is
added in quadrature, so the depth band with the default EMA is
`√(10.7² + 4.53²) = 11.6°` rather than 10.7°.

## What this layer found out about FormFix's own thresholds

Building it produced two findings that no paper contains, because they concern
FormFix's numbers rather than MediaPipe's:

1. **The squat's left/right evenness warning bar (12°) sat below the noise
   floor of its own measurement (±15.1°).** A bilateral comparison is a
   difference of two independent estimates, so it is *less* precise than
   either. The check as configured was capable of reporting an asymmetry that
   was entirely measurement error. **This one was acted on: the rule was
   removed.** The measurement is still computed and exported; it is simply no
   longer turned into a verdict. (The shoulder press keeps its own symmetry
   rule, where a front-on view is the recommended one and both arms are
   visible by design.)

2. **The heel-lift threshold (0.06 lower-leg lengths) sits well below the
   ±0.15 instability of the normalising dimension.** MediaPipe's estimate of a
   fixed body width wanders by 5–34% of its true value across camera angles,
   and any quantity divided by such a dimension inherits that.

The heel threshold has not been changed in the default configuration, for the
reason the tuning log already gives: no labelled recordings exist yet, and
moving a
threshold to fix a problem found in a spreadsheet is still moving a threshold
without evidence. Both are raised above their noise floor in the
[literature preset](../exercises/squat/literature_config.py), so the two can be
run over the same recordings and compared. Both are pinned by tests in
`tests/test_uncertainty.py::TestWhatTheLayerRevealsAboutTheThresholds`, so if
anyone later tightens them the suite says plainly what they are tightening
into.

## What it does not do

* It does not turn RMSE into a probability. The band is used as a scale, not
  as a confidence interval; the sources report RMSE against motion capture,
  not a distribution, and treating one as the other would be inventing
  precision at the exact point this module exists to avoid inventing it.
* It does not apply to the two range-of-motion rules. Each combines three
  criteria with different units and different bars, so no single measured
  value can be compared against the one acceptable range a `RuleSpec`
  declares. Their band is published on the result and the margin is reported
  as unavailable — rather than computed from whichever criterion happens to
  be nearest, which would look precise and mean nothing.
* It does not replace the reliability layer. Reliability answers "how much
  evidence stands behind this measurement in this recording?" Uncertainty
  answers "how large a deviation can this instrument resolve at all?" A
  perfectly visible, fully reliable knee angle still carries ±10.7°.
