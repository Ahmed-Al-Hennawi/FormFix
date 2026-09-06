# Threshold provenance and tuning log

Every technique threshold in FormFix is an **operational prototype value**. It
was chosen from the exercise definition, the FormFix reference demonstration
and iterative testing; it is not a biomechanical constant, and the system never
presents it as one. The interface's wording is "the configured range"
throughout, and every rule carries its `threshold_source` string into the
developer panel (`FORMFIX_DEBUG`) and the export bundle.

This document is the development history: what each value started at, why, what
testing showed, and what it ended at. It is deliberately auditable — a
threshold that moved without a recorded reason is a threshold that cannot be
defended.

---

## 1. Two kinds of value, and why the distinction matters

| Kind | What it controls | Can it be defended biomechanically? |
| --- | --- | --- |
| **Engineering** | detector confidences, gap handling, hysteresis, persistence, plausibility bands, reliability banding | No, and it does not need to be. These make no claim about how an exercise should be performed; they control how noisy the system is. |
| **Technique (operational)** | depth, lean, heel contact, symmetry (press), alignment, range of motion | Only as an operational definition. They state what FormFix was configured to look for, not what is anatomically correct for every body. |

A methodology chapter can say this, and it is defensible:

> An operational threshold was defined for the prototype based on the selected
> exercise definition, the reference demonstration, literature-informed
> movement principles and iterative testing.

It cannot say "the scientifically correct lat-pulldown trunk angle is 15°",
because no such constant exists, and nothing in the ten reviewed papers
establishes one for these movements, these camera positions and these users.

---

## 2. Segmentation thresholds are not technique thresholds

A design decision worth stating separately, because it changed during
development and it is the reason two rules work at all.

The repetition state machine's four hysteresis levels were originally fixed
absolute angles (lat pulldown: rest 148°, start 140°, contracted 115°, end
143°). Testing against a synthetic athlete generated with a 132° top position
exposed a circular failure:

> The athlete never crossed the fixed "extended" level, so no repetition was
> detected, so the range-of-motion rule — the very check meant to notice that
> habit — never ran. FormFix reported "no repetition found" for exactly the
> fault it exists to find.

The same applies to a press with no lockout. The levels were therefore made
**adaptive**: fixed margins below a high percentile of the athlete's own
measured series, clamped to an anatomically sane band. This separates two
questions that must not be conflated:

| Question | Answered by | Nature |
| --- | --- | --- |
| Was this a repetition? | the adaptive levels | permissive segmentation |
| Did it cover enough range? | the fixed `ROM_*` criteria | technique judgement |

Regression tests pin both halves down
(`test_pulldown_metrics.py::test_a_restricted_athlete_still_has_their_repetitions_counted`,
`test_press_metrics.py::test_an_athlete_who_never_locks_out_still_has_reps_counted`).

---

## 3. Lat pulldown

### 3.1 Torso movement — `pulldown_torso`

| Threshold | Initial | Reason for the initial value | Test result | Adjustment | Final |
| --- | --- | --- | --- | --- | --- |
| `TORSO_EXCURSION_WARN` | 15° | The reference demonstration holds roughly 10–20° of trunk inclination throughout, with only a few degrees of *change* between the extended and contracted positions. 15° of change is well beyond that, plus a margin for MediaPipe's own estimation error. | Synthetic recordings: 0° gain → 3.8° measured (pass); 12° gain → pass; 18° gain → 17.3° measured (warning); 28° gain → 26.9° (fail). Behaves as intended across the band. | none | **15°** |
| `TORSO_EXCURSION_FAIL` | 25° | Roughly 1.7× the warning bar, so a clear body swing separates from a moderate one. | 28° gain reached fail as intended. | none | **25°** |
| `TORSO_ABSOLUTE_FAIL` | 45° | A second, absolute guard: a trunk 45° from vertical during the pull is no longer the exercise being analysed, regardless of where the baseline sat. | Never fired on any correct synthetic recording. | none | **45°** |
| `TORSO_MIN_FRAMES` | 4 | Engineering: about 0.13 s at 30 fps — long enough that landmark jitter cannot sustain it. | No false positive on the noisy synthetic (σ = 0.004 normalised). | none | **4** |
| `TORSO_MIN_VIOLATION_RATIO` | 0.20 | Engineering: a swing that occupies a fifth of the pull is movement, not noise. | As above. | none | **0.20** |

**Not yet validated against recorded footage.** The numbers above were chosen
from the reference demonstration and confirmed against synthetic material with
a known ground truth. The synthetic test is a test of the *implementation*, not
of the threshold's suitability for real athletes — that requires the labelled
recordings described in `evaluation/README.md`.

### 3.2 Range of motion — `pulldown_rom`

| Threshold | Initial | Reason for the initial value | Test result | Adjustment | Final |
| --- | --- | --- | --- | --- | --- |
| `ROM_TOP_EXTENSION_PASS` | 150° | The arms should return towards extension between repetitions. Not 180°: nobody is asked to lock out under load, and demanding it would flag correct technique. | 172° top → pass; 132° top → fail. | none | **150°** |
| `ROM_TOP_EXTENSION_WARN` | 138° | The band between warning and failure, roughly 12° wide, matching the tolerance used elsewhere. | 142° → warning; 125° → fail. | none | **138°** |
| `ROM_BOTTOM_FLEXION_PASS` | 100° | The reference demonstration finishes the pull with the elbow close to or past a right angle. 100° is a permissive reading of that. | 80° bottom → pass; 125° bottom → fail. | none | **100°** |
| `ROM_BOTTOM_FLEXION_WARN` | 115° | 15° of tolerance beyond the pass criterion. | 108° → warning. | none | **115°** |
| `ROM_MIN_EXCURSION` | 45° | Catches a repetition that is shallow at both ends without failing either single criterion. | Fires only in combination, as intended. | none | **45°** |

### 3.3 Camera and reliability

Unchanged from the squat's values (`SIDE_VIEW_GOOD_RATIO` 0.45,
`SIDE_VIEW_FRONTAL_RATIO` 1.00): the pulldown wants the same sagittal view a
squat does, so the same bands apply and the orientation vocabulary stays
comparable across FormFix.

---

## 4. Dumbbell shoulder press

### 4.1 Camera bands — changed during development

| Threshold | Initial | What testing showed | Final |
| --- | --- | --- | --- |
| `SIDE_VIEW_GOOD_RATIO` | 0.45 (inherited from the squat) | — | **0.35** |
| `SIDE_VIEW_FRONTAL_RATIO` | 1.00 (inherited from the squat) | A square front-on synthetic recording measured a frontality ratio of **0.71** and was classified `diagonal_side`, not `frontal`. Every left/right comparison in a perfectly good front-on recording was therefore being downgraded to reduced reliability. The 1.00 band assumes shoulder separation of roughly 1.2× trunk length; adult biacromial width is closer to 0.8–0.9× hip-to-shoulder trunk length. | **0.70** |

Regression test:
`test_press_integration.py::test_a_square_front_view_is_recognised_as_frontal`,
which also asserts that every rule then reports `High` reliability. **This is a
threshold change made to fix a misclassification, not to make a fault
detectable** — no rule verdict changed, only the confidence attached to it.

### 4.2 Arm symmetry — `press_symmetry`

| Threshold | Initial | Reason | Test result | Adjustment | Final |
| --- | --- | --- | --- | --- | --- |
| `SYMMETRY_ANGLE_WARN` | 15° | A few degrees of asymmetry is normal human movement *and* lies inside MediaPipe's estimation error, so a tighter bar would measure the pose estimator rather than the athlete. | 4% lag → 3.3° (pass); 28% lag → 23.2° (fail). | none | **15°** |
| `SYMMETRY_ANGLE_FAIL` | 25° | Roughly 1.7× the warning bar. | As above. | none | **25°** |
| `SYMMETRY_HEIGHT_WARN` | 0.12 shoulder widths | ≈ 5 cm of wrist-height difference for an adult — visible to an observer, well above landmark noise. | Synchronised press → < 0.01; 28% lag → 0.22. | none | **0.12** |
| `SYMMETRY_HEIGHT_FAIL` | 0.20 shoulder widths | ≈ 8 cm. | As above. | none | **0.20** |
| `SYMMETRY_ROM_WARN` / `FAIL` | 15° / 25° | Matched to the angle thresholds, since both are elbow-angle differences. | Right arm capped at 130° → range difference flagged. | none | **15° / 25°** |
| `SYMMETRY_MIN_FRAMES` | 5 | Engineering: ≈ 0.17 s at 30 fps. | No false positive on noise or on a four-frame landmark drop. | none | **5** |
| `SYMMETRY_MIN_VIOLATION_RATIO` | 0.25 | Engineering. | As above. | none | **0.25** |

### 4.3 Elbow / wrist alignment — `press_alignment` — changed during development

| Threshold | Initial | What testing showed | Final |
| --- | --- | --- | --- |
| `ALIGNMENT_OFFSET_WARN` | 0.30 shoulder widths (≈ 12 cm) | A synthetic press with a 138° top and **no wrist drift at all** measured a sustained offset of 0.34 and was flagged for misalignment. This is geometrically correct — a repetition that stops short of overhead necessarily leaves the wrist beside the elbow, because the elbow is still bent — but it double-reports one habit as two faults and buries the correction that matters. | **0.38** |
| `ALIGNMENT_OFFSET_FAIL` | 0.45 | Widened proportionally. | **0.52** |

After the change: no-drift limited-range recording → 0.34 (**pass**, correctly
attributing the fault to range of motion alone); genuine 0.6 drift → 0.55
(**fail**). The residual correlation for *severely* limited presses cannot be
removed without making the check meaningless, and is documented in
`press_analysis.md` rather than hidden.

**This change was made to remove a false positive on a correct-for-that-check
recording, not to make a labelled test video pass.** It is the distinction
between calibration and overfitting, and it is why the reason is recorded here.

### 4.4 Range of motion — `press_rom`

| Threshold | Initial | Reason | Test result | Adjustment | Final |
| --- | --- | --- | --- | --- | --- |
| `ROM_TOP_EXTENSION_PASS` | 155° | The reference demonstration presses to near-full extension. Deliberately **not** 180°: a fully locked elbow is neither required nor desirable under load. | 168° → pass; 157° → pass; 148° → warning; 138° → fail. | none | **155°** |
| `ROM_TOP_EXTENSION_WARN` | 143° | ≈ 12° of tolerance. | As above. | none | **143°** |
| `ROM_BOTTOM_FLEXION_PASS` | 100° | The reference lowers the dumbbells to about shoulder/ear height, putting the elbow near a right angle. 100° is a permissive reading. | 85° → pass; 108° → warning; 125° → fail. | none | **100°** |
| `ROM_BOTTOM_FLEXION_WARN` | 115° | 15° of tolerance. | As above. | none | **115°** |
| `ROM_MIN_EXCURSION` | 50° | Catches a press that is shallow at both ends. | Fires only in combination. | none | **50°** |

---

## 5. Squat

The squat's technique thresholds were established before this work and are
**unchanged**. They are documented in `squat_analysis.md` §15 and remain
labelled as provisional calibration values in `exercises/squat/config.py`.

One squat-adjacent inconsistency was found and **fixed**: a frontal-plane
measurement should score the complement of `side_view_confidence` on a
diagonal recording (as the press's frontal-plane rules do) rather than the
confidence itself. Before the fix, the *worst* diagonal recordings for a
left/right comparison — the ones nearly side-on, where one leg lines up
behind the other — were credited with the highest confidence. This was found
on the squat's `knee_symmetry` rule; that rule has since been **removed**
(§7), but the correction stands and the press depends on it.

`exercises/squat/landmarks.py` now declares `FRONTAL_PLANE_METRICS`, and the
confidence adapter passes the flag through. The change affects reliability
*labels* only, on diagonal recordings only — never a verdict, and never a
threshold. The full squat regression suite passes unchanged, and
`tests/test_confidence.py::TestFrontalPlaneMetricsOnADiagonalCamera` pins the
new behaviour in both directions.

---

## 6. Reliability gates corrected after real-video testing

Not thresholds in the technique sense — these decide whether a check *runs at
all* — but they changed after running real footage, so they belong in the same
audit trail. Both are recorded in `evaluation/reference_clip_runs.md` with the
before/after verdicts.

| Gate | Was | Now | Why |
| --- | --- | --- | --- |
| Lat pulldown `arms_reliable` | both arms usable on ≥ 50% of the repetition | the **analysed** arm usable on ≥ 50% | Range of motion is a single-arm joint angle. In the side view this exercise asks for, the far arm is hidden behind the near one for much of the pull, so the check reported "not assessed" on a well-recorded side view — the opposite of what the camera guidance is for. Both-arm availability is still recorded for the export. |
| Shoulder press `alignment_reliable` | mean visibility of **both** arms' elbow/wrist pairs ≥ 0.5 | the **best** side's pair ≥ 0.5 | Alignment is judged per side and the worse side is reported, so one visible arm is enough. The shared gate let a hidden arm switch the check off for the arm that was perfectly visible. |

Neither change moves a technique threshold, and neither makes a fault easier to
report: both restore a check that the recording *could* support and that the
gate was wrongly suppressing.

---

## 7. Measurement uncertainty, added after the literature review

Two of the reviewed papers measure MediaPipe's error on the exact quantities
these thresholds are compared against, and the figures are large enough to
change what some of these thresholds can honestly claim:

| Measurement | Published error | Source |
| --- | --- | --- |
| Knee angle, near limb, single lateral camera | RMSE 10.7° | Dill et al. (2024) |
| Knee angle, far (occluded) limb | RMSE 25.1° | Dill et al. (2024) |
| Knee angle during squats, optimal camera angle | RMSE 9.14° (14.48° poor angle) | Dill et al. (2023) |
| Fixed body widths, as a fraction of true | RMSE 5–34% | Dill et al. (2023) |

`exercises/common/uncertainty.py` now carries these through to the verdict.
See [measurement_uncertainty.md](measurement_uncertainty.md) for the full
account. Two consequences belong in this log:

**Two thresholds sat below the noise floor of their own measurement.**

| Threshold | Value | Its measurement's band | |
| --- | --- | --- | --- |
| `KNEE_SYMMETRY_WARN` (removed) | 12° | ±15.1° | below |
| `HEEL_LIFT_THRESHOLD` | 0.06 | ±0.15 | below |
| `DEPTH_KNEE_ANGLE_WARN` − `PASS` | 15° band | ±11.6° | comparable |

The symmetry bar was the clearer problem: a left/right difference is the
difference of two independent estimates, so it carries √2 times the error of
either — a bilateral comparison is *less* precise than the single measurement
it is built from, which is the opposite of the intuition.

**The squat's evenness rule was removed rather than re-tuned.** Raising the
bar above 15.1° would have left a check that only fires on differences already
obvious without a system, so the rule was deleted and the measurement kept as
an exported quantity (`squat_analysis.md` §10.1). The press keeps its own
symmetry rule: it is recorded front-on by design, with both arms visible.

**The heel threshold has not been changed here.** Section 8 below says why; it
is raised above its noise floor in the literature preset instead, so the two
can be compared over the same recordings.

**A systematic term was added to the depth band.** The smoothing filter has
its own measured bias on the minimum knee angle (see
[filter_selection.md](filter_selection.md)); with the default EMA the depth
band is √(10.7² + 4.53²) = 11.6° rather than 10.7°.

---

## 8. The literature preset, and why it is a preset

`exercises/squat/literature_config.py` holds a second squat configuration
whose technique thresholds are derived from Kotiuk et al. (2022) — via Rao et
al. (2025) — Dill et al. (2024) and Simoes et al. (2024).

| Threshold | Default | Literature preset | Derived from |
| --- | ---: | ---: | --- |
| `DEPTH_KNEE_ANGLE_PASS` | 100° | 81° | Kotiuk et al.: knee flexion 113 ± 7° = interior 67 ± 7°; the preset takes mean + 2 SD, the most permissive traceable reading |
| `DEPTH_KNEE_ANGLE_WARN` | 115° | 101° | Dill et al.: correct repetitions sat ~20° from faulty ones, so the band is one empirical separation wide |
| `TORSO_LEAN_FAIL` | 60° | 55° | Dill et al. E2 (excessive forward bending) as a fault condition; no trunk angle published, so only the failure bar moves |
| `HEEL_LIFT_THRESHOLD` | 0.06 | 0.16 | Raised above the ±0.15 noise floor of a normalised length |
| `FULL_EXTENSION_TOLERANCE` | 12° | 20° | Simoes et al.: 20° beginner allowance |

Engineering thresholds are untouched — no paper has anything to say about EMA
weights or hysteresis levels, and changing one under a citation would borrow
authority the source never gave. `tests/test_literature_preset.py` enforces
that, and enforces that every changed value has a recorded reason naming its
source.

**It is a preset and not a replacement** for the reason §9 gives: swapping the
defaults for these values would replace one unvalidated set with another and
destroy this audit trail in the process. Run both over the same manifest and
report the difference:

```bash
python scripts/evaluate_videos.py manifest.csv --preset default
python scripts/evaluate_videos.py manifest.csv --preset literature
python scripts/evaluate_videos.py manifest.csv --strict-uncertainty
```

The harness prints the preset and its full provenance before any result, so a
run can never be reported without the numbers that produced it.

One recorded illustration of how far the policies diverge, from three
synthetic 105° repetitions:

| Policy | Depth verdict | Score |
| --- | --- | --- |
| Default thresholds, annotate only | warning | 90 |
| Default thresholds, strict uncertainty | pass | 100 |
| Literature preset, annotate only | fail | 83 |

---

## 9. What has *not* been done, and must not be claimed


* **No threshold has been validated against labelled real recordings.** Every
  value above was chosen from the exercise definition and the reference
  demonstration, then confirmed against synthetic material with a known ground
  truth. A synthetic test demonstrates that the implementation behaves as
  specified; it says nothing about whether the specification suits real
  athletes.
* The real-video runs in `evaluation/reference_clip_runs.md` verify that the
  **pipeline** works on genuine MediaPipe output and refuses unusable footage.
  They are verification, not evaluation: the reference clips are animated
  instructional montages, not labelled athlete recordings, and no threshold was
  moved on the basis of them.
* **No accuracy figure exists**, and none should be quoted until the labelled
  recordings in `evaluation/README.md` have been produced and run through
  `scripts/evaluate_videos.py`.
* **Thresholds must not be tuned until every test video passes.** With a
  handful of recordings that is overfitting, not calibration, and it would make
  the resulting numbers meaningless. If a value does move, add a row to this
  document with the reason.
* **The literature preset is not validated either.** Its values are traceable
  to published measurements, which is better than not being traceable to
  anything, but no labelled recording has been run under them. A citation is
  not a validation.
* **The uncertainty bands are RMSE figures used as a scale, not a probability
  distribution.** Saying "inside the measurement error" is not saying "95%
  confident it is fine", and the code does not claim otherwise.
* **The filter comparison ran on synthetic traces.** The true minimum has to
  be known exactly for a bias to be measurable at all, which no real recording
  provides. It establishes that the ranking depends on the shape of the
  repetition — enough to justify making the filter a setting — and not which
  filter is best for real athletes.
