# Literature basis

What each reviewed paper contributed to FormFix, what was taken from it, and
where FormFix does something the paper does not. The provenance table in
`FormFix_Research_Provenance.docx` is the version written for the report; this
one is written for the code.

## The papers

| # | Reference | What it is |
| --- | --- | --- |
| 1 | Kotte, Kravčík & Duong-Trung (2023), *MILeS'23 / ECTEL 2023* | YOLOv7-pose gym-posture feedback; joint-angle ranges as expert-configurable hyper-parameters |
| 2 | Mercadal-Baudart, Liu, Farrell, Boyne, González Escribano, Smolic & Simms (2024), *Heliyon* 10:e27596 | Single-camera markerless 3D pose for exercise quantification; defines the metrics physiotherapists actually use and evaluates them against VICON |
| 4 | Chae, Kim, Park, O'Sullivan, Seo & Park (2023), *Interact J Med Res* 12:e37604 | Deep-learning squat classifier (CNN+LSTM, 85% test accuracy) and a two-week randomised controlled trial of the resulting app |
| 5 | Simoes, Reis, Araujo & Maia Jr. (2024), *Procedia Comput Sci* 251:446–453 | MediaPipe + Naïve Bayes physiotherapy posture assessment with a user-configurable angular tolerance |
| 6 | Rao, Asha & Raghavendra Rao (2025), *IEEE Access* 13:39557–39571 | Stereo-camera squat classification (Bi-CGRU, 96.1%) and regression-based posture correction; quotes published squat joint-angle ranges |
| 7 | Dill, Rösch, Rohr, Güney, De Witte, Schwartz & Hoog Antink (2023), *Curr Dir Biomed Eng* 9(1):563–566 | Quantitative accuracy evaluation of MediaPipe Pose against marker-based motion capture |
| 8 | Dill, Ahmadi, Grimmer, Haufe, Rohr, Zhao, Sharbafi & Hoog Antink (2024), *Sensors* 24(23):7772 | Stereo 3D reconstruction from MediaPipe, evaluated on 810 squat repetitions against Qualisys motion capture |
| 9 | Yadav, Agarwal, Kumar, Tiwari, Pandey & Akbar (2022), *Knowledge-Based Systems* 250:109097 | YogNet: two-stream yoga recognition with an angle-threshold posture-correction table |

Numbering follows the supplied files. There is no paper 3 — eight files were
provided, numbered 1, 2, 4–9.

## What was taken, and what was done with it

### An acceptability criterion → the bar FormFix is read against

*Mercadal-Baudart et al. (2024)*

Evaluating a single-camera 3D pose model against VICON, they set two bars for
whether a metric is usable: below **12°** is "good", because that is better
than a physiotherapist's by-eye judgement in a low-speed functional movement
(Abbott et al., their ref. 1), and below **6°** is "very good", being less
than half of it.

**Taken:** the criterion itself, as `ACCEPTABILITY_THRESHOLDS` in
`exercises/common/uncertainty.py`. It is what turns FormFix's bands from bare
numbers into something a reader can judge. Every angular band is now labelled
against it in the debug view and the export:

| FormFix measurement | Band | Verdict |
| --- | --- | --- |
| Trunk inclination | ±6.5° | good |
| Knee angle, near limb (depth) | ±10.7° (±11.6° with filter bias) | good |
| Bilateral knee difference (left/right) | ±15.1° | **worse than by-eye assessment** |
| Knee angle, far limb | ±25.1° | **worse than by-eye assessment** |

That table is the single most useful thing the literature review produced. It
says plainly which FormFix measurements are worth acting on and which are not,
using a bar the field set rather than one chosen to flatter the system.

**Also taken:** the definition of trunk angle to vertical as **0° in quiet
standing**. FormFix measures trunk lean against the athlete's own standing
baseline, which was originally an engineering decision about camera tilt; it
turns out to be the convention this literature uses, and now has a citation.

**Explicitly not taken:** their *accuracy* figures. They report RMSEs around
1° for trunk and shin angle, which is far better than anything FormFix can
claim — but those come from a Strided Transformer trained on the very
exercises it was tested on, with virtual cameras, and the paper itself warns
that "generalization needs to be considered in future". Quoting a 1° error
next to a MediaPipe pipeline would be borrowing a number from a different
system. The MediaPipe figures used in FormFix all come from Dill et al.

**Contrast worth stating:** their metric set includes knee varus/valgus, which
FormFix explicitly refuses to report. They can measure it because they lift to
3D from a model trained on marker data; FormFix has one 2D camera, and a
valgus collapse is a motion out of the image plane. The difference between
what the two systems claim is a direct consequence of what each can see, and
that is worth a sentence in the methodology.

### Measurement error → a decision input

*Dill et al. (2023, 2024)*

Both papers measure MediaPipe's error against motion capture. Paper 5 reports
squat knee-angle RMSE of 9.14° at a good camera angle and 14.48° at a poor
one, plus 5–34% instability in MediaPipe's estimate of fixed body widths.
Paper 6 reports 10.7° for the near limb and 25.1° for the occluded far limb in
the same lateral recording, and quotes Hancock et al.'s clinical goniometry
benchmarks (6° / 10° / 14°).

**Taken:** the numbers, as the uncertainty bands in
`exercises/common/uncertainty.py`.

**Extended:** both papers report accuracy in one section and neither feeds it
back into a decision rule anywhere. FormFix carries the published error of
each measurement through to the point where the verdict is formed, and marks
any finding whose margin falls inside that error as indicative rather than
established. See [measurement uncertainty](measurement_uncertainty.md).

**Found as a consequence:** FormFix's own left/right evenness threshold (12°)
and heel-lift threshold (0.06) both sat *below* the noise floor of their own
measurements. Neither paper could have found this; it falls out of applying
their numbers to this system's thresholds. The squat's evenness rule was
removed on the strength of it.

### Filtering → a measured trade-off

*Dill et al. (2024)*

They grid-searched a moving-average, a Butterworth low-pass and a
Savitzky–Golay filter, reported that the moving average shows "a visible delay
in the filtered signal" at the top and bottom of a squat, and selected a
4th-order Butterworth at 2 Hz.

**Taken:** the diagnosis (causal filters lag turning points, and turning
points are where FormFix measures) and the three candidate filters.

**Extended:** their selection criterion was overall landmark RMSE; FormFix's
is the bias in the minimum knee angle at the bottom of each repetition.
Re-running the comparison on that criterion
(`scripts/compare_filters.py`) shows the ranking **reverses** with the shape of
the turnaround — the 2 Hz Butterworth is nearly unbiased on a smooth
repetition and the worst of the four on a sharp one. FormFix therefore makes
the filter a configuration value, keeps its existing default, and adds each
filter's measured bias to the depth uncertainty band. See
[filter selection](filter_selection.md).

**Implemented rather than imported:** the Butterworth cascade, zero-phase
forward–backward filtering and the Savitzky–Golay kernel are derived in
`analysis/filters.py` in NumPy, because adding SciPy would put the pinned
macOS NumPy-1 install at risk.

### Squat joint-angle ranges → a comparison preset

*Rao et al. (2025), quoting Kotiuk et al. (2022)*

Published horizontal-squat angles: knee flexion 113 ± 7°, hip flexion
128 ± 9°, ankle dorsiflexion 23 ± 6°.

**Taken:** the values, into `exercises/squat/literature_config.py`.

**Not taken:** they were not made the defaults. Two reasons, both recorded in
that file. The angle convention differs (Kotiuk et al. report flexion; FormFix
measures the interior hip–knee–ankle angle, so 113° of flexion is 67° of
interior angle), and the source measured a supervised parallel squat while
FormFix analyses unsupervised phone recordings. Adopting 67° as a pass bar
would fail almost every real recording. The preset takes the most permissive
traceable reading instead and exists to be *compared* against the defaults —
`scripts/evaluate_videos.py --preset literature` — rather than to replace them.

### Correct-squat criteria → confirmation of the rule set

*Dill et al. (2024)*

Their protocol defines a correct squat as: shoulder-width stance with parallel
or slightly V-shaped feet; heels remaining on the floor throughout; the spine
remaining straight throughout; both legs loaded symmetrically. Their two fault
conditions are excessive forward bending and a lateral weight shift.

**Taken:** confirmation. Three of FormFix's squat rules — heel stability,
torso lean, depth — were arrived at independently and match a protocol
designed by a biomechanics laboratory. Their fourth criterion, symmetric
loading, FormFix also measured, but the measurement could not support a
verdict from one camera and is no longer graded. That is worth stating
in the report as convergent validity, and it is the honest description of what
happened: nothing was copied, and the agreement is the finding.

**Taken as a number:** their correct repetitions reached a mean peak knee
angle about 20° away from the faulty variants. That 20° empirical separation
is what the literature preset's warning band is built from.

### Per-user tolerance → the width of the bands

*Simoes et al. (2024)*

They let the user set the acceptable angular deviation — 20° for a beginner,
under 10° for an experienced user — on the explicit ground that flexibility
differs between people.

**Taken:** the reasoning, as the justification for FormFix's tolerance bands
being wide rather than tight, and the 20° beginner allowance as the literature
preset's `FULL_EXTENSION_TOLERANCE`.

**Not taken:** the user-facing strictness control itself. It is a real gap and
it is named as one in [limitations](limitations.md).

### Threshold-plus-instruction tables → already the architecture

*Yadav et al. (2022); Kotte et al. (2023)*

YogNet pairs each joint-angle threshold with a fixed correction string
("Straighten your left elbow"). Kotte et al. treat ideal joint-angle ranges as
hyper-parameters a fitness expert configures.

**Taken:** confirmation that FormFix's declarative `RuleSpec` — threshold,
phase, supported views, feedback template, all in one configuration object —
is the established shape for this kind of system rather than an idiosyncratic
one. FormFix's version carries more per rule (movement phase, camera-view
support, persistence requirements, provenance string) because it has to say
what it *cannot* assess, which neither paper's table does.

**Taken as a technique:** Yadav et al. average the keypoints that ought to
coincide in an ideal side view but do not in practice
(`s_mid = (p1+p2+p3)/3`). FormFix uses mid-hip and mid-shoulder points for
trunk inclination for the same reason.

### Evaluation design → what FormFix does not claim

*Chae et al. (2023)*

A two-week randomised controlled trial: 20 participants, experimental group
using the app, control group following videos. Squat score improved
significantly in the experimental group (p = .001) and not in the control
(p = .13); knee range of motion improved 12.8% and 15.9% against 0.9% and
3.6%. They also report a baseline comparison against Google AutoML because no
suitable prior baseline existed.

**Taken:** the standard. This is what "does the system help anyone?" looks
like when it is answered properly, and it is the shape a future evaluation of
FormFix would take.

**Not attempted:** FormFix has run no user study and makes no efficacy claim.
Naming Chae et al.'s design as the standard FormFix has *not* met is more
useful in a report than quietly not mentioning it. Their preprocessing —
normalising to a standing reference frame, Gaussian smoothing per keypoint
sequence — is convergent with what FormFix already does (per-video standing
baseline, EMA smoothing), arrived at independently.

## What none of them do

Two things FormFix does that no reviewed paper does, which is where the
original contribution sits:

1. **Feeding published measurement error back into the decision rule.**
   Several papers report their own accuracy; several apply fixed thresholds;
   none connects the two. A finding whose margin is smaller than the
   instrument's own error is reported by every one of these systems as though
   it were established.

2. **Refusing to assess rather than guessing.** Every paper here produces a
   verdict for whatever it is given. FormFix declares, per rule, which camera
   views can support it, and reports "not assessed" with a reason otherwise —
   which is why it never claims to detect knee valgus or spinal rounding from
   a single 2D camera.
