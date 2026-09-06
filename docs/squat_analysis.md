# The FormFix squat analyser — technical documentation

**Scope.** How FormFix turns an uploaded squat video into explainable
corrective feedback: the pipeline, the landmarks it reads, how measurements
are normalised and smoothed, how repetitions and phases are segmented, which
rules are evaluated and when, where every threshold comes from, how camera
view and reliability are handled, and what the system deliberately does not
claim.

**Positioning.** MediaPipe is the pose-estimation *component*, not the
contribution. The contribution is the layer built on top of it: exercise-
specific landmark selection, quality validation, normalisation, temporal
smoothing, repetition and phase segmentation, multi-feature biomechanical
measurement, transparent rule evaluation, evidence aggregation and
explainable feedback. Each of those is a separate, separately testable
module, and the separation is deliberate — a threshold can be re-tuned, or a
rule defended, without touching how the numbers were obtained.

---

## 1. Pipeline

```
Uploaded squat video
        ↓
OpenCV video decoding                    analysis/video_processor.py
        ↓
File-level validation                    analysis/validation.py
        ↓
MediaPipe Pose Landmarker (VIDEO mode)   analysis/pose_detector.py
   up to 4 people; the athlete is the one nearest the camera, then
   followed frame to frame so a bystander cannot take over mid-rep
        ↓
Raw body landmarks (33 per frame)
        ↓
Recording-quality validation             analysis/validation.py
   pose coverage · key landmarks · usable frames · framing ·
   tracking continuity · camera orientation + stability
        ↓
Squat-specific landmark selection        exercises/squat/landmarks.py
        ↓
Short-gap interpolation + EMA smoothing  analysis/smoothing.py
        ↓
Analysis-side selection                  analysis/validation.py
        ↓
Per-frame biomechanical measurement      exercises/squat/metrics.py
   knee · hip · trunk · shin · depth · heel · symmetry · stance
   normalised by body scale               analysis/normalisation.py
   physiological plausibility gating      exercises/squat/config.py
        ↓
Repetition detection (state machine)     exercises/squat/phases.py
        ↓
Exercise plausibility gate               exercises/common/plausibility.py
   does the movement match the exercise the user selected?
        ↓
Phase segmentation per repetition        exercises/squat/metrics.py
   standing → descent → bottom → ascent → finish
        ↓
Per-repetition measurement aggregation   exercises/squat/metrics.py
        ↓
View-aware, phase-scoped rule evaluation exercises/squat/rules.py
   + persistent-violation filtering        exercises/common/persistence.py
        ↓
Metric-level reliability                 exercises/common/confidence.py
        ↓
Per-repetition findings + overall verdict exercises/common/feedback.py
        ↓
Explainable corrective feedback          exercises/squat/feedback.py
        ↓
Annotated evidence video                 analysis/annotation.py
   trimmed to the detected set            analysis/trimming.py
        ↓
Reference exercise video                 components/analysis_results.py
```

The orchestrator (`exercises/squat/analyser.py`) sequences these stages and
owns none of their logic.

---

## 2. Landmark selection

MediaPipe returns 33 landmarks. They are not equally useful: the face mesh,
the fingers and most of the upper-limb chain contribute nothing to a squat,
and counting them as evidence would flatter every reliability figure (a
clearly visible nose cannot make an occluded ankle trustworthy).

`exercises/squat/landmarks.py` is the only place MediaPipe indices appear in
the exercise package.

| Name | Index | Name | Index |
| --- | --- | --- | --- |
| `left_shoulder` | 11 | `right_shoulder` | 12 |
| `left_hip` | 23 | `right_hip` | 24 |
| `left_knee` | 25 | `right_knee` | 26 |
| `left_ankle` | 27 | `right_ankle` | 28 |
| `left_heel` | 29 | `right_heel` | 30 |
| `left_foot_index` | 31 | `right_foot_index` | 32 |

`SideChain` groups one side into `core` (shoulder, hip, knee, ankle — needed
by every sagittal measurement) and `foot` (heel, toe — needed only by the
heel measurement). `METRIC_REQUIREMENTS` maps each measurement to the
landmarks it genuinely depends on, which is what makes **per-metric**
reliability possible: depth can be reported as reliable in the same run where
the heel measurement cannot be assessed at all.

**Analysis side.** In a side view one side of the body is naturally occluded,
so the two sides are not required to be equally good. The side with the
higher usable-frame ratio is chosen (mean visibility breaks ties); both sides
are still measured, because the export and the plausibility gating need
them.

---

## 3. Landmark reliability and recording validation

MediaPipe's output is not treated as ground truth. Four independent checks
run before any technique judgement, each configurable
(`exercises/squat/config.py`):

| Stage | Question | Config |
| --- | --- | --- |
| A | Was a person detected at all, often enough? | `MIN_POSE_FRAME_RATIO` |
| B | Are shoulder/hip/knee/ankle usable per frame? | `MIN_KEY_LANDMARK_VISIBILITY` |
| C | Is enough of the clip measurable? | `MIN_USABLE_FRAME_RATIO`, `MIN_VALID_FRAME_RATIO` |
| D | Does the body stay inside the frame? | `FRAMING_MARGIN`, `FRAMING_WARN_TOLERANCE`, `FRAMING_FAIL_TOLERANCE` |
| E | What camera view is this, and is it stable? | `SIDE_VIEW_GOOD_RATIO`, `SIDE_VIEW_FRONTAL_RATIO`, `VIEW_STABILITY_SPREAD` |

Missing-landmark gaps of at most `MAX_SHORT_GAP_FRAMES` are linearly
interpolated and marked as such; longer gaps stay missing. Interpolated cells
are counted and exported — interpolation is only honest when it is visible.

### Which person is the athlete

The detector is asked for up to four people per frame rather than one, so that
the choice of who to analyse is made explicitly instead of by accident. On the
first frame the athlete is the largest body in shot, measured as the height of
its landmark bounding box, which with a phone propped up to film a set is the
person nearest the lens.

Size alone cannot keep the right person, though, and this was the part that
had to be corrected. A bounding box shrinks as its owner squats, so at the
bottom of a repetition someone standing upright behind the athlete can briefly
become the largest body in frame. Choosing by size on every frame would hand
the analysis to that bystander mid-set and blend two people's measurements
into one result, without anything on screen indicating it had happened. Size
therefore only starts the track: from then on the athlete is whoever is
closest to where the athlete was on the previous frame, hip-centre to
hip-centre, expressed in units of their own torso length so the test does not
depend on how far away they are standing. A track with no candidate close
enough is held for about a second before being abandoned, which is long enough
for somebody to walk between the athlete and the camera.

This also changed what makes a recording unusable. Judging a clip by how many
people appeared in it rejected recordings filmed in a real gym where the
athlete had been tracked correctly from the first frame to the last, which is
the normal case rather than the exceptional one. Other people in shot now
produce a note on the result; a run is stopped only when the track itself
repeatedly comes apart, measured as re-seeds per second of tracked video,
because at that point the numbers no longer describe one person.

### Graceful degradation

Validation produces one of three outcomes, never a bare boolean:

* **Good** — analyse normally.
* **Limited** — analyse, and name what the recording cost. Example: *"Squat
  depth was analysed successfully, but frontal knee alignment could not be
  assessed from this camera view."*
* **Unusable** — stop, with a typed `RejectionCode` and plain-language help.
  Reserved for cases where meaningful analysis is genuinely impossible: no
  person detected, key joints missing for most of the clip, an unreadable or
  extremely short file, no complete repetition, a track that repeatedly loses
  which body it is following, or a movement that contradicts the exercise the
  user selected (below).

A recording is never rejected because one optional measurement is
unavailable. A camera angle on its own never rejects anything.

### Physiological plausibility gating

MediaPipe reports a landmark position — and a high visibility score — even
when it is extrapolating a limb hidden behind the body. The resulting angle
is geometrically valid but anatomically impossible, and a rule reading it
would state a confident finding about something that never happened. Values
outside a **wide** anatomical band are therefore recorded as *not measured*
(`NaN`), never as a fault:

| Gate | Value | Meaning |
| --- | --- | --- |
| `PLAUSIBLE_KNEE_ANGLE_MIN` / `MAX` | 25° / 190° | a knee cannot fold past this |
| `HEEL_LIFT_MAX_PLAUSIBLE` | 0.5 | half a lower-leg length of heel rise is a tracking failure |
| `KNEE_SYMMETRY_MAX_PLAUSIBLE` | 45° | a bigger left/right gap is one leg being extrapolated |

These bands catch tracking failures; they do not police technique.

---

## 4. Normalisation

Angles are scale-invariant and are used directly — they are never converted
into some abstract normalised unit. Only *distances* are normalised, and
always against a body dimension measured in the same frame
(`analysis/normalisation.py`):

| Reference | Chosen when | Why |
| --- | --- | --- |
| torso length (mid-hip → mid-shoulder) | usual case | longest stable segment, least affected by one noisy landmark |
| hip width | torso unavailable | stable in a frontal view |
| lower-leg length | neither available | already needed by the depth and heel measurements |

`hip_centred()` expresses a point relative to the mid-hip in body-scale
units, removing where the athlete stands in the frame while keeping
direction. Every normalised value is dimensionless: a stance width of 1.4
means "1.4 torso lengths", not 1.4 of anything on one particular phone.

Measurements are always taken in **pixel space**, never on the raw normalised
coordinates: MediaPipe normalises x to the frame width and y to the frame
height independently, so an angle computed on normalised values is distorted
on any non-square frame.

---

## 5. Temporal smoothing

Exponential moving average, applied to landmark **coordinates** only —
visibility scores are trust estimates, not positions, and smoothing them
would hide tracking failures.

```
smoothed[t] = α · observed[t] + (1 − α) · smoothed[t−1]
```

| Setting | Value | Applies to |
| --- | --- | --- |
| `EMA_ALPHA` | 0.45 | landmark coordinates |
| `ANGLE_EMA_ALPHA` | 0.5 | the derived knee-angle series |

Both are configuration, not constants buried in the algorithm. The EMA
**resets after a gap** instead of dragging a stale pre-gap position forward,
which would draw a landmark teleporting across the frame. Raw, cleaned and
smoothed coordinate arrays are all retained (`xy_raw` / `xy`), so each
stage's effect can be inspected during evaluation.

---

## 6. Measurements

`exercises/squat/metrics.py` is measurement only: no thresholds, no verdicts,
no text. A knee angle alone cannot describe a squat — two athletes can reach
an identical knee angle with very different hip, trunk and shin positions —
so a set of features is measured:

| Measurement | Definition | Convention |
| --- | --- | --- |
| Knee flexion | hip → knee → ankle interior angle, **per leg** | ~175–180° standing, smaller = more flexion |
| Hip flexion | shoulder → hip → knee interior angle, **per leg** | smaller = more flexion |
| Trunk inclination | mid-hip → mid-shoulder against vertical | 0° upright, 90° horizontal; sign discarded so left- and right-facing subjects measure identically |
| Shin inclination | ankle → knee against vertical | 0° vertical shin; larger = knee further forward over the foot |
| Depth relation | (knee_y − hip_y) / lower-leg length | image y grows downward, so **positive = hip above knee**, ~0 = level, negative = below |
| Heel displacement | heel height above the toe of the same foot, over lower-leg length, minus its standing value | positive = heel higher than when standing |
| Knee / hip symmetry | \|left − right\| flexion | measured and exported only; no rule reads it — see §10.1 |
| Stance width | ankle separation / body scale | frontal plane |
| Knee-over-ankle offset | mean \|knee_x − ankle_x\| / body scale | **descriptive only** — see §11 |
| Timing | descent / bottom / ascent durations | seconds, from timestamps |

An unmeasurable quantity is `NaN`, never a substituted `0` — a rule reads a
zero as a real measurement.

### Why the heel measurement is taken inside the foot

An earlier version compared heel height to its standing height at an absolute
image position. That cannot distinguish a heel lift from the athlete drifting
across the frame or the camera being nudged, and on real footage it produced
heel "lifts" of over 250% of a lower-leg length. Measuring the heel against
the **toe of the same foot** makes the quantity translation-invariant while
keeping the same units (a fraction of a lower leg) and the same threshold.
Regression-tested in `tests/test_integration.py::TestHeelMeasurementRobustness`.

---

## 7. Repetition detection

An explainable state machine over the smoothed knee-angle series
(`exercises/squat/phases.py`):

```
STANDING → DESCENDING → BOTTOM → ASCENDING → STANDING
                ↑_________________________|
```

Threshold-crossing counts would register several "repetitions" from one knee
angle fluttering around a single number. Robustness comes from five separate,
configurable mechanisms:

| Mechanism | Config | Purpose |
| --- | --- | --- |
| Four different thresholds (hysteresis) | `STANDING_KNEE_ANGLE` 160°, `REP_START_KNEE_ANGLE` 150°, `BOTTOM_CANDIDATE_ANGLE` 130°, `REP_END_KNEE_ANGLE` 155° | the machine can never oscillate around one number |
| Transition persistence | `PHASE_MIN_FRAMES` 3 | a transition must hold before it commits |
| Minimum range of motion | `MIN_RANGE_OF_MOTION` 35° | a small dip is not a squat |
| Bottom reversal delta | `BOTTOM_REVERSAL_DELTA` 5° | a one-frame wobble cannot end the descent |
| Duration bounds | `MIN_REP_DURATION` 1.0 s, `MAX_REP_DURATION` 12 s | frame-rate independent sanity |
| Tracking-loss tolerance | `MAX_TRACKING_LOSS_FRAMES` 10 | a long gap abandons the rep rather than bridging it silently |

A video that starts mid-squat never produces a repetition whose beginning was
not observed. Movements that start but do not qualify are counted as **partial
movements** and reported with a reason, never silently included.

---

### Is this actually a squat?

Repetition detection answers whether the movement had rep-shaped structure,
not whether it was the exercise the user picked from the dropdown. Nothing
downstream asks that question either: the analyser measures knee angles and
writes specific squat feedback about whatever it is pointed at. That is fine
while the label is right, and stops being fine as soon as the system is open
to people who did not build it — a bench press labelled as a shoulder press,
say, produces a confident critique of an overhead press that was never
performed, with nothing in the wording to suggest it is wrong.

`exercises/common/plausibility.py` runs after repetition detection and asks
only whether the movement is consistent with the exercise selected. It does
not classify: it never tries to work out what the exercise actually was, which
would have meant training a model and reintroducing exactly the kind of
component this project avoids on explainability grounds.

The bias is one-directional on purpose. Rejecting a genuine squat that
happened to be filmed awkwardly is a worse outcome than accepting an unusual
clip, so each check looks for a *contradiction* — movement the chosen exercise
cannot produce — rather than for a good match. For the squat that means the
knees must bend below 155°, well clear of any depth the rules assess, so a
shallow squat is reported as a depth finding rather than as the wrong
exercise. More than 60% of measurable repetitions must contradict the label
before a run is stopped, and never fewer than two, so one badly tracked
repetition cannot reject a real recording. Where the evidence is thin —
too few measurable repetitions, an unreadable signal — the gate abstains.

---

## 8. Phase segmentation

Every accepted repetition carries both frame indices and durations in
seconds:

```python
SquatRep(
    start_frame, descent_start_frame,
    bottom_start_frame, bottom_frame, bottom_end_frame,
    ascent_start_frame, end_frame,
    descent_duration, bottom_duration, ascent_duration, duration,
)
```

The bottom is a **window** (`BOTTOM_WINDOW_FRAMES` = 3 frames either side of
the minimum), and bottom-position measurements are the window's median — not
one noisy frame. Peak values across a repetition use a *sustained* maximum
(the largest value that held for a configured number of consecutive frames),
so no single jittered landmark can become "the worst moment" of a repetition.

Durations come from timestamps, never from frame counts, so 24, 30 and 60 fps
recordings of the same movement are judged identically
(`tests/test_metrics.py::TestFrameRateIndependence`).

---

## 9. Rules

Every rule is **declared** in `exercises/squat/config.py` and **evaluated** in
`exercises/squat/rules.py`. The declaration is the traceability record:

```python
SquatRule(
    name, rule_id, title,
    metric,                       # which measurement it reads
    phase,                        # when the question is meaningful
    acceptable_min, acceptable_max, tolerance,
    minimum_persistence_frames, min_violation_ratio,
    minimum_visibility,
    supported_views,              # where the measurement is meaningful
    feedback_key,                 # which wording it produces
    threshold_source,             # where the numbers came from
)
```

### The implemented rules

| Rule | Metric | Phase | Views | Thresholds | Persistence | Feedback |
| --- | --- | --- | --- | --- | --- | --- |
| **Squat depth** | min knee flexion **and** hip-vs-knee height at the bottom | bottom window | side, diagonal, unknown | pass ≤ 100°; warn ≤ 115°; or hip within 0.12 of knee level | ≥ 1 frame and ≥ 50% of the bottom window | *"…did not reach the configured target depth. The shallowest (rep 2, 00:04.2) reached a minimum knee angle of about 118°."* |
| **Forward torso lean** | trunk inclination | descent → ascent | side, diagonal, unknown | warn ≥ 45°; fail ≥ 60°, or ≥ 55° above the athlete's own standing baseline | ≥ 4 consecutive frames and ≥ 15% of the repetition | *"Your torso leaned further forward than the configured range in 2 of your 4 repetitions…"* |
| **Heel stability** | heel rise over lower-leg length | descent → ascent | all | warn ≥ 0.06 | ≥ 3 consecutive frames and ≥ 10% of the repetition | *"Your heel appears to rise during 2 of your 4 repetitions…"* — warning only, never a failure |
| **Return to standing** | finish knee angle vs the athlete's own standing baseline | finish | side, diagonal, unknown | pass within 12°; warn within 24°; fail beyond | single completion measurement | *"…ended before you fully returned to your starting position."* |
| **Descent control** | descent duration | descent | side, diagonal, unknown | pass ≥ 0.45 s; warn ≥ 0.25 s; fail below | single duration | *"You dropped down quickly… the fastest descent took about 0.2 seconds."* |

#### 10.1 A rule that was removed: left/right evenness

An earlier version graded \|left − right\| knee flexion as a sixth rule
(`knee_symmetry`), warning at 12° and failing at 20°. It was removed.

The uncertainty layer is what settled it. A left/right difference is the
difference of two independent estimates, so it carries √2 × the single-limb
error: **±15.1°** (`docs/measurement_uncertainty.md`). The warning bar sat
*below* the noise floor of its own comparison, which means the rule as
configured could report an asymmetry that was entirely measurement error. The
honest options were to raise the bar above 15.1° — at which point it only
catches differences already obvious to the eye — or to stop grading it. The
second is the one consistent with the rest of the system: FormFix reports what
it can measure and says so when it cannot.

The measurement itself is kept. `knee_asymmetry`, `hip_asymmetry`,
`max_knee_asymmetry` and `symmetry_reliable` are still computed per frame and
per repetition, and still appear in the export bundle and the calibration
scripts — they are simply no longer turned into a verdict or shown to the
user.

A consequence worth stating: the squat now has **no frontal-plane rule**. A
front-on recording is still analysed, but only the view-independent heel check
survives it.

Aggregation across a set: **fail** when at least half the evaluable
repetitions failed, **warning** when any failed or warned, **pass**
otherwise, **cannot assess** when no repetition offered reliable data.

### Phase scoping

Rules ask "is this correct *during the phase where it matters*", not "is this
correct everywhere". A trunk angle at the bottom of a squat is supposed to
differ from a standing one, so evaluating the whole video against one number
would manufacture findings.

### Persistent-error detection

`exercises/common/persistence.py` requires two independent conditions before a
rule fires:

* `longest_run ≥ minimum_persistence_frames` — guards against isolated
  spikes, whatever their magnitude;
* `violating_frames / measurable_frames ≥ min_violation_ratio` — guards
  against a violation that flickers on and off across most of a phase.

Frames whose measurement is missing are excluded from both numerator and
denominator: *"we could not see it"* is not *"it was fine"*. The counts reach
the debug output and the export bundle, so every finding can be audited
(`debug["rule_persistence"]`).

---

## 10. Camera view

`CameraOrientation` is estimated, not assumed:

```
frontality = max(shoulder separation, hip separation) / torso length
```

both in **pixels**. MediaPipe normalises x to the frame width and y to the
frame height independently, so comparing a normalised horizontal separation
to a normalised vertical one inflated the ratio by height/width — about 1.8×
on a portrait phone clip — and pushed ordinary side-on recordings into the
"frontal" band, silently switching off every sagittal measurement. The torso
length is the full segment length, so an athlete leaning forward at the
bottom of a squat does not appear to grow wider.

| Band | Orientation | Side-view confidence |
| --- | --- | --- |
| ≤ `SIDE_VIEW_GOOD_RATIO` (0.45) | `SIDE` | 1.0 |
| between the bands | `DIAGONAL_SIDE` | linear ramp 1 → 0 |
| ≥ `SIDE_VIEW_FRONTAL_RATIO` (1.00) | `FRONTAL` | 0.0 |
| not measurable | `UNKNOWN` | 0.0 |

The view is a **confidence, not a gate**: it decides *what can be assessed*,
never *whether the video is rejected*.

* **Side view** supports the sagittal measurements: knee and hip flexion,
  depth, trunk and shin inclination, lockout, movement timing.
* **Front / near-front view** supports left/right comparisons and stance
  relationships. These are measured and exported, but since the removal of
  the evenness rule (§10.1) no squat rule reads them.
* **Diagonal** supports both families at reduced confidence.
* **Unknown** is analysed conservatively — never at full confidence, because
  the system genuinely does not know what it is looking at.

A frontal-plane finding is never computed from a side-view recording, and
sagittal angles are never reported from a front-on one. Each unavailable rule
states its own reason (`VIEW_LIMITATION_TEXT`), which the interface shows in
a **Not assessed** block.

**Stability.** The orientation is estimated once per video, so a clip that
cuts between shots, or a camera moved mid-set, cannot be described by one
label. The interquartile spread of the frontality ratio is checked against
`VIEW_STABILITY_SPREAD`, and an unstable recording is flagged in plain
language rather than averaged away.

---

## 11. What is deliberately not claimed

A single 2D camera cannot support these, so FormFix does not implement them:

* lumbar or spinal rounding;
* injury risk, joint loading, or any medical or diagnostic statement;
* knee valgus ("knees caving in") — a 3D motion one camera cannot resolve.
  The left/right knee difference *is* measured, but it describes evenness of
  knee **flexion**, not valgus, and it is no longer graded at all (§10.1);
* **"knees must never pass toes"** as a blanket error. Anterior knee travel
  *is* measured (`knee_over_ankle_offset`) and exported for research, but it
  is never used as a correctness condition: squat mechanics legitimately
  differ with body proportions, stance, squat style, mobility and camera
  perspective.

Feedback language avoids injury claims. The system compares movement against
*configured technique targets*; it does not diagnose anything.

---

## 12. Reliability

Reliability is **transparent, not probabilistic** — nothing here is learned,
so a number like "0.87 confident" would imply a calibration the system does
not have (`exercises/common/confidence.py`, with the squat's bands and
landmark requirements supplied by `exercises/squat/confidence.py`).

One correction worth recording, from when the squat still had a
frontal-plane rule: `side_view_confidence` measures how *side-on* the camera
is, so using it unchanged meant a diagonal recording that was nearly side-on
— the worst case for comparing two legs, because one lines up behind the other
— scored the *highest* confidence. Frontal-plane metrics score the complement
instead; `exercises/squat/landmarks.py` still declares which they are, and the
shoulder press relies on it (see `docs/threshold_tuning.md` §5).

Four inputs, all of which can be pointed at in the data:

| Input | Source |
| --- | --- |
| Landmark visibility | MediaPipe's visibility for *exactly* the landmarks the metric requires, over the frames it reads |
| Measurable frames | the share of those frames that produced a value |
| Camera-view support | 1.0 for a supporting view, the estimated side-view confidence for a diagonal one, 0.0 for a view that cannot support it |
| Evidence breadth | how many repetitions produced a verdict |

Banding is a **conjunction of floors**, not a weighted average: excellent
visibility cannot compensate for a camera that cannot see the measurement,
and an average would hide exactly that.

Two ceilings are then applied, because MediaPipe's visibility score expresses
neither:

* **Subject size** — a person filmed from across the gym occupies few pixels,
  so every landmark carries more absolute error while the visibility score
  (which reports *occlusion*) stays high. Below
  `RELIABILITY_MIN_SUBJECT_PIXELS` nothing is reported as high; below
  `RELIABILITY_POOR_SUBJECT_PIXELS` nothing above low.
* **Recording quality** — if validation already told the athlete the
  recording was limited, no measurement from it comes back labelled "high".

| Level | Meaning |
| --- | --- |
| **High** | clear landmarks, most frames measurable, a supporting view, several repetitions |
| **Medium** | measurable on thinner evidence |
| **Low** | measurable, but the finding is a hint rather than a fact |
| **Cannot assess** | no trustworthy evidence — the rule reports nothing |

Reliability is computed **per measurement**. The session figure is the median
of the metrics that produced a verdict, so one weak check does not condemn a
good recording and one strong check cannot rescue a poor one.

---

## 13. Feedback

`exercises/squat/feedback.py` holds the squat's wording and its evidence
formats; the aggregation itself — the transparent score, per-repetition
verdicts, findings, overview and "not assessed" block — is shared with the
other exercises in `exercises/common/feedback.py`. Between them they are the
only layer allowed to write sentences.
Each correction has four parts:

1. **Finding** — what happened, in plain language.
2. **Evidence** — the measured values, with timestamps
   (*"Rep 2 — minimum knee angle 118° — 00:04.2"*).
3. **Correction** — what to do next time.
4. **Reference** — the rule that produced it, so the chain
   *landmarks → measurement → phase → rule → evidence → feedback* is
   complete; plus the existing correct-form reference clip.

Wording comes from fixed `FeedbackTemplate` entries keyed by each rule's
`feedback_key`; only the *numbers* vary between runs, so every sentence a
beginner can see has been reviewed once. Internal vocabulary
(`metric threshold 2 failed`) never reaches the interface — technical values
stay in the evidence lines and the debug panel.

### Per-repetition and overall results

* **Per repetition** — status, a one-line headline, the checks it passed, the
  issues found on it, and what could not be assessed on it.
* **Overall findings** — *"Depth: 3 of 4 repetitions acceptable"*, one line
  per evaluated check, which separates a habit from a one-off.
* **Not assessed** — its own block, so an unavailable measurement is never
  mistaken for a failed one.
* **Score** — a transparent formula, printed in the interface, not a learned
  output:

  ```
  score = 100 × (passed checks + 0.5 × warnings) / evaluable checks
  ```

  where a check is one rule on one repetition. Checks that could not be
  evaluated are excluded from **both** sides of the fraction. When nothing
  was evaluable, no score is claimed at all.

---

## 14. Annotated evidence

`analysis/annotation.py` draws, per frame: a simplified skeleton in the
FormFix palette with the analysed side emphasised, a HUD (repetition counter,
movement phase, live knee and trunk angles), a bottom-position marker, and
**timed issue banners** shown only around the frames the finding belongs to —
a bottom-position issue never smears across a whole repetition.

When a finding is on screen, the joints that rule actually read are ringed
and linked (hips and knees for depth, shoulders and hips for trunk, hip–knee–
ankle for lockout, heel/ankle/toe for heel contact).
Colouring the whole skeleton "wrong" would explain nothing and would imply
the system judged parts of the body it never measured.

Two decisions about the overlay are worth recording, because both were made
after looking at real output rather than in advance.

The first is occlusion. In a side view the camera-far arm and leg sit behind
the body, and MediaPipe still returns coordinates for them — usually at high
confidence, because the model is confident about where it *inferred* the joint
to be rather than where it saw one. Gating on that confidence let a guessed
half-skeleton draw at full strength on top of the half that was measured, and
it jittered frame to frame because nothing anchored it. Occlusion is therefore
decided geometrically: filmed side-on, a hidden limb projects onto the limb in
front of it, so a far landmark sitting within about an eighth of a torso
length of its near twin horizontally is treated as hidden and is not drawn.
That test reads the track the measurements already came from, not the
detector's opinion of itself.

The second is the render window. Everything before the first detected
repetition and after the last is setup — walking up, getting into position,
racking the weight — and none of it can be assessed. The annotated clip is
rendered only from a second before the first repetition to a second after the
last (`analysis/trimming.py`). Frame indices stay those of the original
recording, so every timestamp in the feedback and the exports still refers to
the video as uploaded, and anything uncertain — no repetitions, an implausible
window, too little to gain — renders the whole video instead.

---

## 15. Configuration and threshold provenance

Everything tunable lives in `exercises/squat/config.py`, split into two
deliberately labelled kinds:

* **Engineering** — detector confidences, visibility floors, gap handling,
  smoothing, hysteresis, persistence, plausibility bands, reliability
  banding. These make no claim about how a squat should be performed.
* **Technique (provisional)** — `DEPTH_*`, `TORSO_LEAN_*`, `HEEL_LIFT_*`,
  `DESCENT_*`, `FULL_EXTENSION_TOLERANCE`.

> The technique values are **provisional prototype calibration values**, not
> clinically validated constants. They were chosen as conservative starting
> points and are expected to be re-tuned against literature and test
> recordings. Each rule carries a `threshold_source` string, every value
> appears in the debug view and the export bundle, and changing one requires
> no code change anywhere else.

Two design choices keep them defensible:

* thresholds are **ranges with a tolerance band** between warning and
  failure — there is deliberately no exact target such as
  `knee_angle == 90`, because no single number is correct for every body;
* comparisons are made against the athlete's **own standing baseline**
  wherever possible (lockout, trunk lean), so nobody is asked to hyperextend
  and a slightly tilted camera is not read as poor posture.

`SIDE_VIEW_RULES`, `FRONT_VIEW_RULES` and `VIEW_INDEPENDENT_RULES` expose the
default rule sets for documentation and tests.

---

## 16. Failure handling

Every stopping condition is a typed `FailureCode` with beginner-readable text
and retry tips — never a traceback in the interface:

| Code | Cause |
| --- | --- |
| `INVALID_VIDEO` | unreadable, corrupt, too short, too long, too low resolution |
| `NO_POSE` | no person detected |
| `INSUFFICIENT_VISIBILITY` | key joints missing for too much of the clip |
| `BODY_OUT_OF_FRAME` | head/torso or hips outside the frame for most of the clip |
| `MULTIPLE_PEOPLE` | tracking kept losing which person to follow (other people merely being in shot only warns) |
| `NO_COMPLETE_SQUAT` | no repetition completed (partial movements are reported with reasons) |
| `ANALYSIS_ERROR` | anything unexpected — logged in full, shown as one friendly line |

A failed annotated-video render degrades to a warning; the analysis result
survives. Pose detection runs in a separate worker process, because
MediaPipe's native code can hard-crash the interpreter on some platforms.

Upload restrictions are deliberately minimal: if OpenCV can decode the video
and there is enough usable pose data, FormFix analyses it. Quality influences
*what can be assessed*, not *whether the video is accepted*.

---

## 17. Debug and evaluation output

`FORMFIX_DEBUG=1` adds developer metrics to the "Technical details" panel and
writes a bundle to `analysis_results/<run id>/`:

* `summary.json` — validation diagnostics, rule specifications, per-rule
  reliability evidence, per-rep persistence counts, thresholds, baselines,
  rep boundaries with phase durations;
* `frame_metrics.csv` — every per-frame measurement, both legs;
* `reps.csv` — every per-repetition value.

`python scripts/calibrate_squat.py video.mp4 --frames` prints the same
information as a table, per repetition and per rule (including the violating-
frame counts behind each verdict), and can save annotated key frames. None of
this is shown to normal users.

---

## 18. Testing

`pytest` — geometry, normalisation, smoothing, the landmark subset,
persistence, the repetition state machine, phase segmentation and durations,
each rule independently, view gating, reliability banding, feedback
generation, the UI mapping, and end-to-end runs on synthetic pose tracks
(including annotated-video rendering and the export bundle).

The squat suite is also the **regression guard** for the shared layer. When
the lat pulldown and shoulder press were added, persistence, reliability and
feedback aggregation moved into `exercises/common/` and the squat modules
became thin adapters over them. No squat threshold or rule logic changed, and
the whole squat suite passing unchanged is what establishes that.

MediaPipe detection itself is exercised by an optional real-video test, and by
the verification runs recorded in `evaluation/reference_clip_runs.md`:

```bash
FORMFIX_TEST_VIDEO=path/to/side-view-squat.mp4 pytest
```

`ruff check .` for lint. Formatting follows `black -l 104` (matching
`ruff.toml`'s line length).

---

## 19. Known limitations

* **2D, single camera.** No true 3D joint angles, no forces or loads. Depth
  along the camera axis is invisible.
* **Estimation error.** MediaPipe landmarks carry error even in good
  conditions; occlusion, loose clothing, poor lighting and camera movement
  make it worse. The validation layer rejects the worst cases and the
  plausibility gates discard impossible ones, but neither can make a marginal
  recording accurate.
* **Camera view is heuristic.** The orientation is estimated from body
  proportions, not classified perfectly, and it is estimated once per video.
  A clip that cuts between camera angles is flagged, but still gets a single
  label.
* **Scene cuts read as movement.** A hard cut inside a clip produces an
  instantaneous change in the knee-angle trace, which the timing rule can
  read as a very fast descent. A single continuous recording is assumed.
* **Anatomy differs between people.** The per-video standing baseline
  compensates partially; the technique thresholds remain provisional.
* **No frontal-plane rule remains.** Left/right evenness was measured but
  never reliably enough to grade (§10.1), and it was removed. A front-on
  recording is still analysed, but only the heel check survives it, so the
  side-on view is now the only one that produces a full set of findings.
* **Real-footage coverage.** The clips available in this repository are
  front-on/mixed-angle rendered animations. The sagittal path was verified
  end-to-end on a side-on segment of one of them; broader validation against
  real gym footage across body types, clothing and lighting is outstanding.
* **No repetition-level view estimation.** A set filmed while the camera is
  carried around would need per-repetition view estimation, which is future
  work.

> FormFix provides exercise-technique guidance from visible movement and
> is not a replacement for professional coaching or medical advice.
