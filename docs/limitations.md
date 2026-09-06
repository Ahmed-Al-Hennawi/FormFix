# Research limitations

What FormFix can and cannot support, stated plainly. This document is intended
to be quoted from the Limitations section of the dissertation, and to be the
reference the interface's wording is checked against.

---

## 1. The system does not claim

FormFix compares a movement against **configured technique targets** and
explains what it measured. It does not:

* prevent, predict or assess injury risk;
* provide medical, physiotherapeutic or diagnostic assessment of any kind;
* replace a coach or a personal trainer;
* measure joint loading, spinal position, muscle activation or mobility;
* establish that a movement is globally correct — "no issue was detected for
  the checks that were run" is a different and much weaker statement than
  "your form is perfect", and the interface never uses the second;
* achieve laboratory-grade biomechanical accuracy.

Every technique threshold is an **operational prototype value** (see
`threshold_tuning.md`), not a biomechanical constant.

---

## 2. Single-camera and pose-estimation limitations

**One RGB camera, no depth.** Every measurement is a 2D image-plane
relationship. Motion along the camera's optical axis is either invisible or
severely foreshortened. This is why the lat pulldown needs a side view and the
shoulder press needs a front view — and why each exercise reports "not
assessed" rather than a guess when it does not get one.

**MediaPipe world landmarks are not motion capture.** The Pose Landmarker
exposes 3D world coordinates. FormFix uses **none of them**: every angle and
distance in all three exercises is computed from 2D image coordinates
converted to pixels. Single-camera depth estimates are not equivalent to
marker-based 3D motion capture, and treating them as such would be the single
easiest way to produce confident, unsupported findings. Where a 3D quantity
would be needed — knee valgus, sagittal trunk lean in a front-on press — the
measurement is simply not made.

**Landmark error.** MediaPipe reports a landmark position, and a high
visibility score, even when it is extrapolating an occluded limb. The estimate
is then geometrically valid and anatomically impossible. FormFix mitigates this
with wide plausibility bands (a value outside them becomes "not measured",
never a fault), per-frame visibility floors, and persistence filtering, but the
underlying error cannot be removed.

**Published accuracy figures do not transfer.** The reviewed literature reports
MediaPipe accuracy for its own exercises, camera positions, participants and
environments. FormFix's exercises, thresholds, recordings and users are
different, and no accuracy figure from that literature should be attributed to
this system.

---

## 3. Recording conditions

| Condition | Effect |
| --- | --- |
| **Camera angle** | The single largest factor. An oblique view foreshortens joint angles; the wrong plane makes a check unavailable entirely. Estimated per recording and reported as a confidence, never as a fact. |
| **Camera distance** | A subject occupying few pixels carries more absolute landmark error than the visibility score admits. Reliability is explicitly capped by subject size in pixels. |
| **Camera movement** | A handheld or drifting camera can imitate body movement. The trunk measurements are taken as *changes from the athlete's own baseline*, which reduces but does not remove this. |
| **Equipment occlusion** | A pulldown machine's frame, a bench, a dumbbell in front of a wrist. Handled as a visibility problem — the affected check reports "not assessed" — rather than as a wrong number. |
| **Self-occlusion** | In a side view one arm and one leg hide the other, which is why left/right comparisons are unavailable there — and, for the squat, why they are no longer graded at all. |
| **Clothing** | Loose clothing degrades landmark estimation, particularly at the hips and shoulders. Not detectable by the system. |
| **Lighting** | Poor or uneven lighting reduces detection coverage; below the configured floor the recording is rejected with an explanation. |
| **Movement speed** | A very fast repetition gives fewer frames per phase, weakening every persistence measure. Duration bounds reject implausibly fast movements as partial. |
| **Multiple people** | Detected and reported; the most prominent figure is analysed, and a crowded recording is rejected. |

---

## 4. Individual and exercise variation

**Body proportions.** Normalisation by a body dimension measured in the same
frame (trunk length, shoulder width) removes the effect of size and camera
distance on *distances*, and angles are already scale-free. It does not remove
genuine anthropometric variation: limb-length ratios legitimately change what a
given joint angle looks like, and no single threshold is correct for every
body.

**Exercise variation.** Each analyser is written for one specific movement
variant — the one its FormFix reference clip demonstrates. A behind-the-neck or
close-grip pulldown, an Arnold or neutral-grip press, a single-arm version of
either, a low-bar or front squat: these are different movements, and the rules
would misdescribe them. The variant is stated in each exercise's documentation
and in its configuration.

**Sample size.** The thresholds have been validated against synthetic
recordings with a known ground truth, which tests the *implementation*. They
have **not** been validated against a corpus of labelled real recordings, and
no accuracy or reliability rate is claimed until that has been done.

---

## 5. Method limitations

**Rule-based by design, with the trade-off that implies.** FormFix uses
transparent geometric rules rather than a learned classifier. Every finding can
be traced to specific landmarks, a specific measurement, a specific movement
phase, a specific threshold and a specific number of frames — which is the
project's explainability contribution. The trade-off is that the system detects
only what it was explicitly designed to detect, and a fault outside its rule
set is silently absent rather than reported as unknown.

**Thresholds are boundaries.** A movement sitting either side of a threshold
receives different verdicts for a difference the athlete may not perceive. The
warning/failure bands and the persistence requirements soften this, and the
measured value is always shown alongside the verdict so the reader can see how
marginal a call was.

**Correlated checks.** The shoulder press's alignment and range-of-motion
measurements are not statistically independent: a press that stops well short
of overhead necessarily leaves the wrist beside the elbow. The tolerance was
widened to decouple the moderate cases; the residual coupling for severe cases
is documented in `press_analysis.md` and should be read as one habit reported
twice, not as two independent detections.

**Feedback is not coaching.** The corrective messages are fixed, reviewed
templates filled with measured values. They cannot adapt to an individual's
history, goals, equipment or constraints, and they are not a substitute for
watching a person move.

---

## 6. What the design does about all of this

The limitations above are the reason for four decisions that run through the
whole system, and each is testable:

1. **Not-assessed is a first-class outcome.** A measurement the recording
   cannot support is reported as unavailable, with the reason, and is excluded
   from both sides of the score fraction. It is never a failure and never a
   silent pass.
2. **Reliability is per measurement, not per video**, and is derived from
   things that can be pointed at — landmark visibility, measurable frames,
   camera-view support, evidence breadth — rather than from a probability the
   system has no basis to state.
3. **Graceful degradation.** One unavailable measurement costs one check, never
   the whole run: a front-on pulldown still gets a range-of-motion verdict, a
   side-on press still gets one, and hidden hips cost the trunk check alone.
4. **Findings must persist.** A single frame can never produce a correction;
   every finding states how many frames and what share of the phase supported
   it.

---

## Measurement uncertainty (added after the literature review)

MediaPipe's error on the quantities FormFix reads has been measured against
marker-based motion capture and is large relative to some of the thresholds
in use: **±10.7°** on a knee angle from a single lateral camera for the near
limb, **±25.1°** for the occluded far limb, and **5–34%** instability in its
estimate of a fixed body width (Dill et al., 2023, 2024). For context, clinical
goniometry on a *stationary* patient resolves 6–14° depending on the
instrument (Hancock et al., 2018).

FormFix now carries these figures through to every verdict rather than
treating its measurements as exact — see
[measurement_uncertainty.md](measurement_uncertainty.md) — but modelling an
error is not removing it. Three limitations follow directly and belong in the
dissertation:

* **Deviations smaller than roughly 10° in a joint angle cannot be resolved.**
  FormFix reports them, labelled as indicative; it cannot establish them.
* **Thresholds sitting below the noise floor of their own measurement.** The
  uncertainty layer found two: the squat's left/right evenness warning bar
  (12°, against a ±15.1° band for a bilateral comparison) and the heel-lift
  threshold (0.06, against ±0.15 for a normalised length). The evenness rule
  was **removed** as a result — the measurement is still exported, but a
  single camera cannot support a verdict on it. The heel threshold is
  unchanged in the default configuration — there are no labelled recordings
  against which to justify moving it — and raised above its noise floor in
  the literature preset so the two can be compared. A user should read a heel
  finding as a prompt to look, not as a measurement.
* **The far limb is never usable for a joint angle from a side view.** At
  ±25.1° it carries more than twice the error of the near limb, which is why
  FormFix selects the more visible side rather than averaging the two, and
  part of why the squat no longer grades left/right evenness.

## Smoothing bias

Every smoothing filter reports the bottom of a squat as slightly shallower
than it was, because the bottom is a turning point. With the default EMA the
measured worst case is **+4.5°** on a sharply reversed repetition
([filter_selection.md](filter_selection.md)). It is included in the depth
uncertainty band rather than left implicit, but it is a real bias in the
direction that produces false "shallow" findings, and a fast, sharply
reversed squat will attract them more than a slow controlled one.

## Not implemented, and known to be missing

* **No per-user strictness setting.** Simoes et al. (2024) let the user choose
  their angular tolerance — 20° for a beginner, under 10° for an experienced
  user — because flexibility differs between people. FormFix applies one
  threshold set to everyone. Body proportions differ too: Rao et al. (2025)
  note that a short torso with long femurs produces a legitimately different
  squat from the reverse, and FormFix's per-video standing baseline only
  partly compensates.
* **No user study, and therefore no efficacy claim.** Chae et al. (2023) ran a
  two-week randomised controlled trial with a control group and pre/mid/post
  measurements, and found a significant improvement in the app group and none
  in the control. FormFix has run nothing equivalent, so it can say what it
  measures and cannot say that using it improves anyone's technique.
* **No labelled-recording evaluation.** The harness exists
  (`scripts/evaluate_videos.py`) and produces a per-rule TP/FP/FN/TN matrix;
  the recordings do not. Until they do, no accuracy figure exists for FormFix
  and none should be quoted.
* **Person selection is proximity, not identity.** With several people in
  shot the athlete is tracked from frame to frame, but the very first frame is
  still decided on apparent size alone, so somebody standing closer to the
  camera than the athlete at the moment the clip opens would be picked and
  then followed correctly for the rest of the video. Weighting that first
  choice towards whoever is actually moving would remove this. The detector
  also only looks for four people, so a busier scene is under-counted.
* **The plausibility gate checks the label, not the movement.** It can tell
  that a recording is not the exercise that was selected; it cannot say what
  the exercise was. It is deliberately one-directional — it looks for
  contradictions and abstains whenever the evidence is thin — so an unusual
  but genuine attempt gets through, at the price of some genuinely mislabelled
  clips getting through as well. A clip that is rep-shaped, filmed cleanly and
  wrong in a way none of the checks cover will still be analysed as the
  exercise that was picked.
