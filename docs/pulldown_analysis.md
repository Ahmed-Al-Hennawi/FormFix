# The FormFix lat-pulldown analyser — technical documentation

This document is the reference for the lat-pulldown implementation: what it
measures, how it decides, what every threshold means and where it came from,
and what it deliberately does not claim. It is written to be quoted from the
Methodology and Evaluation chapters.

The companion documents are `squat_analysis.md`, `press_analysis.md`,
`threshold_tuning.md` (the development history of every technique threshold)
and `limitations.md` (the limitations shared by all three exercises).

---

## 1. Exercise variant

The analysis is written for one specific movement, the one the FormFix
reference clip demonstrates:

> A **seated, bilateral, pronated-grip lat pulldown**, with the bar travelling
> **in front of the head** to the upper chest.

This matters more than it might appear. A behind-the-neck pulldown puts the
arms in a different plane; a close-grip or neutral-grip pulldown changes the
elbow path; a single-arm pulldown breaks the assumption that both arms are
doing the same thing; a standing cable pulldown removes the fixed seat that
the trunk measurement is referenced against. The rules below describe the
seated bilateral front pulldown and would misdescribe those variations. The
interface names the variant, and this document records it.

---

## 2. Pipeline

```
video file
   → probe metadata                     analysis/video_processor.py
   → file validation                    analysis/validation.py
   → pose detection (MediaPipe VIDEO)   analysis/pose_detector.py
   → recording validation               analysis/validation.py
   → short-gap interpolation + EMA      analysis/smoothing.py
   → side selection                     analysis/validation.py
   → per-frame measurement              exercises/pulldown/metrics.py
   → repetition state machine           exercises/common/phases.py
   → top-position baseline              exercises/pulldown/metrics.py
   → per-repetition measurement         exercises/pulldown/metrics.py
   → rule evaluation (view-gated)       exercises/pulldown/rules.py
   → reliability per measurement        exercises/common/confidence.py
   → explainable feedback               exercises/pulldown/feedback.py
   → annotated video                    analysis/annotation.py
   → JSON / CSV export                  analysis/export.py
```

Each stage owns one job. Measurement never judges, judgement never measures,
and only the feedback layer writes sentences. That separation is what allows a
threshold to be re-tuned, or a rule to be defended in a viva, without touching
how any number was obtained.

---

## 3. Camera view

**Required: a side or three-quarter view, roughly level with the chest.**

The reason is not preference, it is observability. Trunk movement during a
pulldown happens in the sagittal plane — the athlete leans *backwards*, away
from the bar. From directly in front of or behind the athlete that motion is
almost entirely along the camera's optical axis, where a single RGB camera
cannot resolve it. The elbow angle, by contrast, is an in-plane joint angle
that a front, side or three-quarter view can all observe.

The consequence is built into the rules rather than left to the user:

| Camera estimate | Range of motion | Torso movement |
| --- | --- | --- |
| Side | assessed, full view support | assessed, full view support |
| Three-quarter (diagonal) | assessed | assessed, reduced reliability |
| Front-on / rear-on | assessed | **not assessed**, with the reason shown |
| Unknown | assessed, reduced reliability | assessed, reduced reliability |

A front-on pulldown recording is therefore still analysed — the athlete gets a
real range-of-motion verdict — and the torso check reports what it could not
see instead of guessing. The camera view is *estimated*, from the ratio of
shoulder-or-hip separation to trunk length in pixels, and is always reported
with a confidence rather than as a fact.

The recording instructions shown before upload come from
`exercises/pulldown/config.RECORDING_TIPS`, the same constant the analyser
attaches to a rejection, so the guidance and the requirements cannot drift.

---

## 4. Landmarks

MediaPipe returns 33 landmarks. The pulldown reads eight, plus three that are
used for one narrow purpose:

| Role | MediaPipe landmarks | Used for |
| --- | --- | --- |
| Shoulders | `LEFT_SHOULDER` (11), `RIGHT_SHOULDER` (12) | elbow angle, trunk reference, height normalisation |
| Elbows | `LEFT_ELBOW` (13), `RIGHT_ELBOW` (14) | elbow angle, elbow travel |
| Wrists | `LEFT_WRIST` (15), `RIGHT_WRIST` (16) | elbow angle, wrist travel |
| Hips | `LEFT_HIP` (23), `RIGHT_HIP` (24) | trunk reference, body scale |
| Head | `NOSE` (0), `LEFT_EAR` (7), `RIGHT_EAR` (8) | facing direction only — never a joint angle |

`exercises/pulldown/landmarks.py` is the only place in the package that
contains a bare landmark number, and it also declares which landmarks each
*measurement* depends on. That is what allows reliability to be reported per
measurement: range of motion can be high-confidence in the same recording
where trunk movement cannot be assessed at all, because a clearly visible
wrist cannot make an occluded hip trustworthy.

**The bar is not tracked.** MediaPipe estimates a body, not equipment, so
every measurement below is a relationship between body landmarks. A rule that
waited for the bar to touch the chest could not be implemented honestly.

---

## 5. Measurements

All measurements are taken in **pixel space**. MediaPipe normalises x to the
frame width and y to the frame height independently, so an angle computed on
the normalised values is distorted on any non-square frame — which every phone
video is. Coordinates are converted first, then measured.

| Measurement | Definition | Units |
| --- | --- | --- |
| `left/right_elbow_angle` | shoulder → elbow → wrist interior angle | degrees, scale-free |
| `elbow_angle` | mean of whichever arms are individually usable | degrees |
| `elbow_angle_difference` | \|left − right\| | degrees (exported, not ruled on) |
| `torso_angle` | mid-hip → mid-shoulder against image vertical, unsigned | degrees |
| `torso_angle_signed` | the same segment with its image direction kept | degrees |
| `torso_posterior` | the signed angle re-expressed anatomically: **positive = leaning back** | degrees |
| `torso_excursion` | posterior lean minus this athlete's own top-position baseline | degrees |
| `torso_velocity` | rate of change of the trunk angle, from timestamps | deg/s (exported, no rule) |
| `wrist_rise` | mean wrist height above the shoulder line ÷ trunk length | dimensionless |
| `elbow_rise` | the same for the elbows | dimensionless |
| `body_scale` | trunk length | pixels |

Anything that cannot be measured on a frame is `NaN`, never a substituted
zero — a zero would read downstream as a genuine measurement of "no movement".

### Why the trunk is measured as a change, not a posture

A seated pulldown is performed with a small, deliberate backward inclination,
and the bench itself often sets one. Comparing the trunk against "vertical"
would flag correct technique on a reclined seat, on a taller athlete, or on a
slightly tilted camera. Every trunk verdict is therefore a comparison against
**this athlete's own posture at their controlled top position**, taken as the
median over genuine top-position frames (never frame 0 alone, since the
athlete is usually still reaching for the bar).

### Why the direction of lean needs the head

"Leaning backwards" is a direction, and an image alone cannot tell a backward
lean from a forward one without knowing which way the athlete faces. The head
sits in front of the shoulder line, so the median horizontal offset of the
visible head landmarks from the shoulder midpoint gives the facing direction.

Two modes follow, and the result records which was used
(`PulldownRep.torso_mode`, and `torso_measurement_mode` in the debug block):

* **`posterior`** — facing established; the excursion is a *signed backward*
  lean, and the feedback says "moved backward".
* **`unsigned`** — the head was not visible enough; the excursion is the
  absolute change in trunk inclination, and the feedback says "moved away from
  its starting position" instead. The magnitude survives; only the direction
  is unavailable, and the wording reflects exactly that.

---

## 6. Repetition detection

The state machine is the shared one (`exercises/common/phases.py`), run on the
mean elbow angle:

```
TOP (extended) → PULLING → BOTTOM (contracted) → RETURNING → TOP
                                ↑______________________________|
```

Five guards make the count trustworthy, and all five are configurable:
hysteresis (four different thresholds), persistence before a transition
commits, a reversal delta before the contracted position is accepted, a
minimum angular excursion, and FPS-independent duration bounds taken from
timestamps rather than frame counts. Movements that fail any of them are
counted as **partial movements** and reported with a reason.

### The thresholds are adaptive, and this is a design decision

Fixed absolute thresholds create a circular failure. An athlete who never
straightens their arms would never cross a fixed "extended" level, so no
repetition would be detected, so the range-of-motion rule — the very check
meant to notice that habit — would never run. FormFix would report "no
repetition found" for exactly the fault it exists to find.

Segmentation is therefore relative to the athlete's own rest position:

```
reference     = 90th percentile of the measured elbow-angle series,
                clamped to [110°, 180°]
rest level    = reference − 6°
start level   = reference − 14°
extreme level = reference − 30°
end level     = reference − 10°
```

This separates two questions that must not be conflated:

| Question | Answered by | Nature |
| --- | --- | --- |
| Was this a repetition? | the adaptive levels above | permissive segmentation |
| Did it cover enough range? | the fixed `ROM_*` criteria | technique judgement |

`tests/test_pulldown_metrics.py::TestPhaseSegmentation` pins this down: an
athlete generated with a 132° top still has three repetitions counted, and the
range-of-motion rule then reports the restriction.

---

## 7. Rules

Two rules, deliberately. Each answers one narrow question about one
measurement, in the one phase where that question is meaningful, and only when
the camera view can support it.

### Rule table

| Field | `pulldown_rom` | `pulldown_torso` |
| --- | --- | --- |
| **Rule ID** | `pulldown_rom` | `pulldown_torso` |
| **Exercise** | Lat Pulldown | Lat Pulldown |
| **Mistake** | Incomplete range of motion, reported as *which end* fell short | Excessive torso movement / body swing during the pull |
| **Camera view** | Any (side, three-quarter, front); reliability reduced when the view is uncertain | Side or three-quarter only |
| **Required landmarks** | 11/12 shoulders, 13/14 elbows, 15/16 wrists | 11/12 shoulders, 23/24 hips (+ 0/7/8 head for the direction) |
| **Phase** | whole repetition (top read in the rest windows either side) | the pull and the contracted position |
| **Measurement** | elbow angle at the extended and contracted positions, and the excursion between them | trunk inclination change from this athlete's own top-position baseline |
| **Threshold** | top ≥ 150° (warn below 138°); bottom ≤ 100° (warn above 115°); excursion ≥ 45° | warn ≥ 15°, fail ≥ 25°; absolute fail at 45° from vertical |
| **Persistence** | per-repetition extremes, already sustained-filtered (3 consecutive frames) | ≥ 4 consecutive frames **and** ≥ 20% of the pulling phase |
| **Confidence requirement** | the **analysed** arm usable on ≥ 50% of the repetition (single-arm on purpose — see below) | shoulders and hips ≥ 0.5 mean visibility on ≥ 50% of frames |
| **Rationale** | The body's own joint excursion is what a pose estimate can measure; the bar cannot be tracked. Judging both ends separately distinguishes "stopping the pull early" from "never letting the arms straighten" — different habits, different corrections. | The fault is a body swing that moves the load instead of the back. Measuring the *change* from the athlete's own top posture rather than an absolute angle keeps a reclined seat, a long torso and a tilted camera from reading as a fault. |
| **Feedback** | *"The pull stopped early in 2 of your 3 repetitions… against the configured criterion of 100°."* → drive the elbows down through a controlled range | *"Your torso moved backward considerably during the pulling phase…"* → keep the torso stable, pull by driving the elbows down |
| **Limitation** | Foreshortening at an oblique angle biases the elbow angle; a bar held very wide or very close changes the geometry; equipment occlusion of a wrist makes the repetition unassessable. | Not observable from a front-on or rear-on camera. Without the head visible, direction is lost and only the magnitude of the change is reported. A camera that moves during the clip can imitate trunk motion. |

### Range-of-motion categories

The rule reports a category, not a score, because "limited range of motion"
alone tells a beginner nothing about what to change:

`complete_rom`, `limited_top_extension`, `limited_bottom_range`,
`limited_overall_rom`, `not_assessable`.

Vertical wrist travel is measured alongside and reported as supporting
evidence, so a reader can sanity-check the angular figures against real
displacement — but it is never the pass condition on its own.

### Why range of motion needs one arm, not two

Range of motion is a joint angle, not a comparison. In the side or
three-quarter view this exercise asks for, the far arm is hidden behind the
near one for much of the pull — which is precisely why the reliability gate
requires only the *analysed* arm. An earlier version demanded both, and
reported "not assessed" on a correctly recorded side view: the check was being
switched off by the very camera position the guidance recommends. Found by
running a real recording (`evaluation/reference_clip_runs.md`). Both-arm
availability is still measured and exported as `both_arms_ratio`, because the
left/right elbow difference series depends on it — but no pulldown rule reads
that series.

### Persistence

Every threshold crossing is filtered through `exercises/common/persistence.py`
before it can become a finding. A violation must both hold for a configured
number of consecutive frames *and* cover a configured share of the phase. One
jittered shoulder landmark can therefore never produce a correction, and the
per-repetition outcome records the numbers (`violating_frames`,
`phase_frames`, `violation_ratio`) that let it fire.

---

## 8. What is deliberately not implemented

**Torso swinging *between* repetitions.** The measurement it would need — peak
trunk angular velocity — is computed and exported on every repetition
(`peak_torso_velocity`), but no rule reads it. It was not validated against
labelled recordings, and shipping an untested third check would weaken the two
that were tested. It is recorded here as future work with the data already in
place to develop it.

**Also not claimed:** scapular movement or "shoulder depression", lat
activation, whether the grip width is correct, whether the bar touched the
chest, spinal position, joint loading, injury risk, and any medical or
physiotherapeutic assessment.

---

## 9. Reliability

Reported per measurement, from four things that can be pointed at in the data:
the visibility of exactly the landmarks that measurement needs, the share of
read frames that produced a value, how well the estimated camera view supports
it, and how many repetitions produced a verdict. Two ceilings are then applied
— subject size in pixels, and whether validation already called the recording
limited.

The four levels are `High`, `Medium`, `Low` and `Cannot assess`. They are
deliberately **not** a probability: nothing here is learned, so a figure such
as "0.87 confident" would imply a calibration the system does not have.

---

## 10. Failure handling

| Situation | Outcome |
| --- | --- |
| No person detected | `NO_POSE` — "We couldn't detect a person in this video" |
| Arms not trackable in enough frames | `INSUFFICIENT_VISIBILITY`, naming shoulders, elbows and wrists |
| Hands leave the top of the frame | warning; range of motion marked not assessed if persistent |
| Shoulders/hips outside the frame for most of the clip | `BODY_OUT_OF_FRAME` |
| Front-on camera | analysed; torso check reported as not assessed, with the reason |
| Hips hidden by the bench | analysed; torso check not assessed, range of motion unaffected |
| No complete repetition | `NO_COMPLETE_REPETITION`, with the partial-movement count and reasons |
| Video too short / unreadable | `INVALID_VIDEO` |

In every case the recording problem is described and the recording advice
attached. **No technique feedback is ever produced from a failed analysis.**

---

## 11. Annotated output

The rendered clip carries the skeleton with the analysed arm emphasised, a HUD
showing the repetition counter, the movement phase and the live elbow and
trunk angles, a `CONTRACTED` marker through the elbow during the turning-point
window, and an issue banner shown **only around the evidence frame of a
finding** — so a contracted-position issue is not smeared across a whole
repetition. Only the landmarks the firing rule actually read are highlighted.

Palette: Deep Space Blue `#0B132B`, Graphite `#1C2541`, Electric Cyan
`#00F0FF`, Neon Lime `#39FF14`, white, with amber and red reserved for
warnings and failures.

---

## 12. Testing

| File | Covers |
| --- | --- |
| `tests/test_common_phases.py` | the shared state machine: hysteresis, persistence, reversal, noise, gaps, truncated clips |
| `tests/test_geometry_upper_body.py` | signed inclination and angular rate, including degenerate inputs |
| `tests/test_pulldown_metrics.py` | measurement, facing estimation, baseline, normalisation, segmentation, per-rep facts |
| `tests/test_pulldown_rules.py` | both rules and their view gating, from constructed repetitions |
| `tests/test_pulldown_integration.py` | the whole pipeline on synthetic recordings, including false-positive protection and graceful degradation |
| `tests/test_evaluation_harness.py` | the labelled-video evaluation harness |

The synthetic generator (`tests/synthetic_pulldown.py`) is aspect-corrected so
a skeleton built with a 172° elbow measures 172° after the pipeline converts
the landmarks back to pixels. A guard test asserts that, because without it
every threshold test would silently be measuring the wrong scale.
