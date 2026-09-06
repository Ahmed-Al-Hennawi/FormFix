# The FormFix dumbbell shoulder-press analyser — technical documentation

This document is the reference for the shoulder-press implementation: what it
measures, how it decides, what every threshold means and where it came from,
and what it deliberately does not claim.

Companion documents: `squat_analysis.md`, `pulldown_analysis.md`,
`threshold_tuning.md` and `limitations.md`.

---

## 1. Exercise variant

The analysis is written for the movement the FormFix reference clip
demonstrates:

> A **seated dumbbell shoulder press with the back supported**, a **pronated
> grip**, both arms pressing **together**, from about shoulder height to
> overhead.

An Arnold press rotates the forearm through the movement; a neutral-grip press
changes the frontal-plane geometry the alignment rule measures; a single-arm
press breaks the assumption that both arms are doing the same thing at the
same time; a standing or push press adds trunk and leg motion the seated model
does not describe. The rules below would misdescribe those variations.

---

## 2. Pipeline

```
video file
   → probe metadata                     analysis/video_processor.py
   → file validation                    analysis/validation.py
   → pose detection (MediaPipe VIDEO)   analysis/pose_detector.py
   → recording validation               analysis/validation.py
   → short-gap interpolation + EMA      analysis/smoothing.py
   → per-frame measurement              exercises/press/metrics.py
   → repetition state machine           exercises/common/phases.py
   → per-repetition measurement         exercises/press/metrics.py
   → rule evaluation (view-gated)       exercises/press/rules.py
   → reliability per measurement        exercises/common/confidence.py
   → explainable feedback               exercises/press/feedback.py
   → annotated video                    analysis/annotation.py
   → JSON / CSV export                  analysis/export.py
```

---

## 3. Camera view

**Required: a front-on view, roughly level with the chest, camera centred.**

Two of the three checks are comparisons *between* the arms. From the side one
arm hides the other, so the measured "difference" between them would be
describing the camera angle rather than the athlete. The wrist-over-elbow
alignment check has the same problem: from the side, the horizontal
relationship between wrist and elbow is depth, which one RGB camera cannot
resolve. Only the range-of-motion check — a single arm's joint angle — survives
a side-on recording.

This is the mirror image of the squat's camera policy, and the generalised
validation layer expresses it as configuration rather than as a special case:

| Camera estimate | Arm symmetry | Elbow/wrist alignment | Range of motion |
| --- | --- | --- | --- |
| Front-on | assessed, full view support | assessed, full view support | assessed |
| Diagonal | assessed, reduced reliability | assessed, reduced reliability | assessed |
| Side-on | **not assessed**, with the reason | **not assessed**, with the reason | assessed |
| Unknown | assessed, reduced reliability | assessed, reduced reliability | assessed |

A side-on press recording is therefore still analysed and still scored — on the
one check it can support — and the other two report what they could not see.

### A subtlety worth recording

`side_view_confidence` measures how *side-on* the camera is. For a sagittal
measurement such as trunk lean, a diagonal recording that is nearly side-on is
the good case and scores high. For a **frontal-plane** measurement such as
comparing two arms, nearly side-on is the *bad* case, and using the same
number unchanged would credit the worst diagonal recordings with the highest
confidence. `common.confidence.view_support` therefore takes a `frontal_plane`
flag and scores the complement for those metrics
(`tests/test_press_rules.py::TestFrontalPlaneViewSupport`).

---

## 4. Landmarks

| Role | MediaPipe landmarks | Used for |
| --- | --- | --- |
| Shoulders | `LEFT_SHOULDER` (11), `RIGHT_SHOULDER` (12) | elbow angle, height reference, **shoulder width** |
| Elbows | `LEFT_ELBOW` (13), `RIGHT_ELBOW` (14) | elbow angle, alignment offset, elbow height |
| Wrists | `LEFT_WRIST` (15), `RIGHT_WRIST` (16) | elbow angle, alignment offset, wrist height |
| Hips | `LEFT_HIP` (23), `RIGHT_HIP` (24) | scale fallback only, when the shoulders overlap |

The dumbbells are not tracked. A dumbbell that hides a wrist therefore shows up
as a *visibility* problem — reported as "not assessed" — rather than as a wrong
number.

---

## 5. Measurements and normalisation

| Measurement | Definition | Units |
| --- | --- | --- |
| `left/right_elbow_angle` | shoulder → elbow → wrist interior angle | degrees |
| `elbow_flexion` | 180° − mean elbow angle — **the movement signal** | degrees |
| `elbow_angle_difference` | \|left − right\| elbow angle | degrees |
| `left/right_wrist_height` | wrist height above the shoulder line ÷ shoulder width | dimensionless |
| `wrist_height_difference` | left − right; **positive means the left wrist is higher** | dimensionless |
| `left/right_elbow_height` | the same for the elbows | dimensionless (exported) |
| `left/right_alignment_offset` | \|wrist x − elbow x\| ÷ shoulder width | dimensionless |
| `shoulder_width` | shoulder-to-shoulder distance | pixels |

### Why shoulder width is the scale reference

Both normalised measurements are relationships in the plane facing the camera,
and shoulder width is the body dimension measured in that same plane. Trunk
length foreshortens the moment the athlete leans or the seat reclines, which
would make the "normalised" values drift during the very phase they are read
in. Using shoulder width also gives the thresholds a physical reading: an adult
shoulder width is roughly 0.40 m, so an offset of 0.38 shoulder widths is
about 15 cm of horizontal wrist-to-elbow displacement.

A guard exists for the near side-on case, where the apparent shoulder
separation collapses towards zero and every normalised value would explode: the
trunk-derived estimate is used as a floor. The frontal-plane rules are already
reporting "not assessed" for such a recording, so the guard exists to keep the
*exported numbers* finite and comparable, not to rescue a verdict.

---

## 6. Repetition detection

```
READY (at the shoulders) → PRESSING → TOP → LOWERING → READY
                                        ↑_______________|
```

The press travels **upwards** where the squat and pulldown travel down, so it
would appear to need its own state machine. It does not: expressing the
movement as elbow **flexion** (180° − elbow angle) gives the same "high at
rest, falling into the repetition" shape the shared machine expects, and the
press reuses it unchanged. One machine, one set of robustness guarantees, one
set of tests.

The thresholds are adaptive for the same reason as the pulldown's — an athlete
who never reaches lockout must still have their repetitions counted, or the
range-of-motion rule meant to notice that habit would never run:

```
reference     = 90th percentile of the measured flexion series,
                clamped to [45°, 140°]
rest level    = reference − 6°
start level   = reference − 14°
extreme level = reference − 30°
end level     = reference − 10°
```

### Where the top and bottom are read

Neither is read at the repetition's own boundary frames. The state machine
commits a repetition once the press has clearly begun, a fraction of a second
*after* the athlete left the bottom position; reading the bottom angle there
would over-report everyone's depth. The bottom is therefore measured in the
rest windows on either side of the repetition, bounded by the neighbouring
repetitions so one rep can never borrow another's depth.

Each arm's **top** extension is the best value *it* sustained anywhere in the
repetition, not the value both arms happened to hold during a shared window.
If one arm arrives a fraction of a second later, a shared window would read the
late arm mid-press and report a limited range for a repetition whose only fault
is timing — which the symmetry rule is already there to say. Measuring each arm
on its own terms keeps the two checks answering two different questions
(`tests/test_press_metrics.py::TestRepMeasurements`).

---

## 7. Rules

### Rule table

| Field | `press_symmetry` | `press_alignment` | `press_rom` |
| --- | --- | --- | --- |
| **Rule ID** | `press_symmetry` | `press_alignment` | `press_rom` |
| **Exercise** | Shoulder Press | Shoulder Press | Shoulder Press |
| **Mistake** | One arm leads, lags or travels less than the other | A wrist drifts away from being stacked over its elbow | Incomplete range of motion, reported as *which end* fell short |
| **Camera view** | Front-on or diagonal only | Front-on or diagonal only | Any; reliability reduced when the view is uncertain |
| **Required landmarks** | 11/12 shoulders, 13/14 elbows, 15/16 wrists (**both** arms) | shoulder, elbow, wrist of the judged side | shoulder, elbow, wrist |
| **Phase** | the press and the overhead position | the press and the overhead position | whole repetition (bottom read in the rest windows either side) |
| **Measurement** | three interpretable signals: \|left − right\| elbow angle; \|left − right\| wrist height ÷ shoulder width; \|left ROM − right ROM\| | \|wrist x − elbow x\| ÷ shoulder width, per arm, worst side reported | elbow angle overhead and at the shoulders, and the excursion between them |
| **Threshold** | angle warn 15° / fail 25°; height warn 0.12 / fail 0.20 shoulder widths; ROM warn 15° / fail 25° | warn 0.38, fail 0.52 shoulder widths | top ≥ 155° (warn below 143°); bottom ≤ 100° (warn above 115°); excursion ≥ 50° |
| **Persistence** | ≥ 5 consecutive frames **and** ≥ 25% of the phase (either in-movement signal may carry it; a whole-repetition range difference also qualifies) | ≥ 5 consecutive frames **and** ≥ 25% of the phase | per-repetition extremes, already sustained-filtered (3 consecutive frames) |
| **Confidence requirement** | both arms usable on ≥ 60% of the repetition, and mean arm visibility ≥ 0.5 | the **best** side's elbow/wrist pair ≥ 0.5 — one visible arm is enough, since the sides are judged independently | landmark visibility floor 0.4 |
| **Rationale** | Three separate interpretable signals rather than one combined index: a beginner can act on "your left arm stayed lower" and cannot act on "symmetry score 0.72". The angle and height signals describe the movement *while* it happens; the range difference catches an arm that travels the same way but not as far. | In the frontal plane the wrist should stay reasonably stacked over its own elbow. Normalising by shoulder width makes the measure independent of camera distance, resolution and body size. The sides are judged independently because a drift usually belongs to one arm, and averaging would hide it. | Judging both ends separately distinguishes "stopping short of overhead" from "not lowering the dumbbells back to the shoulders" — different habits, different corrections. |
| **Feedback** | *"Your right arm stayed higher than your left arm during 2 of your 3 repetitions… a left/right elbow-angle difference of up to about 23°."* → press at a controlled pace, both arms together | *"Your left wrist drifted noticeably away from being stacked over your left elbow… about 46% of your shoulder width."* → keep each dumbbell stacked above its forearm | *"Your presses stopped before reaching the configured top range in 3 of 5 repetitions."* → finish each press through a comfortable, controlled range |
| **Limitation** | Needs both arms visible simultaneously; a dumbbell that hides a wrist makes the repetition unassessable. A few degrees of asymmetry is normal human movement and also lies inside MediaPipe's estimation error, which is why the bar sits well above zero. | A markedly limited top range necessarily leaves the wrist beside the elbow (see below). Wrist landmarks are among the least reliable when a dumbbell occludes them. | Foreshortening at an oblique angle biases the elbow angle; the top criterion is not 180° and full lockout is never required. |

### A known correlation, stated rather than engineered around

A repetition that stops well short of overhead *necessarily* leaves the wrist
beside the elbow, because the elbow is still bent there. The alignment
measurement and the range-of-motion measurement are therefore **not
statistically independent**, and a strongly limited press can register both.

The tolerance was widened from an initial 0.30/0.45 to 0.38/0.52 for exactly
this reason (see `threshold_tuning.md`), which removes the overlap for
moderate cases. It cannot remove it entirely without making the alignment
check meaningless, so the coupling is documented rather than hidden — and the
evaluation should read a simultaneous alignment finding on a
severely-limited-range recording as one habit reported twice, not as two
independent detections.

### One hidden arm costs only the check that needs both

Symmetry is a comparison and genuinely requires both arms. Alignment judges
each side independently and reports the worse one, and range of motion is a
single-arm joint angle — neither should be disabled by a hidden arm. An earlier
version averaged both arms' visibility into a single alignment gate, which let
a hidden arm switch the check off for the arm that was perfectly visible;
`tests/test_press_integration.py::test_one_hidden_arm_costs_only_the_check_that_needs_both`
pins the corrected behaviour.

### Deliberately not implemented

**Sagittal trunk lean** — the "arching the back" fault. It happens in the plane
a front-on camera cannot see, and this exercise needs a front-on camera for its
other checks. Claiming to measure it from this view would be exactly the kind
of unsupported assessment the project argues against.

**Also not claimed:** shoulder mobility, impingement, scapular mechanics, joint
loading, injury risk, and any medical or physiotherapeutic assessment.

---

## 8. Failure handling

| Situation | Outcome |
| --- | --- |
| No person detected | `NO_POSE` |
| Arms not trackable in enough frames | `INSUFFICIENT_VISIBILITY`, naming shoulders, elbows and wrists |
| Hands leave the top of the frame | warning; range of motion marked not assessed if persistent |
| One arm hidden | symmetry and alignment not assessed; range of motion still reported |
| Side-on camera | analysed; symmetry and alignment reported as not assessed, with reasons |
| No complete repetition | `NO_COMPLETE_REPETITION`, with the partial count and reasons |
| Video too short / unreadable | `INVALID_VIDEO` |

An unavailable measurement is **never** scored as a failure. A side-on press
recording with three good repetitions scores 100 on the one check that had
evidence (`tests/test_press_integration.py::test_a_side_on_recording_is_not_scored_as_a_failure`).

---

## 9. Annotated output

Both arms are emphasised — a press is a bilateral movement and two of its three
checks are comparisons — with a HUD showing the repetition counter, the phase
and both live elbow angles, a `TOP` marker across the wrists during the
overhead window, and an issue banner only around the evidence frame of a
finding. An alignment finding highlights the arm it is about; a symmetry or
range finding highlights both.

---

## 10. Testing

| File | Covers |
| --- | --- |
| `tests/test_common_phases.py` | the shared state machine |
| `tests/test_press_metrics.py` | measurement, normalisation, scale guard, segmentation, per-rep facts |
| `tests/test_press_rules.py` | all three rules and their view gating, from constructed repetitions |
| `tests/test_press_integration.py` | the whole pipeline on synthetic recordings, including false-positive protection and graceful degradation |
| `tests/test_evaluation_harness.py` | the labelled-video evaluation harness |

`tests/synthetic_press.py` is aspect-corrected so a skeleton built with a 90°
elbow measures 90° after the pipeline converts the landmarks back to pixels; a
guard test asserts it for both arms.
