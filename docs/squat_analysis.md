# Squat analyser

How FormFix turns a squat video into feedback: the steps, what it measures,
the rules, where the thresholds come from, and what it doesn't claim.

MediaPipe is only the part that finds the landmarks. My contribution is
everything built on top: choosing the landmarks, checking the recording,
smoothing, counting reps, measuring, the rules, reliability and the feedback.
Each part is a separate module with its own tests, so I can change a threshold
without touching how the numbers are measured.

---

## 1. Pipeline

```
video
 -> read the video                        analysis/video_processor.py
 -> check the file                        analysis/validation.py
 -> MediaPipe pose detection              analysis/pose_detector.py
    (up to 4 people; follows the one nearest the camera)
 -> check the recording                   analysis/validation.py
    (coverage, landmarks, framing, camera angle, tracking)
 -> drop impossible landmark jumps        analysis/stabilise.py
 -> fill short gaps + EMA smoothing       analysis/smoothing.py
 -> pick the more visible side            analysis/validation.py
 -> measure every frame                   exercises/squat/metrics.py
 -> count reps (state machine)            exercises/squat/phases.py
 -> is it actually a squat?               exercises/common/plausibility.py
 -> split each rep into phases            exercises/squat/metrics.py
 -> run the rules                         exercises/squat/rules.py
 -> reliability per measurement           exercises/common/confidence.py
 -> feedback                              exercises/squat/feedback.py
 -> annotated video (trimmed to the set)  analysis/annotation.py, trimming.py
```

`exercises/squat/analyser.py` just runs these in order.

---

## 2. Landmarks

The squat only uses 12 of MediaPipe's 33 landmarks. Counting the face or
fingers as evidence would make the reliability look better than it is (a
visible nose doesn't make a hidden ankle trustworthy).

| Left | Index | Right | Index |
| --- | --- | --- | --- |
| shoulder | 11 | shoulder | 12 |
| hip | 23 | hip | 24 |
| knee | 25 | knee | 26 |
| ankle | 27 | ankle | 28 |
| heel | 29 | heel | 30 |
| foot index (toe) | 31 | foot index (toe) | 32 |

`METRIC_REQUIREMENTS` in `landmarks.py` says which landmarks each measurement
needs, so reliability is worked out **per measurement** (depth can be reliable
while the heel check can't be assessed).

**Side:** from the side, one half of the body is hidden, so the side with more
usable frames is analysed. Both sides are still measured and exported.

---

## 3. Checking the recording

MediaPipe's output isn't treated as ground truth. Before any technique check:

| Stage | Question | Setting |
| --- | --- | --- |
| A | Was a person found often enough? | `MIN_POSE_FRAME_RATIO` |
| B | Are shoulder, hip, knee and ankle visible? | `MIN_KEY_LANDMARK_VISIBILITY` (0.4) |
| C | Is enough of the clip measurable? | `MIN_USABLE_FRAME_RATIO`, `MIN_VALID_FRAME_RATIO` |
| D | Does the body stay in frame? | `FRAMING_MARGIN`, `FRAMING_WARN/FAIL_TOLERANCE` |
| E | What's the camera angle, and does it change? | `SIDE_VIEW_GOOD_RATIO`, `SIDE_VIEW_FRONTAL_RATIO`, `VIEW_STABILITY_SPREAD` |

Gaps of up to 5 frames are filled in; longer gaps stay missing. The result is
one of three:

- **Good** - analyse normally.
- **Limited** - analyse, and say which checks the recording affected.
- **Unusable** - stop with a clear reason (no person, joints missing most of
  the time, unreadable file, no complete rep, tracking keeps losing the
  person, or the wrong exercise).

A camera angle on its own never rejects a video.

### Which person is analysed

The detector looks for up to four people. On the first frame it picks the
biggest person (nearest the camera). After that it follows whoever is closest
to where that person was on the last frame, measured hip to hip in torso
lengths.

I had to add this because picking the biggest person on *every* frame broke:
at the bottom of a squat your bounding box shrinks, so someone standing behind
you could suddenly be "biggest" and take over the analysis. Other people in
shot now only add a note; a video is only rejected if the tracking keeps
losing the person.

### Plausibility limits

MediaPipe gives confident positions even for limbs it is guessing. Values
outside wide anatomical limits become "not measured", never a fault:

| Limit | Value |
| --- | --- |
| Knee angle | 25° to 190° |
| Heel lift | at most 0.5 of a lower leg |
| Left/right knee difference | at most 45° |

---

## 4. Normalisation

Angles don't depend on size, so they're used directly. Distances are divided by
a body length from the same frame: torso length, or hip width if the torso isn't
visible, or lower-leg length as a last option.

Everything is measured in **pixels**, because MediaPipe normalises x and y
separately and angles on normalised values are distorted on a phone video.

---

## 5. Smoothing

An EMA on the landmark coordinates (`EMA_ALPHA` 0.45) and on the knee-angle
series (`ANGLE_EMA_ALPHA` 0.5). Visibility isn't smoothed, because that would
hide tracking problems. The EMA restarts after a gap instead of dragging an
old position forward. The filter can be changed with `ANGLE_FILTER` (see
[filter_selection.md](filter_selection.md)).

---

## 6. Measurements

`metrics.py` only measures - no thresholds or verdicts.

| Measurement | How | Notes |
| --- | --- | --- |
| Knee angle | hip-knee-ankle, each leg | ~175-180° standing, smaller = deeper |
| Hip angle | shoulder-hip-knee, each leg | smaller = more bend |
| Torso lean | hip → shoulder vs vertical | 0° = upright |
| Shin angle | ankle → knee vs vertical | bigger = knee further forward |
| Depth | (knee y - hip y) / lower-leg length | positive = hip above knee |
| Heel lift | heel height above the toe / lower-leg length, minus standing value | positive = heel raised |
| Left/right difference | \|left - right\| knee angle | exported only, no rule uses it (§8) |
| Stance width | ankle gap / body size | front view |
| Knee over ankle | \|knee x - ankle x\| / body size | exported only (§9) |
| Timing | descent, bottom, ascent | seconds, from timestamps |

Anything that can't be measured is `NaN`, never 0.

**Why the heel is measured against the toe:** my first version compared the
heel to its standing position in the image, so moving across the frame or a
bumped camera looked like a heel lift (over 250% of a lower leg on real
footage). Comparing heel to toe on the same foot fixed that.

Three more things make the number usable. The zero point is the person's own
settled standing offset, not frame 0, so shuffling into position doesn't set
it. The scale is their standing lower-leg length, not the current frame's -
dividing by the current frame inflated the ratio at the bottom of the rep,
where the shin is tilted and shorter in the image. And it is measured on
whichever foot MediaPipe actually tracked, since the side the angles come from
is chosen on the hip, knee and ankle and says nothing about the feet.

---

## 7. Counting reps and phases

A state machine on the smoothed knee angle:

```
STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING
```

Just counting threshold crossings turned one jittery squat into several reps,
so there are several guards:

| Guard | Setting |
| --- | --- |
| Four different thresholds (hysteresis) | standing 160°, start 150°, bottom 130°, end 155° |
| Change must hold before it counts | `PHASE_MIN_FRAMES` 3 |
| Minimum range | `MIN_RANGE_OF_MOTION` 35° |
| Bottom must reverse by | `BOTTOM_REVERSAL_DELTA` 5° |
| Rep length | 1-12 s (from timestamps) |
| Allowed tracking loss | `MAX_TRACKING_LOSS_FRAMES` 10 |

A video starting mid-squat never counts that rep. Movements that start but
don't qualify are reported as **partial**, with a reason.

Each rep is split into descent, bottom, ascent and finish. The bottom is a
small window around the lowest point (median of ±3 frames), and peaks use the
highest value that was **held** for a few frames, so one jittery frame can't be
the "worst moment". Because durations come from timestamps, 24, 30 and 60 fps
videos give the same result.

### Is this actually a squat?

`common/plausibility.py` checks the movement matches the exercise picked. It
doesn't classify the exercise - it only looks for things a squat can't do. For
the squat, the knees have to bend below 155°, so a shallow squat is still a
depth finding, not "wrong exercise". More than 60% of reps (at least two) have
to contradict it before a video is stopped.

---

## 8. Rules

Each rule is declared in `config.py` (what it reads, which phase, which camera
views, the thresholds, how long a fault must last, and where the numbers came
from) and evaluated in `rules.py`.

| Rule | Measures | Phase | Thresholds | Must last |
| --- | --- | --- | --- | --- |
| **Depth** | min knee angle, or hip reaching knee level | bottom | pass ≤ 100°, warn ≤ 115°, or hip within 0.12 of knee | ≥ 50% of the bottom window |
| **Torso lean** | trunk angle | descent → ascent | warn ≥ 45°, fail ≥ 60° or ≥ 55° past your standing posture | ≥ 4 frames and ≥ 15% of the rep |
| **Heel lift** | heel rise | descent → ascent | warn ≥ 0.16 (warning only) | ≥ 0.2 s and ≥ 15% of the rep |
| **Return to standing** | finish angle vs your own standing angle | finish | pass within 12°, warn within 24° | - |
| **Descent control** | descent time | descent | pass ≥ 0.45 s, warn ≥ 0.25 s | - |

All except heel lift need a side or diagonal view. Across the set: **fail** if
at least half the reps failed, **warning** if any failed or warned, otherwise
**pass**.

Rules only check the phase where they matter - the torso is supposed to lean
more at the bottom than when standing, so checking the whole video against one
number would invent faults.

A fault has to pass two tests (`common/persistence.py`): it must last enough
frames **in a row**, and cover enough of the phase. Missing frames don't count
either way.

### The rule I removed: left/right evenness

There used to be a sixth rule warning at 12° of left/right knee difference.
A difference of two measurements has √2 × the error of one, so about ±15.1°
([measurement_uncertainty.md](measurement_uncertainty.md)). The warning was
below its own noise, so it could report an asymmetry that was just measurement
error. I removed the rule but kept the measurement in the export. This means
the squat has no front-view rule now - a front-on video only gets the heel check.

---

## 9. What it doesn't claim

One 2D camera can't support these, so they aren't checked:

- back rounding
- injury risk, joint loading or any medical statement
- knee valgus (knees caving in) - it's a 3D movement
- "knees past toes" as a fault - it's measured and exported, but squat style
  and body proportions legitimately change it

---

## 10. Camera angle

```
frontality = max(shoulder gap, hip gap) / torso length     (in pixels)
```

| Ratio | Angle | Side-view confidence |
| --- | --- | --- |
| ≤ 0.45 | side | 1.0 |
| between | diagonal | 1 → 0 |
| ≥ 1.00 | front | 0.0 |
| can't measure | unknown | 0.0 |

The angle only decides **what can be assessed**, never whether the video is
rejected. It's estimated once per video, so if the angle changes during the
clip that's flagged instead of averaged.

(An early bug: I compared normalised coordinates, which inflated the ratio
about 1.8× on a portrait phone video and made normal side views look frontal.
It's calculated in pixels now.)

---

## 11. Reliability

Four levels - **High, Medium, Low, Cannot assess** - not a probability,
because nothing here is learned. Based on:

- visibility of exactly the landmarks that measurement uses
- how many frames produced a value
- how well the camera angle supports it
- how many reps gave a verdict

Every level's minimums must all be met (not averaged). Then two caps: a person
filmed from far away (few pixels) can't get High, and nothing from a recording
already marked "limited" can be High. The overall figure is the median across
the checks.

---

## 12. Feedback

The wording comes from fixed templates (`feedback.py`); only the numbers change.
Each correction has:

1. what happened, in plain English
2. the evidence ("Rep 2 - minimum knee angle 118° - 00:04.2")
3. what to try
4. which rule produced it

The results show one verdict per rep, a line per check ("Depth: 3 of 4 reps
acceptable"), a separate **Not assessed** block, and the score:

```
score = 100 x (passes + 0.5 x warnings) / checks that could be assessed
```

If nothing could be assessed, there's no score at all.

---

## 13. The annotated video

- The whole skeleton is drawn, not just the joints the squat uses.
- The skeleton has its own display smoothing, separate from the analysis: a
  running median to kill single-frame noise, then a zero-phase Butterworth, so
  it doesn't jitter and doesn't lag behind the movement either.
- The side away from the camera is drawn fainter, and fainter still if it is
  behind the body for most of the clip - MediaPipe is largely guessing at it.
- One style throughout: thin cyan links, small white joints, nothing changes
  colour when a rep is flagged. The verdict belongs on the results page.
- The video is trimmed from 1 s before the first rep to 1 s after the last,
  but frame numbers stay the same, so every timestamp matches your upload.

---

## 14. Thresholds

Everything is in `exercises/squat/config.py`:

- **Engineering** values (detection, smoothing, rep counting, persistence,
  plausibility, reliability) - these just keep the system stable.
- **Technique** values (`DEPTH_*`, `TORSO_LEAN_*`, `HEEL_LIFT_*`, `DESCENT_*`,
  `FULL_EXTENSION_TOLERANCE`) - my own provisional values, not clinical
  constants.

Thresholds are ranges with a warning band, not exact targets, and where
possible they compare against your own standing posture. There's also a
literature-based set in `literature_config.py` for comparison (see
[threshold_tuning.md](threshold_tuning.md)).

---

## 15. Failures

| Code | Why |
| --- | --- |
| `INVALID_VIDEO` | unreadable, too short/long, too low resolution |
| `NO_POSE` | no person found |
| `INSUFFICIENT_VISIBILITY` | key joints missing too often |
| `BODY_OUT_OF_FRAME` | head/torso or hips out of frame most of the time |
| `MULTIPLE_PEOPLE` | tracking kept losing the person |
| `NO_COMPLETE_SQUAT` | no complete rep (partials are listed) |
| `EXERCISE_MISMATCH` | doesn't look like a squat |
| `ANALYSIS_ERROR` | anything unexpected, shown as a friendly message |

Detection runs in a separate process because MediaPipe can crash Python on
some systems.

---

## 16. Debug output and testing

`FORMFIX_DEBUG=1` shows the full technical details and saves `summary.json`,
`frame_metrics.csv` and `reps.csv` to `analysis_results/<run id>/`.
`python scripts/calibrate.py video.mp4 squat --frames` prints the same as a
table and saves key frames.

The tests cover geometry, smoothing, landmarks, persistence, rep counting,
each rule, camera-view gating, reliability, feedback and full end-to-end runs
on synthetic data. When I moved shared code into `exercises/common/` for the
other two exercises, the squat tests passing unchanged proved nothing broke.

```bash
FORMFIX_TEST_VIDEO=path/to/side-view-squat.mp4 pytest   # adds a real-video test
```

---

## 17. Limitations

- One 2D camera: no 3D angles, no forces.
- MediaPipe error, made worse by occlusion, loose clothes, bad lighting and a
  moving camera.
- The camera angle is estimated once per video.
- A hard cut in the video can look like a very fast descent.
- One set of thresholds for everyone.
- Not yet tested on labelled real gym videos (different bodies, clothing,
  lighting).

See [limitations.md](limitations.md) for the full list.
