# Limitations

What FormFix can and can't do, stated plainly. I use this for the limitations
section of the thesis and to check the app's wording against.

## 1. What it doesn't claim

FormFix compares your movement to **configured technique targets** and explains
what it measured. It does **not**:

- predict or assess injury risk
- give medical or physiotherapy advice
- replace a coach
- measure joint loading, spine position, muscle activation or mobility
- say your form is "perfect" - only that no issue was found in the checks it
  could run
- reach lab-level accuracy

All technique thresholds are my own prototype values
([threshold_tuning.md](threshold_tuning.md)).

## 2. One camera and pose estimation

- **2D only.** Everything is measured in the image. Movement towards or away
  from the camera is invisible or squashed - that's why the pulldown needs a
  side view, the press needs a front view, and a check says "not assessed"
  when it doesn't get one.
- **No 3D landmarks.** MediaPipe does give 3D "world" coordinates, but FormFix
  doesn't use them - single-camera depth isn't motion capture. Anything that
  needs 3D (knee valgus, back arching in a front-on press) isn't measured.
- **Landmark error.** MediaPipe is confident even about limbs it's guessing.
  Plausibility limits, visibility floors and persistence reduce this, but can't
  remove it.
- **Other papers' accuracy doesn't transfer.** Their exercises, cameras and
  people are different from mine.

## 3. Recording conditions

| Condition | Effect |
| --- | --- |
| Camera angle | The biggest factor. Estimated per video and shown as a confidence. |
| Distance | A small person in the frame has more error; reliability is capped. |
| Moving camera | Can look like body movement. Measuring change from your own baseline helps but doesn't fix it. |
| Equipment in the way | Machine frame, bench, dumbbell over a wrist - the check says "not assessed". |
| Self-occlusion | From the side one arm/leg hides the other, so left/right checks aren't possible there. |
| Clothing | Loose clothes make hips and shoulders less accurate. Not detectable. |
| Lighting | Poor light means fewer detections; too little and the video is rejected. |
| Speed | Very fast reps give fewer frames; implausibly fast ones count as partial. |
| Other people | Fine - the person nearest the camera is followed. Only rejected if tracking keeps losing them. |

## 4. People and exercise variations

- **Body proportions.** Scaling by body size fixes distance and size, but limb
  proportions really do change what an angle looks like. No single threshold
  suits everyone.
- **Exercise variations.** Each analyser is for one version of the exercise
  (the one in its reference clip). A behind-the-neck pulldown, an Arnold press
  or a front squat would be described wrongly.
- **No real-video validation.** Tested on synthetic data with known faults, not
  on labelled real videos. No accuracy figure is claimed.

## 5. Method

- **Rule-based.** Every finding can be traced to landmarks, a measurement, a
  phase, a threshold and a number of frames. The trade-off: it only finds what
  I designed it to find, and a fault outside the rules just doesn't appear.
- **Thresholds are hard lines.** A tiny difference either side gives a
  different verdict. Warning bands and persistence soften this, and the
  measured value is always shown.
- **Correlated checks.** In the press, a short rep also leaves the wrist beside
  the elbow, so a very short press can trigger both alignment and ROM
  ([press_analysis.md](press_analysis.md)).
- **Feedback isn't coaching.** Fixed templates with measured numbers - they
  can't adapt to your history, goals or equipment.

## 6. Measurement uncertainty

MediaPipe's error on a knee angle from a side camera is about **±10.7°** (near
leg) and **±25.1°** (far leg), and body widths vary by **5-34%** (Dill et al.,
2023, 2024). Clinical goniometry on a still patient resolves 6-14° (Hancock et
al., 2018). FormFix includes these errors in every verdict
([measurement_uncertainty.md](measurement_uncertainty.md)), but that doesn't
remove them:

- Differences under about 10° in a joint angle can't be resolved - they're
  reported as indicative.
- Two thresholds were below their own noise: the squat left/right warning (12°
  vs ±15.1°) - that rule was **removed** - and heel lift (0.06 vs ±0.15), which
  I kept in the defaults and raised in the literature preset. A heel finding is
  a prompt to look, not a measurement.
- The far leg from the side is never used for an angle.

## 7. Smoothing bias

Every filter makes the bottom of a squat look slightly shallower. With the
default EMA the worst case is **+4.5°** on a bouncy rep
([filter_selection.md](filter_selection.md)). It's included in the depth error,
but fast, bouncy squats will get more "shallow" findings than slow ones.

## 8. Not done yet

- **No strictness setting.** Simoes et al. (2024) let users pick their
  tolerance; FormFix uses one set for everyone. Rao et al. (2025) note that
  body proportions change how a squat looks, and the standing baseline only
  partly helps.
- **No user study,** so no claim that it improves anyone's technique (unlike
  Chae et al.'s 2023 trial). Friends testing the live app gave informal
  feedback, but that isn't a study.
- **No labelled-video evaluation.** The script is ready
  (`scripts/evaluate_videos.py`); the videos aren't recorded.
- **Person selection is by position, not identity.** The first frame picks the
  biggest person, so someone closer to the camera at the start would be picked
  and followed. The detector also only looks for four people.
- **The wrong-exercise check** can tell a video isn't the exercise picked, but
  not what it is. It only looks for clear contradictions, so some mislabelled
  videos will still get through.
- **One camera angle per video** - a clip that changes angle is flagged, not
  split up.
