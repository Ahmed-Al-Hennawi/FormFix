# What the labelled videos showed

The nine clips [evaluation/README.md](../evaluation/README.md) asks for, plus five extra ones, filmed on
a phone in September 2026 and run through `scripts/evaluate_videos.py`. The
numbers here come from [summary.md](../evaluation/summary.md), [results.csv](../evaluation/results.csv) and
[error_matrix.csv](../evaluation/error_matrix.csv); this file is what I make of them.

Run on 12 September 2026 with Python 3.11, MediaPipe 0.10.32, NumPy 2.4.4 and
OpenCV 4.13 on Linux (CPU). The earlier reference-clip runs used MediaPipe
0.10.14 on macOS, so the two are not directly comparable.

## The recordings

| Clip | Exercise | Camera | Labelled |
| --- | --- | --- | --- |
| `LP01_correct` | Lat pulldown | side | correct |
| `LP02_excessive_lean` | Lat pulldown | side | pulldown_torso |
| `LP03_limited_rom` | Lat pulldown | side | pulldown_rom |
| `LP04_front_view` | Lat pulldown | front-ish | correct (wrong camera angle on purpose) |
| `SP01_correct` | Shoulder press | front | correct |
| `SP02_asymmetric` | Shoulder press | front | press_symmetry |
| `SP04_incomplete_rom` | Shoulder press | front | press_rom |
| `SP05_side_view` | Shoulder press | side | correct (wrong camera angle on purpose) |
| `SQ01a_correct_weighted` | Squat | side | correct |
| `SQ01b_correct_bodyweight` | Squat | side | correct |
| `SQ02_to_parallel` | Squat | side | correct |
| `SQ03_heel_lift` | Squat | side | heel_lift |
| `SQ04_setup_heel_lift` | Squat | side | heel_lift |
| `XX01_wrong_exercise` | Bench press, run as a shoulder press | side | correct (should be refused) |

`SP03_elbow_misalignment` was never filmed, so `press_alignment` has no
positive case and is untested.

**Two clips were relabelled after I watched them next to the measurements.**
`SQ02` was filmed as a shallow squat, but at the bottom the thighs are at
parallel and the knee angle is 90°, the same depth as the reps in
`SQ01a` (86°, 84°, 96°). It is a correct squat, so that is what it is labelled.
`SQ04` was filmed as a long setup followed by clean reps, but the heel comes
off the platform at 0:04, so it is labelled as a heel lift. Relabelling after
seeing the measurements is a real risk to the evaluation, so: the labels were
changed to match what the video shows, not to match what the system said, and
no threshold was changed because of them.

## What happened

13 of 14 recordings were analysed. One was refused. Across 48 rule-by-video
cells: **3 true positives, 2 false positives, 2 false negatives, 35 true
negatives, 4 not assessed and 2 not analysed**.

| Rule | TP | FP | FN | TN | Not assessed |
| --- | --- | --- | --- | --- | --- |
| pulldown_rom | 0 | 0 | 1 | 2 | 0 |
| pulldown_torso | 0 | 1 | 0 | 2 | 0 |
| press_symmetry | 1 | 0 | 0 | 2 | 2 |
| press_alignment | 0 | 0 | 0 | 3 | 2 |
| press_rom | 1 | 1 | 0 | 3 | 0 |
| squat_depth | 0 | 0 | 0 | 5 | 0 |
| torso_lean | 0 | 0 | 0 | 5 | 0 |
| heel_lift | 1 | 0 | 1 | 3 | 0 |
| return_to_standing | 0 | 0 | 0 | 5 | 0 |
| descent_control | 0 | 0 | 0 | 5 | 0 |

There is no accuracy percentage here. Fourteen clips of one person in one
session cannot support one, and most of the true negatives come from rules that
had nothing to find.

## What worked

- **No false finding on any of the three correct squats.** All five squat rules
  passed on `SQ01a`, `SQ01b` and `SQ02`.
- **Setup is kept out of the baseline on real footage.** The four squat clips
  where I walk into position dropped 56, 28, 24 and 92 frames of unsettled
  standing before measuring the baseline. Three produced no finding at all; the
  fourth produced only the heel lift that is visible in the frame.
- **`press_symmetry` and `press_rom` both fired on the clip they were meant
  to.**
- **The heel rule separated the clean squats from a real lift.** Silent on all
  three correct squats, fired on `SQ04`.
- **Wrong camera angles were handled rather than ignored.** `SP05`, a press
  filmed from the side, had symmetry and alignment marked "not assessed"
  instead of guessed. `LP04` was marked as a limited recording.

## What it found

### 1. The pulldown range bar cannot separate a full rep from a short one

| Clip | Elbow at the top | Elbow at the bottom | Range |
| --- | --- | --- | --- |
| `LP01` full reps | 175-178° | **52-55°** | 121-125° |
| `LP03` deliberately short reps | 174° | **73-74°** | 100-101° |

The rule passes a rep whose elbow closes to 100° or less. Both clips clear that
bar by a wide margin, so the check cannot tell them apart, and `LP03` came back
"correct". The measurement is fine - the two cases are 20° apart - the
threshold is simply in the wrong place.

### 2. The pulldown trunk bar flags normal technique

| Clip | Trunk movement from the top position |
| --- | --- |
| `LP01` normal reps | **25-27°** |
| `LP03` short reps | 13-15° |
| Current warning bar | 15° |

`LP01` is a clean pulldown and every rep failed the trunk check. A lat pulldown
involves some backward lean, and 15° is inside it.

### 3. A real heel lift was missed by the persistence requirement

`SQ03` measured a heel rise of **0.160** against a 0.16 bar, held for **10.1%**
of the rep. The rule needs 15% of the rep, so it passed. The setting was raised
from 10% to 15% shortly before this run on the reasoning that a real lift holds
longer than noise; on this clip a real lift held 10.1%.

### 4. A fault can be thrown away with the repetition that contained it

In `SQ03` the clearest heel lift is around 0:02. That movement was discarded as
a partial repetition ("tracking was lost during the movement"), so the heel rule
never saw it. Only one of the two attempts was scored.

### 5. A fault the rep detector cannot see is refused rather than reported

`LP02` was refused with "no complete repetition". The clip contains a trunk
excursion of **51°**, against 25-27° in the normal `LP01`, so the fault is there
and it is large. But the elbow moves only 34°, against 121-125° in a real rep:
I leaned back instead of pulling. Repetitions are defined on elbow flexion, so a
rep where the lean replaces the pull never becomes a repetition, and the trunk
rule never runs. Gating a pulldown rep on wrist travel as well as elbow angle
would let this clip be judged.

### 6. Two guard rails let a recording through

- `XX01` is a **bench press**, analysed as a shoulder press. It was accepted:
  2 repetitions, score 100. The plausibility check has no test for lying down.
- `SP05` is a press filmed from the side. Symmetry and alignment were correctly
  not assessed, and the one remaining check passed - so the result reads
  **100/100**. The score is the share of *evaluable* checks that passed, so
  fewer checks make a better-looking score.

`SP02` also fired `press_rom`, which the label does not list. Looking at the
measurements, two of its four repetitions reached only 108° and 112° of elbow
extension against 160-165° on the others, so the finding is defensible; one arm
lagging far enough also shortens the range.

## What this does not show

- No accuracy rate, for the reasons above.
- `press_alignment` was never exercised by a positive case.
- Every clip is one person, one phone, one session. Nothing here says how
  FormFix behaves on a different body, gym or camera.
- Three clips filmed front-on were classified as `diagonal_side`, which
  downgraded them to "limited" and medium reliability. The verdicts were still
  right, but the camera-angle classifier is approximate.

## Measured values, for threshold work

Recorded here so the next change to a threshold has something behind it. Two or
three clips per case is not enough to set a value on its own
(see [threshold_tuning.md](threshold_tuning.md)).

| Measurement | Clean clips | Faulty clips | Current bar |
| --- | --- | --- | --- |
| Pulldown elbow at the bottom | 52-55° | 73-74° | pass at ≤100° |
| Pulldown trunk movement | 25-27° | 13-15° (short reps), 51° (lean instead of pull) | warn at 15° |
| Squat knee angle at the bottom | 61-96° | - | pass at ≤100° |
| Squat heel rise | not flagged | 0.160 (missed), 0.202-0.296 (caught) | warn at 0.16, held 15% of the rep |
| Press elbow at the top | 160-168° | 100-113° | pass at ≥150° |

## Reproducing this

```bash
python scripts/evaluate_videos.py evaluation/labels.csv
```

The clips go in `evaluation/videos/` under the names in
[labels.csv](../evaluation/labels.csv). They are not in the repository.
