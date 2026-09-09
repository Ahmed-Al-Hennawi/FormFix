# Technical evaluation

This folder is where the labelled recordings and their results live. It is not
used by the application at runtime.

## 1. Record the videos

Record each clip as a single set of several controlled repetitions, in the
camera view the exercise needs. The view is not a detail: two of the three
shoulder-press checks are comparisons between the arms and cannot be made from
the side, and the lat-pulldown trunk check is a sagittal measurement that
cannot be made from the front. A recording in the wrong view will be analysed,
but the affected checks will report "not assessed" - which is correct
behaviour, and which makes the clip useless as a test of those checks.

| Clip | Exercise | Camera | Perform |
| --- | --- | --- | --- |
| `LP01_correct` | Lat pulldown | Side or three-quarter | Several controlled reps: arms extended at the top, bar pulled to the upper chest, torso reasonably still |
| `LP02_excessive_lean` | Lat pulldown | Side or three-quarter | Same range, but deliberately swing the torso backwards on every pull |
| `LP03_limited_rom` | Lat pulldown | Side or three-quarter | Deliberately short reps - stop the pull early, and do not let the arms straighten between reps |
| `SP01_correct` | Shoulder press | Front-on | Several controlled reps: both arms together, dumbbells to shoulder height and pressed overhead |
| `SP02_asymmetric` | Shoulder press | Front-on | Deliberately let one arm lead and finish higher than the other on every rep |
| `SP03_elbow_misalignment` | Shoulder press | Front-on | Deliberately let one wrist drift out to the side, away from the line of its forearm |
| `SP04_incomplete_rom` | Shoulder press | Front-on | Deliberately stop short of overhead, and do not lower the dumbbells back to the shoulders |
| `SQ01_correct` | Squat | Side-on | Several controlled full-depth reps, starting and finishing standing |
| `SQ02_shallow` | Squat | Side-on | Deliberately stop well above the depth target |

Record each fault clip with **one** deliberate fault. A clip with two faults
still works, but its row then tests two rules at once and cannot separate them.

Two clips per condition is better than one; three is better still. The point of
the exercise is not a headline accuracy figure - the sample is far too small to
support one - but a defensible account of what the system detected, what it
missed, and what it declined to judge.

## 2. Write the manifest

Copy `labels.example.csv` to `labels.csv` and edit it to match the files you
recorded. `expected` is `correct`, or one or more rule ids separated by `;`:

| Exercise | Rule ids |
| --- | --- |
| Squat | `squat_depth`, `torso_lean`, `heel_lift`, `return_to_standing`, `descent_control` |
| Lat pulldown | `pulldown_rom`, `pulldown_torso` |
| Shoulder press | `press_symmetry`, `press_alignment`, `press_rom` |

## 3. Run the harness

```bash
python scripts/evaluate_videos.py evaluation/labels.csv
```

It writes three files beside the manifest:

* `results.csv` - one row per recording: expected label, detected checks, rep
  count, score, reliability, recording quality, camera orientation, the status
  and reliability of every rule, and the per-repetition measurements;
* `error_matrix.csv` - one row per (recording x rule), labelled TP / FP / FN /
  TN / NOT_ASSESSED;
* `summary.md` - both tables in Markdown, ready for the evaluation chapter.

Add `--export` to also write the full JSON/CSV analysis bundle for each run.

## 4. Reading the results honestly

**`NOT_ASSESSED` is not a miss.** It means the recording could not support that
measurement and the system said so rather than guessing. Counting it as a false
negative would penalise exactly the behaviour the design is built around. It is
tallied in its own column for that reason.

**A recording that was not analysed at all is not a miss either.** It is
reported in its own section of `summary.md`, with the typed failure code, so
the evaluation can distinguish "the system was wrong" from "the system refused
an unusable recording".

**Do not tune thresholds until every clip passes.** With a handful of
recordings that is overfitting, not calibration, and it would make the
resulting numbers meaningless. If a threshold does move, record the change and
the reason in `docs/threshold_tuning.md` so the development history is
auditable.
