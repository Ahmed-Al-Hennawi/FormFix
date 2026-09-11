# Evaluation

This folder is for evaluating FormFix on labelled videos. The app doesn't use
it.

## 1. Record the videos

Film each clip as one set of a few controlled reps, **in the camera angle the
exercise needs**. If you film from the wrong angle the clip still gets
analysed, but the affected checks will say "not assessed", so it can't test
them.

| Clip | Exercise | Camera | What to do |
| --- | --- | --- | --- |
| `LP01_correct` | Lat pulldown | Side | Normal reps: arms straight at the top, bar to the upper chest, torso still |
| `LP02_excessive_lean` | Lat pulldown | Side | Same, but swing the torso back on every pull |
| `LP03_limited_rom` | Lat pulldown | Side | Short reps: stop the pull early and don't straighten the arms |
| `SP01_correct` | Shoulder press | Front | Normal reps: both arms together, shoulders to overhead |
| `SP02_asymmetric` | Shoulder press | Front | Let one arm lead and finish higher |
| `SP03_elbow_misalignment` | Shoulder press | Front | Let one wrist drift out to the side |
| `SP04_incomplete_rom` | Shoulder press | Front | Stop short of overhead and don't lower fully |
| `SQ01_correct` | Squat | Side | Full-depth reps, start and finish standing |
| `SQ02_shallow` | Squat | Side | Stop well above the target depth |

Put **one** fault in each clip, otherwise one clip tests two rules at once.
Two or three clips per case is better than one. The goal isn't an accuracy
percentage (the sample is far too small for that) - it's a clear record of
what was detected, missed, or not assessed.

## 2. Write the labels file

Copy `labels.example.csv` to `labels.csv` and edit it. `expected` is either
`correct` or the rule ids of the fault, separated by `;`:

| Exercise | Rule ids |
| --- | --- |
| Squat | `squat_depth`, `torso_lean`, `heel_lift`, `return_to_standing`, `descent_control` |
| Lat pulldown | `pulldown_rom`, `pulldown_torso` |
| Shoulder press | `press_symmetry`, `press_alignment`, `press_rom` |

## 3. Run it

```bash
python scripts/evaluate_videos.py evaluation/labels.csv
```

It writes three files next to the labels file:

- `results.csv` - one row per video (label, what was detected, reps, score,
  reliability, camera angle, every rule's result)
- `error_matrix.csv` - one row per video and rule: TP / FP / FN / TN / NOT_ASSESSED
- `summary.md` - both tables, ready for the report

Useful options: `--preset literature`, `--strict-uncertainty`,
`--filter savgol`, and `--export` for the full JSON/CSV of each run.

## 4. Reading the results

- **NOT_ASSESSED isn't a miss.** The video couldn't support that check and the
  system said so. It has its own column.
- **A rejected video isn't a miss either.** It's listed separately in
  `summary.md` with the reason.
- **Don't tune thresholds until every clip passes.** With a few videos that's
  overfitting. If a threshold changes, write down why in
  `docs/threshold_tuning.md`.

These videos haven't been recorded yet, so there's no accuracy figure for
FormFix.
