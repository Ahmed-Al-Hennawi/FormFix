# Thresholds: where they came from and what changed

Every technique threshold in FormFix is a **prototype value** I chose from the
exercise definition, the reference clip and testing. None of them is a
biomechanical constant, and the app always calls them "the configured range".
Each rule's `threshold_source` shows up in debug mode and the export.

This is the log of what each value started at, why, what testing showed, and
where it ended up. If a threshold changes, it should get a line here.

---

## 1. Two kinds of value

| Kind | Controls | Biomechanically defensible? |
| --- | --- | --- |
| **Engineering** | detection, gaps, rep counting, persistence, plausibility, reliability | Doesn't need to be - they only control noise. |
| **Technique** | depth, lean, heel, symmetry, alignment, range of motion | Only as an operational definition of what FormFix looks for. |

So in the report I can say *"an operational threshold was defined from the
exercise definition, the reference demonstration, published movement
principles and testing"* - but not "the correct pulldown trunk angle is 15°",
because no paper I reviewed gives one for these movements and camera angles.

---

## 2. Rep counting is not technique

The rep counter originally used fixed angles (pulldown: rest 148°, start 140°,
bottom 115°, end 143°). Testing with a synthetic person whose arms only
straightened to 132° showed a loop: they never reached the "extended" level,
so no rep was counted, so the range-of-motion rule meant to catch exactly that
never ran.

So the levels are now **adaptive** - fixed margins below the person's own
resting position. "Was this a rep?" is adaptive; "was the range enough?" uses
the fixed `ROM_*` values. Both are pinned by tests
(`test_a_restricted_athlete_still_has_their_repetitions_counted`,
`test_an_athlete_who_never_locks_out_still_has_reps_counted`).

---

## 3. Lat pulldown

### Torso movement (`pulldown_torso`)

| Threshold | Value | Why | Test result |
| --- | --- | --- | --- |
| `TORSO_EXCURSION_WARN` | 15° | The reference clip keeps about 10-20° of lean with only a few degrees of change; 15° is well past that plus MediaPipe error. | 0° → 3.8° pass; 18° → 17.3° warn; 28° → 26.9° fail |
| `TORSO_EXCURSION_FAIL` | 25° | About 1.7× the warning. | as above |
| `TORSO_ABSOLUTE_FAIL` | 45° | 45° from vertical isn't really the exercise any more. | never fired on a correct rep |
| `TORSO_MIN_FRAMES` | 4 | ~0.13 s at 30 fps, longer than jitter. | no false positives on noisy data |
| `TORSO_MIN_VIOLATION_RATIO` | 0.20 | A fifth of the pull is movement, not noise. | as above |

### Range of motion (`pulldown_rom`)

| Threshold | Value | Why | Test result |
| --- | --- | --- | --- |
| `ROM_TOP_EXTENSION_PASS` / `WARN` | 150° / 138° | Arms back towards straight, but not locked out. | 172° pass, 142° warn, 125-132° fail |
| `ROM_BOTTOM_FLEXION_PASS` / `WARN` | 100° / 115° | The reference clip ends the pull around a right angle. | 80° pass, 108° warn, 125° fail |
| `ROM_MIN_EXCURSION` | 45° | Catches reps that are a bit short at both ends. | only fires in combination |

The camera bands are the same as the squat (0.45 / 1.00) since both want a side view.

**None of these have been tested on labelled real videos yet** - the synthetic
tests check the code, not whether the values suit real people.

---

## 4. Shoulder press

### Camera bands - changed

| Threshold | Started | Now | Why |
| --- | --- | --- | --- |
| `SIDE_VIEW_GOOD_RATIO` | 0.45 | **0.35** | |
| `SIDE_VIEW_FRONTAL_RATIO` | 1.00 | **0.70** | A straight front-on synthetic video measured 0.71 and was classed as diagonal, so every left/right check got lower reliability. Shoulder width is only about 0.8-0.9 of trunk length. |

This fixed a misclassification - no verdict changed, only the reliability label
(`test_a_square_front_view_is_recognised_as_frontal`).

### Symmetry (`press_symmetry`)

| Threshold | Value | Why | Test result |
| --- | --- | --- | --- |
| `SYMMETRY_ANGLE_WARN` / `FAIL` | 15° / 25° | A few degrees is normal and inside MediaPipe's error. | 4% lag → 3.3° pass; 28% lag → 23.2° fail |
| `SYMMETRY_HEIGHT_WARN` / `FAIL` | 0.12 / 0.20 shoulder widths | ≈ 5 cm / 8 cm - visible, well above noise. | in sync < 0.01; 28% lag → 0.22 |
| `SYMMETRY_ROM_WARN` / `FAIL` | 15° / 25° | Matches the angle thresholds. | right arm capped at 130° → flagged |
| `SYMMETRY_MIN_FRAMES` / `RATIO` | 5 / 0.25 | ~0.17 s at 30 fps. | no false positives on noise |

### Alignment (`press_alignment`) - changed

| Threshold | Started | Now | Why |
| --- | --- | --- | --- |
| `ALIGNMENT_OFFSET_WARN` | 0.30 | **0.38** | A press stopping at 138° with *no* wrist drift measured 0.34 and got flagged, because a bent elbow puts the wrist beside it. That reported one habit as two faults. |
| `ALIGNMENT_OFFSET_FAIL` | 0.45 | **0.52** | Widened by the same amount. |

After the change: no-drift short press → 0.34 pass; real drift → 0.55 fail. I
changed it to remove a false positive, not to make a test pass.

### Range of motion (`press_rom`)

| Threshold | Value | Why | Test result |
| --- | --- | --- | --- |
| `ROM_TOP_EXTENSION_PASS` / `WARN` | 155° / 143° | Near full extension, but not 180°. | 168° pass, 148° warn, 138° fail |
| `ROM_BOTTOM_FLEXION_PASS` / `WARN` | 100° / 115° | Dumbbells back to about shoulder height. | 85° pass, 108° warn, 125° fail |
| `ROM_MIN_EXCURSION` | 50° | Short at both ends. | only fires in combination |

---

## 5. Squat

The squat's technique thresholds haven't changed (see
[squat_analysis.md](squat_analysis.md) §14). One fix: front-view metrics now
use `1 - side_view_confidence` on diagonal videos. Before, the worst diagonal
videos for a left/right comparison got the *highest* confidence. I found it on
the old knee-evenness rule; that rule is gone now, but the press relies on the
fix (`TestFrontalPlaneMetricsOnADiagonalCamera`).

---

## 6. Gates fixed after real-video testing

These decide whether a check runs at all. Both are in
[reference_clip_runs.md](../evaluation/reference_clip_runs.md).

| Gate | Was | Now | Why |
| --- | --- | --- | --- |
| Pulldown `arms_reliable` | both arms usable ≥ 50% | the analysed arm ≥ 50% | From the side the far arm is hidden, so ROM said "not assessed" on the recommended angle. |
| Press `alignment_reliable` | both arms' visibility averaged | the best arm ≥ 0.5 | A hidden arm switched the check off for the visible one. |

---

## 7. Measurement uncertainty

After the literature review I added MediaPipe's published error to every
finding ([measurement_uncertainty.md](measurement_uncertainty.md)). It showed
two thresholds were below their own measurement noise:

| Threshold | Value | Measurement error |
| --- | --- | --- |
| `KNEE_SYMMETRY_WARN` (squat, **removed**) | 12° | ±15.1° |
| `HEEL_LIFT_THRESHOLD` (now 0.16) | 0.06 | ±0.15 |
| depth warning band | 15° | ±11.6° (similar) |

- I **removed** the squat evenness rule instead of raising it above 15°, since
  by then it would only catch differences you can already see. The measurement
  is still exported. The press keeps its symmetry rule because it's filmed
  front-on with both arms visible.
- I **left the heel threshold** alone at first (no labelled videos to justify
  a change) and raised it in the literature preset instead. In September 2026 I
  raised the default to **0.16** as well, after real clips confirmed the old bar
  warned on planted heels; see §9.
- The filter's own depth bias is added to the depth band:
  √(10.7² + 4.53²) = 11.6°.

---

## 8. The literature preset

`exercises/squat/literature_config.py` is a second squat config based on
papers, kept for **comparison**, not as a replacement:

| Threshold | Default | Literature | From |
| --- | ---: | ---: | --- |
| `DEPTH_KNEE_ANGLE_PASS` | 100° | 81° | Kotiuk et al. (2022, via Rao et al., 2025): 113 ± 7° flexion = 67 ± 7° interior; upper end used |
| `DEPTH_KNEE_ANGLE_WARN` | 115° | 101° | Dill et al. (2024): correct reps ~20° from faulty ones |
| `TORSO_LEAN_FAIL` | 60° | 55° | Dill et al. (2024) "excessive forward bending" fault |
| `HEEL_LIFT_THRESHOLD` | 0.16 | 0.16 | above the ±0.15 noise floor (the default now matches) |
| `FULL_EXTENSION_TOLERANCE` | 12° | 20° | Simoes et al. (2024) beginner allowance |

Engineering values aren't touched - no paper covers EMA weights, and changing
them under a citation would be misleading (`test_literature_preset.py` checks
this).

```bash
python scripts/evaluate_videos.py manifest.csv --preset default
python scripts/evaluate_videos.py manifest.csv --preset literature
python scripts/evaluate_videos.py manifest.csv --strict-uncertainty
```

On three synthetic 105° squats the three policies disagree:

| Policy | Depth | Score |
| --- | --- | --- |
| Default, annotate only | warning | 90 |
| Default, strict uncertainty | pass | 100 |
| Literature preset | fail | 83 |

---

## 9. What the labelled videos measured (September 2026)

Fourteen labelled clips, written up in
[evaluation_results.md](evaluation_results.md). One threshold changed because of
the literature (§7); the values below are what the clips actually measured, so
the next change to a threshold has something behind it. **Nothing here was
tuned to these clips** - two or three recordings of one person can't set a
value.

| Measurement | Clean clips | Faulty clips | Current bar |
| --- | --- | --- | --- |
| Pulldown elbow at the bottom | 52-55° | 73-74° | pass at ≤ 100° |
| Pulldown trunk movement | 25-27° | 13-15° (short reps), 51° (lean instead of pull) | warn at 15° |
| Squat knee angle at the bottom | 61-96° | - | pass at ≤ 100° |
| Squat heel rise | not flagged | 0.160 (missed), 0.202-0.296 (caught) | warn at 0.16, held 15% of the rep |
| Press elbow at the top | 160-168° | 100-113° | pass at ≥ 150° |

Two of these are clearly in the wrong place: the pulldown range bar sits
outside the range both classes produce, and the pulldown trunk bar is below
what normal technique measures. I have left both alone and written down why.

---

## 10. What I can't claim

- **No threshold is tuned to labelled real videos.** The clips in §9 measure
  where the thresholds sit; they are too few to move them.
- The real-video runs are verification only - the reference clips are
  animated montages, and no threshold was changed because of them.
- **There is no accuracy figure**, and fourteen clips of one person could not
  support one ([evaluation_results.md](evaluation_results.md)).
- **Don't tune thresholds until every test video passes** - with a few videos
  that's overfitting.
- The literature preset isn't validated either. A citation isn't a validation.
- The uncertainty bands are RMSE used as a scale, not a probability.
- The filter comparison used synthetic data (you need a known true value to
  measure bias).
