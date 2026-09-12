# Dumbbell shoulder press analyser

What the press analyser measures, how it decides, where the thresholds come
from, and what it doesn't claim.

---

## 1. Which press

Written for the movement in the reference clip:

> **Seated dumbbell press, back supported, overhand grip, both arms together,
> from shoulder height to overhead.**

Arnold, neutral-grip, single-arm and standing presses move differently and the
rules would describe them wrongly.

---

## 2. Pipeline

```
video -> check file -> MediaPipe pose -> check recording -> drop impossible
 landmark jumps -> gap filling + EMA
 -> measure every frame -> count reps (shared state machine)
 -> per-rep measurements -> rules (camera-gated) -> reliability
 -> feedback -> annotated video -> JSON/CSV export
```

---

## 3. Camera

**Film from the front, level with your chest.** Two of the three checks compare
your arms, and from the side one arm hides the other. The wrist-over-elbow
check also can't work from the side, because that direction is depth. Only the
range-of-motion check works from any angle.

This is the opposite of the squat, and it's just a setting in the validation:

| Camera | Symmetry | Alignment | Range of motion |
| --- | --- | --- | --- |
| Front | assessed | assessed | assessed |
| Diagonal | lower reliability | lower reliability | assessed |
| Side | **not assessed** | **not assessed** | assessed |

For these front-view checks, a nearly side-on diagonal is the *bad* case, so
they use `1 - side_view_confidence`. (Using the same number as the side-view
checks gave the worst angles the highest confidence.)

The press also uses tighter camera bands than the squat (0.35 / 0.70), because
a straight front view only reaches a ratio of about 0.8 - with the squat's
bands, good front-on videos were classed as diagonal.

---

## 4. Landmarks

| Role | Landmarks | Used for |
| --- | --- | --- |
| Shoulders | 11, 12 | elbow angle, height reference, **shoulder width** |
| Elbows | 13, 14 | elbow angle, alignment |
| Wrists | 15, 16 | elbow angle, alignment, wrist height |
| Hips | 23, 24 | only a backup scale when the shoulders overlap |

The dumbbells aren't tracked. If one hides a wrist, that check is "not
assessed" rather than a wrong number.

---

## 5. Measurements

| Measurement | What it is |
| --- | --- |
| `elbow_angle` (each arm) | shoulder-elbow-wrist |
| `elbow_flexion` | 180° - elbow angle - **the signal used to count reps** |
| `elbow_angle_difference` | \|left - right\| |
| `wrist_height` (each arm) | height above the shoulders / shoulder width |
| `wrist_height_difference` | left - right (positive = left higher) |
| `alignment_offset` (each arm) | \|wrist x - elbow x\| / shoulder width |

**Why shoulder width:** these are all measured in the plane facing the camera,
and shoulder width is in that plane. Trunk length looks shorter as soon as you
lean. It also gives the thresholds a real meaning: shoulder width is about
0.40 m, so 0.38 shoulder widths ≈ 15 cm.

If you turn side-on the shoulder width shrinks towards zero, so the trunk
length is used as a floor - just to keep the exported numbers sensible.

---

## 6. Counting reps

The press moves **up** while the others move down, but using elbow **flexion**
(180° - angle) gives the same "high at rest, drops into the rep" shape, so it
reuses the shared state machine unchanged.

The thresholds are adaptive (same reason as the pulldown - someone who never
locks out still needs their reps counted):

```
reference = 90th percentile of flexion, clamped to 45-140°
rest = ref - 6°   start = ref - 14°   top = ref - 30°   end = ref - 10°
```

**Where top and bottom are measured:**

- The **bottom** is read in the rest frames either side of the rep, not the
  rep's first frame - the rep only starts once you're already pressing, which
  would hide a shallow press.
- The **top** is each arm's own best held value. If one arm arrives later, a
  shared window would give it a range fault for what is really a timing issue
  (which the symmetry rule already reports).

---

## 7. Rules

| | `press_symmetry` | `press_alignment` | `press_rom` |
| --- | --- | --- | --- |
| **Fault** | one arm leads, lags or moves less | wrist drifts away from over the elbow | range too short - says which end |
| **Camera** | front or diagonal | front or diagonal | any |
| **Measures** | elbow angle difference, wrist height difference, ROM difference | wrist-elbow offset per arm, worst one reported | elbow angle at top and bottom, and the range |
| **Thresholds** | angle warn 15° / fail 25°; height warn 0.12 / fail 0.20; ROM warn 15° / fail 25° | warn 0.38, fail 0.52 shoulder widths | top ≥ 155° (warn < 143°), bottom ≤ 100° (warn > 115°), range ≥ 50° |
| **Must last** | ≥ 5 frames and ≥ 25% of the phase | ≥ 5 frames and ≥ 25% | values held for 3 frames |
| **Needs** | both arms usable on ≥ 60% of the rep | one arm's elbow and wrist visible | visibility ≥ 0.4 |

Symmetry uses three separate signals instead of one score, because "your left
arm stayed lower" is useful and "symmetry index 0.72" isn't. The thresholds are
well above zero because nobody presses perfectly evenly and a few degrees is
inside MediaPipe's error. The top isn't 180° - a locked elbow under load isn't
the goal.

**Alignment and ROM overlap.** A press that stops well short of overhead leaves
the wrist beside the elbow, because the elbow is still bent. I widened the
alignment thresholds from 0.30/0.45 to 0.38/0.52 so moderate cases don't get
reported twice, but a very short press can still trigger both. Read that as one
habit reported twice.

**A hidden arm only costs symmetry.** My first version averaged both arms for
the alignment check, so a hidden arm switched it off for the visible one. Now
alignment and ROM work per arm.

**Not included:** trunk lean / arching the back (can't be seen from the front),
shoulder mobility, impingement, loading, injury risk, or anything medical.

---

## 8. Failures and the video

| Situation | Result |
| --- | --- |
| No person | `NO_POSE` |
| Arms not visible enough | `INSUFFICIENT_VISIBILITY` |
| One arm hidden | symmetry and alignment not assessed, ROM still reported |
| Side-on camera | analysed, symmetry and alignment not assessed |
| No complete rep | `NO_COMPLETE_REPETITION` |
| Doesn't look like an overhead press | `EXERCISE_MISMATCH` (e.g. a bench press - the wrists never go above the shoulders) |

A check that couldn't be assessed is never scored as a fail - a side-on video
with three good reps scores 100 on the one check it could do.

The annotated video draws the whole skeleton in one style, with the side away
from the camera fainter. Nothing changes colour when a rep is flagged - the
findings are worded on the results page instead.

---

## 9. Tests

| File | Covers |
| --- | --- |
| `test_common_phases.py` | the shared rep counter |
| `test_press_metrics.py` | measurements, normalisation, rep counting, per-rep values |
| `test_press_rules.py` | all three rules and camera gating |
| `test_press_integration.py` | the full pipeline, including false-positive tests |

The synthetic generator is aspect-corrected so a 90° elbow really measures 90°.
