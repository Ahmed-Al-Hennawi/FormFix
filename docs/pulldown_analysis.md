# Lat pulldown analyser

What the pulldown analyser measures, how it decides, where the thresholds come
from, and what it doesn't claim. See also [squat_analysis.md](squat_analysis.md),
[press_analysis.md](press_analysis.md), [threshold_tuning.md](threshold_tuning.md)
and [limitations.md](limitations.md).

---

## 1. Which pulldown

It's written for the movement in the reference clip:

> **Seated, both arms, overhand grip, bar pulled in front of the head to the
> upper chest.**

Behind-the-neck, close-grip, single-arm or standing pulldowns move differently
and the rules would describe them wrongly.

---

## 2. Pipeline

```
video -> check file -> MediaPipe pose -> check recording -> gap filling + EMA
 -> pick side -> measure every frame -> count reps (shared state machine)
 -> top-position baseline -> per-rep measurements -> rules (camera-gated)
 -> reliability -> feedback -> annotated video -> JSON/CSV export
```

Measuring never judges, judging never measures, and only the feedback step
writes text.

---

## 3. Camera

**Film from the side, level with your chest.** You can turn the camera a bit
towards your back, but not towards the front, where the machine, bar and
weight stack get in the way. Past about 135° the head is lost and the torso
rule can't tell which way you leaned.

The torso lean happens towards/away from the camera if you film from the front,
so one camera can't see it. The elbow angle is visible from any angle.

| Camera | Range of motion | Torso movement |
| --- | --- | --- |
| Side | assessed | assessed |
| Three-quarter | assessed | assessed, lower reliability |
| Front / back | assessed | **not assessed** (reason shown) |
| Unknown | lower reliability | lower reliability |

Before upload the page shows three short tips and a diagram of where to put
the phone (`QUICK_TIPS` in `config.py`, drawn by `utils/camera_guide.py`). The
full `RECORDING_TIPS` are shown if a video gets rejected.

---

## 4. Landmarks

| Role | Landmarks | Used for |
| --- | --- | --- |
| Shoulders | 11, 12 | elbow angle, trunk, height reference |
| Elbows | 13, 14 | elbow angle, elbow travel |
| Wrists | 15, 16 | elbow angle, wrist travel |
| Hips | 23, 24 | trunk, body size |
| Head | 0, 7, 8 | only to work out which way you're facing |

**The bar isn't tracked** - MediaPipe only tracks the body. So there's no "bar
touched the chest" rule; everything is based on body landmarks.

---

## 5. Measurements

All in pixels (normalised coordinates distort angles on phone videos).

| Measurement | What it is |
| --- | --- |
| `elbow_angle` | shoulder-elbow-wrist, averaged over the usable arms |
| `torso_angle` | hip → shoulder vs vertical |
| `torso_posterior` | the same, positive = leaning back |
| `torso_excursion` | change from your own top-position posture |
| `torso_velocity` | trunk speed (exported, no rule uses it) |
| `wrist_rise` / `elbow_rise` | height above the shoulders / trunk length |

**Why change and not posture:** a seated pulldown normally has a small lean,
and the seat often sets it. Comparing to vertical would flag correct technique,
so the trunk is compared to **your own top position** (median over real top
frames, not frame 0 when you're still reaching for the bar).

**Why the head matters:** an image can't tell leaning back from leaning forward
without knowing which way you face. The head sits in front of the shoulders,
so that gives the direction. If the head isn't visible, the lean has no
direction and the feedback says "moved" instead of "leaned back".

---

## 6. Counting reps

The shared state machine (`exercises/common/phases.py`) runs on the elbow
angle:

```
TOP (arms straight) -> PULLING -> BOTTOM -> RETURNING -> TOP
```

**The thresholds are adaptive.** With fixed levels, someone who never
straightens their arms would never count a rep - so the ROM rule that's meant
to catch that would never run. So the levels are margins below your own
resting extension:

```
reference = 90th percentile of the elbow angle, clamped to 110-180°
rest = ref - 6°   start = ref - 14°   bottom = ref - 30°   end = ref - 10°
```

"Was this a rep?" uses these adaptive levels; "was the range enough?" uses the
fixed `ROM_*` thresholds.

---

## 7. Rules

| | `pulldown_rom` | `pulldown_torso` |
| --- | --- | --- |
| **Fault** | range too short - says which end | swinging the body to move the weight |
| **Camera** | any | side or three-quarter |
| **Measures** | elbow angle at top and bottom, and the range between | trunk change from your top position |
| **Thresholds** | top ≥ 150° (warn < 138°), bottom ≤ 100° (warn > 115°), range ≥ 45° | warn ≥ 15°, fail ≥ 25°, or 45° from vertical |
| **Must last** | values held for 3 frames | ≥ 4 frames and ≥ 20% of the pull |
| **Needs** | the analysed arm usable on ≥ 50% of the rep | shoulders and hips visible (≥ 0.5) |

The ROM rule gives a category (`complete_rom`, `limited_top_extension`,
`limited_bottom_range`, `limited_overall_rom`) because "limited range" alone
doesn't tell a beginner what to change. Wrist travel is shown as extra
evidence, but never passes a rep on its own.

**Why ROM only needs one arm:** from the side the far arm is hidden for most of
the pull. My first version needed both arms and said "not assessed" on the
exact camera angle I recommend - I only found this on real footage
([reference_clip_runs.md](../evaluation/reference_clip_runs.md)).

---

## 8. Not included

- **Swinging between reps.** Trunk speed is measured and exported, but I didn't
  have labelled videos to test a rule on, so it's future work.
- Shoulder blade movement, lat activation, grip width, bar to chest, spine
  position, loading, injury risk, or anything medical.

---

## 9. Reliability, failures and the video

Reliability works the same as the squat: High / Medium / Low / Cannot assess,
per measurement.

| Situation | Result |
| --- | --- |
| No person | `NO_POSE` |
| Arms not visible enough | `INSUFFICIENT_VISIBILITY` |
| Front-on camera | analysed, torso check not assessed |
| Hips hidden by the seat | analysed, torso check not assessed |
| No complete rep | `NO_COMPLETE_REPETITION` |
| Doesn't look like a pulldown | `EXERCISE_MISMATCH` (e.g. a row - the hands never start above the shoulders) |

A failed analysis never gives technique feedback.

The annotated video only draws the upper body (shoulders, elbows, wrists, hips,
head) - the legs are under the seat pad where tracking is worst. Joints are
ringed while a finding is shown.

---

## 10. Tests

| File | Covers |
| --- | --- |
| `test_common_phases.py` | the shared rep counter |
| `test_geometry_upper_body.py` | signed angles and angular speed |
| `test_pulldown_metrics.py` | measurements, facing direction, baseline, rep counting |
| `test_pulldown_rules.py` | both rules and camera gating |
| `test_pulldown_integration.py` | the full pipeline, including false-positive tests |

The synthetic generator is aspect-corrected so a 172° elbow really measures
172°, and a test checks that.
