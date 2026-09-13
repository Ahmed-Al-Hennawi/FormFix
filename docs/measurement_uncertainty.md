# Measurement uncertainty

## Why this is here

Every rule compares a measurement to a threshold. The thresholds were already
labelled provisional; this file extends the same treatment to the other side of
the comparison, the **measurement** itself, which is the less certain of the two.

MediaPipe's error on the quantities FormFix measures has been published, and it
is large enough next to some thresholds to be worth building around:

| Measurement | Published error | Where FormFix uses it |
| --- | --- | --- |
| Knee angle, near leg, side camera | RMSE **10.7°** | depth (pass 100°, warn 115°) |
| Knee angle, far (hidden) leg | RMSE **25.1°** | not used - this is why |
| Knee angle in squats, good camera angle | RMSE **9.14°** (14.48° at a bad angle) | depth |
| Shoulder / hip width | RMSE **5.7-29.8%** / **5.0-34.4%** of the true width | every normalised length |

Sources: Dill et al. (2024), *Sensors* 24(23):7772; Dill et al. (2023),
*Curr Dir Biomed Eng* 9(1):563-566.

A rep measured at 108° and one measured at 98° therefore sit inside the same
error band.

For comparison, Hancock et al. (2018, quoted in Dill et al.) give the smallest
difference clinical tools can detect on a patient lying still: 6° with a
digital inclinometer, 10° with a long-arm goniometer, 14° by eye. FormFix's
10.7° sits between a goniometer and a physio's eye, which is a reasonable place
for a single phone camera to land - and it sets the smallest difference the
system should claim to see.

Mercadal-Baudart et al. (2024) turn that into a rule: under **12°** error is
"good" (better than by eye) and under **6°** is "very good". Against that:

| FormFix measurement | Error | Verdict |
| --- | --- | --- |
| Trunk angle | ±6.5° | good |
| Knee angle, near leg (depth) | ±10.7° (±11.6° with filter bias) | good |
| Left/right knee difference | ±15.1° | outside the band - no rule relies on it |
| Knee angle, far leg | ±25.1° | outside the band - no rule relies on it |

## What FormFix does with it

`exercises/common/uncertainty.py` works out the **margin** for each flagged
rep - how far past the threshold it actually was - and compares it to the error:

- **beyond measurement error** - the finding stands.
- **inside measurement error** - still reported, but marked as *indicative,
  not certain*.

The full details (error, source, margins) go into the rule's evidence, the
debug panel and the export.

### Two modes

- **Default:** it only adds the note. No verdict or score changes.
- **`UNCERTAINTY_STRICT = True`** (or `--strict-uncertainty`): a borderline
  finding also drops one level (FAIL → WARNING, WARNING → PASS).

I kept both so they can be compared on the same videos. On three synthetic 105°
squats, the options differ by 17 points:

| Policy | Depth | Score |
| --- | --- | --- |
| Default, annotate only | warning | 90 |
| Default, strict | pass | 100 |
| Literature preset | fail | 83 |

## Where each number comes from

| Error band | Value | Source |
| --- | --- | --- |
| Joint angle, near limb | ±10.7° | Dill et al. (2024), Fig. 8 |
| Joint angle, far limb | ±25.1° | Dill et al. (2024), Fig. 8b |
| Squat knee angle, good view | ±9.14° | Dill et al. (2023), Tab. 3 |
| Trunk angle | ±6.5° | **my derivation**: atan(56.3 mm / 500 mm) from the median landmark error and an adult torso length |
| Left/right difference | ±15.1° | **my derivation**: √2 × 10.7° (difference of two measurements) |
| Normalised length | ±0.15 | **my derivation**: middle of the published 5-34% range |
| Duration | ±0.067 s | **my derivation**: two frames at 30 fps |

The derivations are written out in the code too. Two adjustments on top:

- **Camera angle:** Dill et al. (2023) saw the error go from 9.14° to 14.48°
  between a good and bad angle (×1.58). I scale the band linearly up to that
  as the view gets worse. The linear scaling is my own choice, and it's only
  used for angles.
- **Filter bias:** the smoothing filter's own measured bias on depth is added,
  so the default depth band is √(10.7² + 4.53²) = 11.6°
  ([filter_selection.md](filter_selection.md)).

## How this shaped the thresholds

Two thresholds sat inside their own error band, and putting the bands on paper
is what made that visible:

1. **The squat left/right warning (12°) sat below its own error (±15.1°)**, so
   it could not separate a genuine asymmetry from noise. The rule was
   **removed** and the measurement is still exported. The press keeps its
   symmetry rule, because it is filmed front-on with both arms visible.
2. **The heel-lift threshold (0.06) sat below the ±0.15 error** of the body
   length it is divided by, so a planted heel could fall inside the flagging
   band. It is now **0.16**, clear of the band and the same value as the
   [literature preset](../exercises/squat/literature_config.py) - the one
   threshold where my default and the preset agree.

This is a deliberate trade of sensitivity for confidence, and worth stating
plainly: 0.16 of a lower leg is roughly 6 cm of heel rise, so a small genuine
lift is not reported. From one 2D camera a lift that size cannot be separated
from the noise, so the extra sensitivity of the old bar was apparent rather than
real. A heel finding stays a prompt to look, not a measurement. Tests in
`test_uncertainty.py` fail if the threshold is ever moved back inside the band.

## Scope

- The margins are error bands, not probabilities: the papers report RMSE rather
  than a distribution, so the output stays in those terms.
- It applies to single-margin rules. The two range-of-motion rules combine three
  criteria in different units, so there is no single margin to compare against.
- It sits alongside reliability rather than replacing it. Reliability asks how
  much evidence a given video offers; uncertainty asks how small a difference
  the tool can resolve at all. A perfectly visible knee angle is still ±10.7°.
