# Measurement uncertainty

## Why I added this

Every rule compares a measurement to a threshold. I'd already labelled the
thresholds as provisional, but I was treating the **measurement** as exact -
and it's actually the less certain side.

MediaPipe's error on the things FormFix measures has been published, and it's
big compared to some of my thresholds:

| Measurement | Published error | Where FormFix uses it |
| --- | --- | --- |
| Knee angle, near leg, side camera | RMSE **10.7°** | depth (pass 100°, warn 115°) |
| Knee angle, far (hidden) leg | RMSE **25.1°** | not used - this is why |
| Knee angle in squats, good camera angle | RMSE **9.14°** (14.48° at a bad angle) | depth |
| Shoulder / hip width | RMSE **5.7-29.8%** / **5.0-34.4%** of the true width | every normalised length |

Sources: Dill et al. (2024), *Sensors* 24(23):7772; Dill et al. (2023),
*Curr Dir Biomed Eng* 9(1):563-566.

So a rep measured at 108° can't really be told apart from one at 98°.

For comparison, Hancock et al. (2018, quoted in Dill et al.) give the smallest
difference clinical tools can detect on a patient lying still: 6° with a
digital inclinometer, 10° with a long-arm goniometer, 14° by eye. FormFix's
10.7° is between a goniometer and a physio's eye - fine for a phone, but not
good enough for 3° differences.

Mercadal-Baudart et al. (2024) turn that into a rule: under **12°** error is
"good" (better than by eye) and under **6°** is "very good". Against that:

| FormFix measurement | Error | Verdict |
| --- | --- | --- |
| Trunk angle | ±6.5° | good |
| Knee angle, near leg (depth) | ±10.7° (±11.6° with filter bias) | good |
| Left/right knee difference | ±15.1° | **worse than by eye** |
| Knee angle, far leg | ±25.1° | **worse than by eye** |

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

## What it showed about my own thresholds

1. **The squat left/right warning (12°) was below its own error (±15.1°)**, so
   it could report an asymmetry that was just noise. I **removed that rule**;
   the measurement is still exported. The press keeps its symmetry rule because
   it's filmed front-on with both arms visible.
2. **The heel-lift threshold (0.06) was below the ±0.15 error** of the body
   length it's divided by, so it reported planted heels as lifting. It is now
   **0.16**, clear of the band and the same value as the
   [literature preset](../exercises/squat/literature_config.py) - the one
   threshold where my default and the preset agree.

That costs sensitivity, and I'd rather say so: 0.16 of a lower leg is roughly
6 cm of heel rise, so a small genuine lift is not reported. From one 2D camera
a smaller one can't be separated from the noise, and the old bar only appeared
to find them. A heel finding stays a prompt to look, not a measurement. Tests
in `test_uncertainty.py` fail if anyone puts it back inside the band.

## What it doesn't do

- It doesn't turn RMSE into a probability - the papers report RMSE, not a
  distribution.
- It doesn't apply to the two range-of-motion rules, because they combine
  three criteria in different units, so there's no single margin.
- It doesn't replace reliability. Reliability is "how much evidence is there
  in this video"; uncertainty is "how small a difference can this tool see at
  all". A perfectly visible knee angle is still ±10.7°.
