# Choosing the smoothing filter

## The problem

FormFix reads its two most important squat numbers at the turning points of
the knee angle: depth at the bottom and extension at the top. Turning points
are exactly where a causal filter (like an EMA) is worst - it rounds off the
peak and reports it late. So the depth FormFix sees is a bit **shallower** than
what really happened, which is the direction that creates false "too shallow"
findings.

My default was an EMA just because it was the obvious first choice.

Dill et al. (2024) had the same issue. They compared a moving average, a
Butterworth low-pass and a Savitzky-Golay filter on MediaPipe squat data,
said the moving average shows "a visible delay in the filtered signal" at the
top and bottom, and chose a **4th-order Butterworth at 2 Hz**.

I didn't just copy their choice, because we care about different things.

## Different criterion

Dill et al. picked the filter with the lowest overall landmark error across
the whole recording. I care about the error in **one number per rep**. A filter
can be better at one and worse at the other, so I tested it.

`scripts/compare_filters.py` runs each filter on synthetic knee-angle traces
where the true minimum is known, at 30 fps with 1.5° of noise, at three speeds
and two shapes:

- **smooth** - a sine wave, like a controlled rep with a pause at the bottom
- **cusped** - a triangle, like bouncing straight out of the bottom

## Results

Depth bias in degrees (positive = reported **shallower** than it was).

**Smooth (paused bottom)**

| Filter | 0.33 Hz | 0.5 Hz | 1.0 Hz |
| --- | ---: | ---: | ---: |
| EMA (default) | −1.09 | −0.40 | +0.87 |
| Moving average (5, trailing) | +0.42 | −0.24 | +1.70 |
| **Butterworth 4th order, 2 Hz** | **+0.04** | **−0.03** | **+0.25** |
| Savitzky-Golay (9, 2) | −1.11 | +0.16 | −0.57 |

**Cusped (sharp bounce)**

| Filter | 0.33 Hz | 0.5 Hz | 1.0 Hz |
| --- | ---: | ---: | ---: |
| EMA (default) | +1.40 | +2.47 | +4.53 |
| Moving average (5, trailing) | +1.65 | +3.49 | +6.94 |
| Butterworth 4th order, 2 Hz | +2.63 | +3.61 | **+7.67** |
| **Savitzky-Golay (9, 2)** | +1.58 | **+2.27** | **+3.27** |

## What I found

**The ranking flips between the two shapes.** The Butterworth is almost
perfect on a smooth rep and the *worst* on a sharp one - worse than the moving
average Dill et al. rejected. A sharp bounce has real signal above 2 Hz, and a
2 Hz filter removes it with the noise. Their recordings were supervised lab
squats (smooth); a beginner filming on a phone often bounces.

Savitzky-Golay is never the worst - it fits a curve through each window
instead of averaging it flat, so it keeps peaks better. Its worst case (+3.27°)
is the smallest of the four.

## What I did

1. **The filter is a setting.** `ANGLE_FILTER` can be `ema`, `butterworth`,
   `savgol` or `moving_average`, and the evaluation script has `--filter`.
2. **The default stays `ema`.** Changing it would change every result so far,
   and I have no labelled videos to justify it. Savitzky-Golay is *probably*
   better, but "probably, on synthetic data" isn't enough to change a default.
3. **The bias goes into the verdict.** Each filter's worst-case depth bias is in
   `analysis/filters.py` (`MEASURED_DEPTH_BIAS_DEG`) and is added to the depth
   error band: √(10.7² + 4.53²) = 11.6° for the EMA
   ([measurement_uncertainty.md](measurement_uncertainty.md)).

## How it's implemented

No SciPy - it isn't in the requirements, and the macOS install is pinned to
NumPy 1 for MediaPipe, so I didn't want to risk it for one function.
`analysis/filters.py` builds the Butterworth from the bilinear transform, does
forward-backward filtering for zero lag, and builds the Savitzky-Golay kernel
from a least-squares fit.

Because I wrote it myself, `tests/test_filters.py` checks it against known
maths rather than saved output:

- Butterworth gain is 1 at DC, −3.01 dB at the cut-off and 0 at Nyquist
- forward-backward filtering doesn't move a peak; a single pass does
- a quadratic passes through Savitzky-Golay unchanged, including the ends
- noise is reduced by the theoretical √(2/15)
- gaps in tracking stay gaps - no filter draws a landmark through them

The Butterworth is also used (separately) to smooth the skeleton drawn on the
result video. That's display only and doesn't change any measurement.

## Reproduce

```bash
python scripts/compare_filters.py                 # the tables above
python scripts/compare_filters.py --noise 3.0     # more noise
python scripts/compare_filters.py --csv evaluation/filters.csv
```

## Limits

The traces are synthetic on purpose - you need the true minimum to measure a
bias, and real videos don't come with one. So this shows the ranking depends on
the rep shape (enough to make the filter a setting), not which filter is best
for real people.
