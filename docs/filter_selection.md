# Choosing the smoothing filter

## Why this needed measuring rather than deciding

FormFix reads its two most consequential squat numbers at the two turning
points of the knee-angle signal: the depth reached at the bottom, and the
extension reached at the top. Turning points are the worst place for a causal
smoothing filter. It rounds the extreme off and reports it late, so the depth
FormFix sees is systematically **shallower** than the depth that happened —
and shallower is the direction that produces false findings, because it makes
a good repetition look like a shallow one.

FormFix's default has always been an exponential moving average, which is
causal. That was never argued for; it was just the obvious first choice.

Dill et al. (2024) hit the same problem from the measurement side. Grid-
searching a moving-average, a Butterworth low-pass and a Savitzky–Golay filter
over MediaPipe squat data, they noted that the moving average "suffer[s] from
the fact that the squat movement features two sudden shifts in direction, when
the subject reaches the highest and lowest point of the movement, leading to a
visible delay in the filtered signal", and selected a **4th-order Butterworth
low-pass at 2 Hz** instead.

The obvious move is to adopt their answer. This document is why that would
have been wrong, and what was done instead.

## The criterion is not the same one

Dill et al. selected on overall landmark RMSE across the whole recording.
FormFix's criterion is narrower: the bias in **one number, at one instant, on
each repetition**. A filter can be better on the first and worse on the
second, and the comparison is cheap to run, so it was run.

`scripts/compare_filters.py` puts each filter over synthetic knee-angle traces
whose true minimum is known by construction, at 30 fps with 1.5° of landmark
noise, across three repetition rates and two shapes of turnaround:

* **smooth** — a sinusoid: a controlled repetition that pauses at the bottom.
  Nearly all of its energy is below 2 Hz.
* **cusped** — a triangle: the athlete reverses direction sharply with no
  pause. A corner contains energy at every frequency.

## What it measured

Depth bias in degrees. Positive means the filter reports a **shallower** squat
than was performed.

**Smooth (paused bottom)**

| Filter | 0.33 Hz | 0.5 Hz | 1.0 Hz |
| --- | ---: | ---: | ---: |
| EMA (FormFix default) | −1.09 | −0.40 | +0.87 |
| Moving average w=5 (trailing) | +0.42 | −0.24 | +1.70 |
| **Butterworth 4th/2 Hz zero-phase** | **+0.04** | **−0.03** | **+0.25** |
| Savitzky–Golay 9/2 | −1.11 | +0.16 | −0.57 |

**Cusped (sharp reversal)**

| Filter | 0.33 Hz | 0.5 Hz | 1.0 Hz |
| --- | ---: | ---: | ---: |
| EMA (FormFix default) | +1.40 | +2.47 | +4.53 |
| Moving average w=5 (trailing) | +1.65 | +3.49 | +6.94 |
| Butterworth 4th/2 Hz zero-phase | +2.63 | +3.61 | **+7.67** |
| **Savitzky–Golay 9/2** | +1.58 | **+2.27** | **+3.27** |

## The finding

**The ranking reverses between the two shapes.** The 2 Hz Butterworth is
essentially unbiased on a smooth repetition — better than everything else by
an order of magnitude — and the *worst* of the four on a sharp one, worse even
than the trailing moving average that Dill et al. rejected.

The reason is not subtle once stated: a sharp reversal carries real signal
above 2 Hz, and a 2 Hz low-pass removes it along with the noise. Dill et al.'s
recordings were made under laboratory supervision to a defined protocol, where
repetitions are smooth. A beginner filming themselves on a phone does not
reverse smoothly, and the sharper the reversal the more depth a low-pass
filter eats.

Savitzky–Golay is the only filter that is never worst. It fits a polynomial
through each window and evaluates it, rather than averaging the window flat,
so it preserves the height of an extreme far better than any filter of
comparable smoothing power. Its worst case (+3.27°) is the smallest worst case
of the four.

## What FormFix does with that

Three things, and deliberately not a silent swap of the default.

**1. The filter is a configuration value.** `ANGLE_FILTER` in each exercise's
config takes `"ema"`, `"butterworth"`, `"savgol"` or `"moving_average"`, and
the evaluation harness takes `--filter` so a manifest can be run under each.
A question this open should be a setting, not a decision buried in a pipeline.

**2. The default is unchanged.** It stays `"ema"`. Changing it would
invalidate every result FormFix has produced so far, and there are no labelled
recordings against which the change could be justified — the same rule the
threshold tuning log already imposes on itself. The evidence above says
Savitzky–Golay is probably the better choice; "probably, on synthetic data"
is not enough to move a default.

**3. The residual bias is carried into the verdict.** This is the part that
matters. Each filter's measured worst-case depth bias lives in
`analysis/filters.py::MEASURED_DEPTH_BIAS_DEG` and is added, in quadrature, to
the depth measurement's uncertainty band. With the default EMA the depth band
is `√(10.7² + 4.53²) = 11.6°` rather than 10.7°. The cost of the smoothing
choice therefore reaches the point of decision instead of being described in a
document nobody reads while the verdict is being formed. See
[measurement uncertainty](measurement_uncertainty.md).

## The implementation

No new dependency. SciPy is not in `requirements.txt`, and the macOS install
is pinned to a NumPy-1 ABI for MediaPipe's sake, so pulling SciPy in for
`filtfilt` would risk the one fragile part of the install for one function.
`analysis/filters.py` therefore derives the Butterworth cascade from the
bilinear transform of the analogue prototype, implements forward–backward
filtering for zero phase, and builds the Savitzky–Golay kernel from the
pseudo-inverse of the window's Vandermonde matrix.

Because it is written rather than imported, it is tested against behaviour
that is known analytically rather than against a stored snapshot of its own
output (`tests/test_filters.py`):

* DC gain exactly 1, response exactly −3.0103 dB at the cut-off, exactly zero
  at Nyquist, monotonic roll-off throughout — the defining properties of a
  Butterworth, any of which a coefficient error would break.
* Forward–backward filtering leaves an asymmetric pulse's peak at the frame it
  started on; a single forward pass moves it later.
* A quadratic survives a quadratic Savitzky–Golay filter exactly, at every
  sample including the first and last. (This is what forced proper end
  handling: padding the ends fabricates samples, and for a curved signal it
  fabricates them wrongly. The first and last windows are fitted once and
  evaluated off-centre instead.)
* Noise is reduced by the factor theory predicts, `√(2/15)`, not merely "by
  some amount".
* Every filter leaves a tracked-landmark gap as a gap. Each contiguous run of
  finite samples is filtered independently, so no filter can draw a landmark
  through a stretch where nobody was detected — the rule the rest of the
  pipeline already follows.

## Reproducing the table

```bash
python scripts/compare_filters.py                        # the table above
python scripts/compare_filters.py --noise 3.0            # heavier landmark noise
python scripts/compare_filters.py --csv evaluation/filters.csv
```

## Honest limits of this comparison

The traces are synthetic. That is deliberate — the true minimum has to be
known exactly for a bias to be measurable at all, and no real recording comes
with one — but it means these numbers describe how each filter behaves on a
*model* of a squat, not on a squat. The two shapes were chosen to bracket real
execution rather than to represent it. What the comparison establishes is that
the ranking is shape-dependent, which is enough to justify making the filter a
setting; it does not establish which filter is best for real athletes, and
nothing here should be quoted as though it did.
