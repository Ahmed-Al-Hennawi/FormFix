"""
Filters for the landmark and joint-angle series, written out so there is no
extra dependency.

An EMA is causal, so it lags, and it lags worst at the turning points - which
for a squat are lockout and depth, the measurement everything hangs on. Dill
et al. (2024) hit the same problem and picked a 4th-order Butterworth at 2 Hz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "FilterSpec",
    "BUTTERWORTH_2HZ",
    "SAVGOL_DEFAULT",
    "butterworth_sos",
    "sosfilt",
    "filtfilt",
    "butterworth_lowpass",
    "savitzky_golay",
    "moving_average",
    "apply_named_filter",
    "FILTERS",
    "MEASURED_DEPTH_BIAS_DEG",
    "MEASURED_DEPTH_BIAS_SMOOTH_DEG",
    "depth_bias_for",
]


# --- contiguous-run helper: stops any filter crossing a gap ---


def _finite_runs(values: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    finite = np.isfinite(values)
    if not finite.any():
        return []
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, ok in enumerate(finite):
        if ok and start is None:
            start = index
        elif not ok and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(finite)))
    return [run for run in runs if run[1] - run[0] >= min_length]


# --- Butterworth low-pass, by bilinear transform ---


def butterworth_sos(order: int, cutoff_hz: float, fs: float) -> np.ndarray:
    if order < 2 or order % 2 != 0:
        raise ValueError(f"order must be a positive even integer, got {order}")
    if fs <= 0:
        raise ValueError(f"sampling rate must be positive, got {fs}")
    nyquist = fs / 2.0
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(
            f"cutoff must lie strictly between 0 and Nyquist ({nyquist:.3f} Hz), got {cutoff_hz}"
        )

    k = math.tan(math.pi * cutoff_hz / fs)
    k2 = k * k
    sections = []
    for index in range(1, order // 2 + 1):
        zeta = math.sin((2 * index - 1) * math.pi / (2 * order))
        denom = k2 + 2.0 * zeta * k + 1.0
        sections.append(
            [
                k2 / denom,
                2.0 * k2 / denom,
                k2 / denom,
                1.0,
                2.0 * (k2 - 1.0) / denom,
                (k2 - 2.0 * zeta * k + 1.0) / denom,
            ]
        )
    return np.asarray(sections, dtype=np.float64)


def _sosfilt_run(sos: np.ndarray, signal: np.ndarray, zero_phase_init: bool) -> np.ndarray:
    """One forward pass of an SOS cascade, direct form II transposed."""
    out = signal.astype(np.float64, copy=True)
    for section in sos:
        b0, b1, b2, _, a1, a2 = section
        if zero_phase_init:
            # start the delay line where it would settle if the first sample had been
            # held forever - from zero every run opens with a big false transient
            gain = (b0 + b1 + b2) / (1.0 + a1 + a2)
            first = out[0]
            z1 = first * (b1 + b2 - gain * (a1 + a2))
            z2 = first * (b2 - gain * a2)
        else:
            z1 = z2 = 0.0
        filtered = np.empty_like(out)
        for index, sample in enumerate(out):
            value = b0 * sample + z1
            z1 = b1 * sample + z2 - a1 * value
            z2 = b2 * sample - a2 * value
            filtered[index] = value
        out = filtered
    return out


def sosfilt(sos: np.ndarray, values: np.ndarray) -> np.ndarray:

    out = np.asarray(values, dtype=np.float64).copy()
    for start, stop in _finite_runs(out, min_length=1):
        out[start:stop] = _sosfilt_run(sos, out[start:stop], zero_phase_init=True)
    return out


def _odd_reflect(signal: np.ndarray, pad: int) -> np.ndarray:

    if pad <= 0:
        return signal
    left = 2.0 * signal[0] - signal[pad:0:-1]
    right = 2.0 * signal[-1] - signal[-2 : -pad - 2 : -1]
    return np.concatenate([left, signal, right])


def filtfilt(sos: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Zero-phase filtering: forward then backward over each finite stretch. Runs
    too short to pad come back untouched."""
    out = np.asarray(values, dtype=np.float64).copy()
    pad = 3 * (2 * len(sos) + 1)
    for start, stop in _finite_runs(out, min_length=1):
        segment = out[start:stop]
        if segment.size <= pad + 1:
            continue
        padded = _odd_reflect(segment, pad)
        forward = _sosfilt_run(sos, padded, zero_phase_init=True)
        backward = _sosfilt_run(sos, forward[::-1], zero_phase_init=True)[::-1]
        out[start:stop] = backward[pad : pad + segment.size]
    return out


def butterworth_lowpass(
    values: np.ndarray, fs: float, cutoff_hz: float = 2.0, order: int = 4
) -> np.ndarray:
    """Dill et al.'s choice: 4th-order Butterworth at 2 Hz, run both ways so there
    is no lag. fs has to be the clip's real frame rate - 24 and 60 fps clips need
    the same movement bandwidth, not the same fraction of their sample rate."""
    fs = float(fs)
    if fs <= 0 or not math.isfinite(fs):
        return np.asarray(values, dtype=np.float64).copy()
    # Keep the cut-off under Nyquist for absurdly low frame rates.
    cutoff = min(cutoff_hz, 0.45 * fs)
    if cutoff <= 0:
        return np.asarray(values, dtype=np.float64).copy()
    return filtfilt(butterworth_sos(order, cutoff, fs), values)


# --- Savitzky-Golay ---


def _savgol_coefficients(window: int, polyorder: int, position: float) -> np.ndarray:
    """
    Coefficients that evaluate a window's least-squares polynomial fit at
    position, an offset in samples from the centre. The fit is linear in the
    samples, so it collapses to [1, p, p^2, ...] @ pinv(V). position = 0 is the
    normal smoothing kernel; other positions handle the ends without padding.
    """
    if window % 2 == 0 or window < 3:
        raise ValueError(f"window must be an odd integer >= 3, got {window}")
    if polyorder >= window:
        raise ValueError(f"polyorder must be < window, got {polyorder} >= {window}")
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    vandermonde = np.vander(offsets, polyorder + 1, increasing=True)
    basis = np.array([position**power for power in range(polyorder + 1)], dtype=np.float64)
    return basis @ np.linalg.pinv(vandermonde)


def _savgol_kernel(window: int, polyorder: int) -> np.ndarray:
    """The centre-evaluated smoothing kernel (position = 0)."""
    return _savgol_coefficients(window, polyorder, 0.0)


def savitzky_golay(values: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing, per finite stretch. Fits a curve through the
    window instead of flattening it, so peak heights survive - and the peaks are
    the measurement here. Symmetric, so no lag."""
    out = np.asarray(values, dtype=np.float64).copy()
    kernel = _savgol_kernel(window, polyorder)
    half = window // 2
    for start, stop in _finite_runs(out, min_length=1):
        segment = out[start:stop]
        if segment.size < window:
            continue
        smoothed = np.convolve(segment, kernel[::-1], mode="valid")
        result = np.empty_like(segment)
        result[half : half + smoothed.size] = smoothed
        # ends: evaluate the first/last fitted window off-centre instead of padding
        first, last = segment[:window], segment[-window:]
        for offset in range(half):
            result[offset] = _savgol_coefficients(window, polyorder, offset - half) @ first
            result[-1 - offset] = _savgol_coefficients(window, polyorder, half - offset) @ last
        out[start:stop] = result
    return out


# --- Moving average - here so the harness can measure what it costs ---


def moving_average(values: np.ndarray, window: int = 5, centred: bool = True) -> np.ndarray:
    """Boxcar mean. centred=True is the zero-phase form; centred=False trails by
    (window - 1) / 2 samples, which is the lag Dill et al. describe."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    out = np.asarray(values, dtype=np.float64).copy()
    if window == 1:
        return out
    kernel = np.full(window, 1.0 / window)
    for start, stop in _finite_runs(out, min_length=1):
        segment = out[start:stop]
        if segment.size < window:
            continue
        if centred:
            half = window // 2
            padded = _odd_reflect(segment, half)
            smoothed = np.convolve(padded, kernel, mode="valid")
            out[start:stop] = smoothed[: segment.size]
        else:
            padded = np.concatenate([np.full(window - 1, segment[0]), segment])
            out[start:stop] = np.convolve(padded, kernel, mode="valid")
    return out


# --- Named filters, so a config value can pick one ---


@dataclass(frozen=True)
class FilterSpec:
    """A filter choice plus where its parameters came from."""

    name: str
    description: str
    causal: bool
    source: str
    params: dict


BUTTERWORTH_2HZ = FilterSpec(
    name="butterworth",
    description="4th-order Butterworth low-pass, 2 Hz, applied forward-backward (zero phase)",
    causal=False,
    source=(
        "Dill et al. (2024), Sensors 24(23):7772 - selected by grid search over a "
        "moving-average, Butterworth and Savitzky-Golay filter on MediaPipe squat data"
    ),
    params={"cutoff_hz": 2.0, "order": 4},
)

SAVGOL_DEFAULT = FilterSpec(
    name="savgol",
    description="Savitzky-Golay, 9-sample window, quadratic (zero phase)",
    causal=False,
    source="Dill et al. (2024) - one of the three filters compared; parameters chosen here",
    params={"window": 9, "polyorder": 2},
)

EMA_DEFAULT = FilterSpec(
    name="ema",
    description="Exponential moving average (FormFix default; causal, lags turning points)",
    causal=True,
    source="FormFix engineering default, see analysis/smoothing.py",
    params={"alpha": 0.5},
)

MOVING_AVERAGE_TRAILING = FilterSpec(
    name="moving_average",
    description="Trailing 5-sample boxcar mean (causal; the filter Dill et al. rejected)",
    causal=True,
    source="Dill et al. (2024) - reported visible delay at squat turning points",
    params={"window": 5, "centred": False},
)

FILTERS: dict[str, FilterSpec] = {
    spec.name: spec for spec in (BUTTERWORTH_2HZ, SAVGOL_DEFAULT, EMA_DEFAULT, MOVING_AVERAGE_TRAILING)
}


def apply_named_filter(name: str, values: np.ndarray, fs: float) -> np.ndarray:
    """Run one of the named filters over a 1-D series."""
    from .smoothing import smooth_series  # local import to dodge a cycle

    spec = FILTERS.get(name)
    if spec is None:
        raise ValueError(f"unknown filter {name!r}; known: {sorted(FILTERS)}")
    if spec.name == "butterworth":
        return butterworth_lowpass(values, fs, **spec.params)
    if spec.name == "savgol":
        return savitzky_golay(values, **spec.params)
    if spec.name == "moving_average":
        return moving_average(values, **spec.params)
    return smooth_series(np.asarray(values, dtype=np.float64), spec.params["alpha"])


# --- What each filter costs the measurement ---

# worst-case degrees of depth each filter loses at the bottom of a rep, from
# scripts/compare_filters.py. Feeds the systematic term of the depth band.
MEASURED_DEPTH_BIAS_DEG: dict[str, float] = {
    "ema": 4.53,
    "moving_average": 6.94,
    "butterworth": 7.67,
    "savgol": 3.27,
}

# same figures over smooth reps only, where the ranking flips round
MEASURED_DEPTH_BIAS_SMOOTH_DEG: dict[str, float] = {
    "ema": 1.09,
    "moving_average": 1.70,
    "butterworth": 0.25,
    "savgol": 1.11,
}


def depth_bias_for(filter_name: str) -> float:
    """Worst-case depth bias of a named filter, in degrees."""
    return MEASURED_DEPTH_BIAS_DEG.get(filter_name, 0.0)
