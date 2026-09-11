"""
Filters tested against known maths, not a saved copy of their own output
(which would only show the numbers haven't changed, not that they're right).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.filters import (
    FILTERS,
    apply_named_filter,
    butterworth_lowpass,
    butterworth_sos,
    depth_bias_for,
    filtfilt,
    moving_average,
    savitzky_golay,
    sosfilt,
)


def _response(sos: np.ndarray, frequency_hz: float, fs: float) -> complex:
    """Complex frequency response of the SOS cascade."""
    z = np.exp(2j * np.pi * frequency_hz / fs)
    response = 1 + 0j
    for b0, b1, b2, _a0, a1, a2 in sos:
        response *= (b0 + b1 / z + b2 / z**2) / (1 + a1 / z + a2 / z**2)
    return response


class TestButterworthCoefficients:
    """The defining properties of a Butterworth low-pass, from its response."""

    def test_it_passes_dc_untouched(self):
        sos = butterworth_sos(4, 2.0, 30.0)
        assert abs(_response(sos, 0.0, 30.0)) == pytest.approx(1.0, abs=1e-12)

    def test_it_is_three_decibels_down_at_the_cutoff(self):
        # -3.0103 dB at the cut-off by definition (the pre-warping makes it exact)
        sos = butterworth_sos(4, 2.0, 30.0)
        decibels = 20 * math.log10(abs(_response(sos, 2.0, 30.0)))
        assert decibels == pytest.approx(-3.0103, abs=1e-3)

    def test_it_fully_rejects_nyquist(self):
        sos = butterworth_sos(4, 2.0, 30.0)
        assert abs(_response(sos, 15.0, 30.0)) == pytest.approx(0.0, abs=1e-12)

    def test_the_response_falls_monotonically(self):
        # no passband ripple is what makes it a Butterworth, so a coefficient
        # mistake shows up here first
        sos = butterworth_sos(4, 2.0, 30.0)
        magnitudes = [abs(_response(sos, f, 30.0)) for f in np.linspace(0.0, 15.0, 200)]
        assert all(b <= a + 1e-12 for a, b in zip(magnitudes, magnitudes[1:], strict=False))

    def test_a_higher_cutoff_attenuates_less(self):
        low = abs(_response(butterworth_sos(4, 1.0, 30.0), 3.0, 30.0))
        high = abs(_response(butterworth_sos(4, 5.0, 30.0), 3.0, 30.0))
        assert high > low

    @pytest.mark.parametrize("order", [0, 1, 3, -2])
    def test_it_refuses_an_order_it_cannot_build(self, order):
        with pytest.raises(ValueError):
            butterworth_sos(order, 2.0, 30.0)

    def test_it_refuses_a_cutoff_at_or_above_nyquist(self):
        with pytest.raises(ValueError):
            butterworth_sos(4, 15.0, 30.0)


class TestZeroPhase:
    """Zero phase lag - the whole reason for the module."""

    def test_forward_backward_filtering_does_not_move_a_peak(self):
        fs = 30.0
        t = np.arange(0, 300) / fs
        # asymmetric pulse on purpose, a symmetric one would hide a phase shift
        signal = np.exp(-(((t - 4.0) / 0.35) ** 2)) + 0.4 * np.exp(-(((t - 4.6) / 0.9) ** 2))
        sos = butterworth_sos(4, 3.0, fs)
        zero_phase = filtfilt(sos, signal)
        causal = sosfilt(sos, signal)
        assert int(np.argmax(zero_phase)) == int(np.argmax(signal))
        # the single-pass version does shift it
        assert int(np.argmax(causal)) > int(np.argmax(signal))

    def test_it_removes_the_amount_of_noise_theory_predicts(self):
        # white noise over 0-15 Hz through a ~2 Hz filter should come out at
        # about sqrt(2/15) = 0.37 of its amplitude
        fs = 30.0
        t = np.arange(0, 600) / fs
        clean = 135 + 40 * np.cos(2 * np.pi * 0.5 * t)
        noisy = clean + np.random.default_rng(7).normal(0, 3.0, clean.size)
        filtered = butterworth_lowpass(noisy, fs)
        before = float(np.sqrt(np.mean((noisy - clean) ** 2)))
        after = float(np.sqrt(np.mean((filtered - clean) ** 2)))
        assert after / before == pytest.approx(math.sqrt(2.0 / 15.0), rel=0.25)

    def test_a_symmetric_moving_average_is_also_zero_phase(self):
        t = np.arange(0, 200) / 30.0
        signal = np.exp(-(((t - 3.0) / 0.4) ** 2))
        assert int(np.argmax(moving_average(signal, 5, centred=True))) == int(np.argmax(signal))

    def test_a_trailing_moving_average_lags(self):
        t = np.arange(0, 200) / 30.0
        signal = np.exp(-(((t - 3.0) / 0.4) ** 2))
        assert int(np.argmax(moving_average(signal, 9, centred=False))) > int(np.argmax(signal))


class TestSavitzkyGolay:
    def test_it_reproduces_a_polynomial_exactly(self):
        # Savitzky-Golay fits a quadratic exactly, so it keeps the peak height
        # where a moving average flattens it
        x = np.arange(60, dtype=np.float64)
        quadratic = 3.0 + 0.5 * x - 0.02 * x**2
        smoothed = savitzky_golay(quadratic, 9, 2)
        assert np.allclose(smoothed, quadratic, atol=1e-8)

    def test_it_preserves_a_peak_better_than_a_boxcar_of_the_same_width(self):
        t = np.arange(0, 200) / 30.0
        signal = np.exp(-(((t - 3.0) / 0.25) ** 2))
        savgol_peak = float(np.max(savitzky_golay(signal, 9, 2)))
        boxcar_peak = float(np.max(moving_average(signal, 9, centred=True)))
        assert savgol_peak > boxcar_peak

    def test_it_refuses_an_even_window(self):
        with pytest.raises(ValueError):
            savitzky_golay(np.zeros(50), 8, 2)

    def test_it_refuses_a_polynomial_it_cannot_fit(self):
        with pytest.raises(ValueError):
            savitzky_golay(np.zeros(50), 5, 5)


class TestGapsAreNeverBridged:
    """A gap in tracking stays missing, otherwise the filter invents landmarks."""

    @pytest.mark.parametrize("name", sorted(FILTERS))
    def test_a_gap_survives_every_filter(self, name):
        values = np.linspace(100.0, 160.0, 200)
        values[80:100] = np.nan
        out = np.asarray(apply_named_filter(name, values, 30.0))
        assert np.all(np.isnan(out[80:100]))
        assert np.isfinite(out[:80]).all()
        assert np.isfinite(out[100:]).all()

    def test_each_side_of_a_gap_is_filtered_independently(self):
        # joining the two runs would make the step between them ring across the gap
        left = np.full(120, 170.0)
        right = np.full(120, 100.0)
        values = np.concatenate([left, [np.nan] * 20, right])
        out = butterworth_lowpass(values, 30.0)
        assert out[100] == pytest.approx(170.0, abs=0.5)
        assert out[-20] == pytest.approx(100.0, abs=0.5)

    def test_a_run_too_short_to_filter_is_returned_untouched(self):
        short = np.array([120.0, 118.0, 121.0])
        assert np.allclose(butterworth_lowpass(short, 30.0), short)


class TestStartOfSignal:
    def test_it_does_not_invent_a_transient_at_the_first_sample(self):
        # starting from zero the filter swings over the first frames, which are
        # the standing baseline for a squat
        constant = np.full(200, 175.0)
        out = butterworth_lowpass(constant, 30.0)
        assert np.allclose(out, 175.0, atol=1e-6)

    def test_a_single_pass_also_starts_from_the_signal(self):
        constant = np.full(200, 175.0)
        out = sosfilt(butterworth_sos(4, 2.0, 30.0), constant)
        assert np.allclose(out, 175.0, atol=1e-6)


class TestNamedFilters:
    def test_every_named_filter_runs_and_keeps_the_length(self):
        values = np.linspace(90.0, 175.0, 240)
        for name in FILTERS:
            out = np.asarray(apply_named_filter(name, values, 30.0))
            assert out.shape == values.shape

    def test_an_unknown_name_is_an_error_not_a_silent_passthrough(self):
        with pytest.raises(ValueError):
            apply_named_filter("gaussian", np.zeros(50), 30.0)

    def test_ema_is_delegated_to_the_one_existing_implementation(self):
        from analysis.smoothing import smooth_series

        values = np.linspace(90.0, 175.0, 100)
        assert np.allclose(apply_named_filter("ema", values, 30.0), smooth_series(values, 0.5))

    def test_every_named_filter_has_a_measured_depth_bias(self):
        # every filter needs a measured depth cost for the uncertainty band, so an
        # unmeasured new filter should fail here
        for name in FILTERS:
            assert depth_bias_for(name) > 0.0

    def test_an_impossible_frame_rate_degrades_gracefully(self):
        values = np.linspace(90.0, 175.0, 100)
        assert np.allclose(butterworth_lowpass(values, 0.0), values)
        assert np.allclose(butterworth_lowpass(values, float("nan")), values)
