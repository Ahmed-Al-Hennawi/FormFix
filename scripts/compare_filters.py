#!/usr/bin/env python3
"""
How much depth each smoothing filter costs us.

Depth and extension are both read at turning points of the knee-angle signal,
which is where a causal filter behaves worst: it rounds the extreme off and
reports it late. Dill et al. (2024) picked a 4th-order Butterworth at 2 Hz, but
they optimised overall landmark RMSE and what matters here is the bias in one
number at one instant per rep, so this runs each filter over traces whose true
minimum is known by construction.

    python scripts/compare_filters.py [--noise 2.5] [--csv out.csv]

Nothing consumes the output. It is the evidence for why ANGLE_FILTER is a
setting, not a hard-coded choice.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.filters import (  # noqa: E402
    butterworth_lowpass,
    moving_average,
    savitzky_golay,
)
from analysis.smoothing import smooth_series  # noqa: E402

FILTERS = {
    "ema (FormFix default)": lambda x, fs: smooth_series(x, 0.5),
    "moving average w=5 (trailing)": lambda x, fs: moving_average(x, 5, centred=False),
    "butterworth 4th/2Hz zero-phase": lambda x, fs: butterworth_lowpass(x, fs, 2.0, 4),
    "savitzky-golay 9/2": lambda x, fs: savitzky_golay(x, 9, 2),
}


def knee_trace(
    reps_per_second: float,
    seconds: float,
    fs: float,
    standing: float,
    bottom: float,
    cusped: bool,
) -> np.ndarray:
    """
    A synthetic knee-angle trace with a known minimum. The two shapes bracket real
    reps: cusped=False is a sinusoid, a controlled rep that pauses at the bottom,
    almost all of it below 2 Hz; cusped=True is a triangle, a sharp reversal whose
    corner has energy at every frequency, so any low-pass has to round it off.
    """
    t = np.arange(0.0, seconds, 1.0 / fs)
    amplitude = (standing - bottom) / 2.0
    midpoint = (standing + bottom) / 2.0
    phase = (t * reps_per_second) % 1.0
    if cusped:
        return bottom + (standing - bottom) * np.abs(2.0 * phase - 1.0)
    return midpoint + amplitude * np.cos(2.0 * np.pi * reps_per_second * t)


def measure(
    truth: np.ndarray, filtered: np.ndarray, fs: float, reps_per_second: float
) -> tuple[float, float]:
    """
    (depth_bias_deg, lag_ms) measured over the third rep - the third rather than
    the first, so no filter gets judged on its start-up transient.
    """
    period = 1.0 / reps_per_second
    lo = int(2.0 * period * fs)
    hi = int(3.0 * period * fs)
    hi = min(hi, len(truth))
    if hi - lo < 5:
        return float("nan"), float("nan")
    true_index = lo + int(np.argmin(truth[lo:hi]))
    got_index = lo + int(np.argmin(filtered[lo:hi]))
    bias = float(np.min(filtered[lo:hi]) - np.min(truth[lo:hi]))
    lag = (got_index - true_index) / fs * 1000.0
    return bias, lag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs", type=float, default=30.0, help="frame rate (default 30)")
    parser.add_argument(
        "--noise",
        type=float,
        default=1.5,
        help="landmark noise as a standard deviation in degrees of knee angle (default 1.5)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    print(f"Frame rate {args.fs:g} fps, landmark noise sigma = {args.noise:g} deg\n")
    print(
        "Depth bias is POSITIVE when the filter reports a SHALLOWER squat than "
        "the athlete performed.\n"
    )

    for cusped in (False, True):
        shape = "cusped (sharp reversal)" if cusped else "smooth (paused bottom)"
        print(f"--- {shape} " + "-" * (52 - len(shape)))
        header = f"{'filter':32s}" + "".join(f"{r:>10s}" for r in ("0.33 Hz", "0.5 Hz", "1.0 Hz"))
        print(header)
        for name, function in FILTERS.items():
            cells = []
            for reps_per_second in (1 / 3, 0.5, 1.0):
                truth = knee_trace(reps_per_second, 12.0, args.fs, 175.0, 95.0, cusped)
                noisy = truth + rng.normal(0.0, args.noise, truth.size)
                filtered = np.asarray(function(noisy, args.fs), dtype=np.float64)
                bias, lag = measure(truth, filtered, args.fs, reps_per_second)
                cells.append(f"{bias:>+9.2f}")
                rows.append(
                    {
                        "shape": "cusped" if cusped else "smooth",
                        "reps_per_second": round(reps_per_second, 3),
                        "filter": name,
                        "depth_bias_deg": round(bias, 3),
                        "lag_ms": round(lag, 1),
                    }
                )
            print(f"{name:32s}" + "".join(cells))
        print()

    best = {}
    for row in rows:
        key = row["shape"]
        if key not in best or abs(row["depth_bias_deg"]) < abs(best[key]["depth_bias_deg"]):
            best[key] = row
    print("Lowest depth bias observed:")
    for shape, row in best.items():
        print(f"  {shape:8s}  {row['filter']}  ({row['depth_bias_deg']:+.2f} deg)")

    print(
        "\nRead this as a trade-off, not a ranking. A 2 Hz low-pass is the best choice\n"
        "for a smooth repetition and among the worst for a sharp one, because a sharp\n"
        "reversal carries real signal above 2 Hz that the filter removes along with the\n"
        "noise. Savitzky-Golay fits a curve through each window instead of flattening\n"
        "it, which is why it holds up on both. Whichever is chosen, the residual bias\n"
        "belongs in the depth measurement's uncertainty band, not in a footnote - see\n"
        "exercises/common/uncertainty.py."
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nFull table written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
