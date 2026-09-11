"""
Prints the frontality ratio for one or more clips, using the app's own detector
and heuristic. Not used by the app itself.

    python scripts/check_camera_view.py clip.mp4 [clip2.mp4 ...]

The SIDE / DIAGONAL_SIDE / FRONTAL band edges came from a synthetic test pose,
so the unit tests can't show a band is wrong. I use this on clips labelled by
eye to check them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from analysis import pose_detector  # noqa: E402
from analysis.validation import (  # noqa: E402
    classify_orientation,
    frontality_series,
    pose_detection_ratio,
)
from analysis.video_processor import probe_video  # noqa: E402

EXERCISE_CONFIGS = {
    "squat": "exercises.squat.config",
    "press": "exercises.press.config",
    "pulldown": "exercises.pulldown.config",
}


def _load_config(exercise: str):
    import importlib

    module = importlib.import_module(EXERCISE_CONFIGS[exercise])
    return module.DEFAULT_CONFIG


class _Bands:
    """The two band edges in the format classify_orientation expects."""

    def __init__(self, good: float, frontal: float) -> None:
        self.side_view_good_ratio = good
        self.side_view_frontal_ratio = frontal


def check(path: Path, config, bands: _Bands, stability_spread: float) -> dict | None:
    video = probe_video(path)
    if not video.readable:
        print(f"{path.name}: unreadable")
        return None

    pose = pose_detector.detect_poses(
        video,
        detection_confidence=config.POSE_DETECTION_CONFIDENCE,
        presence_confidence=config.POSE_PRESENCE_CONFIDENCE,
        tracking_confidence=config.TRACKING_CONFIDENCE,
    )
    series = frontality_series(pose, video.width, video.height)
    if series.size == 0:
        print(f"{path.name}: not enough tracked frames to judge the camera angle")
        return None

    median = float(np.median(series))
    spread = float(np.percentile(series, 75) - np.percentile(series, 25))
    orientation, confidence = classify_orientation(median, bands)

    print(f"\n{path.name}  ({video.width}x{video.height}, {video.duration:.1f}s)")
    print(f"  pose found in            {pose_detection_ratio(pose):.0%} of frames")
    print(f"  frontality ratio median  {median:.3f}")
    print(
        f"  p10 / p90                {np.percentile(series, 10):.3f} / "
        f"{np.percentile(series, 90):.3f}"
    )
    print(f"  min / max                {series.min():.3f} / {series.max():.3f}")
    print(
        f"  IQR spread               {spread:.3f}"
        f"  {'UNSTABLE' if spread > stability_spread else 'stable'}"
        f" (limit {stability_spread})"
    )
    print(f"  -> {orientation.value.upper()}  (side-view confidence {confidence:.2f})")
    print(
        f"     bands: SIDE <= {bands.side_view_good_ratio} | "
        f"FRONTAL >= {bands.side_view_frontal_ratio}"
    )

    return {
        "name": path.name,
        "median": median,
        "p90": float(np.percentile(series, 90)),
        "max": float(series.max()),
        "spread": spread,
        "orientation": orientation.value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--exercise", default="squat", choices=sorted(EXERCISE_CONFIGS))
    parser.add_argument(
        "--good",
        type=float,
        default=None,
        help="override SIDE_VIEW_GOOD_RATIO for this run (does not change the app)",
    )
    parser.add_argument(
        "--frontal",
        type=float,
        default=None,
        help="override SIDE_VIEW_FRONTAL_RATIO for this run (does not change the app)",
    )
    args = parser.parse_args()

    config = _load_config(args.exercise)
    bands = _Bands(
        args.good if args.good is not None else config.SIDE_VIEW_GOOD_RATIO,
        args.frontal if args.frontal is not None else config.SIDE_VIEW_FRONTAL_RATIO,
    )

    rows = [
        r for path in args.videos if (r := check(path, config, bands, config.VIEW_STABILITY_SPREAD))
    ]

    if len(rows) > 1:
        print("\n" + "-" * 72)
        print(f"{'clip':<34}{'median':>9}{'p90':>9}{'max':>9}  orientation")
        for r in sorted(rows, key=lambda r: r["median"]):
            print(
                f"{r['name'][:33]:<34}{r['median']:>9.3f}{r['p90']:>9.3f}"
                f"{r['max']:>9.3f}  {r['orientation']}"
            )
        reached = [r for r in rows if r["max"] >= bands.side_view_frontal_ratio]
        if not reached:
            print(
                f"\nNo clip reached the FRONTAL band ({bands.side_view_frontal_ratio}) "
                "in any single frame."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
