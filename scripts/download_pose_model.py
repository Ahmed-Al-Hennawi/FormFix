"""
Download the MediaPipe Pose Landmarker model into
static/assets/models/pose_landmarker_full.task.

    python scripts/download_pose_model.py

Run this after cloning. The actual download code is in
analysis.pose_detector.fetch_model, which the app also uses on start-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from analysis.pose_detector import (  # noqa: E402
    MODEL_DOWNLOAD_URL,
    default_model_path,
    fetch_model,
)


def main() -> int:
    target = default_model_path()
    if target.is_file():
        print(f"Model already present: {target}")
        return 0

    print(f"Downloading {MODEL_DOWNLOAD_URL}\n -> {target}")
    if fetch_model(target) is None:
        print(f"Download failed. Fetch the URL manually and save it to {target}")
        return 1

    print(f"Done ({target.stat().st_size / 1_000_000:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
