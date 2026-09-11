"""
Runs pose detection in a child process. MediaPipe's native code can segfault
and take the whole interpreter (and Streamlit) down, and no try/except catches
that.

    python -m analysis.pose_worker <video> <out.npz> --detection 0.5 ...

Prints "PROGRESS <frame> <total>" while working. On exit 0 the .npz holds xy,
visibility, pose_found, n_poses and timestamps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so the app packages import from any working directory
APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np  # noqa: E402

from analysis.models import AnalysisFailure  # noqa: E402
from analysis.pose_detector import detect_poses_inprocess  # noqa: E402
from analysis.video_processor import probe_video  # noqa: E402

PROGRESS_EVERY = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--detection", type=float, required=True)
    parser.add_argument("--presence", type=float, required=True)
    parser.add_argument("--tracking", type=float, required=True)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    video = probe_video(args.video)
    if not video.readable:
        print("ERROR unreadable video", file=sys.stderr)
        return 3

    def on_frame(index: int, total: int) -> None:
        if index % PROGRESS_EVERY == 0 or index + 1 == total:
            print(f"PROGRESS {index} {total}", flush=True)

    try:
        pose = detect_poses_inprocess(
            video,
            detection_confidence=args.detection,
            presence_confidence=args.presence,
            tracking_confidence=args.tracking,
            model_path=args.model,
            on_frame=on_frame,
        )
    except AnalysisFailure as failure:
        # typed failures come back as an exit code plus a stderr line
        print(f"FAILURE {failure.code.value} {failure.message}", file=sys.stderr)
        return 4

    np.savez_compressed(
        args.out,
        xy=pose.xy_raw,
        visibility=pose.visibility,
        pose_found=pose.pose_found,
        n_poses=pose.n_poses,
        timestamps=pose.timestamps,
        identity_switches=np.int32(pose.identity_switches),
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
