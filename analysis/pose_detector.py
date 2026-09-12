"""
Wrapper around the MediaPipe Pose Landmarker (Tasks API, VIDEO mode, since the
old mp.solutions.pose isn't in current wheels).

It looks for more than one person so someone in the gym background doesn't get
analysed instead of the athlete - _PersonTracker picks which one to follow.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .models import (
    LEFT_HIP,
    LEFT_SHOULDER,
    NUM_LANDMARKS,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    AnalysisFailure,
    FailureCode,
    FramePoseData,
    VideoMetadata,
)
from .video_processor import iter_frames, open_video

logger = logging.getLogger(__name__)

# model path, relative to the app root so it isn't tied to my machine
MODEL_RELATIVE_PATH = Path("static/assets/models/pose_landmarker_full.task")

MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / MODEL_RELATIVE_PATH


def check_runtime_compatibility() -> None:
    """
    Stop early on the one combination that segfaults: MediaPipe 0.10.14 (last
    macOS build) with NumPy 2.x. The process just dies with no exception, so it
    has to be checked up front.
    """
    if sys.platform != "darwin":
        return
    try:
        import mediapipe
        import numpy

        numpy_major = int(numpy.__version__.split(".")[0])
        mp_version = tuple(int(p) for p in mediapipe.__version__.split(".")[:3])
    except Exception:  # pragma: no cover - never block on a version parse
        return

    if numpy_major >= 2 and mp_version <= (0, 10, 14):
        raise AnalysisFailure(
            FailureCode.ANALYSIS_ERROR,
            "This computer's MediaPipe build is not compatible with the installed "
            "NumPy version, which makes pose detection crash.",
            suggestions=[
                "In the project's virtual environment run: "
                'pip install "numpy==1.26.4" "opencv-python==4.10.0.84" '
                '"opencv-contrib-python==4.10.0.84" "jax==0.4.30" "jaxlib==0.4.30"',
                "Then restart Streamlit. (requirements.txt now pins these on macOS.)",
            ],
        )


# the real model is about 9 MB, so anything smaller is a broken download
MIN_MODEL_BYTES = 1_000_000


def fetch_model(target: Path | None = None, timeout: float = 120.0) -> Path | None:
    """
    Download the model if it's missing (it isn't in the repo). Writes to a temp
    file and renames it so a half download can't look valid. None on failure.
    """
    import shutil
    import urllib.request

    path = target or default_model_path()
    if path.is_file() and path.stat().st_size >= MIN_MODEL_BYTES:
        return path

    logger.info("Pose model not found at %s - downloading it", path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".part") as handle:
            partial = Path(handle.name)
            with urllib.request.urlopen(MODEL_DOWNLOAD_URL, timeout=timeout) as response:
                shutil.copyfileobj(response, handle)
        if partial.stat().st_size < MIN_MODEL_BYTES:
            partial.unlink(missing_ok=True)
            logger.warning("Downloaded pose model is too small to be valid")
            return None
        partial.replace(path)
    except Exception as exc:  # network, disk, permissions: none of them fatal
        logger.warning("Could not download the pose model: %s", exc)
        return None

    logger.info("Pose model downloaded (%.1f MB)", path.stat().st_size / 1_000_000)
    return path


def _resolve_model(model_path: Path | None) -> Path:
    path = model_path or default_model_path()
    if not path.is_file():
        fetch_model(path)
    if not path.is_file():
        raise AnalysisFailure(
            FailureCode.ANALYSIS_ERROR,
            "The pose-detection model file is missing, so FormFix cannot analyse videos yet.",
            suggestions=[
                f"Download the MediaPipe Pose Landmarker (full) model and save it as {MODEL_RELATIVE_PATH}.",
                f"Model URL: {MODEL_DOWNLOAD_URL}",
                "Or run: python scripts/download_pose_model.py",
            ],
        )
    return path


def detect_poses(
    video: VideoMetadata,
    detection_confidence: float,
    presence_confidence: float,
    tracking_confidence: float,
    model_path: Path | None = None,
    on_frame: Callable[[int, int], None] | None = None,
) -> FramePoseData:
    """
    Run the Pose Landmarker over every frame. Normally this runs in a separate
    process (pose_worker.py) because MediaPipe crashed Streamlit a few times.
    FORMFIX_POSE_SUBPROCESS=0 runs it in-process, which the tests use.
    """
    check_runtime_compatibility()
    if os.environ.get("FORMFIX_POSE_SUBPROCESS", "1") != "0":
        return _detect_poses_subprocess(
            video,
            detection_confidence,
            presence_confidence,
            tracking_confidence,
            model_path,
            on_frame,
        )
    return detect_poses_inprocess(
        video,
        detection_confidence,
        presence_confidence,
        tracking_confidence,
        model_path,
        on_frame,
    )


def _drain(stream, into: list[str]) -> None:
    """Read a pipe to the end so the writer can't block on a full buffer."""
    if stream is None:
        return
    try:
        for line in stream:
            into.append(line)
    except (OSError, ValueError):  # pipe closed under us, nothing to do
        pass
    finally:
        with contextlib.suppress(OSError):
            stream.close()


def _pump(stream, into: queue.Queue[str | None]) -> None:
    """Push a pipe's lines onto a queue, then None at the end. Runs on its own
    thread so the caller can use a timeout and notice a stuck worker."""
    try:
        _drain_into_queue(stream, into)
    finally:
        into.put(None)


def _drain_into_queue(stream, into: queue.Queue[str | None]) -> None:
    if stream is None:
        return
    try:
        for line in stream:
            into.put(line)
    except (OSError, ValueError):
        pass
    finally:
        with contextlib.suppress(OSError):
            stream.close()


def _detect_poses_subprocess(
    video: VideoMetadata,
    detection_confidence: float,
    presence_confidence: float,
    tracking_confidence: float,
    model_path: Path | None,
    on_frame: Callable[[int, int], None] | None,
) -> FramePoseData:
    """Run the worker and rebuild the pose track from the .npz it writes."""
    resolved_model = _resolve_model(model_path)
    app_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory(prefix="formfix_pose_") as tmp:
        out_path = Path(tmp) / "pose.npz"
        command = [
            sys.executable,
            "-m",
            "analysis.pose_worker",
            str(video.path),
            str(out_path),
            "--detection",
            str(detection_confidence),
            "--presence",
            str(presence_confidence),
            "--tracking",
            str(tracking_confidence),
            "--model",
            str(resolved_model),
        ]
        # generous: model load plus a few frames a second
        timeout = 180 + max(video.frame_count, 1) * 0.5

        process = subprocess.Popen(  # noqa: S603 - fixed argv, our own worker
            command,
            cwd=str(app_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # read the pipes on threads and wait here with a deadline, since reading
        # a pipe directly would block forever on a stuck worker
        progress_queue: queue.Queue[str | None] = queue.Queue()
        stderr_lines: list[str] = []
        readers = (
            threading.Thread(target=_pump, args=(process.stdout, progress_queue), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, stderr_lines), daemon=True),
        )
        for reader in readers:
            reader.start()

        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            while True:
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                try:
                    line = progress_queue.get(timeout=0.5)
                except queue.Empty:
                    continue  # nothing yet, re-check the deadline
                if line is None:
                    break  # stdout closed, so the worker is done or gone
                if line.startswith("PROGRESS") and on_frame is not None:
                    try:
                        _, index, total = line.split()
                        on_frame(int(index), int(total))
                    except ValueError:  # malformed line, ignore it
                        pass

            if not timed_out:
                # worker is on its way out, so this only covers the .npz write
                try:
                    process.wait(timeout=max(deadline - time.monotonic(), 5.0))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for reader in readers:
                reader.join(timeout=5.0)
        stderr_text = "".join(stderr_lines)

        if timed_out:
            logger.error("Pose worker exceeded its %.0fs budget and was stopped", timeout)
            raise AnalysisFailure(
                FailureCode.ANALYSIS_ERROR,
                "Analysing this video took far longer than expected, so it was stopped.",
                suggestions=["Try a shorter clip, or a lower-resolution export."],
            )

        if process.returncode != 0 or not out_path.is_file():
            logger.error(
                "Pose worker exited %s. stderr tail: %s",
                process.returncode,
                (stderr_text or "")[-800:],
            )
            crashed = process.returncode is not None and process.returncode < 0
            raise AnalysisFailure(
                FailureCode.ANALYSIS_ERROR,
                (
                    "The pose-detection engine stopped unexpectedly while reading " "this video."
                    if crashed
                    else "Pose detection could not be completed for this video."
                ),
                suggestions=[
                    "Try pressing Analyse Form again.",
                    "Try re-exporting the clip as a standard MP4 (H.264).",
                    "If this keeps happening on macOS, install a known-stable "
                    "MediaPipe build: pip install 'mediapipe==0.10.14'.",
                ],
            )

        data = np.load(out_path)
        xy = data["xy"]
        pose = FramePoseData(
            xy_raw=xy,
            xy=xy.copy(),
            visibility=data["visibility"],
            valid=np.isfinite(xy[:, :, 0]),
            pose_found=data["pose_found"],
            n_poses=data["n_poses"],
            timestamps=data["timestamps"],
            identity_switches=(
                int(data["identity_switches"]) if "identity_switches" in data.files else 0
            ),
        )

    found = int(pose.pose_found.sum())
    total = pose.frame_count
    logger.info("Pose detection (worker): %d/%d frames with a pose", found, total)
    return pose


def detect_poses_inprocess(
    video: VideoMetadata,
    detection_confidence: float,
    presence_confidence: float,
    tracking_confidence: float,
    model_path: Path | None = None,
    on_frame: Callable[[int, int], None] | None = None,
) -> FramePoseData:
    """The actual MediaPipe loop. Returns the raw track (smoothing.py cleans it)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    resolved_model = _resolve_model(model_path)

    frame_count = max(video.frame_count, 1)
    xy = np.full((frame_count, NUM_LANDMARKS, 2), np.nan, dtype=np.float64)
    visibility = np.zeros((frame_count, NUM_LANDMARKS), dtype=np.float64)
    pose_found = np.zeros(frame_count, dtype=bool)
    n_poses = np.zeros(frame_count, dtype=np.int16)
    timestamps = np.arange(frame_count, dtype=np.float64) / video.fps

    options = vision.PoseLandmarkerOptions(
        # CPU on purpose - the GPU delegate isn't supported everywhere and was
        # linked to the macOS crashes
        base_options=mp_python.BaseOptions(
            model_asset_path=str(resolved_model),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_TRACKED_POSES,
        min_pose_detection_confidence=detection_confidence,
        min_pose_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)
    tracker = _PersonTracker()
    frames_seen = 0
    last_timestamp_ms = -1
    try:
        with open_video(video.path) as capture:
            for index, frame in iter_frames(capture):
                if index >= frame_count:
                    break  # more frames than the container claimed; drop them
                # VIDEO mode needs strictly increasing timestamps in ms
                timestamp_ms = int(round(timestamps[index] * 1000.0))
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, timestamp_ms)

                frames_seen += 1
                if on_frame is not None:
                    on_frame(index, frame_count)

                poses = result.pose_landmarks or []
                n_poses[index] = len(poses)
                if not poses:
                    continue

                landmarks = tracker.select(poses, index)
                if not landmarks:
                    # the only person detected is not where the athlete was, so
                    # leave the frame empty instead of tracking someone else
                    continue
                pose_found[index] = True
                for li, lm in enumerate(landmarks[:NUM_LANDMARKS]):
                    xy[index, li, 0] = lm.x
                    xy[index, li, 1] = lm.y
                    # visibility and presence both say how much to trust the landmark
                    vis = lm.visibility if lm.visibility is not None else 0.0
                    presence = getattr(lm, "presence", None)
                    if presence is not None and presence > 0:
                        vis = min(vis, presence) if vis > 0 else presence
                    visibility[index, li] = vis
    finally:
        landmarker.close()

    if frames_seen == 0:
        raise AnalysisFailure(
            FailureCode.INVALID_VIDEO,
            "We couldn't read any frames from this video file.",
            suggestions=["Try re-exporting the clip as an MP4 and uploading it again."],
        )

    logger.info(
        "Pose detection: %d/%d frames with a pose (%.0f%%), %d frame(s) skipped as "
        "the wrong person, %d identity restart(s)",
        int(pose_found.sum()),
        frames_seen,
        100.0 * pose_found.sum() / frames_seen,
        tracker.rejected_frames,
        tracker.switches,
    )

    return FramePoseData(
        xy_raw=xy,
        xy=xy.copy(),  # smoothing.py cleans this copy in place
        visibility=visibility,
        valid=np.isfinite(xy[:, :, 0]),
        pose_found=pose_found,
        n_poses=n_poses,
        timestamps=timestamps,
        identity_switches=tracker.switches,
    )


# how many people the landmarker looks for. With 2 a third person was never
# picked up, so the multi-person check under-reported.
MAX_TRACKED_POSES = 4

# how far the tracked hip centre may move between frames, in torso lengths,
# before it stops being the same person
MAX_IDENTITY_DRIFT_TORSOS = 1.6

# frames the tracked person can be missing before restarting from height
MAX_IDENTITY_GAP_FRAMES = 30

# frames in a row the detections can be rejected before the track restarts. A
# single detection nowhere near the athlete is usually someone in the background;
# if it keeps happening the athlete has really moved and the track restarts.
MAX_IDENTITY_REJECT_FRAMES = 3


def _bbox_height(landmarks: list) -> float:
    ys = [lm.y for lm in landmarks]
    return max(ys) - min(ys)


def _hip_centre(landmarks: list) -> tuple[float, float] | None:
    """Mid-hip in normalised coordinates - the steadiest point on a body."""
    try:
        left, right = landmarks[LEFT_HIP], landmarks[RIGHT_HIP]
    except (IndexError, TypeError):
        return None
    x, y = (left.x + right.x) / 2.0, (left.y + right.y) / 2.0
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return float(x), float(y)


def _torso_length(landmarks: list) -> float:
    """Shoulder-centre to hip-centre, used as the scale for everything else."""
    hips = _hip_centre(landmarks)
    if hips is None:
        return 0.0
    try:
        ls, rs = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
    except (IndexError, TypeError):
        return 0.0
    sx, sy = (ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0
    if not (np.isfinite(sx) and np.isfinite(sy)):
        return 0.0
    return float(math.hypot(sx - hips[0], sy - hips[1]))


class _PersonTracker:
    """
    Decides which detected person is the athlete and sticks with them.

    Tallest person alone doesn't work - at the bottom of a squat someone standing
    behind can be taller and take over. So height only starts the track, then it
    follows whoever is nearest to last frame's hip position (in torso lengths).
    switches counts how often it had to restart.
    """

    def __init__(self) -> None:
        self.switches = 0
        self.rejected_frames = 0
        self._last_centre: tuple[float, float] | None = None
        self._last_frame: int = -1
        self._rejects = 0

    def select(self, poses: list, frame_index: int) -> list:
        """The athlete's landmarks for this frame, or [] if none of the detections
        can be the athlete."""
        if not poses:
            return []

        stale = self._last_centre is None or frame_index - self._last_frame > MAX_IDENTITY_GAP_FRAMES
        if not stale:
            match = self._nearest(poses)
            if match is not None:
                self._rejects = 0
                self._remember(match, frame_index)
                return match
            # every detection is too far from where the athlete was. Taking one
            # anyway is what threw the skeleton across the frame, so skip it and
            # only restart the track if it keeps happening.
            self._rejects += 1
            self.rejected_frames += 1
            if self._rejects <= MAX_IDENTITY_REJECT_FRAMES:
                return []
            self.switches += 1

        self._rejects = 0
        tallest = max(poses, key=_bbox_height)
        self._remember(tallest, frame_index)
        return tallest

    def _nearest(self, poses: list) -> list | None:
        """Closest candidate to the last known position, if any is close enough."""
        assert self._last_centre is not None
        best, best_distance = None, float("inf")
        for candidate in poses:
            centre = _hip_centre(candidate)
            if centre is None:
                continue
            torso = _torso_length(candidate)
            if torso <= 0:
                continue
            drift = math.hypot(centre[0] - self._last_centre[0], centre[1] - self._last_centre[1])
            if drift / torso > MAX_IDENTITY_DRIFT_TORSOS:
                continue
            if drift < best_distance:
                best, best_distance = candidate, drift
        return best

    def _remember(self, landmarks: list, frame_index: int) -> None:
        centre = _hip_centre(landmarks)
        if centre is not None:
            self._last_centre = centre
            self._last_frame = frame_index


def _primary_pose(poses: list) -> list:
    """Tallest person in a single frame, no history. The main loop uses _PersonTracker."""
    if len(poses) == 1:
        return poses[0]
    return max(poses, key=_bbox_height)
