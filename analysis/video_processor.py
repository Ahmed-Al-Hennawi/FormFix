"""
Video reading, probing and H.264 output. Nothing here trusts the reported
metadata, frames are streamed rather than decoded into memory, and the output
is re-encoded with FFmpeg because cv2 only gives mp4v, which most browsers
won't play. If FFmpeg is missing the mp4v file comes back and the UI warns.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from .models import AnalysisFailure, FailureCode, VideoMetadata

logger = logging.getLogger(__name__)

# used only when the container reports no usable FPS
DEFAULT_FPS = 30.0

# scratch space for uploads and rendered output
_WORK_ROOT = Path(tempfile.gettempdir()) / "formfix_analysis"


def new_session_dir() -> Path:
    """A unique scratch directory for one analysis run."""
    session = _WORK_ROOT / uuid.uuid4().hex[:12]
    session.mkdir(parents=True, exist_ok=True)
    return session


# how long a finished run's scratch dir survives before a later run sweeps it
SESSION_TTL_SECONDS: float = 6 * 60 * 60


def purge_stale_sessions(ttl_seconds: float = SESSION_TTL_SECONDS) -> int:
    """Delete scratch dirs left by earlier runs - a closed tab or a restart leaves
    rendered video in temp forever. Never raises."""
    if not _WORK_ROOT.is_dir():
        return 0
    cutoff = time.time() - ttl_seconds
    removed = 0
    try:
        candidates = list(_WORK_ROOT.iterdir())
    except OSError:  # pragma: no cover - unreadable temp space
        return 0
    for session in candidates:
        try:
            if not session.is_dir() or session.stat().st_mtime > cutoff:
                continue
            shutil.rmtree(session, ignore_errors=True)
            removed += not session.exists()
        except OSError:  # pragma: no cover - a concurrent run owns it
            continue
    if removed:
        logger.info("Removed %d stale analysis scratch director(ies)", removed)
    return removed


def cleanup_session_dir(session_dir: Path, keep: tuple[Path, ...] = ()) -> None:
    """Delete a session's scratch files, keeping anything in keep (usually the
    annotated video Streamlit is still serving)."""
    keep_resolved = {p.resolve() for p in keep if p is not None}
    try:
        for item in sorted(session_dir.rglob("*"), reverse=True):
            if item.is_file() and item.resolve() not in keep_resolved:
                item.unlink(missing_ok=True)
        if not any(session_dir.iterdir()):
            session_dir.rmdir()
    except OSError:  # pragma: no cover - cleanup must never break a run
        logger.warning("Could not fully clean session dir %s", session_dir)


def probe_video(path: Path) -> VideoMetadata:
    """Read the file's metadata, treating every value as suspect. A broken file
    comes back with readable False instead of raising."""
    width = height = frame_count = 0
    fps = 0.0
    fourcc = ""

    capture = cv2.VideoCapture(str(path))
    try:
        if capture.isOpened():
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            raw_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
            if raw_fourcc:
                fourcc = "".join(chr((raw_fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()

            # some containers report zero or absurd values, so try a real read first
            if frame_count <= 0 or fps <= 0 or fps > 240:
                ok, _ = capture.read()
                if ok and frame_count <= 0:
                    pass
                if fps <= 0 or fps > 240:
                    logger.warning(
                        "Unusable FPS %.2f reported for %s; assuming %.0f", fps, path, DEFAULT_FPS
                    )
                    fps = DEFAULT_FPS if ok else 0.0
    finally:
        capture.release()

    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return VideoMetadata(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        fourcc=fourcc,
    )


@contextmanager
def open_video(path: Path) -> Iterator[cv2.VideoCapture]:
    """cv2.VideoCapture with guaranteed release."""
    capture = cv2.VideoCapture(str(path))
    try:
        yield capture
    finally:
        capture.release()


def iter_frames(capture: cv2.VideoCapture) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) in original order, streaming."""
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield index, frame
        index += 1


# --- Output ---


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _render_failure() -> AnalysisFailure:
    """The failure raised when the annotated clip can't be written."""
    return AnalysisFailure(
        FailureCode.VIDEO_RENDER_ERROR,
        "The analysed video could not be written on this machine.",
        suggestions=["The analysis itself is unaffected - only the annotated clip is missing."],
    )


@contextmanager
def open_writer(path: Path, fps: float, size: tuple[int, int]) -> Iterator[cv2.VideoWriter]:
    """
    cv2.VideoWriter for the mp4v intermediate, released on exit.

    It doesn't create the parent folder and doesn't error when it can't open the
    file - it just silently drops every write. Hence the mkdir and isOpened check.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _render_failure() from exc

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        writer.release()
        raise _render_failure()
    try:
        yield writer
    finally:
        writer.release()


def convert_to_h264(source: Path, target: Path, timeout: int = 600) -> bool:
    """
    Re-encode to H.264 so browsers can play it. False on failure, and the caller
    keeps the mp4v rather than losing the run. +faststart puts the moov atom at
    the front so playback can start early; -an drops the audio.
    """
    if not ffmpeg_available():
        logger.warning("FFmpeg not found - annotated video stays mp4v (may not play in all browsers)")
        return False

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(target),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
        logger.warning("FFmpeg conversion failed: %s", exc)
        return False

    if completed.returncode != 0:
        logger.warning(
            "FFmpeg exited %s: %s",
            completed.returncode,
            completed.stderr.decode(errors="replace")[:500],
        )
        return False

    playable = probe_video(target)
    if not playable.readable:
        logger.warning("FFmpeg output not readable, keeping mp4v file")
        return False
    return True
