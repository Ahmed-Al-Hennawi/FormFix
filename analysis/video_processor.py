"""
Reading videos and writing the H.264 output. Metadata isn't trusted, frames
are streamed instead of loaded into memory, and the output is re-encoded with
FFmpeg because most browsers won't play cv2's mp4v.
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

# only used when the file reports no usable FPS
DEFAULT_FPS = 30.0

# scratch space for uploads and rendered output
_WORK_ROOT = Path(tempfile.gettempdir()) / "formfix_analysis"


def new_session_dir() -> Path:
    session = _WORK_ROOT / uuid.uuid4().hex[:12]
    session.mkdir(parents=True, exist_ok=True)
    return session


# how long a finished run's scratch dir survives before a later run sweeps it
SESSION_TTL_SECONDS: float = 6 * 60 * 60


def purge_stale_sessions(ttl_seconds: float = SESSION_TTL_SECONDS) -> int:
    """Delete scratch dirs from earlier runs, since a closed tab or restart would
    leave videos in temp forever. Never raises."""
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
    """Read the file's metadata. A broken file returns readable=False instead of raising."""
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
    capture = cv2.VideoCapture(str(path))
    try:
        yield capture
    finally:
        capture.release()


def iter_frames(capture: cv2.VideoCapture) -> Iterator[tuple[int, np.ndarray]]:
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield index, frame
        index += 1


# --- Output ---


def _ffmpeg_executable() -> str | None:
    """
    System FFmpeg if installed, otherwise the one from imageio-ffmpeg (which is
    what runs on Streamlit Community Cloud).
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # imageio raises its own error types
        logger.warning("imageio-ffmpeg could not provide a binary: %s", exc)
        return None


def ffmpeg_available() -> bool:
    return _ffmpeg_executable() is not None


def _render_failure() -> AnalysisFailure:
    return AnalysisFailure(
        FailureCode.VIDEO_RENDER_ERROR,
        "The analysed video could not be written on this machine.",
        suggestions=["The analysis itself is unaffected - only the annotated clip is missing."],
    )


@contextmanager
def open_writer(path: Path, fps: float, size: tuple[int, int]) -> Iterator[cv2.VideoWriter]:
    """
    cv2.VideoWriter for the mp4v file. It silently drops every write if it can't
    open the file, hence the mkdir and isOpened check.
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
    Re-encode to H.264 so browsers can play it. Returns False on failure (the
    caller keeps the mp4v). +faststart lets playback start early, -an drops audio.
    """
    executable = _ffmpeg_executable()
    if executable is None:
        logger.warning("FFmpeg not found - annotated video stays mp4v (may not play in all browsers)")
        return False

    command = [
        executable,
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
