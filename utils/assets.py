"""
Turns an asset file on disk into a URL the browser can fetch. The prototype's
relative paths stop resolving once Streamlit renders the HTML in its own
document.

Preferred route is Streamlit's static serving, so the browser caches the images
and reruns don't re-send them. Without it we base64 the file into a data URI,
which works but is much heavier - hence the cache.
"""

from __future__ import annotations

import base64
import functools
import mimetypes
from pathlib import Path

import streamlit as st

from .paths import ASSETS_DIR, LEGACY_ASSETS, STATIC_DIR

# Filenames as they live in streamlit_app/static/assets/…
IMAGE_SQUAT = "images/squat-front-view.png"
IMAGE_PRESS = "images/shoulder-press-dumbbell.png"
IMAGE_PULLDOWN = "images/lat-pulldown-back-view.png"
LOGO_MARK = "logo/logo-mark.png"
LOGO_FULL = "logo/formfix-logo.png"
RESEARCH_PAPER = "docs/research-paper.pdf"

# Optional reference-technique clips, keyed by exercise id. Drop a file into
# static/assets/videos/ under one of these names and the analysis page
# plays it in the reference modal; without it the modal falls back to a still.
# They have to be H.264 MP4 - Chrome, Firefox and Edge won't decode HEVC in a
# <video> element. The HEVC originals live, unshipped, in
# static/assets/videos/_source_hevc/.
REFERENCE_VIDEOS: dict[str, str] = {
    "squat": "videos/squat-reference.mp4",
    "press": "videos/shoulder-press-reference.mp4",
    "pulldown": "videos/lat-pulldown-reference.mp4",
}

# Original filenames, used only as a read-only fallback if a copy is missing.
_LEGACY_EQUIVALENTS = {
    IMAGE_SQUAT: "squat - front view.png",
    IMAGE_PRESS: "shoulder press dumbbell.png",
    IMAGE_PULLDOWN: "lat pulldown - back view.png",
    LOGO_MARK: "logo-mark.png",
    LOGO_FULL: "new - logo.png",
    RESEARCH_PAPER: "research-paper.pdf",
}

# 1x1 transparent PNG, stood in for a missing asset so the layout doesn't
# collapse around a broken image.
_TRANSPARENT_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _static_serving_enabled() -> bool:
    """True when Streamlit will serve static/ over HTTP."""
    try:
        from streamlit import config

        return bool(config.get_option("server.enableStaticServing"))
    except Exception:  # pragma: no cover - very old / very new Streamlit
        return False


def asset_path(relative: str) -> Path | None:
    """Find an asset on disk, falling back to the old prototype folder."""
    candidate = ASSETS_DIR / relative
    if candidate.is_file():
        return candidate

    legacy_name = _LEGACY_EQUIVALENTS.get(relative)
    if legacy_name:
        legacy = LEGACY_ASSETS / legacy_name
        if legacy.is_file():
            return legacy
    return None


@st.cache_data(show_spinner=False)
def _data_uri(path_str: str, mtime: float) -> str:
    """Base64-encode a file as a data URI (cached on path + mtime)."""
    path = Path(path_str)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def asset_url(relative: str) -> str:
    """
    A URL the browser can use for an asset inside static/assets/.
    relative looks like "images/squat-front-view.png".
    """
    path = asset_path(relative)
    if path is None:
        return _TRANSPARENT_PNG

    if _static_serving_enabled():
        try:
            served = path.resolve().relative_to(STATIC_DIR.resolve())
        except ValueError:
            served = None
        if served is not None:
            # Kept relative so the app still works behind a baseUrlPath.
            return "app/static/" + served.as_posix()

    return _data_uri(str(path), path.stat().st_mtime)


def reference_video_url(exercise_id: str) -> str | None:
    """
    URL of an exercise's reference clip, or None if it isn't there - which
    is the normal state, and just means the modal shows a still frame instead.
    """
    relative = REFERENCE_VIDEOS.get(exercise_id)
    if not relative or asset_path(relative) is None:
        return None
    return asset_url(relative)


@functools.lru_cache(maxsize=1)
def missing_assets() -> tuple[str, ...]:
    """Names of expected assets that could not be found anywhere."""
    expected = (
        IMAGE_SQUAT,
        IMAGE_PRESS,
        IMAGE_PULLDOWN,
        LOGO_MARK,
        LOGO_FULL,
        RESEARCH_PAPER,
    )
    return tuple(name for name in expected if asset_path(name) is None)
