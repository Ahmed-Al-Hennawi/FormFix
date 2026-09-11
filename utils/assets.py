"""
Turns an asset file into a URL the browser can load. Uses Streamlit's static
serving when it's on (so images get cached), otherwise falls back to a base64
data URI, which is heavier - hence the cache.
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

# optional reference clips by exercise id, played in the reference modal (falls
# back to a still). Must be H.264 MP4 since Chrome, Firefox and Edge won't play
# HEVC. The HEVC originals are kept locally in static/assets/videos/_source_hevc/
REFERENCE_VIDEOS: dict[str, str] = {
    "squat": "videos/squat-reference.mp4",
    "press": "videos/shoulder-press-reference.mp4",
    "pulldown": "videos/lat-pulldown-reference.mp4",
}

# original filenames, only a fallback if a copy is missing
_LEGACY_EQUIVALENTS = {
    IMAGE_SQUAT: "squat - front view.png",
    IMAGE_PRESS: "shoulder press dumbbell.png",
    IMAGE_PULLDOWN: "lat pulldown - back view.png",
    LOGO_MARK: "logo-mark.png",
    LOGO_FULL: "new - logo.png",
    RESEARCH_PAPER: "research-paper.pdf",
}

# 1x1 transparent PNG for a missing asset, so the layout doesn't break
_TRANSPARENT_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _static_serving_enabled() -> bool:
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
    """Base64 data URI, cached on path + mtime."""
    path = Path(path_str)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def asset_url(relative: str) -> str:
    """Browser URL for an asset, e.g. asset_url("images/squat-front-view.png")."""
    path = asset_path(relative)
    if path is None:
        return _TRANSPARENT_PNG

    if _static_serving_enabled():
        try:
            served = path.resolve().relative_to(STATIC_DIR.resolve())
        except ValueError:
            served = None
        if served is not None:
            # relative so it still works behind a baseUrlPath
            return "app/static/" + served.as_posix()

    return _data_uri(str(path), path.stat().st_mtime)


def reference_video_url(exercise_id: str) -> str | None:
    """URL of an exercise's reference clip, or None (the modal then shows a still)."""
    relative = REFERENCE_VIDEOS.get(exercise_id)
    if not relative or asset_path(relative) is None:
        return None
    return asset_url(relative)


@functools.lru_cache(maxsize=1)
def missing_assets() -> tuple[str, ...]:
    expected = (
        IMAGE_SQUAT,
        IMAGE_PRESS,
        IMAGE_PULLDOWN,
        LOGO_MARK,
        LOGO_FULL,
        RESEARCH_PAPER,
    )
    return tuple(name for name in expected if asset_path(name) is None)
