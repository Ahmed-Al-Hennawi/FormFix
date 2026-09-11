"""
URL helpers for the two pages. The components are plain HTML, so links like
"Analyse Form" and "Back Home" need a real href. These respect
server.baseUrlPath.
"""

from __future__ import annotations

import streamlit as st

# url_path values used in app.py
HOME_PATH = "home"
ANALYSE_PATH = "analyse"


def _prefix() -> str:
    """The configured base path, normalised to "" or "/base"."""
    try:
        base = st.get_option("server.baseUrlPath") or ""
    except Exception:  # pragma: no cover - option missing on old Streamlit
        base = ""
    base = str(base).strip("/")
    return f"/{base}" if base else ""


def page_url(path: str = "") -> str:
    path = path.strip("/")
    prefix = _prefix()
    if not path:
        return prefix or "/"
    return f"{prefix}/{path}"


def home_url() -> str:
    return page_url()


def analyse_url() -> str:
    return page_url(ANALYSE_PATH)


def link(url: str, anchor: str = "") -> str:
    """Build an href, optionally with a section anchor.

    link(home_url(), "ff-how") -> "/#ff-how"
    """
    return f"{url}#{anchor}" if anchor else url
