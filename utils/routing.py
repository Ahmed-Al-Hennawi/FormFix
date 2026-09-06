"""
URL helpers for the two pages. Streamlit owns the URLs, but the components emit
plain HTML, so the "Analyse Form" and "Back Home" links need a real href rather
than a widget. These build one and honour server.baseUrlPath.
"""

from __future__ import annotations

import streamlit as st

# The url_path values app.py declares the pages with.
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
    """Root-relative URL for one of the app's pages."""
    path = path.strip("/")
    prefix = _prefix()
    if not path:
        return prefix or "/"
    return f"{prefix}/{path}"


def home_url() -> str:
    """The homepage, which Streamlit serves from the app root."""
    return page_url()


def analyse_url() -> str:
    """The upload / analysis page."""
    return page_url(ANALYSE_PATH)


def link(url: str, anchor: str = "") -> str:
    """Build an href, optionally with a section anchor.

    link(home_url(), "ff-how") -> "/#ff-how"
    """
    return f"{url}#{anchor}" if anchor else url
