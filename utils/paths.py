"""
File paths for the app, based on this file's location so it runs from any
working directory.
"""

from __future__ import annotations

from pathlib import Path

# Roots

APP_ROOT: Path = Path(__file__).resolve().parent.parent

# the original HTML/CSS/JS prototype one level up, only used as a fallback if
# an asset is missing
LEGACY_ROOT: Path = APP_ROOT.parent
LEGACY_ASSETS: Path = LEGACY_ROOT / "assets"

# Application directories

STYLES_DIR: Path = APP_ROOT / "styles"
SCRIPTS_DIR: Path = APP_ROOT / "scripts"

# served at app/static/... only when server.enableStaticServing is on in
# .streamlit/config.toml
STATIC_DIR: Path = APP_ROOT / "static"
ASSETS_DIR: Path = STATIC_DIR / "assets"
IMAGES_DIR: Path = ASSETS_DIR / "images"
LOGO_DIR: Path = ASSETS_DIR / "logo"
DOCS_DIR: Path = ASSETS_DIR / "docs"

# reference clips (optional, the analysis page falls back to a still image)
VIDEOS_DIR: Path = ASSETS_DIR / "videos"

MAIN_CSS: Path = STYLES_DIR / "main.css"
MAIN_JS: Path = SCRIPTS_DIR / "main.js"

# only loaded on /analyse, on top of main.css / main.js
ANALYSE_CSS: Path = STYLES_DIR / "analyse.css"
ANALYSE_JS: Path = SCRIPTS_DIR / "analyse.js"


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return default
