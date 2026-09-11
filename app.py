"""
FormFix - run with `streamlit run app.py`.

This file just sets up the pages and routing. views/ has the pages, components/
the sections, utils/ shared helpers, styles/ and scripts/ the CSS and JS, and
static/assets/ the images and reference clips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# so the imports work from any working directory
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import views  # noqa: E402
from utils.assets import LOGO_MARK, asset_path  # noqa: E402
from utils.routing import ANALYSE_PATH, HOME_PATH  # noqa: E402
from utils.styling import inject_styles  # noqa: E402

HOME_TITLE = "FormFix - See your form. Understand your movement."
ANALYSE_TITLE = "Analyse Your Form - FormFix"


def configure_page() -> None:
    icon = asset_path(LOGO_MARK)
    st.set_page_config(
        page_title=HOME_TITLE,
        page_icon=str(icon) if icon else ":material/fitness_center:",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": (
                "FormFix - an explainable computer-vision system for exercise "
                "technique assessment. MSc Computer Science thesis project."
            )
        },
    )


def build_navigation():

    pages = [
        st.Page(views.home.render, title=HOME_TITLE, url_path=HOME_PATH, default=True),
        st.Page(views.analyse.render, title=ANALYSE_TITLE, url_path=ANALYSE_PATH),
    ]
    try:
        return st.navigation(pages, position="hidden")
    except TypeError: 
        return st.navigation(pages)


@st.cache_resource(show_spinner=False)
def ensure_pose_model() -> bool:
    try:
        from analysis.pose_detector import fetch_model

        return fetch_model() is not None
    except Exception:  # the CV backend may not be installed at all
        return False


def main() -> None:
    configure_page()

    ensure_pose_model()

    # site-wide CSS, /analyse adds its own on top
    inject_styles()

    build_navigation().run()


main()
