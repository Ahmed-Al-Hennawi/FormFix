"""
FormFix AI - Streamlit application
==================================

A Python/Streamlit prototype website. 

Run it with::

    streamlit run app.py

Structure
---------

    app.py              this file - page config, then the page router
    views/              one module per page:
                          home.py     /          the marketing site
                          analyse.py  /analyse   upload + analysis studio
    components/         one module per section / feature
    utils/              paths, routing, assets, styling, pose renderer,
                        exercise data, and the placeholder analysis pipeline
    styles/main.css     the full stylesheet
    styles/analyse.css  additions used by the /analyse page only
    scripts/main.js     the small behaviour layer (scroll progress, reveals,
                        exercise explorer, smooth navigation)
    scripts/analyse.js  the /analyse page behaviour (reference lightbox,
                        score count-up)
    static/assets/      images, logo, research paper, reference clips
    .streamlit/         theme + server configuration
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `components` and `utils` importable no matter which directory the app
# was launched from.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.assets import LOGO_MARK, asset_path  # noqa: E402
from utils.routing import ANALYSE_PATH, HOME_PATH  # noqa: E402
from utils.styling import inject_styles  # noqa: E402
import views  # noqa: E402

HOME_TITLE = "FormFix AI - See your form. Understand the mistake."
ANALYSE_TITLE = "Analyse Your Form - FormFix AI"


def configure_page() -> None:
    """Wide, dark, no sidebar - the app should read as a product website."""
    icon = asset_path(LOGO_MARK)
    st.set_page_config(
        page_title=HOME_TITLE,
        page_icon=str(icon) if icon else ":material/fitness_center:",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": (
                "FormFix AI - an explainable computer-vision system for exercise "
                "technique assessment. MSc Computer Science thesis prototype."
            )
        },
    )


def build_navigation():
    """
    The two pages of the application.

    ``position="hidden"`` keeps Streamlit's own navigation out of the way -
    the site provides its own (the floating pill nav on the homepage, the
    "Back Home" button on the analysis page).
    """
    pages = [
        st.Page(views.home.render, title=HOME_TITLE, url_path=HOME_PATH, default=True),
        st.Page(views.analyse.render, title=ANALYSE_TITLE, url_path=ANALYSE_PATH),
    ]
    try:
        return st.navigation(pages, position="hidden")
    except TypeError:  # pragma: no cover - older Streamlit without "hidden"
        return st.navigation(pages)


def main() -> None:
    configure_page()

    # One stylesheet for the whole application; the analysis page adds its own
    # on top of this one.
    inject_styles()

    build_navigation().run()


main()
