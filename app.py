"""
FormFix AI - Streamlit application
==================================

A Python/Streamlit rebuild of the FormFix AI prototype website. The original
HTML/CSS/JavaScript version lives one folder up and is left completely
untouched; this application reproduces its design, content and behaviour.

Run it with::

    streamlit run app.py

Structure
---------

    app.py              this file - page config, then one call per section
    components/         one module per section of the page
    utils/              paths, assets, styling, pose renderer, exercise data,
                        and the placeholder analysis pipeline
    styles/main.css     the full stylesheet, ported from the prototype
    scripts/main.js     the small behaviour layer (scroll progress, reveals,
                        exercise explorer, smooth navigation)
    static/assets/      copies of the original images, logo and research paper
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

from utils.assets import LOGO_MARK, asset_path, missing_assets  # noqa: E402
from utils.styling import inject_behaviour, inject_styles  # noqa: E402
import components  # noqa: E402


def configure_page() -> None:
    """Wide, dark, no sidebar - the app should read as a product website."""
    icon = asset_path(LOGO_MARK)
    st.set_page_config(
        page_title="FormFix AI - See your form. Understand the mistake.",
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


def warn_about_missing_assets() -> None:
    """
    A quiet, non-fatal note if an image could not be found on disk.

    Rendered at the very bottom of the page so it can never interfere with the
    design; missing assets fall back to a transparent pixel rather than a
    broken-image icon.
    """
    missing = missing_assets()
    if missing:
        with st.container(key="ff_asset_warning"):
            st.warning(
                "Some assets could not be found and are rendering as blanks: "
                + ", ".join(missing)
                + ". Copy them into `streamlit_app/static/assets/`."
            )


def main() -> None:
    configure_page()
    inject_styles()

    # Fixed layers first - they get relocated to <body> by scripts/main.js.
    components.background.render()
    components.navbar.render()

    # The page, in the order of the original prototype.
    components.hero.render()                 # 1  Hero
    components.problem.render()              # 2  The problem
    components.how_it_works.render()         # 3  How it works
    components.exercise_section.render()     # 4  Exercise explorer
    components.explainable.render()          # 5  Explainable AI
    components.results_section.render_example()  # 6  Example analysis
    components.upload_section.render()       # -  Video upload / analysis
    components.technology.render()           # 7  Technology
    components.research.render()             # 7b Academic foundation
    components.outro.render()                # 8  Final CTA + footer

    warn_about_missing_assets()

    # Behaviour last, so everything it enhances is already in the DOM.
    inject_behaviour()


main()
