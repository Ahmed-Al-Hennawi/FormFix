"""
The FormFix AI homepage.

Exactly the sections of the original single-page site, in the original order.
The only change is that the upload / analysis experience is no longer one of
them: it has its own page now (``views/analyse.py``), and the "Analyse Form"
call to action links there.
"""

from __future__ import annotations

import streamlit as st

import components
from utils.assets import missing_assets
from utils.styling import inject_behaviour


def _warn_about_missing_assets() -> None:
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


def render() -> None:
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
    components.technology.render()           # 7  Technology
    components.research.render()             # 7b Academic foundation
    components.outro.render()                # 8  Final CTA + footer

    _warn_about_missing_assets()

    # Behaviour last, so everything it enhances is already in the DOM.
    inject_behaviour()
