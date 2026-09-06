"""The FormFix homepage."""

from __future__ import annotations

import streamlit as st

import components
from utils.assets import missing_assets
from utils.styling import inject_behaviour


def _warn_about_missing_assets() -> None:
    """Quiet note at the bottom of the page if an image is missing. Missing assets
    render as a transparent pixel, so without this you would never notice."""
    missing = missing_assets()
    if missing:
        with st.container(key="ff_asset_warning"):
            st.warning(
                "Some assets could not be found and are rendering as blanks: "
                + ", ".join(missing)
                + ". Copy them into `streamlit_app/static/assets/`."
            )


def render() -> None:
    # Fixed layers first; scripts/main.js moves them up to <body>.
    components.background.render()
    components.navbar.render()

    # Sections in the order of the original prototype.
    components.hero.render()
    components.about.render()
    components.problem.render()
    components.how_it_works.render()
    components.exercise_section.render()
    components.explainable.render()
    components.results_section.render_example()
    components.technology.render()
    components.research.render()
    components.outro.render()

    _warn_about_missing_assets()

    # Behaviour last, so everything it hooks into is already in the DOM.
    inject_behaviour()
