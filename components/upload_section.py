"""
The old upload section, replaced by the /analyse page and no longer used.
Styled by section 15 of styles/main.css.
"""

from __future__ import annotations

import streamlit as st

from utils.analysis import SUPPORTED_VIDEO_TYPES, process_uploaded_video
from utils.exercise_data import BY_ID, EXERCISES
from utils.helpers import strip
from utils.styling import html

from . import results_section

PIPELINE_STRIP = (
    "VIDEO UPLOAD",
    "OPENCV",
    "MEDIAPIPE POSE",
    "LANDMARKS",
    "RULE ANALYSIS",
    "FEEDBACK",
)


def _head() -> None:
    strip_html = "<i>&rarr;</i>".join(f"<span>{item}</span>" for item in PIPELINE_STRIP)
    html(strip(f"""
            <div class="ff-page">
            <section class="analyse section" id="ff-analyse">
              <div class="container container--narrow analyse__head">
                <p class="eyebrow" data-animate="fade-up">Analyse your form</p>
                <h2 class="display display--md" data-animate="fade-up">
                  Upload a set.<br />Get an explanation.
                </h2>
                <p class="analyse__lede" data-animate="fade-up">
                  Pick the movement, drop in a short clip, and FormFix will track your
                  body through the lift and explain what it sees.
                </p>
                <div class="analyse__pipe" data-animate="fade-up">{strip_html}</div>
              </div>
            </section>
            </div>
            """))


def render() -> None:
    _head()

    with st.container(key="ff_analyse"):
        choice = st.radio(
            "Exercise",
            options=[exercise.id for exercise in EXERCISES],
            format_func=lambda eid: BY_ID[eid].name,
            horizontal=True,
            key="ff_exercise_choice",
        )

        uploaded = st.file_uploader(
            "Workout video",
            type=list(SUPPORTED_VIDEO_TYPES),
            accept_multiple_files=False,
            key="ff_video",
            help="MP4, MOV or AVI. A single set filmed from the side or front works best.",
        )

        html(
            '<p class="analyse__hint">Accepted formats &middot; '
            + " &middot; ".join(f".{ext}" for ext in SUPPORTED_VIDEO_TYPES)
            + "</p>"
        )

        if uploaded is not None:
            st.video(uploaded)

            analyse = st.button(
                "Analyse Form",
                type="primary",
                key="ff_analyse_button",
                use_container_width=False,
            )

            if analyse:
                with st.spinner("EXTRACTING LANDMARKS..."):
                    result = process_uploaded_video(uploaded, exercise_id=choice)
                st.session_state["ff_result"] = result
                st.session_state["ff_has_run"] = True

        if st.session_state.get("ff_has_run"):
            results_section.render_result(st.session_state.get("ff_result"))

    # spacer before the next section
    html('<div class="ff-page"><div style="height: clamp(96px, 12vw, 176px)"></div></div>')
