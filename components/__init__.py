"""
Page components.

One module per section of the FormFix AI website, in the order they appear:

    background          fixed scroll-progress line + atmosphere layers
    navbar              floating pill navigation
    hero                1  - hero
    problem             2  - the problem
    how_it_works        3  - how FormFix AI works
    exercise_section    4  - exercise explorer (squat / press / pulldown)
    explainable         5  - explainable AI + traceability pipeline
    results_section     6  - example analysis report (and real results)
    upload_section      -  - video upload / analysis (Streamlit widgets)
    technology          7  - under the hood
    research            7b - academic & technical foundation
    outro               8  - final call to action
    footer              -  - footer
"""

from . import (
    background,
    exercise_section,
    explainable,
    footer,
    hero,
    how_it_works,
    navbar,
    outro,
    problem,
    research,
    results_section,
    technology,
    upload_section,
)

__all__ = [
    "background",
    "exercise_section",
    "explainable",
    "footer",
    "hero",
    "how_it_works",
    "navbar",
    "outro",
    "problem",
    "research",
    "results_section",
    "technology",
    "upload_section",
]
