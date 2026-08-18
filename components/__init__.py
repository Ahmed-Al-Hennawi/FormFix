"""
Page components.

Homepage sections, in the order they appear:

    background          fixed scroll-progress line + atmosphere layers
    navbar              floating pill navigation
    hero                1  - hero
    problem             2  - the problem
    how_it_works        3  - how FormFix AI works
    exercise_section    4  - exercise explorer (squat / press / pulldown)
    explainable         5  - explainable AI + traceability pipeline
    results_section     6  - example analysis report (and real results)
    technology          7  - under the hood
    research            7b - academic & technical foundation
    outro               8  - final call to action
    footer              -  - footer

The /analyse page (see ``views/analyse.py``):

    analyse_page        the upload + analysis experience
    analysis_results    the score, feedback and reference-video views

    upload_section      the original in-page uploader. Superseded by
                        analyse_page and no longer rendered on the homepage;
                        kept so nothing that referenced it breaks.
"""

from . import (
    analyse_page,
    analysis_results,
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
    "analyse_page",
    "analysis_results",
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
