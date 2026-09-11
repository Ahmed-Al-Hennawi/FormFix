"""
Page components. Homepage order: background, navbar, hero, about, problem,
how_it_works, exercise_section, explainable, results_section, technology,
research, outro, footer.

/analyse uses analyse_page (upload + analysis) and analysis_results (score,
feedback, reference videos). upload_section is the old uploader and isn't used
any more.
"""

from . import (
    about,
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
    "about",
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
