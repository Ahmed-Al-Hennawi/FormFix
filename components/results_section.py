"""
Report card rendering. render_example draws the example in section 6 and
render_result a real result. Both use report_markup so they look the same.
"""

from __future__ import annotations

from utils.analysis import AnalysisResult, demo_result
from utils.helpers import strip
from utils.styling import html

_BAR_MODIFIER = {
    "successful": " report__bar--cyan",
    "good": "",
    "attention": " report__bar--warn",
}

_BADGE_MODIFIER = {
    "successful": "badge--cyan",
    "good": "badge--lime",
    "attention": "badge--warn",
}


def report_markup(result: AnalysisResult, animate: bool = True) -> str:
    rows = []
    for row in result.rows:
        detail = f'<span class="report__sub">{row.detail}</span>' if row.detail else ""
        rows.append(
            '<li class="report__row">'
            f'<span class="report__label">{row.label}{detail}</span>'
            '<span class="report__meter">'
            f'<span class="report__bar{_BAR_MODIFIER.get(row.status, "")}" '
            f'style="--w:{row.score:g}%"></span></span>'
            f'<span class="badge {_BADGE_MODIFIER.get(row.status, "badge--lime")}">{row.badge}</span>'
            "</li>"
        )

    animate_attr = ' data-animate="report"' if animate else ""

    return (
        f'<div class="report"{animate_attr}>'
        '<header class="report__head">'
        "<div>"
        f'<p class="report__title">{result.exercise_name} Analysis</p>'
        f'<p class="report__file">{result.filename} &middot; {result.duration} '
        f"&middot; reps detected: {result.reps}</p>"
        "</div>"
        '<span class="report__status"><span class="hud__dot"></span>ANALYSIS COMPLETE</span>'
        "</header>"
        f'<ul class="report__rows">{"".join(rows)}</ul>'
        '<footer class="report__feedback">'
        '<span class="report__feedback-tag">FEEDBACK</span>'
        f"<p>{result.feedback}</p>"
        "</footer>"
        "</div>"
    )


def render_example() -> None:
    """Section 6 - the example from the original prototype."""
    html(strip(f"""
            <div class="ff-page">
            <section class="analysis section" id="ff-analysis">
              <div class="container container--narrow">
                <p class="eyebrow" data-animate="fade-up">Example analysis</p>
                <h2 class="display display--md" data-animate="fade-up">What a session looks like.</h2>
                {report_markup(demo_result())}
              </div>
            </section>
            </div>
            """))


def render_result(result: AnalysisResult | None) -> None:
    """Render a real result. None means the analysis couldn't run, and the empty
    state says so instead of making up a result."""
    if result is None:
        html(strip("""
                <div class="ff-page">
                <div class="results__empty">
                  <span class="results__empty-tag">UNAVAILABLE</span>
                  <p>This analysis could not be run on this machine - the pose-estimation
                     dependencies are not available here, or that movement has no analyser
                     registered. Everything else is unchanged: a completed run reports in
                     exactly the format shown further up the page.</p>
                </div>
                </div>
                """))
        return

    html(f'<div class="ff-page">{report_markup(result, animate=False)}</div>')
