"""
Analysis report rendering.

Two uses:

* :func:`render_example` - section 6 of the website, the worked example that
  shows what a session looks like (identical figures to the original prototype)
* :func:`render_result` - the same card, driven by a real
  :class:`~utils.analysis.AnalysisResult` once the computer-vision backend is
  connected

Because both go through :func:`report_markup`, real results will look exactly
like the example the marketing page promises.
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
    """Build the analysis report card for a result."""
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
    """Section 6 - the worked example from the original prototype."""
    html(
        strip(
            f"""
            <div class="ff-page">
            <section class="analysis section" id="ff-analysis">
              <div class="container container--narrow">
                <p class="eyebrow" data-animate="fade-up">Example analysis</p>
                <h2 class="display display--md" data-animate="fade-up">What a session looks like.</h2>
                {report_markup(demo_result())}
              </div>
            </section>
            </div>
            """
        )
    )


def render_result(result: AnalysisResult | None) -> None:
    """
    Render a real analysis result inside the upload section.

    ``None`` means the computer-vision backend has not been connected yet, so
    the placeholder explains that rather than inventing a fake analysis.
    """
    if result is None:
        html(
            strip(
                """
                <div class="ff-page">
                <div class="results__empty">
                  <span class="results__empty-tag">PENDING</span>
                  <p>Pose estimation is not connected in this prototype yet. Once
                     <code>utils/analysis.py</code> is implemented, the analysis report
                     appears here in exactly the format shown further up the page.</p>
                </div>
                </div>
                """
            )
        )
        return

    html(f'<div class="ff-page">{report_markup(result, animate=False)}</div>')
