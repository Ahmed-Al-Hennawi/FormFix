"""
The results view of the /analyse page.

Everything here is driven by a single :class:`~utils.analysis.AnalysisResult`,
so the same markup renders a real pose-estimation run and the sample analysis
the prototype currently shows. Nothing is hard-coded to one exercise or to one
piece of feedback:

    score_block          animated ring + verdict (colour *and* words)
    checks_block         the measured checks behind the score
    positives_block      "What you did well"
    improvements_block   "What to improve" - a list, of any length
    reference_block      "See the correct technique" card
    reference_modal      the lightbox the card opens

The hierarchy follows the brief: verdict first, then what went well, then what
to fix, then the reference movement.
"""

from __future__ import annotations

from utils.analysis import AnalysisResult
from utils.assets import asset_url, reference_video_url
from utils.exercise_data import Exercise
from utils.helpers import esc
from utils.pose import render_pose

CHECK_ICON = (
    '<svg class="ax-icon" viewBox="0 0 16 16" aria-hidden="true">'
    '<path d="M3 8.4 6.2 11.6 13 4.8" fill="none" stroke="currentColor" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>'
)

PLAY_ICON = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6.5 18 12l-9 5.5z" '
    'fill="currentColor" /></svg>'
)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def score_block(result: AnalysisResult) -> str:
    """The animated score ring, the verdict and the run's headline facts."""
    demo_tag = (
        '<span class="ax-tag ax-tag--demo">Sample output</span>' if result.is_demo else ""
    )

    return (
        f'<div class="ax-score ax-score--{result.band}" data-animate="fade-up">'
        '  <div class="ax-score__ring-wrap">'
        '    <svg class="ax-score__ring" viewBox="0 0 120 120" aria-hidden="true">'
        '      <circle class="ax-score__track" cx="60" cy="60" r="52" />'
        f'      <circle class="ax-score__value" cx="60" cy="60" r="52" pathLength="100" '
        f'style="--pct:{result.score}" />'
        "    </svg>"
        '    <div class="ax-score__center">'
        f'      <span class="ax-score__num" data-ax-count-to="{result.score}">0</span>'
        '      <span class="ax-score__den">/ 100</span>'
        "    </div>"
        "  </div>"
        '  <div class="ax-score__meta">'
        '    <p class="ax-score__eyebrow">Analysis complete</p>'
        f'    <p class="ax-score__exercise">{esc(result.exercise_name)}</p>'
        f'    <p class="ax-score__verdict">{esc(result.verdict)}</p>'
        f'    <p class="ax-score__facts">{esc(result.filename)} &middot; '
        f"{esc(result.duration)} &middot; {result.reps} reps detected</p>"
        f"    {demo_tag}"
        "  </div>"
        "</div>"
    )


def checks_block(result: AnalysisResult) -> str:
    """
    The measured checks behind the score.

    Each row is one rule from the analysis, so a score can always be traced
    back to the specific check that produced it.
    """
    if not result.rows:
        return ""

    rows = []
    for row in result.rows:
        detail = f'<span class="ax-check__detail">{esc(row.detail)}</span>' if row.detail else ""
        rows.append(
            f'<li class="ax-check ax-check--{esc(row.status)}">'
            f'<span class="ax-check__label">{esc(row.label)}{detail}</span>'
            '<span class="ax-check__meter">'
            f'<span class="ax-check__bar" style="--w:{row.score:g}%"></span></span>'
            f'<span class="ax-check__value">{row.score:g}</span>'
            "</li>"
        )

    return (
        '<section class="ax-panel ax-panel--quiet" data-animate="fade-up">'
        '<h3 class="ax-panel__title ax-panel__title--sm">Measured checks</h3>'
        f'<ul class="ax-checks">{"".join(rows)}</ul>'
        "</section>"
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


def positives_block(result: AnalysisResult) -> str:
    """What the athlete did well - always shown before the corrections."""
    if not result.positives:
        return ""

    items = "".join(
        f'<li class="ax-win">{CHECK_ICON}<span>{esc(item)}</span></li>'
        for item in result.positives
    )
    return (
        '<section class="ax-panel ax-panel--wins" data-animate="fade-up">'
        '<h3 class="ax-panel__title">What you did well</h3>'
        f'<ul class="ax-wins">{items}</ul>'
        "</section>"
    )


def improvements_block(result: AnalysisResult) -> str:
    """Every detected issue, each with what was seen and how to fix it."""
    if not result.improvements:
        return (
            '<section class="ax-panel" data-animate="fade-up">'
            '<h3 class="ax-panel__title">What to improve</h3>'
            '<p class="ax-empty-note">No technique issues were detected in this set. '
            "Keep the same setup and add load gradually.</p>"
            "</section>"
        )

    cards = []
    for index, item in enumerate(result.improvements, start=1):
        cards.append(
            f'<article class="ax-fix ax-fix--{esc(item.severity)}" data-animate="fade-up">'
            '  <header class="ax-fix__head">'
            f'    <span class="ax-fix__num">{index:02d}</span>'
            f'    <h4 class="ax-fix__title">{esc(item.title)}</h4>'
            f'    <span class="ax-sev ax-sev--{esc(item.severity)}">{esc(item.severity_label)}</span>'
            "  </header>"
            f'  <p class="ax-fix__issue">{esc(item.issue)}</p>'
            '  <p class="ax-fix__how"><span class="ax-fix__how-tag">How to improve</span>'
            f"{esc(item.correction)}</p>"
            "</article>"
        )

    count = len(result.improvements)
    return (
        '<section class="ax-panel" data-animate="fade-up">'
        '<h3 class="ax-panel__title"><span>What to improve</span>'
        f'<span class="ax-count">{count} {"point" if count == 1 else "points"}</span></h3>'
        f'<div class="ax-fixes">{"".join(cards)}</div>'
        "</section>"
    )


# ---------------------------------------------------------------------------
# Reference movement
# ---------------------------------------------------------------------------


def reference_block(exercise: Exercise) -> str:
    """The card that opens the reference-technique lightbox."""
    return (
        '<section class="ax-panel ax-panel--ref" data-animate="fade-up">'
        '<h3 class="ax-panel__title">See the correct technique</h3>'
        '<button class="ax-ref" type="button" data-ax-open '
        f'aria-label="Play the {esc(exercise.name)} reference movement">'
        '  <span class="ax-ref__thumb">'
        f'    <img src="{asset_url(exercise.image)}" alt="" loading="lazy" />'
        '    <span class="ax-ref__scrim"></span>'
        f'    <span class="ax-ref__play">{PLAY_ICON}</span>'
        "  </span>"
        '  <span class="ax-ref__body">'
        '    <span class="ax-ref__eyebrow">Reference movement</span>'
        f'    <span class="ax-ref__title">{esc(exercise.name)} &mdash; Correct Technique</span>'
        '    <span class="ax-ref__meta">Watch how the movement should look, then compare '
        "it with your own set.</span>"
        "  </span>"
        "</button>"
        "</section>"
    )


def reference_modal(exercise: Exercise) -> str:
    """
    The lightbox itself.

    ``scripts/analyse.js`` moves this to ``<body>`` on load so its fixed
    positioning is resolved against the viewport rather than against a
    Streamlit block, then handles open / close / Escape / backdrop clicks.
    """
    video = reference_video_url(exercise.id)

    if video:
        media = (
            f'<video class="ax-modal__video" src="{video}" controls playsinline '
            f'preload="metadata" poster="{asset_url(exercise.image)}"></video>'
        )
        note = "Reference clip"
    else:
        # No clip has been added yet - show the annotated still instead of an
        # empty player, so the modal is still useful and never looks broken.
        media = (
            '<figure class="ax-modal__still">'
            f'<img src="{asset_url(exercise.image)}" alt="{esc(exercise.image_alt)}" />'
            f'{render_pose(exercise.pose, key="ref", extra_class="pose--ref")}'
            '<span class="scanline"></span>'
            "</figure>"
        )
        note = "Reference still &middot; clip pending"

    areas = " &middot; ".join(esc(area) for area in exercise.areas)

    return (
        '<div class="ax-modal" data-ax-modal hidden>'
        '  <div class="ax-modal__backdrop" data-ax-close></div>'
        '  <div class="ax-modal__panel" role="dialog" aria-modal="true" '
        f'aria-label="{esc(exercise.name)} reference technique">'
        '    <button class="ax-modal__close" type="button" data-ax-close aria-label="Close">'
        '      <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" '
        'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>'
        "    </button>"
        f'    <div class="ax-modal__media">{media}</div>'
        '    <div class="ax-modal__caption">'
        f'      <p class="ax-modal__title">{esc(exercise.name)} &mdash; Correct Technique</p>'
        f'      <p class="ax-modal__meta">{note} &middot; {areas}</p>'
        "    </div>"
        "  </div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def markup(result: AnalysisResult, exercise: Exercise) -> str:
    """The whole results column, in the order the brief asks for."""
    return (
        '<div class="ff-page ax-results" data-ax-results>'
        f"{score_block(result)}"
        f"{positives_block(result)}"
        f"{improvements_block(result)}"
        f"{checks_block(result)}"
        f"{reference_block(exercise)}"
        "</div>"
    )
