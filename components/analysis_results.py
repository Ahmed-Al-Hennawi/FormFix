"""
Results view for /analyse. Everything comes from one AnalysisResult, so a real
run and the sample look the same.

Order: analysed video, verdict, corrections and positives, reference clip, then
the per-finding and per-rep detail in "See the full detail" (the full trace
with FORMFIX_DEBUG).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from utils.analysis import AnalysisResult, Improvement
from utils.assets import asset_url, reference_video_url
from utils.exercise_data import Exercise
from utils.helpers import esc
from utils.pose import render_pose
from utils.styling import html

CHECK_ICON = (
    '<svg class="ax-icon" viewBox="0 0 16 16" aria-hidden="true">'
    '<path d="M3 8.4 6.2 11.6 13 4.8" fill="none" stroke="currentColor" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>'
)

PLAY_ICON = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6.5 18 12l-9 5.5z" '
    'fill="currentColor" /></svg>'
)

# beginners care about their form, not a score
VERDICT_EYEBROW = "Overall form"


# --- Level 1 - the verdict ---


def verdict_block(result: AnalysisResult) -> str:
    """Verdict in words plus one sentence on what it's based on. There's a coloured
    ring too, but nothing relies on reading the colour."""
    demo_tag = '<span class="ax-tag ax-tag--demo">Sample output</span>' if result.is_demo else ""
    facts = " &middot; ".join(
        part
        for part in (
            esc(result.exercise_name),
            f"{result.reps} rep{'' if result.reps == 1 else 's'}" if result.reps else "",
            esc(result.duration),
        )
        if part
    )
    headline = f'<p class="ax-score__headline">{esc(result.headline)}</p>' if result.headline else ""

    # nothing reliable to score - a 0 would look like "poor technique"
    if not result.score_available:
        ring = (
            '  <div class="ax-score__ring-wrap">'
            '    <svg class="ax-score__ring" viewBox="0 0 120 120" aria-hidden="true">'
            '      <circle class="ax-score__track" cx="60" cy="60" r="52" />'
            "    </svg>"
            '    <div class="ax-score__center">'
            '      <span class="ax-score__num">&mdash;</span>'
            '      <span class="ax-score__den">not scored</span>'
            "    </div>"
            "  </div>"
        )
        verdict = "Not scored"
        modifier = "attention"
    else:
        ring = (
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
        )
        verdict = result.verdict
        modifier = result.band

    return (
        f'<div class="ax-score ax-score--{modifier}" data-animate="fade-up">'
        f"{ring}"
        '  <div class="ax-score__meta">'
        f'    <p class="ax-score__eyebrow">{VERDICT_EYEBROW}</p>'
        f'    <p class="ax-score__verdict-lg">{esc(verdict)}</p>'
        f"    {headline}"
        f'    <p class="ax-score__facts">{facts}</p>'
        f"    {demo_tag}"
        "  </div>"
        "</div>"
    )


# --- Level 1 - corrections and positives ---


def _correction_card(index: int, item: Improvement) -> str:
    """One correction: what happened, why it matters, what to try. No numbers here,
    they're in the expander."""
    why = f'<p class="ax-fix__why">{esc(item.why)}</p>' if item.why else ""
    return (
        f'<li class="ax-fix ax-fix--{esc(item.severity)}" data-animate="fade-up">'
        '  <div class="ax-fix__head">'
        f'    <span class="ax-fix__num">{index:02d}</span>'
        f'    <h4 class="ax-fix__title">{esc(item.title)}</h4>'
        "  </div>"
        f'  <p class="ax-fix__issue">{esc(item.headline_issue)}</p>'
        f"  {why}"
        '  <p class="ax-fix__how"><span class="ax-fix__how-tag">Try this</span>'
        f"{esc(item.headline_fix)}</p>"
        "</li>"
    )


def corrections_block(result: AnalysisResult) -> str:
    """ "What to fix" - the top few findings, already sorted by priority."""
    visible = result.key_improvements
    if not visible:
        return (
            '<section class="ax-panel ax-panel--clear" data-animate="fade-up">'
            '<h3 class="ax-panel__title">Nothing to fix in this set</h3>'
            '<p class="ax-empty-note">None of the checks FormFix could measure found a '
            "problem. Keep the same setup and add load gradually.</p>"
            "</section>"
        )

    more = len(result.further_improvements)
    note = (
        f'<p class="ax-fixes__more">{more} smaller point{"" if more == 1 else "s"} '
        "in the analysis details below.</p>"
        if more
        else ""
    )
    cards = "".join(_correction_card(index, item) for index, item in enumerate(visible, start=1))
    count = len(visible)
    return (
        '<section class="ax-panel" data-animate="fade-up">'
        '<h3 class="ax-panel__title"><span>What to fix</span>'
        f'<span class="ax-count">{count} thing{"" if count == 1 else "s"}</span></h3>'
        f'<ol class="ax-fixes">{cards}</ol>'
        f"{note}"
        "</section>"
    )


def positives_block(result: AnalysisResult) -> str:
    """What went well - only from checks that actually passed."""
    items = result.key_positives
    if not items:
        return ""
    rows = "".join(f'<li class="ax-win">{CHECK_ICON}<span>{esc(item)}</span></li>' for item in items)
    return (
        '<section class="ax-panel ax-panel--wins" data-animate="fade-up">'
        '<h3 class="ax-panel__title">What you did well</h3>'
        f'<ul class="ax-wins">{rows}</ul>'
        "</section>"
    )


# --- Level 3 - how FormFix detected it ---


# evidence keys that aren't settings and are printed elsewhere
_SETTING_EXCLUSIONS = frozenset({"reliability_evidence", "threshold_source"})

# max number of settings to show
_MAX_SETTINGS = 8


def _format_setting(key: str, value: Any) -> str:
    name = key.replace("_", " ")
    if isinstance(value, float):
        return f"{name}: {value:.3g}"
    if isinstance(value, (list, tuple)):
        return f"{name}: {', '.join(str(v) for v in value)}"
    return f"{name}: {value}"


def _settings_lines(item: Improvement) -> list[str]:
    """The rule's thresholds and where they came from (simple values only)."""
    lines: list[str] = []
    scalars = [
        (key, value)
        for key, value in item.settings.items()
        if key not in _SETTING_EXCLUSIONS and not isinstance(value, dict)
    ]
    if scalars:
        rendered = " &middot; ".join(
            esc(_format_setting(key, value)) for key, value in scalars[:_MAX_SETTINGS]
        )
        lines.append(f'<p class="ax-ex__settings">Rule settings &mdash; {rendered}</p>')
    source = item.settings.get("threshold_source")
    if source:
        lines.append(f'<p class="ax-ex__settings">Threshold source &mdash; {esc(str(source))}</p>')
    return lines


def _detection_card(item: Improvement, minor: bool = False, technical: bool = False) -> str:
    """
    One finding in the details panel. Normally just the mistake and what to try.
    technical=True (FORMFIX_DEBUG) shows the full trace: landmarks, measurement,
    phase, rule and evidence.
    """
    if not technical:
        lines = [
            f'<p class="ax-ex__line">{esc(item.headline_issue)}</p>' if item.headline_issue else "",
            (
                f'<p class="ax-ex__line"><span class="ax-ex__tag">Try this</span>'
                f"{esc(item.headline_fix)}</p>"
                if item.headline_fix
                else ""
            ),
        ]
        tag = '<span class="ax-ex__minor">Also noticed</span>' if minor else ""
        return (
            f'<article class="ax-ex ax-ex--{esc(item.severity)}">'
            f'<h4 class="ax-ex__title">{esc(item.title)}{tag}</h4>'
            f"{''.join(lines)}"
            "</article>"
        )

    lines: list[str] = []
    if item.measures:
        source = f" from {esc(item.measured_from)}" if item.measured_from else ""
        lines.append(
            f'<p class="ax-ex__line"><span class="ax-ex__tag">Measured</span>'
            f"{esc(item.measures)}{source}.</p>"
        )
    scope = []
    if item.phase:
        scope.append(f"during the {esc(item.phase.replace('_', ' '))} phase")
    if item.evaluable_reps:
        scope.append(
            esc(
                f"flagged on {item.flagged_reps} of {item.evaluable_reps} judged repetition"
                f"{'' if item.evaluable_reps == 1 else 's'}"
            )
        )
    if item.reliability:
        scope.append(f"{esc(item.reliability.lower())} evidence")
    if scope:
        # entries are already escaped, so the separator can be an HTML entity
        lines.append(
            '<p class="ax-ex__line"><span class="ax-ex__tag">Where</span>'
            + " &middot; ".join(scope)
            + "</p>"
        )
    if item.issue:
        lines.append(
            f'<p class="ax-ex__line"><span class="ax-ex__tag">Detected</span>{esc(item.issue)}</p>'
        )
    if item.evidence:
        rows = "".join(f"<li>{esc(line)}</li>" for line in item.evidence)
        lines.append(f'<ul class="ax-ex__evidence">{rows}</ul>')
    lines.extend(_settings_lines(item))
    if item.rule_id:
        lines.append(f'<p class="ax-ex__rule">Rule: {esc(item.rule_id)}</p>')

    tag = '<span class="ax-ex__minor">Also noticed</span>' if minor else ""
    return (
        f'<article class="ax-ex ax-ex--{esc(item.severity)}">'
        f'<h4 class="ax-ex__title">{esc(item.title)}{tag}</h4>'
        f'{"".join(lines)}'
        "</article>"
    )


def detection_markup(result: AnalysisResult, technical: bool = False) -> str:
    """All findings, most important first (the smaller ones only appear here)."""
    if not result.improvements:
        return (
            '<p class="ax-empty-note">No check produced a finding in this set, so there is '
            "nothing to list here. The measured checks above show what was evaluated.</p>"
        )
    top = "".join(_detection_card(item, technical=technical) for item in result.key_improvements)
    rest = "".join(
        _detection_card(item, minor=True, technical=technical) for item in result.further_improvements
    )
    lede = (
        "FormFix does not produce a black-box score. Each correction came from one rule, "
        "reading one measurement, taken from specific body landmarks during a specific "
        "part of the movement."
        if technical
        else "Every finding, smaller ones included - each with one thing to try next time."
    )
    return f'<p class="ax-empty-note">{lede}</p><div class="ax-exs">{top}{rest}</div>'


def reps_block(result: AnalysisResult) -> str:
    """One card per rep: passed, not passed and not assessed."""
    if not result.rep_results:
        return ""

    cards = []
    for rep in result.rep_results:
        lines = []
        if rep.issues:
            # escape each issue before joining, or the separator gets escaped too
            lines.append(
                '<p class="ax-rep__meta">'
                + " &middot; ".join(esc(issue) for issue in rep.issues)
                + "</p>"
            )
        if rep.passed:
            lines.append(
                '<p class="ax-rep__meta ax-rep__meta--muted">Passed: '
                + esc(", ".join(rep.passed))
                + "</p>"
            )
        if rep.not_assessed:
            lines.append(
                '<p class="ax-rep__meta ax-rep__meta--muted">Not assessed: '
                + esc(", ".join(rep.not_assessed))
                + "</p>"
            )
        tag = esc(rep.timestamp) if rep.timestamp and rep.timestamp != "-" else ""
        cards.append(
            f'<li class="ax-rep ax-rep--{esc(rep.status)}">'
            f'<span class="ax-rep__num">Rep {rep.number:02d}</span>'
            '<span class="ax-rep__body">'
            f'<span class="ax-rep__headline">{esc(rep.headline)}</span>'
            f'{"".join(lines)}'
            "</span>"
            f'<span class="ax-rep__tag">{tag}</span>'
            "</li>"
        )

    count = len(result.rep_results)
    return (
        '<section class="ax-panel ax-panel--quiet">'
        '<h3 class="ax-panel__title ax-panel__title--sm"><span>Repetition by repetition</span>'
        f'<span class="ax-count">{count} {"rep" if count == 1 else "reps"}</span></h3>'
        f'<ul class="ax-reps">{"".join(cards)}</ul>'
        "</section>"
    )


def checks_block(result: AnalysisResult) -> str:
    """The checks behind the score, one row per rule with its reliability."""
    if not result.rows:
        return ""

    measured = []
    for row in result.rows:
        detail_text = " · ".join(
            part
            for part in (row.detail, f"{row.reliability} reliability" if row.reliability else "")
            if part
        )
        detail = f'<span class="ax-check__detail">{esc(detail_text)}</span>' if detail_text else ""
        measured.append(
            f'<li class="ax-check ax-check--{esc(row.status)}">'
            f'<span class="ax-check__label">{esc(row.label)}{detail}</span>'
            '<span class="ax-check__meter">'
            f'<span class="ax-check__bar" style="--w:{row.score:g}%"></span></span>'
            f'<span class="ax-check__value">{row.score:g}</span>'
            "</li>"
        )

    count = len(measured)
    return (
        '<section class="ax-panel" data-animate="fade-up">'
        '<h3 class="ax-panel__title"><span>Measured checks</span>'
        f'<span class="ax-count">{count} check{"" if count == 1 else "s"}</span></h3>'
        f'<ul class="ax-checks">{"".join(measured)}</ul>'
        "</section>"
    )


def not_assessed_block(result: AnalysisResult) -> str:
    """What wasn't judged and why, in its own block so it doesn't look like a fail."""
    if not result.not_assessed:
        return ""
    rows = "".join(
        '<li class="ax-na__row">'
        f'<span class="ax-na__title">{esc(item.title)}</span>'
        f'<span class="ax-na__reason">{esc(item.reason)}</span>'
        "</li>"
        for item in result.not_assessed
    )
    return (
        '<section class="ax-panel ax-panel--quiet">'
        '<h3 class="ax-panel__title ax-panel__title--sm">Not assessed</h3>'
        '<p class="ax-empty-note">These checks need something this recording could not '
        "show, so FormFix left them unscored rather than guessing.</p>"
        f'<ul class="ax-na">{rows}</ul>'
        "</section>"
    )


# --- Level 2 - the reference movement ---


def reference_block(exercise: Exercise) -> str:
    """Card that opens the reference video, placed under the corrections."""
    return (
        '<section class="ax-panel ax-panel--ref" data-animate="fade-up">'
        f'<h3 class="ax-panel__title">See correct {esc(exercise.name.lower())} form</h3>'
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
        "it with your own set above.</span>"
        "  </span>"
        "</button>"
        "</section>"
    )


def reference_modal(exercise: Exercise) -> str:
    """The reference lightbox. analyse.js moves it to <body> so position: fixed
    works against the viewport."""
    video = reference_video_url(exercise.id)

    if video:
        media = (
            f'<video class="ax-modal__video" src="{video}" controls playsinline '
            f'preload="metadata" poster="{asset_url(exercise.image)}"></video>'
        )
        note = "Reference clip"
    else:
        # no clip added yet, so show the annotated still
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


# --- Recording quality ---


# headline per detected camera angle
_ORIENTATION_HEADLINES = {
    "frontal": "Analysed from a front-on camera",
    "side": "Analysed from a side-on camera",
    "diagonal_side": "Analysis completed with a partial camera angle",
}


def recording_quality_block(result: AnalysisResult, exercise: Exercise) -> str:
    """
    Note above the results when the recording was usable but not ideal ("limited").
    The advice comes from the exercise, since each needs a different angle.
    """
    if result.recording_quality != "limited" or not result.warnings:
        return ""

    headline = _ORIENTATION_HEADLINES.get(
        result.camera_orientation, "Analysis completed with some limitations"
    )
    needed = (
        f"{esc(exercise.name)} is analysed from a {esc(exercise.camera_view.lower())} view."
        if exercise.camera_view
        else ""
    )
    notes = "".join(f'<li class="ax-retry__tip">{esc(w)}</li>' for w in result.warnings)
    return (
        '<section class="ax-panel ax-panel--invalid" data-animate="fade-up">'
        '  <p class="eyebrow">Recording quality</p>'
        f'  <h3 class="ax-panel__title">{esc(headline)}</h3>'
        f'  <p class="ax-video-note">FormFix could analyse this recording. {needed} '
        "Keeping the whole movement in frame, with the camera still, gives the most "
        "accurate measurements.</p>"
        f'  <ul class="ax-retry__tips">{notes}</ul>'
        "</section>"
    )


# --- Validation failure ---


def failure_markup(result: AnalysisResult, exercise: Exercise | None = None) -> str:
    """Shown when the recording couldn't be analysed: what failed, why, and what to change."""
    # analyser's suggestions first, otherwise the full recording tips
    suggestions = result.error_suggestions or (list(exercise.recording_tips) if exercise else [])
    tips = "".join(f'<li class="ax-retry__tip">{esc(tip)}</li>' for tip in suggestions)
    tips_block = (
        '<div class="ax-retry">'
        '<p class="ax-retry__label">For the best result</p>'
        f'<ul class="ax-retry__tips">{tips}</ul>'
        "</div>"
        if tips
        else ""
    )
    return (
        '<div class="ff-page ax-results" data-ax-results>'
        '<section class="ax-panel ax-panel--invalid" data-animate="fade-up">'
        '  <p class="eyebrow">Analysis not possible</p>'
        f'  <h3 class="ax-invalid__title">{esc(result.error_title or "We couldn&rsquo;t analyse this recording")}</h3>'
        f'  <p class="ax-invalid__body">{esc(result.error_message)}</p>'
        f"  {tips_block}"
        '  <p class="ax-invalid__note">Nothing was scored - FormFix never guesses when it '
        "cannot see the movement clearly. Adjust the recording and upload again.</p>"
        "</section>"
        "</div>"
    )


# --- Composition ---


def markup(result: AnalysisResult) -> str:
    """The main feedback: verdict, what to fix, what you did well, then the checks."""
    return (
        '<div class="ff-page ax-results" data-ax-results>'
        f"{verdict_block(result)}"
        f"{corrections_block(result)}"
        f"{positives_block(result)}"
        f"{checks_block(result)}"
        "</div>"
    )


def _annotated_video_section(result: AnalysisResult) -> None:
    """ "Your analysed movement" - the overlay video. Goes first so the feedback
    under it makes sense."""
    path = result.annotated_video
    if path is None or not path.is_file():
        return
    with st.container(key="ff_ax_annotated"):
        html(
            '<div class="ff-page ax-panel ax-panel--video" data-animate="fade-up">'
            '<h3 class="ax-panel__title">Your analysed movement</h3>'
            '<p class="ax-video-note">FormFix tracked your body through the movement. '
            "The highlighted joints are the ones each finding was measured from.</p>"
            "</div>"
        )
        st.video(str(path))


def _technical_lines(result: AnalysisResult) -> list[str]:
    """The run's settings and limits, only shown with FORMFIX_DEBUG."""
    lines = [
        f"**Analysis side:** {result.analysis_side or '-'} side used for movement analysis",
        f"**Repetitions:** {result.reps} complete"
        + (
            f", {result.partial_movements} partial movement(s) ignored"
            if result.partial_movements
            else ""
        ),
    ]
    if result.score_formula and result.score_available:
        lines.append(f"**Score formula:** {result.score_formula}")
    if result.reliability:
        lines.append(f"**Analysis reliability:** {result.reliability}")
    lines.append(f"**Recording quality:** {result.recording_quality}")
    if result.camera_orientation:
        lines.append(
            f"**Camera orientation:** {result.camera_orientation.replace('_', ' ')} "
            f"(side-view confidence {result.side_view_confidence:.2f})"
        )
    if result.limited_metrics:
        lines.append(
            "**Not assessed for this recording:** "
            + ", ".join(m.replace("_", " ") for m in result.limited_metrics)
        )
    for warning in result.warnings:
        lines.append(f"**Note:** {warning}")
    return lines


def _analysis_details(result: AnalysisResult, debug_mode: bool) -> None:
    """The collapsed detail expander: how each finding was detected, each rep, and
    what couldn't be assessed."""
    with st.container(key="ff_ax_tech"):
        with st.expander("See the full detail - how FormFix measured every rep", expanded=False):
            html(
                '<p class="ax-details__lede">How each finding was detected, how every rep '
                "scored, and what this recording could not answer.</p>"
            )
            html(
                '<div class="ax-details">'
                f"{detection_markup(result, technical=debug_mode)}"
                f"{reps_block(result)}"
                f"{not_assessed_block(result)}"
                "</div>"
            )
            if debug_mode:
                st.caption("Run settings and limits (FORMFIX_DEBUG)")
                st.markdown("  \n".join(_technical_lines(result)))
            if debug_mode and result.debug:
                st.caption("Developer metrics (FORMFIX_DEBUG)")
                st.json(result.debug, expanded=False)


def render(result: AnalysisResult, exercise: Exercise, debug_mode: bool = False) -> None:
    """The whole results column, in order."""
    if not result.success:
        html(failure_markup(result, exercise))
        return

    quality = recording_quality_block(result, exercise)
    if quality:
        html(f'<div class="ff-page ax-results">{quality}</div>')
    _annotated_video_section(result)
    html(markup(result))
    html(f'<div class="ff-page ax-results ax-results--ref">{reference_block(exercise)}</div>')
    _analysis_details(result, debug_mode)
