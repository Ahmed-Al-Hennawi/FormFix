"""
The /analyse page - the complete upload and analysis experience.

Moved off the homepage so the whole flow gets a screen of its own:

    Back Home
    Title
    Progress tracker        Upload Video -> Analyse Form -> Feedback
    Upload (left)           exercise, drop zone, preview, confirmation, CTA
    Analysis (right)        waiting -> ready -> analysing -> results

The page is a small state machine held in ``st.session_state``:

    idle       nothing uploaded yet
    ready      a video is selected and confirmed
    analysing  the pipeline is running; stages are reported live
    complete   an AnalysisResult is on screen

The stage names, the reported steps and the result object are all defined in
``utils/analysis.py``, so connecting the real OpenCV / MediaPipe backend means
implementing that module - not editing this one. Nothing here knows anything
about a specific exercise or a specific piece of feedback.
"""

from __future__ import annotations

import time

import streamlit as st

from utils.analysis import (
    ANALYSIS_STAGES,
    SUPPORTED_VIDEO_TYPES,
    AnalysisResult,
    analyse,
)
from utils.assets import LOGO_MARK, asset_url
from utils.exercise_data import EXERCISES, get as get_exercise
from utils.helpers import esc, strip
from utils.pose import render_pose
from utils.routing import home_url
from utils.styling import html
from . import analysis_results

# --- session keys ----------------------------------------------------------
STAGE = "ff_ax_stage"
RESULT = "ff_ax_result"
NONCE = "ff_ax_nonce"
EXERCISE = "ff_ax_exercise"

IDLE, READY, ANALYSING, COMPLETE = "idle", "ready", "analysing", "complete"

#: The three stations of the progress tracker: (full label, short label).
TRACKER_STEPS: tuple[tuple[str, str], ...] = (
    ("Upload Video", "Upload"),
    ("Analyse Form", "Analyse"),
    ("Feedback", "Feedback"),
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _init_state() -> None:
    st.session_state.setdefault(STAGE, IDLE)
    st.session_state.setdefault(RESULT, None)
    st.session_state.setdefault(NONCE, 0)
    st.session_state.setdefault(EXERCISE, EXERCISES[0].id)


def _uploader_key() -> str:
    """
    The file-uploader's widget key.

    It carries a nonce so "Remove video" can genuinely clear the widget by
    giving it a fresh identity, rather than leaving a stale file behind.
    """
    return f"ff_ax_video_{st.session_state[NONCE]}"


def _reset(clear_upload: bool = False) -> None:
    """Back to a clean slate. Used by "Remove video" and "Analyse again"."""
    st.session_state[STAGE] = IDLE
    st.session_state[RESULT] = None
    if clear_upload:
        st.session_state[NONCE] += 1


def _on_exercise_change() -> None:
    """A previous result belongs to the exercise it was produced for."""
    if st.session_state.get(RESULT) is not None:
        st.session_state[RESULT] = None
        st.session_state[STAGE] = READY


def _current_stage(has_video: bool) -> str:
    stage = st.session_state[STAGE]
    if not has_video:
        return IDLE
    if stage == ANALYSING:
        return ANALYSING
    if stage == COMPLETE and st.session_state[RESULT] is not None:
        return COMPLETE
    return READY


# ---------------------------------------------------------------------------
# Chrome - back link, title, tracker
# ---------------------------------------------------------------------------


def _topbar() -> None:
    html(
        strip(
            f"""
            <div class="ff-page">
            <div class="ax-topbar">
              <a class="ax-back" href="{home_url()}" target="_self">
                <span class="ax-back__arrow">&larr;</span>
                <span>Back Home</span>
              </a>
              <a class="ax-brand" href="{home_url()}" target="_self" aria-label="FormFix AI home">
                <img src="{asset_url(LOGO_MARK)}" alt="" width="30" height="23" />
                <span>FORMFIX<span class="ax-brand__ai">AI</span></span>
              </a>
            </div>
            </div>
            """
        )
    )


def _header() -> None:
    html(
        strip(
            """
            <div class="ff-page">
            <header class="ax-head">
              <p class="eyebrow">Analyse your form</p>
              <h1 class="display display--md ax-head__title">Upload a set.<br />Get an explanation.</h1>
              <p class="ax-head__lede">
                Record a single set from the side or the front, drop the clip in, and FormFix AI
                will track your body through the movement and explain what it sees.
              </p>
            </header>
            </div>
            """
        )
    )


def _tracker(stage: str) -> None:
    """
    Upload Video --- Analyse Form --- Feedback

    Three states per station: done (lime), active (cyan, pulsing) and waiting
    (muted). The connector between two stations animates while the analysis is
    running, so the page reads as "working" without a generic spinner.
    """
    index = {IDLE: 0, READY: 0, ANALYSING: 1, COMPLETE: 3}[stage]
    # On the first station, "ready" means the upload is genuinely finished.
    upload_done = stage in (READY, ANALYSING, COMPLETE)

    parts: list[str] = []
    for position, (full, short) in enumerate(TRACKER_STEPS):
        if position < index or (position == 0 and upload_done):
            state, mark = "done", '<span class="ax-node__tick">&#10003;</span>'
        elif position == index:
            state, mark = "active", f'<span class="ax-node__dot"></span>'
        else:
            state, mark = "wait", '<span class="ax-node__ring"></span>'

        if position:
            link_state = "done" if position <= index else "wait"
            if position == index and stage == ANALYSING:
                link_state = "live"
            parts.append(f'<span class="ax-link ax-link--{link_state}"></span>')

        parts.append(
            f'<span class="ax-step ax-step--{state}">'
            f'  <span class="ax-node">{mark}</span>'
            f'  <span class="ax-step__label">'
            f'    <span class="ax-step__full">{full}</span>'
            f'    <span class="ax-step__short">{short}</span>'
            "  </span>"
            "</span>"
        )

    html(
        f'<div class="ff-page"><div class="ax-track" data-stage="{stage}">'
        f'{"".join(parts)}</div></div>'
    )


# ---------------------------------------------------------------------------
# Left column - the upload experience
# ---------------------------------------------------------------------------


def _panel_head(number: str, title: str, note: str = "") -> str:
    note_html = f'<span class="ax-panel__note">{esc(note)}</span>' if note else ""
    return (
        '<div class="ff-page ax-panel__head">'
        f'<span class="ax-panel__num">{number}</span>'
        f'<h2 class="ax-panel__heading">{esc(title)}</h2>'
        f"{note_html}"
        "</div>"
    )


def _confirmation(uploaded) -> str:
    size = getattr(uploaded, "size", 0) or 0
    if size >= 1024 * 1024:
        size_label = f"{size / (1024 * 1024):.1f} MB"
    elif size:
        size_label = f"{size / 1024:.0f} KB"
    else:
        size_label = ""
    meta = " &middot; ".join(part for part in (esc(uploaded.name), size_label) if part)

    return (
        '<div class="ff-page">'
        '<div class="ax-confirm" role="status">'
        '  <span class="ax-confirm__check">'
        '    <svg viewBox="0 0 24 24" aria-hidden="true">'
        '      <path d="M5 12.5 10 17.5 19 7.5" fill="none" stroke="currentColor" '
        'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />'
        "    </svg>"
        "  </span>"
        '  <span class="ax-confirm__text">'
        '    <strong>Video uploaded successfully</strong>'
        f'    <span class="ax-confirm__meta">{meta}</span>'
        "  </span>"
        "</div>"
        "</div>"
    )


def _upload_column(stage: str, uploaded) -> None:
    with st.container(key="ff_ax_upload"):
        html(_panel_head("01", "Your video", "MP4 · MOV · AVI"))

        st.radio(
            "Which exercise?",
            options=[exercise.id for exercise in EXERCISES],
            format_func=lambda eid: get_exercise(eid).name,
            horizontal=True,
            key=EXERCISE,
            on_change=_on_exercise_change,
            disabled=stage == ANALYSING,
        )

        st.file_uploader(
            "Workout video",
            type=list(SUPPORTED_VIDEO_TYPES),
            accept_multiple_files=False,
            key=_uploader_key(),
            help="A single set, filmed from the side or the front, works best.",
            disabled=stage == ANALYSING,
        )

        if uploaded is None:
            html(
                '<p class="ff-page ax-hint">Drag &amp; drop or browse &middot; '
                + " &middot; ".join(f".{ext}" for ext in SUPPORTED_VIDEO_TYPES)
                + "</p>"
            )
            return

        html(_confirmation(uploaded))
        st.video(uploaded)

        actions_left, actions_right = st.columns([3, 2], gap="small")

        with actions_left:
            if stage == COMPLETE:
                if st.button(
                    "Analyse again",
                    key="ff_ax_again",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state[STAGE] = ANALYSING
                    st.session_state[RESULT] = None
                    st.rerun()
            else:
                if st.button(
                    "Analyse Form",
                    key="ff_ax_run",
                    type="primary",
                    use_container_width=True,
                    disabled=stage == ANALYSING,
                ):
                    st.session_state[STAGE] = ANALYSING
                    st.session_state[RESULT] = None
                    st.rerun()

        with actions_right:
            if st.button(
                "Remove video",
                key="ff_ax_remove",
                use_container_width=True,
                disabled=stage == ANALYSING,
            ):
                _reset(clear_upload=True)
                st.rerun()


# ---------------------------------------------------------------------------
# Right column - waiting, working, results
# ---------------------------------------------------------------------------


def _waiting_markup(stage: str) -> str:
    ready = stage == READY
    eyebrow = "Ready to analyse" if ready else "Waiting for a video"
    title = "Press Analyse Form" if ready else "Your results will appear here"
    body = (
        "Your clip is loaded. Start the analysis and FormFix AI will walk through the "
        "movement frame by frame."
        if ready
        else "Choose your exercise and upload a short clip of one set. Nothing is analysed "
        "until you press Analyse Form."
    )

    rows = (
        ("Form score", "An overall score out of 100"),
        ("What you did well", "The parts of the movement that looked right"),
        ("What to improve", "Each issue, with a clear correction"),
    )
    preview = "".join(
        f'<li class="ax-preview__row"><span class="ax-preview__dot"></span>'
        f"<span><strong>{label}</strong>{note}</span></li>"
        for label, note in rows
    )

    return (
        f'<div class="ff-page ax-wait ax-wait--{"ready" if ready else "idle"}">'
        f'  <p class="eyebrow">{eyebrow}</p>'
        f'  <p class="ax-wait__title">{title}</p>'
        f'  <p class="ax-wait__body">{body}</p>'
        f'  <ul class="ax-preview">{preview}</ul>'
        "</div>"
    )


def _scan_markup(exercise) -> str:
    """The 'system is working' visual - the FormFix landmark language, moving."""
    return (
        '<div class="ff-page ax-scan">'
        '  <div class="ax-scan__stage">'
        f'    <img src="{asset_url(exercise.image)}" alt="" />'
        f'    {render_pose(exercise.pose, key="scan", extra_class="pose--scan")}'
        '    <span class="ax-scan__line"></span>'
        '    <span class="ax-scan__grid"></span>'
        "  </div>"
        '  <p class="ax-scan__hud"><span class="hud__dot"></span>TRACKING 33 LANDMARKS</p>'
        "</div>"
    )


def _steps_markup(done: int, active: int | None) -> str:
    """The live stage list rendered underneath the scan visual."""
    items = []
    for position, stage_def in enumerate(ANALYSIS_STAGES):
        if position < done:
            state, mark = "done", "&#10003;"
        elif position == active:
            state, mark = "active", "&bull;"
        else:
            state, mark = "wait", "&mdash;"
        items.append(
            f'<li class="ax-run__step ax-run__step--{state}">'
            f'<span class="ax-run__mark">{mark}</span>'
            f"<span>{stage_def.label}</span></li>"
        )
    return f'<ul class="ff-page ax-run">{"".join(items)}</ul>'


def _run_analysis(uploaded, exercise_id: str) -> None:
    """
    Report each stage as it happens, then hand the result to the results view.

    The ``time.sleep`` below is the prototype's simulation. Once
    ``utils.analysis.process_uploaded_video`` does real work, replace the loop
    with progress reported by the pipeline itself - the markup, the tracker and
    the results view stay exactly as they are.
    """
    exercise = get_exercise(exercise_id)

    html(_scan_markup(exercise))
    slot = st.empty()

    for position, stage_def in enumerate(ANALYSIS_STAGES):
        slot.markdown(_steps_markup(position, position), unsafe_allow_html=True)
        time.sleep(stage_def.seconds)

    slot.markdown(_steps_markup(len(ANALYSIS_STAGES), None), unsafe_allow_html=True)

    result: AnalysisResult = analyse(uploaded, exercise_id=exercise_id)

    st.session_state[RESULT] = result
    st.session_state[STAGE] = COMPLETE
    st.rerun()


def _results_column(stage: str, uploaded, exercise_id: str) -> None:
    exercise = get_exercise(exercise_id)

    if stage == COMPLETE:
        result = st.session_state[RESULT]
        html(analysis_results.markup(result, exercise))
        return

    with st.container(key="ff_ax_state"):
        if stage == ANALYSING:
            html(
                '<div class="ff-page ax-panel__head">'
                '<span class="ax-panel__num ax-panel__num--live">02</span>'
                '<h2 class="ax-panel__heading">Analysing your form</h2>'
                '<span class="ax-panel__note">This takes a few seconds</span>'
                "</div>"
            )
            _run_analysis(uploaded, exercise_id)
        else:
            html(_waiting_markup(stage))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render() -> None:
    _init_state()

    uploaded = st.session_state.get(_uploader_key())
    if uploaded is None and st.session_state[STAGE] != IDLE:
        # The file was cleared with the uploader's own control.
        _reset()

    stage = _current_stage(uploaded is not None)
    exercise_id = st.session_state[EXERCISE]

    _topbar()
    _header()
    _tracker(stage)

    with st.container(key="ff_ax"):
        upload_col, results_col = st.columns([1, 1], gap="large")

        with upload_col:
            _upload_column(stage, uploaded)

        with results_col:
            _results_column(stage, uploaded, exercise_id)

    # The lightbox lives at page level; analyse.js moves it to <body>.
    html(f'<div class="ff-page">{analysis_results.reference_modal(get_exercise(exercise_id))}</div>')
    html('<div class="ff-page"><div class="ax-footer-space"></div></div>')
