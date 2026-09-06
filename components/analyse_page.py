"""
The /analyse page: upload on the left, analysis on the right, progress tracker
across the top. A small state machine in st.session_state runs idle -> ready ->
analysing -> complete. The stages and the result come from utils/analysis.py;
nothing here knows about a particular exercise.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time

import streamlit as st

from analysis.video_processor import purge_stale_sessions
from utils.analysis import (
    ANALYSIS_STAGES,
    STAGE_INDEX,
    SUPPORTED_VIDEO_TYPES,
    AnalysisResult,
    analyse,
    unexpected_failure,
)
from utils.assets import LOGO_MARK, asset_url
from utils.exercise_data import EXERCISES
from utils.exercise_data import get as get_exercise
from utils.helpers import esc, strip
from utils.pose import render_pose
from utils.routing import home_url
from utils.styling import html

from . import analysis_results

logger = logging.getLogger(__name__)

# --- session keys ---
STAGE = "ff_ax_stage"
RESULT = "ff_ax_result"
NONCE = "ff_ax_nonce"
EXERCISE = "ff_ax_exercise"
# completed analyses, keyed by (file hash, exercise), so an unrelated rerun
# can't re-run the whole pipeline
CACHE = "ff_ax_cache"
FORCE = "ff_ax_force"
SWEPT = "ff_ax_swept"

DEBUG_MODE = bool(os.environ.get("FORMFIX_DEBUG"))

IDLE, READY, ANALYSING, COMPLETE = "idle", "ready", "analysing", "complete"

try:  # Streamlit's flow-control signals are ordinary exceptions - never swallow them.
    from streamlit.runtime.scriptrunner_utils.exceptions import (
        RerunException,
        StopException,
    )

    _STREAMLIT_CONTROL_EXCEPTIONS: tuple[type[BaseException], ...] = (
        RerunException,
        StopException,
    )
except ImportError:  # pragma: no cover - private path moved between versions
    _STREAMLIT_CONTROL_EXCEPTIONS = ()

# The three stations of the progress tracker: (full label, short label).
TRACKER_STEPS: tuple[tuple[str, str], ...] = (
    ("Upload Video", "Upload"),
    ("Analyse Form", "Analyse"),
    ("Feedback", "Feedback"),
)


# --- State ---


def _init_state() -> None:
    st.session_state.setdefault(STAGE, IDLE)
    st.session_state.setdefault(RESULT, None)
    st.session_state.setdefault(NONCE, 0)
    st.session_state.setdefault(EXERCISE, EXERCISES[0].id)
    st.session_state.setdefault(CACHE, {})
    st.session_state.setdefault(FORCE, False)

    # only the runs this session's cache evicts get their temp dir removed, so
    # one sweep per browser session keeps the leftovers bounded
    if not st.session_state.get(SWEPT):
        st.session_state[SWEPT] = True
        try:
            purge_stale_sessions()
        except Exception:  # pragma: no cover - tidying is never load-bearing
            logger.exception("Could not sweep stale analysis directories")


def _uploader_key() -> str:
    """The file-uploader's widget key. It carries a nonce because giving the widget
    a fresh identity is the only way "Remove video" can clear it."""
    return f"ff_ax_video_{st.session_state[NONCE]}"


def _release_stale_uploads() -> None:
    """Free the bytes of an upload that "Remove video" replaced - the old key hangs
    onto its UploadedFile and a phone clip is tens of megabytes."""
    current = _uploader_key()
    stale = [
        key
        for key in st.session_state
        if isinstance(key, str) and key.startswith("ff_ax_video_") and key != current
    ]
    for key in stale:
        try:
            del st.session_state[key]
        except Exception:  # pragma: no cover - Streamlit refuses some keys
            logger.debug("Could not release stale upload %s", key)


def _reset(clear_upload: bool = False) -> None:
    """Back to a clean slate. Used by "Remove video" and "Analyse again"."""
    st.session_state[STAGE] = IDLE
    st.session_state[RESULT] = None
    if clear_upload:
        st.session_state[NONCE] += 1


def _on_exercise_change() -> None:
    """Drop any result when the exercise changes - it belongs to the old one."""
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


# --- Chrome - back link, title, tracker ---


def _topbar() -> None:
    html(strip(f"""
            <div class="ff-page">
            <div class="ax-topbar">
              <a class="ax-back" href="{home_url()}" target="_self">
                <span class="ax-back__arrow">&larr;</span>
                <span>Back Home</span>
              </a>
              <a class="ax-brand" href="{home_url()}" target="_self" aria-label="FormFix home">
                <img src="{asset_url(LOGO_MARK)}" alt="" width="30" height="23" />
                <span>FORMFIX</span>
              </a>
            </div>
            </div>
            """))


def _header() -> None:
    html(strip("""
            <div class="ff-page">
            <header class="ax-head">
              <p class="eyebrow">Analyse your form</p>
              <h1 class="display display--md ax-head__title">Upload a set.<br />Get an explanation.</h1>
              <p class="ax-head__lede">
                Record a single set, drop the clip in, and FormFix will track your body
                through the movement and explain what it sees. Each exercise has its own
                camera view - the guidance below the exercise picker tells you which.
              </p>
            </header>
            </div>
            """))


def _tracker(stage: str) -> None:
    """Upload Video - Analyse Form - Feedback. Each station is done, active or
    waiting, and the connector animates while the analysis runs."""
    index = {IDLE: 0, READY: 0, ANALYSING: 1, COMPLETE: 3}[stage]
    # On the first station, "ready" means the upload is genuinely finished.
    upload_done = stage in (READY, ANALYSING, COMPLETE)

    parts: list[str] = []
    for position, (full, short) in enumerate(TRACKER_STEPS):
        if position < index or (position == 0 and upload_done):
            state, mark = "done", '<span class="ax-node__tick">&#10003;</span>'
        elif position == index:
            state, mark = "active", '<span class="ax-node__dot"></span>'
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


# --- Left column - the upload experience ---


def _panel_head(number: str, title: str, note: str = "") -> str:
    note_html = f'<span class="ax-panel__note">{esc(note)}</span>' if note else ""
    return (
        '<div class="ff-page ax-panel__head">'
        f'<span class="ax-panel__num">{number}</span>'
        f'<h2 class="ax-panel__heading">{esc(title)}</h2>'
        f"{note_html}"
        "</div>"
    )


def _recording_guidance(exercise) -> str:
    """"How to record this" - the camera view plus the exercise's own checklist,
    read from its config so it can't drift from the thresholds validation uses."""
    if not exercise.recording_tips:
        return ""
    items = "".join(f'<li class="ax-guide__tip">{esc(tip)}</li>' for tip in exercise.recording_tips)
    return (
        '<div class="ff-page ax-guide">'
        '  <p class="ax-guide__head">'
        '    <span class="ax-guide__label">How to record</span>'
        f'    <span class="ax-guide__view">{esc(exercise.camera_view)} view</span>'
        "  </p>"
        f'  <ul class="ax-guide__tips">{items}</ul>'
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
        "    <strong>Video uploaded successfully</strong>"
        f'    <span class="ax-confirm__meta">{meta}</span>'
        "  </span>"
        "</div>"
        "</div>"
    )


def _upload_column(stage: str, uploaded) -> None:
    with st.container(key="ff_ax_upload"):
        # the note states the limits the analyser actually enforces
        selected = get_exercise(st.session_state[EXERCISE])
        note = " · ".join(
            part for part in ("MP4 · MOV · AVI", selected.duration_label, "one set per clip") if part
        )
        html(_panel_head("01", "Your video", note))

        st.radio(
            "Which exercise?",
            options=[exercise.id for exercise in EXERCISES],
            format_func=lambda eid: get_exercise(eid).name,
            horizontal=True,
            key=EXERCISE,
            on_change=_on_exercise_change,
            disabled=stage == ANALYSING,
        )

        exercise_id = st.session_state[EXERCISE]
        exercise = get_exercise(exercise_id)

        # guidance before the upload, not after - several checks aren't observable
        # from the wrong camera position
        html(_recording_guidance(exercise))

        uploader_help = f"{exercise.camera_view} view. " + (
            exercise.recording_tips[0] if exercise.recording_tips else ""
        )

        st.file_uploader(
            "Workout video",
            type=list(SUPPORTED_VIDEO_TYPES),
            accept_multiple_files=False,
            key=_uploader_key(),
            help=uploader_help,
            disabled=stage == ANALYSING,
        )

        if uploaded is None:
            html(
                '<p class="ff-page ax-hint">Drag &amp; drop or browse &middot; '
                + " &middot; ".join(f".{ext}" for ext in SUPPORTED_VIDEO_TYPES)
                + "</p>"
                # people are uploading video of their own bodies, so say what happens to
                # it before they do. The wording is precise: the upload really is deleted
                # when the run finishes, while the annotated clip is swept a few hours later.
                '<p class="ff-page ax-hint ax-hint--privacy">Your video is deleted as '
                "soon as it has been analysed, the result is erased a few hours later, "
                "and no account is needed.</p>"
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
                    st.session_state[FORCE] = True
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


# --- Right column - waiting, working, results ---


def _waiting_markup(stage: str) -> str:
    ready = stage == READY
    eyebrow = "Ready to analyse" if ready else "Waiting for a video"
    title = "Press Analyse Form" if ready else "Your results will appear here"
    body = (
        "Your clip is loaded. Start the analysis and FormFix will walk through the "
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


def _steps_markup(done: int, active: int | None, active_pct: int | None = None) -> str:
    """The live stage list rendered underneath the scan visual."""
    items = []
    for position, stage_def in enumerate(ANALYSIS_STAGES):
        label = stage_def.label
        if position < done:
            state, mark = "done", "&#10003;"
        elif position == active:
            state, mark = "active", "&bull;"
            if active_pct is not None and 0 < active_pct < 100:
                label = f"{label} &middot; {active_pct}%"
        else:
            state, mark = "wait", "&mdash;"
        items.append(
            f'<li class="ax-run__step ax-run__step--{state}">'
            f'<span class="ax-run__mark">{mark}</span>'
            f"<span>{label}</span></li>"
        )
    return f'<ul class="ff-page ax-run">{"".join(items)}</ul>'


def _file_key(uploaded, exercise_id: str) -> str:
    """Identify one upload+exercise combination, for the analysis cache."""
    digest = hashlib.md5(uploaded.getbuffer()).hexdigest()
    return f"{digest}:{exercise_id}"


def _forget_result(result) -> None:
    """Remove a cached run's rendered video directory from the temp space."""
    annotated = getattr(result, "annotated_video", None)
    if annotated is not None:
        shutil.rmtree(annotated.parent, ignore_errors=True)


def _run_analysis(uploaded, exercise_id: str) -> None:
    """
    Run the pipeline, reporting each stage as it happens. Results are cached on
    (file hash, exercise) so an unrelated rerun never re-runs MediaPipe; "Analyse
    again" drops that entry first.
    """
    exercise = get_exercise(exercise_id)

    html(_scan_markup(exercise))
    slot = st.empty()

    cache: dict = st.session_state[CACHE]
    key = _file_key(uploaded, exercise_id)
    force = st.session_state.get(FORCE, False)
    st.session_state[FORCE] = False

    if force:
        stale = cache.pop(key, None)
        if stale is not None:
            _forget_result(stale)

    cached = cache.get(key)
    if cached is not None:
        st.session_state[RESULT] = cached
        st.session_state[STAGE] = COMPLETE
        st.rerun()

    # --- live progress ---
    last_paint = {"index": -1, "pct": -1, "time": 0.0}

    def report(stage_key: str, fraction: float, message: str) -> None:
        index = STAGE_INDEX.get(stage_key)
        if index is None:  # "complete"
            index, fraction = len(ANALYSIS_STAGES), 1.0
        # the pipeline revisits a stage (it validates before and after detection),
        # but this is a progress bar, not a log, so it never goes backwards
        if index < last_paint["index"]:
            return
        pct = int(max(0.0, min(1.0, fraction)) * 100)
        now = time.monotonic()
        same_stage = index == last_paint["index"]
        if same_stage and (abs(pct - last_paint["pct"]) < 4 or now - last_paint["time"] < 0.15):
            return
        last_paint.update(index=index, pct=pct, time=now)
        if index >= len(ANALYSIS_STAGES):
            slot.markdown(_steps_markup(len(ANALYSIS_STAGES), None), unsafe_allow_html=True)
        else:
            slot.markdown(_steps_markup(index, index, pct), unsafe_allow_html=True)

    report(ANALYSIS_STAGES[0].key, 0.0, "")

    # developer mode also writes a JSON/CSV bundle to analysis_results/
    export_root = None
    if DEBUG_MODE:
        from utils.paths import APP_ROOT

        export_root = APP_ROOT

    try:
        result: AnalysisResult = analyse(
            uploaded, exercise_id=exercise_id, progress=report, export_root=export_root
        )
    except _STREAMLIT_CONTROL_EXCEPTIONS:
        raise  # st.rerun() / st.stop() - Streamlit's own flow control
    except Exception:
        # foreseeable problems already come back as a failure result; this is the
        # rest, and without it the page stays pinned to "analysing"
        logger.exception("Analysis failed unexpectedly")
        slot.empty()
        st.session_state[RESULT] = unexpected_failure(getattr(uploaded, "name", ""), exercise_id)
        st.session_state[STAGE] = COMPLETE
        st.rerun()

    if result.is_demo:
        # no real analyser for this exercise yet, so walk the stages briefly
        for position in range(len(ANALYSIS_STAGES)):
            slot.markdown(_steps_markup(position, position), unsafe_allow_html=True)
            time.sleep(0.3)

    slot.markdown(_steps_markup(len(ANALYSIS_STAGES), None), unsafe_allow_html=True)

    # keep only the few most recent runs; an evicted one loses its video dir
    if result.success and not result.is_demo:
        cache[key] = result
        while len(cache) > 3:
            oldest = next(iter(cache))
            _forget_result(cache.pop(oldest))

    st.session_state[RESULT] = result
    st.session_state[STAGE] = COMPLETE
    st.rerun()


def _results_column(stage: str, uploaded, exercise_id: str) -> None:
    exercise = get_exercise(exercise_id)

    if stage == COMPLETE:
        result = st.session_state[RESULT]
        analysis_results.render(result, exercise, debug_mode=DEBUG_MODE)
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


# --- Page ---


def render() -> None:
    _init_state()
    _release_stale_uploads()

    uploaded = st.session_state.get(_uploader_key())
    if uploaded is None and st.session_state[STAGE] != IDLE:
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
