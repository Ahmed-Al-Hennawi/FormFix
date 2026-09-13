# Technical documentation

How FormFix is put together: the layers, the pipeline stage by stage, and the
implementation details that aren't obvious from the code.

[README.md](../README.md) covers what FormFix does, how to install and run it,
the technology list, deployment and the headline limitations. This file does not
repeat those - it is the internals.

---

## 1. Architecture

There is one link between the UI and the analysis: `utils/analysis.py`. The UI
never knows which exercise it is showing.

| Layer | Folder | What it does |
| --- | --- | --- |
| Interface | `app.py`, `views/`, `components/`, `styles/`, `scripts/*.js` | The two pages, their sections, CSS and JS. |
| Bridge | `utils/analysis.py` | Saves the upload, runs the right analyser, and turns the result into what the UI shows. |
| Shared computer vision | `analysis/` | Video in/out, pose detection, validation, smoothing, filters, geometry, trimming, the result video. |
| Shared exercise logic | `exercises/common/` | Rule spec, rep counting, persistence, reliability, uncertainty, feedback, exercise check. |
| Per exercise | `exercises/squat｜pulldown｜press/` | Each has config, landmarks, metrics, phases, rules, feedback and analyser. |

Adding a fourth exercise means a new folder plus one entry in
`exercises/__init__.py` - the upload page and results view don't change.

All paths come from `utils/paths.py`, so it runs from any folder. The two pages
(`/` homepage, `/analyse` upload and results) are set up with
`st.navigation(position="hidden")`, so Streamlit's own menu never shows.

---

## 2. The pipeline

| Stage | Module | What happens |
| --- | --- | --- |
| Read + check file | `video_processor.py`, `validation.py` | Metadata isn't trusted; length must be 3-120 s. |
| Pose detection | `pose_detector.py`, `pose_worker.py` | MediaPipe in VIDEO mode, up to 4 people, in a separate process. Follows the person nearest the camera from frame to frame. |
| Check recording | `validation.py` | Coverage, key landmarks, usable frames, framing, camera angle and stability. Good / limited / unusable. |
| Clean | `stabilise.py`, `smoothing.py` | Drop landmark positions that can't be real movement, fill gaps up to 5 frames, EMA smoothing (resets after a gap). |
| Measure | `exercises/*/metrics.py` | Angles and distances in pixels, normalised by body size, with plausibility limits. |
| Count reps | `common/phases.py` | One state machine for all three exercises, with hysteresis and duration limits. |
| Exercise check | `common/plausibility.py` | Looks for movement the chosen exercise can't produce. |
| Rules | `exercises/*/rules.py` | Phase-scoped, camera-gated, persistence-filtered. |
| Reliability + uncertainty | `common/confidence.py`, `uncertainty.py` | Per-measurement reliability and published error. |
| Feedback | `common/feedback.py` | Score, per-rep verdicts, corrections, what went well, not assessed. |
| Result video | `annotation.py`, `overlay.py`, `trimming.py` | Trimmed to the set, H.264 via FFmpeg. |

**Adaptive rep counting.** With fixed thresholds, someone who never straightens
their arms would never count a rep, so the range-of-motion rule meant to catch
that would never run. The pulldown and press therefore use levels relative to
the person's own resting position. "Was it a rep?" is adaptive; "was the range
enough?" uses fixed thresholds
([threshold_tuning.md](threshold_tuning.md) §2).

**The result video.**

- Trimmed from 1 s before the first rep to 1 s after the last; frame numbers
  stay the same, so timestamps match the upload.
- The whole skeleton is drawn, with the side away from the camera fainter.
- The drawn skeleton has its own smoothing, separate from the analysis: a
  running median, then a zero-phase Butterworth at 3 Hz, plus hysteresis so
  joints don't flicker.
- A far-side limb that is behind the body for most of the clip is drawn fainter
  still, rather than hidden. One style throughout: nothing changes colour when a
  rep is flagged.

---

## 3. The interface

### The `/analyse` page

`components/analyse_page.py` keeps four states in `st.session_state`: **idle**
(drop zone and camera guide), **ready** (video preview), **analysing** (live
progress through eight stages) and **complete** (results). Results are cached by
file hash and exercise, so clicking around doesn't re-run MediaPipe.

Before upload, the page shows three short tips and an overhead diagram of where
to put the phone (`utils/camera_guide.py`) - testers didn't read the longer
instructions and still filmed from the wrong side. The full tips are shown if a
video gets rejected.

### Making a custom design work in Streamlit

- CSS goes in with `st.markdown`. Selectors are written like `p.eyebrow` because
  Streamlit's own element styles beat plain classes.
- Streamlit strips `<script>`, so JS goes through a zero-height iframe that
  installs it into the parent page, with a per-run nonce so it re-runs after a
  rerun.
- Section ids start with `ff-` so they don't clash with Streamlit's heading
  anchors.
- Fixed layers (progress line, nav, background, video popup) are moved to
  `<body>` so `position: fixed` works.

---

## 4. Two version pins worth knowing

- **macOS uses NumPy 1.** MediaPipe 0.10.14 is the last macOS build and crashes
  with NumPy 2, so the macOS stack is pinned. This is also why the filters are
  written in NumPy instead of adding SciPy.
- **Streamlit is held below 1.63.** The JavaScript is injected with
  `st.components.v1.html`, which is deprecated. A newer version could drop it
  and break the popup, scroll effects and score count-up.

`FORMFIX_POSE_SUBPROCESS=0` runs detection in-process instead of in the worker,
for debugging. The rest of the commands and environment variables are in
[README.md](../README.md).

---

## 5. Testing and evaluation

Over 700 tests cover geometry, smoothing, filters (against known maths),
landmarks, rep counting, every rule, camera gating, reliability, uncertainty,
person selection, occlusion, trimming, the exercise check, the UI mapping and
full end-to-end runs. Each exercise has a synthetic video generator, and every
suite has false-positive tests - noise, a dropped landmark, a value just under a
threshold - that must not produce a finding.

`scripts/evaluate_videos.py` runs FormFix over labelled videos and writes a
per-rule TP / FP / FN / TN / not-assessed table. Fourteen labelled videos were
recorded and run in September 2026 and written up in
[evaluation_results.md](evaluation_results.md) (see
[evaluation/README.md](../evaluation/README.md)). Runs on the three real
reference clips are in
[reference_clip_runs.md](../evaluation/reference_clip_runs.md) - two were
correctly refused, the pulldown was analysed, and four bugs were found and
fixed.

---

## 6. Thresholds, settings and scope

Every threshold is in each exercise's `config.py`, split into engineering values
(stability) and technique values (provisional prototype values). Each rule has a
`threshold_source`. The full history of what changed and why is in
[threshold_tuning.md](threshold_tuning.md).

- `literature_config.py` - a second squat config from published papers, for
  comparison only (`--preset literature`).
- `ANGLE_FILTER` - `ema` (default), `butterworth`, `savgol` or `moving_average`.
  The ranking flips with the shape of the rep, so it is a setting
  ([filter_selection.md](filter_selection.md)).
- `UNCERTAINTY_STRICT` - also downgrades findings inside the measurement error
  ([measurement_uncertainty.md](measurement_uncertainty.md)).

What FormFix is deliberately not: a diagnostic tool (no injury risk, back
rounding or knee valgus), a learned scorer (the score is a printed formula), or
a validated intervention (no user study). The full list is in
[limitations.md](limitations.md); what came from which paper is in
[research_provenance.md](research_provenance.md).
