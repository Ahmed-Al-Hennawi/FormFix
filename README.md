# FormFix — Streamlit version

**Analysed exercises: Squat · Lat Pulldown · Dumbbell Shoulder Press**

Each exercise needs its own camera view, because what a single camera can
measure depends on where it stands. The interface says which before you upload,
and any check the view cannot support is reported as *Not assessed*, with the
reason, rather than guessed:

| Exercise | Camera | Checks |
| --- | --- | --- |
| **Squat** | Side-on | depth · torso lean · heel stability · return to standing · descent control |
| **Lat Pulldown** | Side or three-quarter | torso movement · range of motion |
| **Shoulder Press** | Front-on | arm symmetry · elbow/wrist alignment · range of motion |

Technical documentation: [squat](docs/squat_analysis.md) ·
[lat pulldown](docs/pulldown_analysis.md) ·
[shoulder press](docs/press_analysis.md) ·
[threshold provenance](docs/threshold_tuning.md) ·
[measurement uncertainty](docs/measurement_uncertainty.md) ·
[filter selection](docs/filter_selection.md) ·
[literature basis](docs/literature_basis.md) ·
[limitations](docs/limitations.md) ·
[real-video verification runs](evaluation/reference_clip_runs.md)

## Quick start

```bash
cd streamlit_app

python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# one-time: fetch the MediaPipe Pose Landmarker model (~9 MB)
python scripts/download_pose_model.py

# recommended: FFmpeg makes the annotated result video browser-playable
brew install ffmpeg              # macOS; use your package manager elsewhere

streamlit run app.py
```

The app opens at <http://localhost:8501>. To run the test-suite:

```bash
pytest
```

Two pages:

| URL | Page |
| --- | --- |
| `/` | the marketing site |
| `/analyse` | the upload + analysis studio |

Every "Analyse Form" call to action on the homepage links to `/analyse`, and
that page has a single "← Back Home" button. Both pages are declared with
`st.navigation` in `app.py` (`position="hidden"`), so Streamlit's own page
navigation never appears.

Run it from any directory you like — every path in the application is derived
from `utils/paths.py`, so `streamlit run streamlit_app/app.py` works just as
well.

---

## Project structure

```
streamlit_app/
├── app.py                  page config, then the two-page router
├── requirements.txt        pinned Python dependencies
├── packages.txt            apt packages the deploy installs first (OpenCV, FFmpeg)
├── ruff.toml               lint rules
├── pyproject.toml          Black's formatting config
├── README.md               this file
├── README_FILES.md         one line per file in the project
│
├── .streamlit/
│   └── config.toml         dark theme, wide layout, static file serving
│
├── views/                  one module per page
│   ├── home.py             /          the marketing site
│   └── analyse.py          /analyse   the upload + analysis studio
│
├── components/             one module per section / feature
│   ├── background.py       fixed scroll-progress line + atmosphere layers
│   ├── navbar.py           floating pill navigation
│   ├── hero.py             1  hero
│   ├── about.py            1b what this is / who it is for
│   ├── problem.py          2  the problem
│   ├── how_it_works.py     3  how FormFix works
│   ├── exercise_section.py 4  exercise explorer (squat / press / pulldown)
│   ├── explainable.py      5  explainable feedback + traceability pipeline
│   ├── results_section.py  6  example analysis report (and real results)
│   ├── technology.py       7  under the hood
│   ├── research.py         7b academic & technical foundation
│   ├── outro.py            8  final call to action
│   ├── footer.py           footer
│   │
│   ├── analyse_page.py     ← /analyse: upload, tracker, state machine
│   ├── analysis_results.py ← /analyse: score, feedback, reference video
│   └── upload_section.py   the original in-page uploader (superseded)
│
├── utils/
│   ├── paths.py            every filesystem path, via pathlib
│   ├── routing.py          page URLs for the plain-HTML links
│   ├── assets.py           asset → browser URL (static serving or data URI)
│   ├── styling.py          CSS + JavaScript injection
│   ├── helpers.py          small markup helpers
│   ├── pose.py             the FormFix landmark renderer
│   ├── exercise_data.py    the three exercises + their landmark geometry
│   └── analysis.py         ← the computer-vision integration point
│
├── styles/
│   ├── main.css            the full stylesheet
│   └── analyse.css         /analyse only, loaded on top of main.css
│
├── analysis/               shared, exercise-agnostic computer vision
│   ├── pose_detector.py    MediaPipe wrapper + which person is the athlete
│   ├── pose_worker.py      detection in a subprocess, so a crash is catchable
│   ├── validation.py       can this recording be analysed at all?
│   ├── smoothing.py        gap interpolation + EMA
│   ├── filters.py          Butterworth / Savitzky-Golay / MA / EMA, in NumPy
│   ├── geometry.py         angles, inclination, angular rate
│   ├── normalisation.py    body-relative scaling
│   ├── trimming.py         which frames of the upload are the actual set
│   ├── annotation.py       the annotated result video
│   ├── overlay.py          the drawing language it uses
│   ├── video_processor.py  reading, probing, H.264 output
│   ├── models.py           the typed structures every stage passes
│   └── export.py           the JSON/CSV evaluation bundles
│
├── exercises/              one package per exercise + shared machinery
│   ├── common/             spec, phases, persistence, metrics, confidence,
│   │                       uncertainty, feedback, failures, plausibility
│   ├── squat/              side-view squat (6 rules)
│   ├── pulldown/           seated lat pulldown (2 rules)
│   └── press/              dumbbell shoulder press (3 rules)
│
├── tests/                  660+ tests, incl. synthetic generators per exercise
├── docs/                   the technical write-ups behind the dissertation
├── evaluation/             the labelled-recording harness and its manifest
├── wireframes/             every screen and state as SVG, from one generator
├── questions/              my own viva prep notes (git-ignored)
│
├── scripts/
│   ├── main.js             the small behaviour layer
│   ├── analyse.js          /analyse only: reference lightbox, score count-up
│   └── *.py                model download, calibration, filter comparison,
│                           camera-view check, the evaluation runner
│
└── static/
    └── assets/
        ├── images/         squat, shoulder press, lat pulldown
        ├── logo/           logo-mark.png, formfix-logo.png
        ├── videos/         optional reference clips (see below)
        ├── models/         the MediaPipe pose model (downloaded, not in Git)
        └── docs/           research-paper.pdf
```

`README_FILES.md` describes every file in one line each; this tree is the
shape, that is the index.

---

## Where the assets live

The folder is called `static/` because `server.enableStaticServing = true` in
`.streamlit/config.toml` makes Streamlit serve everything under it at
`app/static/…`. That means the browser fetches and caches each photograph once
instead of Streamlit re-sending it on every rerun.

`utils/assets.py` handles this: it returns a served URL when static serving is
on, and falls back to a base64 data URI when it is not. If an asset is missing
entirely it falls back to a transparent pixel and the app prints a note at the
bottom of the page rather than showing a broken image.

### Reference technique clips

The results page ends with a "See the correct technique" card that opens a
lightbox. Drop a clip into `static/assets/videos/` using one of these names and
it plays automatically:

```
squat-reference.mp4
shoulder-press-reference.mp4
lat-pulldown-reference.mp4
```

Until a clip is added, the lightbox shows the annotated still frame instead and
labels itself `CLIP PENDING` — nothing breaks and nothing is faked. The
filenames live in `REFERENCE_VIDEOS` in `utils/assets.py`.

---

## The /analyse page

The whole upload → analyse → feedback experience lives on its own screen so the
homepage stays a homepage.

**Layout.** Upload on the left, analysis on the right, a three-station progress
tracker across the top. Below 900px everything stacks in reading order: back
button, title, tracker, upload, confirmation, analysis, score, what you did
well, what to improve, measured checks, reference video.

**State machine.** `components/analyse_page.py` holds four stages in
`st.session_state`:

| Stage | On screen |
| --- | --- |
| `idle` | drop zone, and a preview of what the results will contain |
| `ready` | upload confirmation, video preview, "Analyse Form" |
| `analysing` | landmark scan animation + the pipeline stages, reported live |
| `complete` | score, feedback, measured checks, reference video |

**The analysis itself is not faked.** `analyse()` in `utils/analysis.py` calls
`process_uploaded_video()` first; only when that returns `None` — i.e. the
computer-vision backend is not connected yet — does it fall back to
`demo_analysis()`, and the result is flagged `is_demo`, which is why the score
card carries a `SAMPLE OUTPUT` badge. Connect the backend and the badge and the
sample data both disappear on their own.

**Stages.** `ANALYSIS_STAGES` in `utils/analysis.py` is the list the page
reports while it works. The prototype walks it on a timer; the real pipeline
should report each stage as it genuinely completes. Nothing else changes.

**Score bands.** `BAND_THRESHOLDS` maps a score onto a band *and* a verdict, so
colour is never the only signal: 80+ is "Good Form" (Neon Lime), 60–79 is
"Needs Improvement" (amber), below 60 is "Poor Form" (red). Edit that one tuple
to change the system everywhere.

**Feedback is a list, not a headline.** An `AnalysisResult` carries
`positives: list[str]` and `improvements: list[Improvement]`, each improvement
holding a title, what was seen, the correction and a severity
(`minor` / `moderate` / `important`). Any number of findings renders correctly,
for any of the three exercises.

---

## How the design survives Streamlit

Streamlit renders the app inside its own document, which takes some care:

**CSS** is injected with `st.markdown(..., unsafe_allow_html=True)`. Streamlit
styles bare elements (`p`, `h1`–`h6`, `ul`, `li`, `a`) at class-level
specificity, so every rule in `main.css` that targets one of those directly is
written element-qualified — `p.eyebrow`, `h2.display`, `li.report__row`,
`ul.nav__links`. That is why the selectors look slightly unusual; changing them
back to plain class selectors will silently lose to Streamlit's own styles.

**JavaScript** cannot go through `st.markdown` (Streamlit strips `<script>`),
so `utils/styling.py` uses a zero-height `components.html` iframe as a delivery
mechanism: the script inside it installs `scripts/main.js` into the parent
document, next to the real page content. The bootstrap carries a per-run nonce —
without it the iframe's content is byte-identical on every rerun, Streamlit
reuses the frame, the script never re-executes, and anything a rerun renders
(a fresh analysis) is never picked up by the scroll-reveal observers.

**Section ids are namespaced `ff-…`** because Streamlit generates anchor ids
from heading text, and a heading called "Analyse" would otherwise claim
`#analyse`.

**Fixed layers** (the scroll-progress line, the navigation, the atmosphere) are
moved to `document.body` on load, so `position: fixed` always resolves against
the viewport regardless of what containing blocks Streamlit introduces.

---

## What the JavaScript does

`scripts/main.js` replaces the GSAP + ScrollTrigger layer of the original with
native browser APIs — no CDN dependency:

| Original (GSAP) | Here |
| --- | --- |
| scroll progress line | one rAF-throttled scroll handler, `transform: scaleX()` |
| sticky nav surface | class toggle on the same handler |
| active section indication | `IntersectionObserver` |
| scroll reveals / fade-ups / masked lines | `IntersectionObserver` → `.is-inview`, transitions in CSS |
| "how it works" rail draw | scroll-scrubbed `scaleY()` |
| step highlighting | `IntersectionObserver` with a viewport-band root margin |
| exercise cross-fade timeline | class-driven CSS transitions |
| hero float / glow / landmark pulses | CSS keyframes |
| hero mouse parallax | pointer listener, `pointer: fine` only |

Everything degrades: the reveal rules are scoped to `html.ff-motion`, a class
`main.js` adds. If the script never runs, nothing is ever hidden.
`prefers-reduced-motion` is respected throughout.

---

## The analysers (the real computer-vision backend)

All three exercises are analysed for real. `utils/analysis.py` remains the
single integration point the UI talks to — nothing in the interface knows which
exercise it is showing — and underneath it sit three layers: shared
computer-vision building blocks, shared exercise machinery, and one package per
exercise.

```
analysis/                  shared, exercise-agnostic building blocks
├── video_processor.py     OpenCV reading, defensive metadata, H.264 output
├── pose_detector.py       MediaPipe Pose Landmarker (Tasks API, VIDEO mode)
├── validation.py          recording-suitability checks → ValidationResult
├── smoothing.py           short-gap interpolation + EMA smoothing
├── filters.py             Butterworth / Savitzky-Golay / zero-phase filtering
├── geometry.py            calculate_angle(), torso inclination, distances
├── annotation.py          FormFix-branded skeleton + HUD overlay renderer
├── models.py              the typed data structures between the stages
└── export.py              JSON/CSV evaluation exports

exercises/
├── common/                shared exercise machinery (written and tested once)
│   ├── spec.py            the declarative RuleSpec + camera-view vocabulary
│   ├── phases.py          the hysteresis repetition state machine
│   ├── persistence.py     persistent-violation filtering
│   ├── metrics.py         phase scoping, NaN-safe series, plausibility gating
│   ├── confidence.py      per-measurement reliability from real evidence
│   ├── uncertainty.py     published measurement error, carried into the verdict
│   ├── feedback.py        transparent score, per-rep verdicts, findings
│   └── failures.py        typed rejection → user-facing failure code
├── squat/                 side-view squat        (6 rules)
├── pulldown/              seated lat pulldown    (2 rules)
└── press/                 dumbbell shoulder press(3 rules)

each exercise package:
    ├── config.py          every threshold + the rule specifications
    ├── landmarks.py       its own MediaPipe landmark subset
    ├── metrics.py         per-frame + per-repetition measurement (no verdicts)
    ├── phases.py          its thresholds for the shared state machine
    ├── confidence.py      its reliability bands
    ├── rules.py           its technique rules, view- and phase-scoped
    ├── feedback.py        its wording and its evidence formats
    └── analyser.py        the pipeline orchestrator for that exercise
```

One state machine segments all three exercises. Each expresses its movement as
a signal that is **high at rest and falls as the repetition progresses** — the
squat's knee angle, the pulldown's elbow angle, the press's elbow *flexion* —
so the properties that make repetition counting trustworthy (hysteresis,
persistence, a reversal delta, a minimum excursion, FPS-independent duration
bounds) are argued and tested in one place.

Full technical documentation:
**[squat](docs/squat_analysis.md)** ·
**[lat pulldown](docs/pulldown_analysis.md)** ·
**[shoulder press](docs/press_analysis.md)**.

### The pipeline

```
upload → probe video → validate file → MediaPipe pose detection (VIDEO mode,
  up to 4 people; the athlete is the one nearest the camera, then followed
  frame to frame so a bystander cannot take over mid-repetition)
→ validate recording (coverage · framing · camera view + stability · tracking
  continuity — other people in shot are a caveat, not a rejection)
→ select the squat's landmark subset → interpolate short gaps → EMA smoothing
→ select the more visible body side
→ per-frame measurement, both legs (knee · hip · trunk · shin · depth · heel ·
  left/right difference · stance), in pixel space, normalised by body scale, with
  physiological plausibility gating
→ phase state machine → repetition detection
→ exercise plausibility gate (does this movement match the exercise the user
  selected? a contradiction stops the run rather than producing confident
  feedback about a movement that was never performed)
→ per-repetition phase segmentation (descent · bottom · ascent · finish)
→ robust standing baseline (median over genuine standing frames)
→ six technique rules, each scoped to its phase and to the camera views that
  support it, each requiring a *persistent* violation
→ per-metric reliability → per-repetition verdicts + aggregated feedback
→ annotated H.264 video, trimmed to the detected set (relevant joints
  highlighted per finding; occluded far-side limbs are not drawn)
→ AnalysisResult for the existing results view
```

`exercises/__init__.py` holds the analyser and feedback registries — the single
place the interface looks up which code runs for which exercise. Adding a
fourth exercise means adding a package and two lines there, not editing the
upload page, the results view or the analysis bridge.

### Lat Pulldown and Shoulder Press

The two upper-body analysers follow the same pipeline. What differs is the
movement, and therefore the camera:

* **Lat pulldown** (`exercises/pulldown/`) — seated, bilateral, pronated grip,
  bar to the front. Measures elbow flexion per arm and trunk inclination
  against the athlete's *own* top-position posture, signed towards the back
  when the facing direction can be established from the head. Two rules:
  excessive torso movement, and range of motion at both ends of the pull.
  A front-on recording still gets a range-of-motion verdict; the torso check
  reports what it could not see.

* **Shoulder press** (`exercises/press/`) — seated dumbbell press, back
  supported, both arms. Measures elbow angle, wrist and elbow heights and
  wrist-over-elbow offsets, every distance normalised by shoulder width in the
  same frame. Three rules: arm symmetry, elbow/wrist alignment, and range of
  motion. A side-on recording still gets a range-of-motion verdict; the two
  bilateral checks report what they could not see.

Both use **adaptive segmentation thresholds** — fixed margins below the
athlete's own resting extension — for a specific reason: with fixed absolute
levels, an athlete who never straightens their arms would produce no detected
repetitions at all, so the range-of-motion rule meant to notice that habit
would never run. Segmentation asks "was this a repetition?"; judgement asks
"did it cover enough range?", and only the second uses fixed criteria.

### Recording guidance (what the analysers expect)

Closest person to the camera · camera still · one continuous shot ·
reasonable lighting · several complete repetitions. Each exercise's own
instructions live in its config module (`RECORDING_TIPS`) and are shown above
the uploader, so the guidance the interface gives and the requirements the
analyser enforces cannot drift apart.

An empty gym is not required. Other people may be in shot: the athlete is
whoever is nearest the camera, and they are then followed from frame to frame
rather than re-chosen on each one, so a bystander standing upright cannot take
over the analysis at the bottom of a repetition when the athlete's bounding
box is at its shortest.

For the squat: whole body and feet visible, start and finish standing.

A **side view** supports the squat's measurements (depth, trunk and shin
inclination, lockout, timing); a **front-on** view supports only the
view-independent heel check, since the squat has no frontal-plane rule left
(see below). Neither is rejected — the camera angle decides
*what can be assessed*, not whether the video is accepted, and anything that
cannot be measured is reported as "Not assessed" with a reason. Only
genuinely unusable recordings are stopped, with a plain-English explanation and
retry tips: no person, key joints missing for most of the clip, an unreadable
file, no complete repetition, tracking that repeatedly loses which body it is
following, or a movement that contradicts the exercise the user selected.

### Technique rules and thresholds

Every threshold lives in `exercises/squat/config.py`, split into
**engineering** values (detector confidences, hysteresis, gap limits,
persistence, plausibility bands — system stability) and **technique** values
(depth, torso lean, heel lift, extension, descent timing). The technique values are **provisional calibration values**, clearly
labelled as such, pending literature/empirical validation — changing them
requires no code change anywhere else, and the debug output shows which value
each rule used and where it came from.

Rules are *declared*, not coded: each `SquatRule` names the measurement it
reads, the movement phase it is evaluated on, the camera views under which it
is meaningful, its acceptable range and tolerance, and how persistently a
violation must hold before it becomes a finding. A rule the camera view
cannot support reports "Not assessed" with a reason instead of a verdict.

The rules deliberately do *not* claim to detect spinal rounding, injury risk
or knee valgus, and never use "knees past toes" as a blanket error — a single
2D pose cannot support those claims. Anterior knee travel is measured and
exported for research, but it is never a correctness condition.

The overall score shown on the results ring is **not** a learned, black-box score: it
is `100 × (passed checks + 0.5 × warnings) / evaluable checks`, where a check
is one rule on one repetition. The formula itself is printed in the developer
panel (`FORMFIX_DEBUG`) and travels with the export bundle; the results page
keeps to plain language.

### Measurement uncertainty

Every finding carries the published measurement error of the quantity it rests
on. MediaPipe's knee-angle RMSE from a single lateral camera is **10.7°** for
the near limb and **25.1°** for the occluded far limb (Dill et al., 2024);
during squats it is **9.14°** at a good camera angle and **14.48°** at a poor
one (Dill et al., 2023). A deviation smaller than that cannot be resolved, so
FormFix reports it as *indicative rather than established* instead of stating
it as fact.

`exercises/common/uncertainty.py` holds the bands and their sources; every
derived figure carries its derivation. By default the layer annotates and
changes no verdict; `UNCERTAINTY_STRICT = True` additionally downgrades
marginal findings one step, so the two policies can be compared over the same
recordings rather than one being assumed. Full account:
**[measurement uncertainty](docs/measurement_uncertainty.md)**.

Building it surfaced two findings about FormFix's own numbers: the left/right
evenness bar (12°) and the heel-lift threshold (0.06) both sat below the noise
floor of their own measurements (±15.1° and ±0.15). The squat's evenness rule
was **removed** as a result — the measurement is still exported, but one
camera cannot support a verdict on it. The heel threshold is unchanged in the
default configuration and raised above its floor in the literature preset.

### Threshold presets

`exercises/squat/literature_config.py` is a second squat configuration whose
technique thresholds are derived from Kotiuk et al. (2022), Dill et al. (2024)
and Simoes et al. (2024), each with its provenance recorded. It exists to be
*compared* against the operational defaults, not to replace them:

```bash
python scripts/evaluate_videos.py manifest.csv --preset default
python scripts/evaluate_videos.py manifest.csv --preset literature
python scripts/evaluate_videos.py manifest.csv --strict-uncertainty
python scripts/evaluate_videos.py manifest.csv --filter savgol
```

The harness prints the preset and its full provenance before any result.

### Smoothing filter

`ANGLE_FILTER` selects how the movement signal is smoothed: `ema` (the
default), `butterworth` (4th-order, 2 Hz, zero-phase), `savgol` or
`moving_average`. `analysis/filters.py` implements all of them in NumPy — no
SciPy, so the pinned macOS NumPy-1 install stays intact — and
`scripts/compare_filters.py` measures what each costs the depth measurement.

The measured result is that the ranking **reverses** with the shape of the
turnaround: the 2 Hz Butterworth that Dill et al. selected is nearly unbiased
on a smooth repetition and the worst of the four on a sharp one. The default
is therefore unchanged, and each filter's measured bias is added to the depth
uncertainty band instead. Full account:
**[filter selection](docs/filter_selection.md)**.

### Model asset

The MediaPipe Pose Landmarker (full) model is stored at
`static/assets/models/pose_landmarker_full.task` (git-ignored; ~9 MB). Fetch
it once with `python scripts/download_pose_model.py`. If it is missing the
app fails with a clear message telling you exactly that.

### Crash isolation

Pose detection runs in a separate worker process (`analysis/pose_worker.py`).
MediaPipe's native code can hard-crash the interpreter on some platforms
(seen on macOS as "Python quit unexpectedly" during landmark detection);
isolating it means a crash becomes a friendly in-app error with retry tips
instead of killing Streamlit. Set `FORMFIX_POSE_SUBPROCESS=0` to force the
in-process path when debugging. If the engine still stops repeatedly on
macOS, pin a known-stable build: `pip install 'mediapipe==0.10.14'`.

### Annotated result video & FFmpeg

OpenCV writes an mp4v intermediate; FFmpeg (a **system** dependency, not a
pip package) converts it to H.264 + faststart so `st.video` can actually play
it in the browser. Without FFmpeg the mp4v file is kept and the limitation is
logged. The annotated clip is silent — the analysis needs no audio, and a
reliable video was prioritised over carrying the original track (documented
decision).

The clip is rendered only over the frames the set actually occupies
(`analysis/trimming.py`): the window runs from a second before the first
detected repetition to a second after the last, so the walk-up and the re-rack
are skipped and the render does proportionally less work. Frame indices stay
those of the original recording, so every timestamp in the feedback, the
evidence strings and the exports still refers to the video as uploaded. Any
uncertainty — no repetitions, an implausible window, too little to gain —
renders the whole video instead.

### What happens to an uploaded video

The upload is written to a scratch directory, analysed, and deleted as soon as
the run finishes (`utils/analysis.py`). The annotated result stays in its own
session directory so it can be replayed, and is swept a few hours later
(`SESSION_TTL_SECONDS` in `analysis/video_processor.py`). Nothing is written to
a database, no account exists, and the upload screen states this before the
file picker rather than after — the wording there is deliberately precise about
the two different lifetimes, because it is effectively a promise.

### Debug / calibration

* `FORMFIX_DEBUG=1 streamlit run app.py` — the "Technical details" panel on
  the results page gains the full developer metrics (thresholds, baselines,
  rep boundaries, coverage, side scores), and every run writes a JSON/CSV
  bundle to `analysis_results/<run id>/` (summary.json, frame_metrics.csv,
  reps.csv) for evaluation and the dissertation.
* `python scripts/calibrate.py video.mp4 pulldown --frames` — the developer
  calibration tool for any exercise. Prints the recording diagnostics, a
  per-repetition measurement table, and every rule verdict *beside the
  threshold it was compared against and the number of frames that supported
  it*. `--frames` saves annotated key frames (start / turning point / end of
  each rep) so the printed numbers can be checked against what the video
  shows. `scripts/calibrate_squat.py` is kept for the squat's own tooling.
* Session caching: results are cached on (file hash, exercise), so unrelated
  UI interactions never re-run MediaPipe; "Analyse again" forces a real
  re-run. Uploads and rendered clips live in temporary session directories.

### Testing

```bash
pytest                     # 660+ tests: geometry, normalisation, smoothing,
                           # landmarks, persistence, the shared state machine,
                           # phase durations, every rule of every exercise,
                           # view gating, reliability, feedback, UI mapping,
                           # exercise wiring, end-to-end pipelines, the
                           # evaluation harness, the digital filters
                           # (against their analytic response), the
                           # measurement-uncertainty layer, person selection
                           # in multi-person recordings, far-side occlusion,
                           # the render window and the exercise-plausibility
                           # gate
FORMFIX_TEST_VIDEO=path/to/side-view-squat.mp4 pytest   # + real-video test
ruff check .               # lint (config in ruff.toml)
black -l 104 .             # formatting (line length matches ruff.toml)
```

The end-to-end tests drive the full pipeline (validation → smoothing →
metrics → reps → rules → feedback → annotated render) with kinematically
plausible synthetic pose tracks — one generator per exercise — injecting the
faults each analyser is meant to find. Every suite also has a
**false-positive protection** group: landmark noise, a brief landmark drop, a
value hovering just under a threshold and a mirrored recording must all leave
the verdicts unchanged. A rule engine tested only on faults will report faults
on everything. MediaPipe detection itself is exercised via the optional
real-video test.

### Technical evaluation

`scripts/evaluate_videos.py` runs FormFix over a manifest of labelled
recordings and writes `results.csv`, `error_matrix.csv` (TP/FP/FN/TN per rule)
and `summary.md`. See **[evaluation/README.md](evaluation/README.md)** for what
to record and how to read the output honestly — in particular, why a
*not assessed* measurement is neither a detection nor a miss, and why the
harness deliberately computes no accuracy percentage.

### Real-video verification

**[evaluation/reference_clip_runs.md](evaluation/reference_clip_runs.md)**
records genuine end-to-end runs — real MediaPipe detection, not synthetic pose
tracks — over the three reference clips. The squat and press clips are
correctly *refused* (a montage is not a set; a third of a clip is not enough to
judge someone), and the pulldown clip is analysed with its limits stated. Two
implementation defects that only real footage surfaced were found and fixed
there. These runs verify the pipeline; they are not an evaluation, and no
threshold was moved on the basis of them.

### Known limitations (state these in the dissertation)

* 2D single-camera perspective — no true joint angles in 3D, no forces/loads.
* Landmark occlusion, loose clothing, poor lighting and camera movement all
  degrade MediaPipe's estimates; the validation layer rejects the worst cases
  but cannot make a marginal recording accurate. MediaPipe reports high
  confidence for a limb it has *inferred* behind the body rather than seen, so
  occlusion is decided geometrically — a far landmark projecting onto its near
  twin is not drawn — instead of from the detector's own visibility score.
* The camera angle is detected heuristically, not classified perfectly, and
  is estimated once per video — a clip that cuts between angles is flagged
  but still gets a single label.
* A hard scene cut inside a clip changes the knee-angle trace instantly,
  which the timing rule can read as a very fast descent; one continuous
  recording is assumed.
* MediaPipe landmarks carry estimation error even in good conditions, and it
  is now quantified rather than merely acknowledged: 10.7 deg RMSE on a knee
  angle from a single lateral camera, 25.1 deg for the occluded far limb.
  Deviations smaller than that are reported as indicative, not established.
* Two thresholds (left/right evenness at 12 deg, heel lift at 0.06) sat below
  the noise floor of their own measurements. The evenness rule was removed;
  the heel threshold is unchanged in the defaults and raised in the literature
  preset. Both are documented.
* Every smoothing filter reports the bottom of a squat slightly shallower than
  it was; the default EMA's measured worst case is +4.5 deg, and it is carried
  into the depth uncertainty band.
* No per-user strictness setting: one threshold set is applied to everyone,
  though flexibility and body proportions legitimately differ.
* No user study and no labelled-recording evaluation, so no efficacy claim and
  no accuracy figure exist.
* Person selection is resolved by proximity to the camera, not by identity. The
  first frame is decided on apparent size alone, so somebody standing closer to
  the lens than the athlete at the moment the clip opens would be picked and
  then followed; weighting that first choice towards whoever is moving is
  future work.
* The exercise-plausibility gate checks the label rather than classifying the
  movement: it can tell that a clip is not the exercise that was selected, but
  never what it actually was. It is deliberately one-directional — it looks for
  contradictions and abstains on thin evidence — so an unusual-but-genuine
  attempt passes at the cost of some genuinely mislabelled clips also passing.
* Anatomy differs between people; the per-video standing baseline compensates
  partially, but the technique thresholds remain provisional calibration
  values, not clinically validated constants.
* The system cannot diagnose injury, measure spinal curvature, or assess
  knee valgus (a 3D motion one camera cannot resolve), and never claims to.
  The left/right knee difference is measured and exported, but it describes
  evenness of knee *flexion* rather than valgus, and it is no longer graded:
  in a side view one leg hides the other, and even from the front the
  comparison's error exceeds the threshold that would have to fire.

> FormFix provides exercise-technique guidance from visible movement and is
> not a replacement for professional coaching or medical advice.

---

## The landmark system

All three exercise figures are drawn by one renderer, `utils/pose.py`, from
coordinates stored in `utils/exercise_data.py`. That is what guarantees a single
FormFix visual language rather than generic pose-estimation dots:

* Electric Cyan hairline links between joints, `non-scaling-stroke` so they stay
  crisp at every figure size
* a cyan → lime gradient on the limb the rules engine is reading, with a single
  light travelling along it
* Neon Lime nodes with an outer glow, a ping ring and a slow orbit ring on the
  joint under analysis
* small, uniform node sizes with a bright centre point
* a dashed structural centre line and a lime arc for the measured joint angle

---

## Deployment

The app is deployed on Streamlit Community Cloud from the `main` branch of the
GitHub mirror, with `app.py` as the entry point and Python 3.11.

`packages.txt` lists the apt packages the deployment image installs before pip
runs. It deliberately contains nothing but bare package names:

* `libgl1` and `libglib2.0-0` - OpenCV's Python wheel links against libGL and
  glib. They are present on a normal desktop but not in the deployment image,
  and without them `import cv2` fails, which makes the app fall back to its
  labelled sample output instead of analysing anything.
* `ffmpeg` - converts the annotated clip to H.264 so it plays in the browser.
  It is a system binary, not a pip package.

Do not add comments to `packages.txt`. Community Cloud feeds the file to
`apt-get` through `xargs`, which does not strip `#` lines and treats
apostrophes and stray `/` characters as syntax - a commented file fails the
build with `xargs: unmatched single quote` and `E: Unsupported file /`.
`requirements.txt` is read by pip and does support comments.

The MediaPipe model is not in the repository; `analysis.pose_detector.fetch_model`
downloads it on first start-up, so a fresh deploy needs no extra step.

---

## Notes

* Requires Streamlit 1.36 or newer: `st.container(key=…)` scopes the widget
  styling and `st.navigation(..., position="hidden")` provides the two pages.
  `app.py` falls back gracefully if `position` is unsupported.
* The typefaces (Space Grotesk, Inter) load from Google Fonts, as in the
  original. Offline, the app falls back to the system sans-serif stack.
* This is an educational exercise-technique prototype, not a replacement for
  professional coaching or medical advice.
