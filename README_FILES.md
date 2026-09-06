# FormFix — file guide

One sentence per file, so you can find anything in the project without opening it.
For how the system works, see [README.md](README.md).

---

## Root

| File | What it does |
| --- | --- |
| `app.py` | The entry point: sets the page config, injects the stylesheet, and declares the two pages (`/` and `/analyse`) with `st.navigation`. |
| `requirements.txt` | The pinned Python dependencies, with the reasoning for the macOS NumPy-1 pin and the Streamlit upper bound written in as comments. |
| `packages.txt` | The system packages Streamlit Community Cloud installs with apt before pip runs: `libgl1` and `libglib2.0-0`, which OpenCV's wheel links against and which the deployment image does not otherwise have, plus `ffmpeg` for the H.264 conversion of the annotated clip. Without it `import cv2` fails on the deployed app and no analysis runs. |
| `pyproject.toml` | Black's configuration (line length 104, target Python 3.11). |
| `ruff.toml` | Ruff's lint rule selection and its matching line length. |
| `README.md` | The full project documentation — architecture, pipeline, thresholds, uncertainty, testing and limitations. |
| `README_FILES.md` | This file: a one-line description of every file in the project. |
| `questions/` | Personal reference sheets written while preparing for the viva. Git-ignored: they are notes to myself, not part of the submission. |
| `.gitignore` | Keeps the virtual environment, caches, the ~9 MB pose model and generated analysis output out of Git. |
| `.streamlit/config.toml` | Streamlit's own settings: dark theme, wide layout, and static file serving from `static/`. |

---

## `views/` — one module per page

| File | What it does |
| --- | --- |
| `views/__init__.py` | Exposes the two page modules so `app.py` can register them. |
| `views/home.py` | Assembles the homepage by calling each section component in order. |
| `views/analyse.py` | Assembles the `/analyse` page: loads its extra stylesheet and behaviour script, then renders the analysis studio. |

---

## `components/` — one module per section or feature

| File | What it does |
| --- | --- |
| `components/__init__.py` | Collects every section component into one import surface for the two page modules. |
| `components/background.py` | Draws the two fixed layers behind and above everything — the scroll-progress line and the atmosphere gradients. |
| `components/navbar.py` | The floating pill navigation with active-section highlighting. |
| `components/hero.py` | Section 1 — the hero: headline, the animated pose figure, and the primary call to action. |
| `components/about.py` | Section 1b — what FormFix is, who it is for, and what it does not claim to do. |
| `components/problem.py` | Section 2 — the problem the project addresses. |
| `components/how_it_works.py` | Section 3 — the scroll-scrubbed step rail explaining the pipeline. |
| `components/exercise_section.py` | Section 4 — the exercise explorer that cross-fades between squat, press and pulldown. |
| `components/explainable.py` | Section 5 — the explainable-feedback and traceability pipeline. |
| `components/results_section.py` | Section 6 — renders an analysis report, used for both the example and real results. |
| `components/technology.py` | Section 7 — the "under the hood" technology breakdown. |
| `components/research.py` | Section 7b — the academic and technical foundation, including the reviewed papers. |
| `components/outro.py` | Section 8 — the closing call to action. |
| `components/footer.py` | The page footer. |
| `components/analyse_page.py` | The `/analyse` page itself: the uploader, the progress tracker and the four-stage state machine (`idle` → `ready` → `analysing` → `complete`). |
| `components/analysis_results.py` | The results view: score ring, verdict, positives, corrections, measured checks, the "see the full detail" panel and the reference-video card. The panel is plain language for the athlete — every finding as the mistake and the one thing to try, plus the rep-by-rep breakdown and what was not assessed. The traceability record (measurement, landmarks, phase, rule settings, threshold provenance, per-rep evidence, rule id) and the run's own settings and limits (analysed side, camera orientation, score formula, gated metrics, validation warnings) are developer-only, under `FORMFIX_DEBUG`. |
| `components/upload_section.py` | The original in-page uploader, superseded by the `/analyse` page and kept for reference. |

---

## `utils/` — application plumbing

| File | What it does |
| --- | --- |
| `utils/__init__.py` | Marks the shared-utilities package. |
| `utils/paths.py` | Derives every filesystem path with `pathlib`, so the app runs from any working directory. |
| `utils/routing.py` | Holds the page URLs used by the plain-HTML links in the markup. |
| `utils/assets.py` | Turns an asset path into a browser URL — a served static URL when possible, a base64 data URI otherwise, and a transparent pixel if the file is missing. |
| `utils/styling.py` | Injects the CSS with `st.markdown`, and installs the JavaScript through a zero-height iframe with a per-run nonce. |
| `utils/helpers.py` | Small shared helpers for building the HTML markup. |
| `utils/pose.py` | The FormFix landmark renderer that draws every pose figure on the site in one consistent visual language. |
| `utils/exercise_data.py` | The single source of truth for the three exercises and their landmark geometry on the homepage. |
| `utils/analysis.py` | The bridge between the interface and the computer-vision backend — the one function the UI calls, plus the stage list, score bands and result types. |

---

## `analysis/` — shared, exercise-agnostic computer vision

| File | What it does |
| --- | --- |
| `analysis/__init__.py` | Marks the shared computer-vision package. |
| `analysis/video_processor.py` | Reads video with OpenCV, probes its metadata defensively, and writes the H.264 output. |
| `analysis/pose_detector.py` | Wraps the MediaPipe Pose Landmarker (Tasks API, VIDEO running mode) to produce landmarks per frame, and decides which person is the athlete when several are in shot - nearest the camera to begin with, then followed frame to frame so a bystander cannot take over the analysis mid-repetition. |
| `analysis/pose_worker.py` | Runs pose detection in a separate process, so a native MediaPipe crash becomes an in-app error instead of killing Streamlit. |
| `analysis/validation.py` | Decides whether a recording can be analysed at all — coverage, framing, camera orientation and stability, and which body side to use. Other people in shot are a caveat, not a rejection: what stops a run is the tracker repeatedly losing which body it is following. |
| `analysis/smoothing.py` | Interpolates short landmark gaps and applies EMA smoothing to the tracks. |
| `analysis/filters.py` | Implements Butterworth, Savitzky–Golay, moving-average and EMA filtering in pure NumPy, with no SciPy dependency. |
| `analysis/geometry.py` | The geometry primitives every analyser shares: joint angles, torso inclination, signed inclination, angular rate and distances. |
| `analysis/normalisation.py` | Scales pixel measurements by the athlete's own body dimensions so distance from the camera does not change the result. |
| `analysis/annotation.py` | Renders the annotated result video in the FormFix visual language, highlighting the joints behind each finding. Landmarks the camera cannot actually see - a far limb behind the body in a side view - are dropped rather than drawn from the detector's guess. |
| `analysis/trimming.py` | Works out which frames of the upload are the actual set, from the reps already detected, so the annotated video shows the exercise rather than the walk-up and the re-rack. Falls back to the whole video whenever the window is uncertain. |
| `analysis/overlay.py` | The drawing language used by the annotator — the skeleton, nodes, glow and HUD elements. |
| `analysis/models.py` | The typed data structures passed between every stage of the pipeline. |
| `analysis/export.py` | Writes the JSON and CSV analysis bundles used for evaluation and the dissertation. |

---

## `exercises/common/` — machinery written and tested once

| File | What it does |
| --- | --- |
| `exercises/__init__.py` | The analyser and feedback registries — the single place the interface looks up which code runs for which exercise. |
| `exercises/common/__init__.py` | Marks the shared exercise-machinery package. |
| `exercises/common/spec.py` | Defines `RuleSpec` and the camera-view vocabulary — the declarative shape every technique rule is written in. |
| `exercises/common/phases.py` | The hysteresis repetition state machine that segments all three exercises. |
| `exercises/common/persistence.py` | Requires a violation to hold over enough frames and enough of the phase before it can become a finding. |
| `exercises/common/metrics.py` | Shared measurement helpers: phase scoping, NaN-safe series handling and physiological plausibility gating. |
| `exercises/common/confidence.py` | Computes per-measurement reliability (High / Medium / Low / Cannot assess) from visibility, usable frames, camera-view support and evidence breadth. |
| `exercises/common/uncertainty.py` | Holds the published measurement-error bands and marks any finding whose margin falls inside them as indicative rather than established. |
| `exercises/common/feedback.py` | Turns rule outcomes into the transparent score, per-repetition verdicts and the aggregated findings the user sees. |
| `exercises/common/failures.py` | Converts a typed recording rejection into the plain-English failure message and retry tips shown in the interface. |
| `exercises/common/plausibility.py` | Checks that the movement matches the exercise the user selected, so a bench press labelled "shoulder press" is refused instead of being written up as overhead-press technique. Looks only for contradictions and abstains whenever the evidence is thin. |

---

## `exercises/squat/` — side-view squat (6 rules)

| File | What it does |
| --- | --- |
| `exercises/squat/__init__.py` | Exposes the squat analyser and its feedback builder. |
| `exercises/squat/config.py` | Every squat threshold in one place, split into engineering values and technique values, each with its provenance. |
| `exercises/squat/literature_config.py` | A second squat configuration whose thresholds derive from published biomechanics, kept for comparison against the defaults. |
| `exercises/squat/landmarks.py` | The subset of MediaPipe's 33 landmarks the squat reads, and what each measurement genuinely needs. |
| `exercises/squat/metrics.py` | Per-frame and per-repetition squat measurement — knee, hip, trunk, shin, depth, heel, left/right difference and stance — with no verdicts. The left/right difference is exported only; no rule reads it. |
| `exercises/squat/phases.py` | The squat's thresholds for the shared state machine, and its descent / bottom / ascent / finish segmentation. |
| `exercises/squat/persistence.py` | The squat's own persistent-violation filtering. |
| `exercises/squat/confidence.py` | The squat's reliability bands over the shared confidence engine. |
| `exercises/squat/rules.py` | The six squat technique rules, each scoped to a phase and to the camera views that support it. |
| `exercises/squat/feedback.py` | The squat's wording and evidence formats over the shared aggregator. |
| `exercises/squat/analyser.py` | Orchestrates the whole squat pipeline from video to `AnalysisResult`. |

---

## `exercises/pulldown/` — seated lat pulldown (2 rules)

| File | What it does |
| --- | --- |
| `exercises/pulldown/__init__.py` | Exposes the lat-pulldown analyser and its feedback builder. |
| `exercises/pulldown/config.py` | Every lat-pulldown threshold and rule specification in one place. |
| `exercises/pulldown/landmarks.py` | The landmark subset the pulldown reads, and what each measurement needs. |
| `exercises/pulldown/metrics.py` | Measures elbow flexion per arm and trunk inclination against the athlete's own top position. |
| `exercises/pulldown/phases.py` | The pulldown's adaptive segmentation thresholds and repetition counting. |
| `exercises/pulldown/confidence.py` | The pulldown's reliability bands. |
| `exercises/pulldown/rules.py` | The two pulldown rules: excessive torso movement, and range of motion at both ends of the pull. |
| `exercises/pulldown/feedback.py` | The pulldown's wording and evidence formats. |
| `exercises/pulldown/analyser.py` | Orchestrates the whole lat-pulldown pipeline. |

---

## `exercises/press/` — dumbbell shoulder press (3 rules)

| File | What it does |
| --- | --- |
| `exercises/press/__init__.py` | Exposes the shoulder-press analyser and its feedback builder. |
| `exercises/press/config.py` | Every shoulder-press threshold and rule specification in one place. |
| `exercises/press/landmarks.py` | The landmark subset the press reads, and what each measurement needs. |
| `exercises/press/metrics.py` | Measures elbow angles, wrist and elbow heights and wrist-over-elbow offsets, normalised by shoulder width in the same frame. |
| `exercises/press/phases.py` | The press's adaptive segmentation thresholds and repetition counting. |
| `exercises/press/confidence.py` | The press's reliability bands. |
| `exercises/press/rules.py` | The three press rules: arm symmetry, elbow and wrist alignment, and range of motion. |
| `exercises/press/feedback.py` | The press's wording and evidence formats. |
| `exercises/press/analyser.py` | Orchestrates the whole shoulder-press pipeline. |

---

## `tests/` — 660 tests

| File | What it does |
| --- | --- |
| `tests/__init__.py` | Marks the test package. |
| `tests/conftest.py` | Makes the application packages importable no matter where pytest is run from. |
| `tests/synthetic_squat.py` | Generates kinematically plausible synthetic squat pose tracks with a known, deliberately injected fault. |
| `tests/synthetic_pulldown.py` | The same generator for the lat pulldown. |
| `tests/synthetic_press.py` | The same generator for the shoulder press. |
| `tests/test_geometry.py` | Unit-tests the shared geometry utilities. |
| `tests/test_geometry_upper_body.py` | Tests the geometry the upper-body exercises added — signed inclination and angular rate. |
| `tests/test_normalisation.py` | Tests that body-relative normalisation makes measurements independent of camera distance. |
| `tests/test_smoothing.py` | Tests short-gap interpolation and EMA smoothing. |
| `tests/test_filters.py` | Checks each digital filter against its known analytic response rather than a stored snapshot of its own output. |
| `tests/test_landmarks.py` | Tests the squat's landmark subset and its per-measurement requirements. |
| `tests/test_validation.py` | Tests the recording-suitability layer and its reason codes. |
| `tests/test_ui_mapping.py` | Tests how a validation verdict reaches the interface as a user-facing message. |
| `tests/test_metrics.py` | Tests the measurement layer, phase segmentation and frame-rate independence. |
| `tests/test_phases.py` | Tests the state machine on synthetic knee-angle sequences. |
| `tests/test_common_phases.py` | Tests the shared repetition state machine used by all three exercises. |
| `tests/test_persistence.py` | Tests that a violation must persist before it becomes a finding. |
| `tests/test_confidence.py` | Tests metric-level reliability banding. |
| `tests/test_uncertainty.py` | Tests the measurement-uncertainty layer, including the two thresholds that sit below their own noise floor. |
| `tests/test_rules.py` | Tests each squat technique rule independently on constructed repetitions. |
| `tests/test_rule_engine.py` | Tests view gating, persistence gating, the frontal-plane and timing rules, and the feedback they produce. |
| `tests/test_pulldown_metrics.py` | Tests the lat-pulldown measurement layer, including that a restricted athlete still has their reps counted. |
| `tests/test_pulldown_rules.py` | Tests the lat-pulldown rules on constructed repetitions. |
| `tests/test_pulldown_integration.py` | Drives the whole lat-pulldown pipeline end to end on synthetic recordings. |
| `tests/test_press_metrics.py` | Tests the shoulder-press measurement layer, including that an athlete who never locks out still has reps counted. |
| `tests/test_press_rules.py` | Tests the shoulder-press rules on constructed repetitions. |
| `tests/test_press_integration.py` | Drives the whole shoulder-press pipeline end to end, including that a square front view is recognised as frontal. |
| `tests/test_integration.py` | Drives the whole squat pipeline end to end on synthetic recordings, with the optional real-video test. |
| `tests/test_annotation_output.py` | Tests the annotated-video writer. |
| `tests/test_occlusion.py` | Tests that a far-side limb hidden behind the body in a side view is not drawn, and that the test is skipped wherever it cannot be answered. |
| `tests/test_trimming.py` | Tests the render window: what gets trimmed, and the many cases that must fall back to the whole video rather than produce an empty one. |
| `tests/test_person_selection.py` | Tests that the athlete is tracked consistently when others are in shot - including that a bystander cannot take over at the bottom of a squat - and that a busy recording is only rejected when tracking genuinely comes apart. |
| `tests/test_plausibility.py` | Tests the exercise-plausibility gate: a bench press labelled as a press is refused, while a shallow squat or a partial press still passes. |
| `tests/test_exercise_registry.py` | Tests the wiring between the three exercises and the interface. |
| `tests/test_literature_preset.py` | Tests the literature-derived squat preset and its provenance. |
| `tests/test_evaluation_harness.py` | Tests the technical-evaluation harness and its output tables. |

---

## `scripts/`

| File | What it does |
| --- | --- |
| `scripts/main.js` | The homepage behaviour layer: scroll progress, reveals, the exercise explorer and smooth navigation, using native browser APIs only. |
| `scripts/analyse.js` | The `/analyse` page behaviour: the reference-video lightbox and the score count-up. |
| `scripts/download_pose_model.py` | Fetches the MediaPipe Pose Landmarker model file once into `static/assets/models/`. |
| `scripts/calibrate.py` | The developer calibration tool for any exercise — prints diagnostics, per-repetition measurements and every rule verdict beside the threshold it was compared against. |
| `scripts/calibrate_squat.py` | The squat's own earlier calibration tool, kept for its squat-specific output. |
| `scripts/check_camera_view.py` | Prints the camera-orientation heuristic's verdict and confidence for a clip, for checking how a recording will be classified before analysing it. |
| `scripts/compare_filters.py` | Measures what each smoothing filter costs the depth measurement, on FormFix's own criterion rather than overall landmark RMSE. |
| `scripts/evaluate_videos.py` | Runs FormFix over a manifest of labelled recordings and writes `results.csv`, `error_matrix.csv` and `summary.md`. |

---

## `docs/`

| File | What it does |
| --- | --- |
| `docs/squat_analysis.md` | The full technical account of the squat analyser. |
| `docs/pulldown_analysis.md` | The full technical account of the lat-pulldown analyser. |
| `docs/press_analysis.md` | The full technical account of the shoulder-press analyser. |
| `docs/threshold_tuning.md` | The auditable tuning log: what every threshold started at, why, what testing showed, and what it ended at. |
| `docs/measurement_uncertainty.md` | Where each measurement-error band comes from, and how it is carried into the verdict. |
| `docs/filter_selection.md` | The measured trade-off between the four smoothing filters, and why the default was kept. |
| `docs/literature_basis.md` | What each reviewed paper contributed, what was extended, and what was explicitly not taken. |
| `docs/limitations.md` | What the system cannot do, written to be quoted in the dissertation. |
| `docs/FormFix_Research_Provenance.docx` | The provenance table in the form written for the report rather than for the code. |
| `docs/FormFix_Overview.docx` | The project overview written for a non-technical reader. |
| `docs/FormFix_Technical_Documentation.docx` | The technical documentation in report form. |

---

## `wireframes/`

Mid-fidelity wireframes covering the key screens and interaction states were created as SVG diagrams using representative interface content. They document the planned interface structure and user flow that informed the implementation of FormFix AI.

| File | What it does |
| --- | --- |
| `wireframes/generate_wireframes.py` | Generates every SVG below. Re-running it regenerates all of them, so the wording lives in one place. |
| `wireframes/00-screen-flow.svg` | The whole journey on one sheet: which screen leads to which. |
| `wireframes/01-home-full-desktop.svg` | The homepage end to end, all sections in order. |
| `wireframes/02-home-hero-desktop.svg` | The hero section on its own. |
| `wireframes/02b-home-about-desktop.svg` | The "what this is / who it is for" section added after supervisor feedback. |
| `wireframes/03-home-exercise-explorer-desktop.svg` | The exercise explorer that cross-fades between the three exercises. |
| `wireframes/04-home-explainable-and-example-desktop.svg` | The explainable-feedback pipeline and the example report. |
| `wireframes/05-home-technology-research-outro-desktop.svg` | The technology, research and closing sections. |
| `wireframes/06-analyse-idle-desktop.svg` | `/analyse` before anything is uploaded. |
| `wireframes/07-analyse-ready-desktop.svg` | A video chosen, ready to analyse. |
| `wireframes/08-analyse-analysing-desktop.svg` | The progress tracker mid-run. |
| `wireframes/09-analyse-results-desktop.svg` | The results view: score, verdict, corrections. |
| `wireframes/10-analyse-results-expanded-desktop.svg` | The same with the full-detail panel opened. |
| `wireframes/11-analyse-failure-desktop.svg` | A rejected recording and its explanation. |
| `wireframes/12-analyse-limited-quality-desktop.svg` | An analysed run where some checks could not be assessed. |
| `wireframes/13-reference-modal-desktop.svg` | The reference-technique video overlay. |
| `wireframes/m01-home-hero-mobile.svg` | The hero at phone width. |
| `wireframes/m02-home-menu-mobile.svg` | The navigation on a phone. |
| `wireframes/m03-home-sections-mobile.svg` | The homepage sections stacked for a phone. |
| `wireframes/m04-analyse-idle-mobile.svg` | The upload screen on a phone. |
| `wireframes/m05-analyse-analysing-mobile.svg` | The progress tracker on a phone. |
| `wireframes/m06-analyse-results-mobile.svg` | The results view on a phone. |

---

## `styles/`, `static/` and generated output

| Path | What it does |
| --- | --- |
| `styles/main.css` | The full stylesheet, written element-qualified so Streamlit's own element styles cannot override it. |
| `styles/analyse.css` | The `/analyse` page's additions, loaded on top of `main.css`. |
| `static/assets/images/` | The three 3D exercise renders used by the exercise explorer. |
| `static/assets/logo/` | The FormFix logo and logo mark. |
| `static/assets/videos/` | Optional reference technique clips, one per exercise, shown in the results lightbox. |
| `static/assets/models/` | The MediaPipe Pose Landmarker model file, fetched by the download script and git-ignored. |
| `static/assets/docs/research-paper.pdf` | The research paper offered for download on the homepage. |
| `analysis_results/` | Generated per-run debug bundles (`summary.json`, `reps.csv`, `frame_metrics.csv`), written only when `FORMFIX_DEBUG=1`. |
