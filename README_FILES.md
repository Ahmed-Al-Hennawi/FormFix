# FormFix - file guide

One line on every file, so you can find things without opening them.
For how the whole system works, see [README.md](README.md).

## Root

| File | What it is |
| --- | --- |
| `app.py` | Starts the app, loads the CSS and sets up the two pages (`/` and `/analyse`). |
| `requirements.txt` | Python packages, pinned to the versions I tested with. |
| `packages.txt` | System packages the deploy installs (`libgl1`, `libglib2.0-0t64`) - no comments allowed in this file. |
| `pyproject.toml` | Black settings (line length 104). |
| `ruff.toml` | Ruff lint rules. |
| `.streamlit/config.toml` | Streamlit settings: dark theme, static files, 300 MB upload limit. |
| `.gitignore` | Keeps the venv, caches, pose model and generated output out of git. |
| `README.md` | The main project overview. |
| `README_FILES.md` | This file. |

## `views/` - the two pages

| File | What it is |
| --- | --- |
| `views/home.py` | The homepage: calls each section in order. |
| `views/analyse.py` | The `/analyse` page: loads its CSS/JS and the analysis screen. |

## `components/` - page sections

| File | What it is |
| --- | --- |
| `background.py` | Scroll progress line and background glow. |
| `navbar.py` | The floating navigation bar. |
| `hero.py` | Section 1: headline and animated pose figure. |
| `about.py` | Section 1b: what FormFix is and who it's for. |
| `problem.py` | Section 2: the problem. |
| `how_it_works.py` | Section 3: the five steps. |
| `exercise_section.py` | Section 4: exercise explorer (squat / press / pulldown). |
| `explainable.py` | Section 5: explainable feedback example. |
| `results_section.py` | Section 6: the report card (example and real results). |
| `technology.py` | Section 7: technology used. |
| `research.py` | Section 7b: research and papers. |
| `outro.py` / `footer.py` | Section 8: final call to action, and the footer. |
| `analyse_page.py` | The `/analyse` page: uploader, progress tracker, idle → ready → analysing → complete. |
| `analysis_results.py` | The results: verdict, what to fix, what went well, checks, video, reference clip, full detail. |
| `upload_section.py` | The old uploader, replaced by `/analyse` and not used any more. |

## `utils/` - helpers

| File | What it is |
| --- | --- |
| `paths.py` | All file paths, so the app runs from any folder. |
| `routing.py` | URLs for the links between the pages. |
| `assets.py` | Turns images/videos into browser URLs. |
| `styling.py` | Injects the CSS and JavaScript into Streamlit. |
| `helpers.py` | Small HTML helpers. |
| `pose.py` | Draws the landmark overlay on the site's exercise figures. |
| `exercise_data.py` | Data for the three exercises (text, tips, pose geometry, camera position). |
| `camera_guide.py` | The overhead "where to put your phone" diagram. |
| `analysis.py` | The bridge between the UI and the analysis; also the score bands and result types. |

## `analysis/` - shared computer vision

| File | What it is |
| --- | --- |
| `video_processor.py` | Reads videos, and writes the H.264 result video. |
| `pose_detector.py` | MediaPipe wrapper; picks and follows the right person. |
| `pose_worker.py` | Runs detection in a separate process so a MediaPipe crash can't kill the app. |
| `validation.py` | Checks the recording can be analysed (coverage, framing, camera angle). |
| `stabilise.py` | Drops landmark positions that can't be real movement, before anything is filled in or smoothed. |
| `smoothing.py` | Fills short gaps and smooths the landmarks (EMA). |
| `filters.py` | Butterworth, Savitzky-Golay and moving-average filters in NumPy. |
| `geometry.py` | Angles, inclinations and distances. |
| `normalisation.py` | Scales distances by body size. |
| `trimming.py` | Finds the part of the video with the actual set. |
| `annotation.py` | Decides what goes on each frame of the result video. |
| `overlay.py` | Draws the skeleton: one style, thin links, small joints, the far side fainter. |
| `models.py` | The data classes passed between the steps. |
| `export.py` | Saves JSON/CSV results in debug mode. |

## `exercises/` - the analysers

| File | What it is |
| --- | --- |
| `__init__.py` | Which analyser and feedback belong to which exercise. |
| `common/spec.py` | The rule format and camera-view names. |
| `common/phases.py` | The rep-counting state machine used by all three exercises. |
| `common/persistence.py` | A fault has to last a few frames before it counts. |
| `common/metrics.py` | Shared measuring helpers. |
| `common/confidence.py` | Reliability per measurement (High / Medium / Low / Cannot assess). |
| `common/uncertainty.py` | Published measurement error, applied to each finding. |
| `common/feedback.py` | Score, per-rep verdicts and feedback text. |
| `common/failures.py` | Turns a rejected recording into a friendly message. |
| `common/plausibility.py` | Checks the movement matches the exercise picked. |

Each of `squat/`, `pulldown/` and `press/` has:

| File | What it is |
| --- | --- |
| `config.py` | Every threshold and rule for that exercise. |
| `landmarks.py` | Which landmarks it uses. |
| `metrics.py` | Measuring only, no verdicts. |
| `phases.py` | Its settings for the rep counter. |
| `confidence.py` | Its reliability settings. |
| `rules.py` | Its technique rules (squat 5, pulldown 2, press 3). |
| `feedback.py` | Its feedback wording. |
| `analyser.py` | Runs all the steps in order. |

The squat also has `literature_config.py` (thresholds from published papers,
for comparison) and `persistence.py` (re-exports the shared one).

## `tests/`

| File | What it tests |
| --- | --- |
| `synthetic_squat.py`, `synthetic_pulldown.py`, `synthetic_press.py` | Make fake pose data with known faults. |
| `test_integration.py`, `test_pulldown_integration.py`, `test_press_integration.py` | The whole pipeline, end to end, per exercise. |
| `test_rules.py`, `test_pulldown_rules.py`, `test_press_rules.py` | Each rule on hand-built reps. |
| `test_metrics.py`, `test_pulldown_metrics.py`, `test_press_metrics.py` | The measurements. |
| `test_rule_engine.py` | Camera-view checks, persistence and feedback. |
| `test_phases.py`, `test_common_phases.py` | Rep counting. |
| `test_geometry.py`, `test_geometry_upper_body.py` | Angles and inclinations. |
| `test_normalisation.py`, `test_smoothing.py`, `test_filters.py` | Scaling, smoothing and the filters. |
| `test_validation.py`, `test_ui_mapping.py` | Recording checks and how they reach the UI. |
| `test_confidence.py`, `test_uncertainty.py`, `test_persistence.py` | Reliability, measurement error, persistence. |
| `test_landmarks.py` | The squat landmark list. |
| `test_person_selection.py` | Following the right person when others are in shot. |
| `test_plausibility.py` | The wrong-exercise check. |
| `test_occlusion.py` | Spotting a far-side limb that's behind the body in a side view. |
| `test_stabilise.py` | Outlier rejection: a tracking error goes, real movement stays. |
| `test_setup_phase.py` | Setup before the set doesn't reach the baseline or the rules. |
| `test_trimming.py` | Trimming the result video. |
| `test_annotation_output.py` | Writing the result video. |
| `test_exercise_registry.py` | The wiring between exercises and the UI. |
| `test_literature_preset.py` | The literature thresholds. |
| `test_evaluation_harness.py` | The evaluation script. |
| `conftest.py` | Lets pytest import the app from any folder. |

## `scripts/`

| File | What it is |
| --- | --- |
| `main.js` | Homepage behaviour: scroll effects, explorer, navigation. |
| `analyse.js` | `/analyse` behaviour: reference video popup, score count-up. |
| `download_pose_model.py` | Downloads the MediaPipe model. |
| `calibrate.py` | Prints every measurement next to its threshold, for any exercise. |
| `calibrate_squat.py` | Older squat-only version of `calibrate.py`. |
| `check_camera_view.py` | Shows which camera angle a clip is detected as. |
| `compare_filters.py` | Compares the smoothing filters on depth. |
| `evaluate_videos.py` | Runs FormFix over labelled videos and writes the results tables. |

## `docs/`

| File | What it is |
| --- | --- |
| `squat_analysis.md`, `pulldown_analysis.md`, `press_analysis.md` | How each analyser works. |
| `threshold_tuning.md` | Where each threshold came from and what changed. |
| `measurement_uncertainty.md` | MediaPipe's error and how it's used. |
| `filter_selection.md` | Which smoothing filter and why. |
| `literature_basis.md` | What I took from each paper. |
| `limitations.md` | What FormFix can't do. |
| `evaluation_results.md` | What the labelled videos showed, and what it found. |
| `FormFix_Overview.docx` | Short overview for non-technical readers. |
| `FormFix_Technical_Documentation.docx` | Technical documentation for the report. |
| `FormFix_Research_Provenance.docx` | What came from which paper, for the report. |

## `evaluation/`

| File | What it is |
| --- | --- |
| `README.md` | Which videos to record and how to run the evaluation. |
| `labels.example.csv` | Example list of labelled videos. |
| `labels.csv` | The videos I recorded and what each one is. |
| `results.csv`, `error_matrix.csv`, `summary.md` | Written by `scripts/evaluate_videos.py`. The write-up is `docs/evaluation_results.md`. |
| `reference_clip_runs.md` | Results of running FormFix on the real reference clips. |
| `figures/pulldown_annotated_frames.png` | Frames from the pulldown run. |

## `wireframes/`

| File | What it is |
| --- | --- |
| `generate_wireframes.py` | Makes all the SVG wireframes. |
| `00-screen-flow.svg` | How the screens connect. Start here. |
| `01`-`05` `...-desktop.svg` | Homepage sections on desktop. |
| `06`-`13` `...-desktop.svg` | Every state of the `/analyse` page on desktop. |
| `m01`-`m06` `...-mobile.svg` | The same on a phone. |

See [wireframes/README.md](wireframes/README.md) for each sheet.

## `styles/`, `static/` and generated folders

| Path | What it is |
| --- | --- |
| `styles/main.css` | The main stylesheet. |
| `styles/analyse.css` | Extra styles for `/analyse`. |
| `static/assets/images/` | The three exercise images. |
| `static/assets/logo/` | Logo files. |
| `static/assets/videos/` | Reference technique clips. |
| `static/assets/models/` | The MediaPipe model (downloaded, not in git). |
| `static/assets/docs/research-paper.pdf` | Linked from the homepage research section (currently a placeholder). |
| `analysis_results/` | Debug exports, only created with `FORMFIX_DEBUG=1` (not in git). |
| `questions/` | My own viva prep notes (not in git). |
