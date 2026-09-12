# FormFix

**An explainable computer vision system for exercise form assessment and corrective feedback.**

FormFix is my MSc Computer Science thesis project at UAL. You upload a short
video of yourself doing an exercise, and it tells you in plain English what you
did well and what to fix - and *why*, so every piece of feedback can be traced
back to a measured angle and a written rule.

It's aimed at **beginners** who train on their own and don't have a coach to
check their form.

**Live app:** <https://formfix.streamlit.app>

> FormFix gives technique guidance from visible movement. It is not a
> replacement for a coach, and it doesn't give medical advice.

---

## What it analyses

| Exercise | Film from | What it checks |
| --- | --- | --- |
| **Squat** | the side | depth · torso lean · heel lift · return to standing · descent speed |
| **Lat pulldown** | the side (or turned slightly towards your back) | torso movement · range of motion |
| **Dumbbell shoulder press** | the front | arm symmetry · wrist over elbow · range of motion |

Each exercise needs its own camera angle, because one camera can only measure
what it can see. If a check isn't possible from your angle, FormFix shows it as
**Not assessed** with the reason, instead of guessing.

It isn't a trained AI model. It uses **MediaPipe** to find the body landmarks,
and everything after that is my own rule-based Python code, so every result is
explainable.

---

## Run it locally

You need **Python 3.11**.

```bash
cd streamlit_app
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_pose_model.py   # one-time, ~9 MB (the app also tries this on start-up)
streamlit run app.py
```

Then open <http://localhost:8501>. There are two pages:

- `/` - the homepage (what FormFix is and how it works)
- `/analyse` - upload a video and get your feedback

FFmpeg comes with the `imageio-ffmpeg` package, so the result video plays in
the browser without installing anything else.

---

## How it works

```
upload video
  -> check the file and the recording (is a person visible? is enough in frame?)
  -> MediaPipe finds 33 body landmarks on every frame
  -> fill short gaps and smooth the landmarks
  -> measure angles and distances for this exercise
  -> count the reps (a state machine)
  -> check it's actually the exercise you picked
  -> run the technique rules on each rep, in the right phase
  -> work out how reliable each result is
  -> write the feedback + render the annotated video
```

A few things I designed on purpose:

- **"Not assessed" instead of guessing.** A bad camera angle only switches off
  the checks it affects. It never counts as a failure.
- **One result per rep.** You can see which rep went wrong, not just one grade.
- **A simple, visible score:**
  `100 x (passes + 0.5 x warnings) / checks that could be assessed`.
- **Measurement error is included.** MediaPipe's knee angle is only accurate
  to about ±10.7° (Dill et al., 2024), so a finding smaller than that is marked
  as *indicative*, not certain.
- **Reliability per measurement** (High / Medium / Low / Cannot assess), based
  on how visible the landmarks were, the camera angle and how many reps there were.
- **The right person is analysed.** Other people in the gym are fine - FormFix
  follows the person nearest the camera from frame to frame.
- **Wrong exercise check.** If you upload a bench press as a "shoulder press",
  it says so instead of giving you press feedback.
- **The result video is trimmed** to just your set, with a smoothed skeleton
  over your whole body and the side away from the camera drawn fainter.

---

## Project structure

```
streamlit_app/
├── app.py              starts the app and sets up the two pages
├── views/              the two pages (home, analyse)
├── components/         one file per page section (hero, explorer, results, ...)
├── utils/              paths, styling, assets, and the bridge to the analysis
├── styles/             CSS (main.css + analyse.css)
├── scripts/            JS for the site + helper scripts (model download,
│                       calibration, filter comparison, evaluation)
├── analysis/           shared computer vision: video, pose detection,
│                       validation, outlier rejection, smoothing, filters,
│                       geometry, overlay video
├── exercises/
│   ├── common/         shared logic: rule spec, rep counting, reliability,
│   │                   uncertainty, feedback, exercise check
│   ├── squat/          5 rules
│   ├── pulldown/       2 rules
│   └── press/          3 rules
├── tests/              700+ tests, with fake (synthetic) videos for each exercise
├── docs/               technical write-ups for the thesis
├── evaluation/         the labelled-video evaluation harness and its results
├── wireframes/         every screen as an SVG, made by one script
└── static/assets/      images, logo, reference clips, pose model
```

Each exercise folder has the same files: `config.py` (every threshold),
`landmarks.py`, `metrics.py` (measuring only), `phases.py`, `rules.py`,
`feedback.py` (the wording) and `analyser.py` (runs the steps in order).
Adding a fourth exercise means a new folder plus one entry in
`exercises/__init__.py`.

[README_FILES.md](README_FILES.md) has one line on every file.

---

## Your video and privacy

- No accounts and no database.
- Your upload is deleted as soon as the analysis finishes.
- The annotated result video is kept temporarily so you can replay it, then
  cleaned up after a few hours.

---

## Testing and tools

```bash
pytest                  # all tests
ruff check .            # lint
black --check .         # formatting (line length 104)
```

The tests run the whole pipeline on synthetic pose data with known faults.
Every exercise also has **false-positive tests** (noise, a dropped landmark, a
value just under a threshold) that must *not* produce a finding. To also test
real MediaPipe detection:

```bash
FORMFIX_TEST_VIDEO=path/to/side-view-squat.mp4 pytest
```

Helper scripts (not used by the app itself):

| Script | What it does |
| --- | --- |
| `scripts/download_pose_model.py` | downloads the MediaPipe model |
| `scripts/calibrate.py video.mp4 pulldown --frames` | prints every measurement next to its threshold, and saves key frames |
| `scripts/calibrate_squat.py` | older squat-only version of the above |
| `scripts/check_camera_view.py` | shows which camera angle a clip will be detected as |
| `scripts/compare_filters.py` | compares the smoothing filters on depth |
| `scripts/evaluate_videos.py` | runs FormFix over labelled videos (see [evaluation](evaluation/README.md)) |

**Debug mode:** `FORMFIX_DEBUG=1 streamlit run app.py` shows the full technical
details on the results page and saves a JSON/CSV export of each run to
`analysis_results/`.

**Settings worth knowing** (in each exercise's `config.py`):

- `ANGLE_FILTER` - smoothing filter: `ema` (default), `butterworth`, `savgol`
  or `moving_average`
- `UNCERTAINTY_STRICT` - if `True`, findings inside the measurement error are
  downgraded one step
- `exercises/squat/literature_config.py` - a second set of squat thresholds
  taken from published papers, so both can be compared

---

## Deployment

The app is deployed on **Streamlit Community Cloud** from the `main` branch,
with `app.py` as the entry point and Python 3.11. A few things I learned the
hard way:

- `packages.txt` lists `libgl1` and `libglib2.0-0t64`, which OpenCV needs.
  Use the `t64` name, otherwise apt fails.
- **Don't put comments in `packages.txt`** - the build reads every line as a
  package name and breaks.
- OpenCV has to be `opencv-contrib-python` (MediaPipe needs it), not the
  headless version.
- Streamlit is capped below 1.63 because the JavaScript is injected with
  `st.components.v1.html`, which is deprecated.
- The pose model isn't in the repo; the app downloads it on first start-up.

---

## Limitations

- One 2D camera: no real 3D angles, no forces, and nothing along the camera's
  line of sight.
- MediaPipe has measurement error (about ±10.7° on a knee angle), so small
  differences can't be resolved.
- The thresholds are my own prototype values. They were tested on synthetic
  data, **not** validated on labelled real videos, so there is no accuracy
  figure yet.
- No user study has been done, so I don't claim it improves anyone's technique.
- It doesn't detect injury risk, back rounding or knee valgus - one camera
  can't support those.
- One set of thresholds for everyone; bodies and flexibility differ.

More detail in [docs/limitations.md](docs/limitations.md).

---

## Documentation

| Document | What's in it |
| --- | --- |
| [Squat](docs/squat_analysis.md) · [Lat pulldown](docs/pulldown_analysis.md) · [Shoulder press](docs/press_analysis.md) | how each analyser works |
| [Thresholds](docs/threshold_tuning.md) | where every threshold came from and why it changed |
| [Measurement uncertainty](docs/measurement_uncertainty.md) | MediaPipe's error and how it's used |
| [Filter selection](docs/filter_selection.md) | which smoothing filter and why |
| [Literature basis](docs/literature_basis.md) | what I took from each paper |
| [Limitations](docs/limitations.md) | what FormFix can't do |
| [Evaluation results](docs/evaluation_results.md) | what the labelled videos showed |
| [Evaluation](evaluation/README.md) · [Real-video runs](evaluation/reference_clip_runs.md) | how to evaluate it, and the runs on real footage |
| [Wireframes](wireframes/README.md) | every screen and state |

The Word documents in `docs/` (overview, technical documentation, research
provenance) are longer versions written for the report.
