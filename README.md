# FormFix AI — Streamlit version

## Quick start

```bash
cd streamlit_app

python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

The app opens at <http://localhost:8501>.

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
├── requirements.txt
├── README.md
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
│   ├── problem.py          2  the problem
│   ├── how_it_works.py     3  how FormFix AI works
│   ├── exercise_section.py 4  exercise explorer (squat / press / pulldown)
│   ├── explainable.py      5  explainable AI + traceability pipeline
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
├── scripts/
│   ├── main.js             the small behaviour layer
│   └── analyse.js          /analyse only: reference lightbox, score count-up
│
└── static/
    └── assets/
        ├── images/         squat, shoulder press, lat pulldown
        ├── logo/           logo-mark.png, formfix-logo.png
        ├── videos/         optional reference clips (see below)
        └── docs/           research-paper.pdf
```

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

## Connecting the computer-vision backend

`utils/analysis.py` is the integration point. It is entirely placeholders — no
analysis is faked anywhere in the app.

```
Video Upload
      ↓
save_upload()             ← implemented (writes a real temp file)
      ↓
extract_landmarks()       ← OpenCV frame loop + MediaPipe Pose
      ↓
evaluate_rules()          ← exercise-specific biomechanical rules
      ↓
generate_feedback()       ← plain-English, traceable explanations
      ↓
annotate_video()          ← draw the FormFix overlay onto the video
      ↓
process_uploaded_video()  ← wire the above together, return AnalysisResult
```

To connect it:

1. Uncomment `opencv-python`, `mediapipe` and `numpy` in `requirements.txt` and
   install them.
2. Implement the stages above.
3. Return a populated `AnalysisResult` from `process_uploaded_video()`.

Nothing else changes. The UI only ever calls `process_uploaded_video()` (via
`analyse()`) and renders an `AnalysisResult` — through
`components/results_section.py` on the homepage and
`components/analysis_results.py` on `/analyse` — so real results appear in
exactly the layouts the site already shows. Populate `score`, `positives` and
`improvements` on the result and the analysis page fills itself in. Each
`RuleResult` carries a `rule_id`, which is what keeps the feedback traceable —
the explainability claim the site makes.

Per-exercise rule identifiers are already listed on each `Exercise` in
`utils/exercise_data.py`.

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

## Notes

* Requires Streamlit 1.36 or newer: `st.container(key=…)` scopes the widget
  styling and `st.navigation(..., position="hidden")` provides the two pages.
  `app.py` falls back gracefully if `position` is unsupported.
* The typefaces (Space Grotesk, Inter) load from Google Fonts, as in the
  original. Offline, the app falls back to the system sans-serif stack.
* This is an educational exercise-technique prototype, not a replacement for
  professional coaching or medical advice.
