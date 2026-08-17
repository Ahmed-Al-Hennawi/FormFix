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

Run it from any directory you like — every path in the application is derived
from `utils/paths.py`, so `streamlit run streamlit_app/app.py` works just as
well.

---

## Project structure

```
streamlit_app/
├── app.py                  page config, then one call per section
├── requirements.txt
├── README.md
│
├── .streamlit/
│   └── config.toml         dark theme, wide layout, static file serving
│
├── components/             one module per section of the page
│   ├── background.py       fixed scroll-progress line + atmosphere layers
│   ├── navbar.py           floating pill navigation
│   ├── hero.py             1  hero
│   ├── problem.py          2  the problem
│   ├── how_it_works.py     3  how FormFix AI works
│   ├── exercise_section.py 4  exercise explorer (squat / press / pulldown)
│   ├── explainable.py      5  explainable AI + traceability pipeline
│   ├── results_section.py  6  example analysis report (and real results)
│   ├── upload_section.py   -  video upload / analysis  ← the new section
│   ├── technology.py       7  under the hood
│   ├── research.py         7b academic & technical foundation
│   ├── outro.py            8  final call to action
│   └── footer.py           footer
│
├── utils/
│   ├── paths.py            every filesystem path, via pathlib
│   ├── assets.py           asset → browser URL (static serving or data URI)
│   ├── styling.py          CSS + JavaScript injection
│   ├── helpers.py          small markup helpers
│   ├── pose.py             the FormFix landmark renderer
│   ├── exercise_data.py    the three exercises + their landmark geometry
│   └── analysis.py         ← the computer-vision integration point
│
├── styles/
│   └── main.css            the full stylesheet
│
├── scripts/
│   └── main.js             the small behaviour layer
│
└── static/
    └── assets/
        ├── images/         squat, shoulder press, lat pulldown
        ├── logo/           logo-mark.png, formfix-logo.png
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

To swap an image, drop the replacement into `static/assets/images/` under the
same filename — no code changes needed.

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
document, next to the real page content.

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

Nothing else changes. The UI only ever calls `process_uploaded_video()` and
renders an `AnalysisResult` through `components/results_section.py`, so real
results appear in exactly the report card the marketing page promises. Each
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

To add a fourth exercise, add an `Exercise` (with its `Pose`) to
`utils/exercise_data.py` and drop its photograph into `static/assets/images/` —
the explorer tabs, counter, HUD read-out and preview card all build themselves
from that tuple.

---

## Notes

* Requires Streamlit 1.32 or newer (`st.container(key=…)` is used to scope
  widget styling).
* The typefaces (Space Grotesk, Inter) load from Google Fonts, as in the
  original. Offline, the app falls back to the system sans-serif stack.
* This is an educational exercise-technique prototype, not a replacement for
  professional coaching or medical advice.
