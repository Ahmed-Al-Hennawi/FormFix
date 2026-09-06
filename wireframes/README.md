# FormFix — Wireframes

Mid-fidelity wireframes covering the key screens and interaction states, created
as SVG diagrams using representative interface content. They document the
interface structure and user flow of FormFix.

## How these were produced

The interface was developed iteratively rather than specified up front. Working
in short cycles, I designed a screen, built it, used it against real recordings,
and changed it — the `/analyse` state machine, the amount of detail shown after
an analysis, and the recording guidance shown before upload were all revised
several times in response to what the running application actually made clear or
confusing. Deciding by looking at a working screen, rather than by drawing one
first, is what kept the interface aligned with what the analyser could honestly
report.

These wireframes were then produced to document the interface that process
arrived at: one sheet per screen and per state, including the states that are
easy to leave undocumented — a rejected recording, an analysed run where some
checks could not be assessed, and the mobile layouts. Writing them as a
generator rather than drawing them was a deliberate choice: the copy is taken
from the application's own modules, so a wording change is a code change and the
diagrams cannot quietly drift away from the built interface, which is exactly
what happens to wireframes drawn once and left behind.

No design tool is needed to open, edit or embed them: any browser, Word, LaTeX
(`\includegraphics` with `svg`, or export to PDF), or Preview will render them,
and they stay sharp at any size.

`generate_wireframes.py` builds the whole set from one small layout toolkit,
using the real interface copy taken from the application modules. Change a label
in the script and re-run it, and every sheet that shows that label updates:

```bash
python3 wireframes/generate_wireframes.py
```

Only the Python standard library is used.

## The sheets

### Overview

| File | What it shows |
| --- | --- |
| `00-screen-flow.svg` | How the two pages connect, the nine homepage sections, the `/analyse` state machine (idle → ready → analysing → complete), the three things the complete state can show, and the four levels of progressive disclosure. Start here. |

### Homepage — desktop (1440px)

| File | What it shows |
| --- | --- |
| `01-home-full-desktop.svg` | The whole page, all nine sections in order, as one tall sheet. |
| `02-home-hero-desktop.svg` | Fixed pill navigation, hero headline, CTAs, athlete figure with landmark overlay and HUD chips. |
| `02b-home-about-desktop.svg` | Section 1b, "What this is": the orientation block between the hero and the problem — what FormFix is, the three movements, who it is for and what it deliberately does not do. |
| `03-home-exercise-explorer-desktop.svg` | The three-exercise explorer: tabs, copy panel, figure, arrows and the NEXT preview card. |
| `04-home-explainable-and-example-desktop.svg` | Black-box verdict vs the four-part FormFix correction, the three-step trace behind it, and the worked example analysis card. |
| `05-home-technology-research-outro-desktop.svg` | Technology grid, the academic foundation card, closing CTA and footer. |

### Analyse page — desktop (1440px)

Each sheet is one state of `st.session_state["ff_ax_stage"]`.

| File | State |
| --- | --- |
| `06-analyse-idle-desktop.svg` | `idle` — no video selected; the right column previews what the result will contain. |
| `07-analyse-ready-desktop.svg` | `ready` — clip confirmed, preview shown, "Analyse Form" armed. |
| `08-analyse-analysing-desktop.svg` | `analysing` — scan visual, the eight pipeline stages reporting live, controls greyed out. |
| `09-analyse-results-desktop.svg` | `complete` — levels 1 and 2: annotated video, verdict and score ring, what to fix, what you did well, measured checks, reference card. |
| `10-analyse-results-expanded-desktop.svg` | Level 3 — the "See the full detail" panel: every finding, rep by rep, and what was not assessed. |
| `11-analyse-failure-desktop.svg` | Validation failure — what could not be measured, why, and what to change. Nothing is scored. |
| `12-analyse-limited-quality-desktop.svg` | Analysis ran, but the camera angle limited it — including the "— not scored" verdict. |
| `13-reference-modal-desktop.svg` | The reference-technique lightbox over the results. |

### Mobile (390px)

| File | What it shows |
| --- | --- |
| `m01-home-hero-mobile.svg` | Hero in a single column, navigation collapsed to a menu button, followed by the orientation block. |
| `m02-home-menu-mobile.svg` | The overlay menu (below 900px). |
| `m03-home-sections-mobile.svg` | How it works and the exercise explorer, stacked. |
| `m04-analyse-idle-mobile.svg` | Analyse page, idle — the columns stack, so upload comes first. |
| `m05-analyse-analysing-mobile.svg` | Analyse page, pipeline running. |
| `m06-analyse-results-mobile.svg` | Analyse page, results — the same progressive disclosure in one column. |

## Reading the wireframes

| Convention | Meaning |
| --- | --- |
| Grey filled block | A container, card or panel |
| Stacked grey bars | Body copy (real sentences are used wherever the wording matters) |
| Box with a diagonal cross | Image, video frame or media placeholder |
| Dark filled button | Primary action |
| Outlined button | Secondary action |
| Greyed button | Disabled control |
| Dashed outline | An element conditional on interface state |
| Italic grey text with a dashed rule | An annotation — never part of the interface |

Every sheet is titled, labelled with its viewport, and names the module it
documents, so a sheet lifted into a report still explains itself.
