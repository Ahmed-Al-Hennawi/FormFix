# FormFix - Wireframes

Mid-fidelity wireframes of every screen and state, as SVG files.

## How I made them

I built the interface iteratively: design a screen, build it, try it on real
videos, then change it. The `/analyse` flow, how much detail to show after an
analysis, and the recording guidance all changed several times this way.

The wireframes document where that process ended up, including the states
that are easy to forget - a rejected video, a run where some checks couldn't
be assessed, and the mobile layouts. Instead of drawing them in a design tool,
I wrote a script that generates them with the real interface text, so they're
easy to update:

```bash
python3 wireframes/generate_wireframes.py
```

It only uses the Python standard library. The SVGs open in any browser, in
Word, or as images in a report, and stay sharp at any size.

> The wireframes were generated on 6 September 2026. After that, the "how to
> record" panel on `/analyse` was changed to three short tips plus a camera
> diagram, and the squat no longer has a knee-evenness check, so those two
> details differ from the live app.

## The sheets

### Overview

| File | What it shows |
| --- | --- |
| `00-screen-flow.svg` | How the pages connect, the homepage sections, and the `/analyse` states (idle → ready → analysing → complete). Start here. |

### Homepage - desktop (1440px)

| File | What it shows |
| --- | --- |
| `01-home-full-desktop.svg` | The whole homepage in one tall sheet. |
| `02-home-hero-desktop.svg` | Navigation, headline, buttons and the pose figure. |
| `02b-home-about-desktop.svg` | "What this is": what FormFix is, who it's for, what it doesn't do. |
| `03-home-exercise-explorer-desktop.svg` | The three-exercise explorer. |
| `04-home-explainable-and-example-desktop.svg` | Explainable feedback and the example report. |
| `05-home-technology-research-outro-desktop.svg` | Technology, research, final call to action and footer. |

### Analyse page - desktop (1440px)

| File | State |
| --- | --- |
| `06-analyse-idle-desktop.svg` | No video yet. |
| `07-analyse-ready-desktop.svg` | Video chosen, ready to analyse. |
| `08-analyse-analysing-desktop.svg` | Analysis running, stages updating live. |
| `09-analyse-results-desktop.svg` | Results: video, verdict, what to fix, what went well, checks. |
| `10-analyse-results-expanded-desktop.svg` | "See the full detail" opened. |
| `11-analyse-failure-desktop.svg` | Video rejected, with the reason and what to change. |
| `12-analyse-limited-quality-desktop.svg` | Analysed, but the camera angle limited some checks. |
| `13-reference-modal-desktop.svg` | The reference technique video popup. |

### Mobile (390px)

| File | What it shows |
| --- | --- |
| `m01-home-hero-mobile.svg` | Hero in one column. |
| `m02-home-menu-mobile.svg` | The mobile menu. |
| `m03-home-sections-mobile.svg` | How it works and the explorer, stacked. |
| `m04-analyse-idle-mobile.svg` | Upload screen. |
| `m05-analyse-analysing-mobile.svg` | Analysis running. |
| `m06-analyse-results-mobile.svg` | Results. |

## How to read them

| Symbol | Meaning |
| --- | --- |
| Grey block | A container or card |
| Grey bars | Body text (real text is used where the wording matters) |
| Box with a cross | Image or video |
| Dark button | Main action |
| Outlined button | Secondary action |
| Greyed button | Disabled |
| Dashed outline | Only shown in some states |
| Italic grey text | A note, not part of the interface |
