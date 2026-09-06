"""
Section 4 - the exercise explorer. All three exercises are rendered into the
DOM at once; main.js just changes which one is current and the CSS handles the
crossfade. It has to be CSS, not a Python callback, because a Streamlit
rerun resets the scroll position and throws you out of the section.
"""

from __future__ import annotations

from utils.assets import asset_url
from utils.exercise_data import EXERCISES
from utils.helpers import strip
from utils.pose import render_pose
from utils.styling import html


def _tabs() -> str:
    return "".join(
        '<button class="tab{active}" role="tab" aria-selected="{selected}" '
        'data-index="{index}" id="tab-{eid}" aria-controls="panel-{eid}">'
        '<span class="tab__label">{name}</span></button>'.format(
            active=" is-active" if index == 0 else "",
            selected="true" if index == 0 else "false",
            index=index,
            eid=exercise.id,
            name=exercise.name,
        )
        for index, exercise in enumerate(EXERCISES)
    )


def _panels() -> str:
    """Left-hand copy column: one panel per exercise, all but one hidden."""
    panels = []
    for index, exercise in enumerate(EXERCISES):
        areas = "".join(f"<li>{area}</li>" for area in exercise.areas)
        hidden = "" if index == 0 else " hidden"
        panels.append(
            f'<div class="explorer__panel" data-index="{index}"{hidden}>'
            f'<h3 class="explorer__name">{exercise.name}</h3>'
            f'<p class="explorer__desc">{exercise.description}</p>'
            f'<ul class="explorer__areas">{areas}</ul>'
            "</div>"
        )
    return "".join(panels)


def _figures() -> str:
    figures = []
    for index, exercise in enumerate(EXERCISES):
        pose = render_pose(exercise.pose, key=exercise.id)
        figures.append(
            f'<figure class="ex-figure" data-exercise="{exercise.id}" '
            f'data-index="{index}" id="panel-{exercise.id}" role="tabpanel" '
            f'aria-labelledby="tab-{exercise.id}" '
            f'data-atm-a="{exercise.atmosphere_a}" data-atm-b="{exercise.atmosphere_b}">'
            f'<img src="{asset_url(exercise.image)}" alt="{exercise.image_alt}" />'
            f"{pose}"
            '<div class="scanline"></div>'
            "</figure>"
        )
    return "".join(figures)


def _metrics() -> str:
    return "".join(
        '<span class="explorer__metric" data-index="{index}"{hidden}>{metric}</span>'.format(
            index=index,
            hidden="" if index == 0 else " hidden",
            metric=exercise.metric_html,
        )
        for index, exercise in enumerate(EXERCISES)
    )


def _previews() -> str:
    """One card per exercise, but the visible one is the *next* exercise."""
    cards = []
    for index, exercise in enumerate(EXERCISES):
        # We start on the squat, so index 1 is what's up next.
        hidden = "" if index == 1 % len(EXERCISES) else " hidden"
        cards.append(
            f'<span class="explorer__preview-item" data-index="{index}"{hidden}>'
            '<span class="explorer__preview-img">'
            f'<img src="{asset_url(exercise.image)}" alt="" /></span>'
            f'<span class="explorer__preview-name">{exercise.name}</span>'
            "</span>"
        )
    return "".join(cards)


def render() -> None:
    html(strip(f"""
            <div class="ff-page">
            <section class="explorer section" id="ff-exercises">
              <div class="explorer__head container">
                <p class="eyebrow" data-animate="fade-up">Exercise explorer</p>
                <h2 class="display display--md" data-animate="fade-up">
                  Three movements.<br />One analysis engine.
                </h2>
              </div>
              <div class="explorer__stage-wrap" tabindex="0">
                <div class="explorer__tabs" role="tablist" aria-label="Choose an exercise"
                     data-animate="fade-up">{_tabs()}</div>
                <div class="explorer__stage">
                  <div class="explorer__info" aria-live="polite" data-animate="rise">
                    <p class="explorer__count">
                      <span class="explorer__count-current">01</span> / 0{len(EXERCISES)}
                    </p>
                    {_panels()}
                  </div>
                  <div class="explorer__figures" data-animate="figures">
                    <div class="explorer__glow"></div>
                    {_figures()}
                    <div class="explorer__shadow"></div>
                    <div class="hud hud--stage"><span class="hud__dot"></span>{_metrics()}</div>
                  </div>
                  <div class="explorer__side" data-animate="rise">
                    <div class="explorer__arrows">
                      <button class="arrow" data-dir="-1" aria-label="Previous exercise">&larr;</button>
                      <button class="arrow" data-dir="1" aria-label="Next exercise">&rarr;</button>
                    </div>
                    <button class="explorer__preview" aria-label="Go to next exercise">
                      <span class="explorer__preview-label">NEXT</span>
                      {_previews()}
                    </button>
                  </div>
                </div>
              </div>
            </section>
            </div>
            """))
