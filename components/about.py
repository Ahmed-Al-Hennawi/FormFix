"""
Section 1b - what this is, and who it is for. Added after feedback that the
hero said what the system does but not what kind of thing it is.
"""

from __future__ import annotations

from utils.helpers import strip
from utils.styling import html

# <em> is the house emphasis style, same as the hero headline.
TITLE = "A <em>second pair of eyes</em> on the lifts you're learning."

LEDE = (
    "Upload a short video of your set. FormFix follows how you move, then tells "
    "you which part of your technique to work on - and why it matters."
)

CHIPS_LABEL = "The three movements"
EXERCISES: tuple[str, ...] = ("Barbell squat", "Shoulder press", "Lat pulldown")

# Both claims hold in the app as built: there is no account, and utils.analysis
# unlinks the upload when the run finishes. Check that before changing either.
ASIDE = "No account, no setup. Your video is deleted the moment the analysis finishes."

# (panel title, marker modifier, lines)
PANELS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Built for you if",
        "yes",
        (
            "You are learning these three lifts",
            "You train without a coach watching",
            "You would rather know what to fix than get a score",
        ),
    ),
    (
        "What it does not do",
        "no",
        (
            "Write your training programme",
            "Tell you how much to lift",
            "Make any claim about injury or health",
        ),
    ),
)


def _chips() -> str:
    return "".join(f"<li>{name}</li>" for name in EXERCISES)


def _panels() -> str:
    parts = []
    for title, marker, lines in PANELS:
        items = "".join(f"<li>{line}</li>" for line in lines)
        parts.append(
            f'<div class="about__panel about__panel--{marker}">'
            f'<p class="about__panel-title">{title}</p>'
            f'<ul class="about__list about__list--{marker}">{items}</ul>'
            "</div>"
        )
    return "".join(parts)


def render() -> None:
    html(strip(f"""
            <div class="ff-page">
            <section class="about section section--tight" id="ff-about">
              <div class="container">
                <div class="about__grid">
                  <div class="about__intro">
                    <p class="eyebrow" data-animate="fade-up">What this is</p>
                    <h2 class="about__title" data-animate="fade-up">{TITLE}</h2>
                    <span class="about__rule" data-animate="fade-up" aria-hidden="true"></span>
                    <p class="about__text" data-animate="fade-up">{LEDE}</p>
                    <p class="about__chips-label" data-animate="fade-up">{CHIPS_LABEL}</p>
                    <ul class="about__chips" data-animate="fade-up">{_chips()}</ul>
                    <p class="about__aside" data-animate="fade-up">{ASIDE}</p>
                  </div>
                  <div class="about__panels" data-animate="fade-up">{_panels()}</div>
                </div>
              </div>
            </section>
            </div>
            """))
