"""Section 2 - The problem."""

from __future__ import annotations

from utils.helpers import masked_lines, strip
from utils.styling import html

HEADLINE = (
    "Watching the movement is easy.",
    "Understanding your own movement is harder.",
)


def render() -> None:
    html(strip(f"""
            <div class="ff-page">
            <section class="problem section" id="ff-problem">
              <div class="container container--narrow">
                <p class="eyebrow" data-animate="fade-up">The problem</p>
                <h2 class="display display--md">{masked_lines(HEADLINE, dim_from=1)}</h2>
                <div class="problem__grid">
                  <p class="problem__text" data-animate="fade-up">
                    Most beginners learn exercises from short videos and copy what they see.
                    Without feedback, small technique errors - a knee drifting inward, an elbow
                    slipping out of position - go unnoticed and quietly become habits.
                  </p>
                  <p class="problem__text" data-animate="fade-up">
                    A coach would catch them. Not everyone has one - and asking a stranger
                    mid-workout is harder than it sounds.
                    <span class="text-accent">FormFix bridges that gap:</span>
                    it watches your movement the way a coach would, and tells you what it sees.
                  </p>
                </div>
              </div>
            </section>
            </div>
            """))
