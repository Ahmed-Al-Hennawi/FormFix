"""Section 7 - Technology / under the hood."""

from __future__ import annotations

from utils.helpers import strip
from utils.styling import html

ITEMS: tuple[tuple[str, str, str], ...] = (
    (
        "01",
        "Computer Vision",
        "Your uploaded video is processed frame by frame, then checked for whether it can "
        "actually be measured - coverage, framing and camera angle - before any judgement.",
    ),
    (
        "02",
        "Pose Estimation",
        "A 33-landmark human pose model maps your joints through the entire movement. "
        "Short tracking gaps are filled and the coordinates smoothed; confidence is left alone.",
    ),
    (
        "03",
        "Movement Rules",
        "Each lift - squat, shoulder press, lat pulldown - has its own analyser: transparent, "
        "testable rules over angles, alignment and symmetry. No black box.",
    ),
    (
        "04",
        "Explainable Feedback",
        "Every message traces back to the exact rule - and the exact moment - that triggered it.",
    ),
)


def render() -> None:
    items = "".join(f"""
        <div class="tech__item" data-animate="fade-up">
          <span class="tech__num">{number}</span>
          <h3>{title}</h3>
          <p>{body}</p>
        </div>
        """ for number, title, body in ITEMS)

    html(strip(f"""
            <div class="ff-page">
            <section class="tech section" id="ff-technology">
              <div class="container">
                <p class="eyebrow" data-animate="fade-up">Under the hood</p>
                <h2 class="display display--md" data-animate="fade-up">Transparent by design.</h2>
                <div class="tech__grid">{items}</div>
              </div>
            </section>
            </div>
            """))
