"""
Section 3 - How it works. Five steps on a vertical line that fills as you
scroll (main.js), lighting up the step in view.
"""

from __future__ import annotations

from utils.helpers import strip
from utils.styling import html

# (number, title, text, tag, tag modifier)
STEPS: tuple[tuple[str, str, str, str, str], ...] = (
    ("01", "Record", "Film a short clip of your set. Any phone camera works.", "VIDEO IN", ""),
    ("02", "Upload", "Drop the video into FormFix. No account, no setup.", "UPLOAD.MP4", ""),
    (
        "03",
        "Analyse",
        "Computer vision tracks 33 body landmarks through every frame of the movement.",
        "POSE ESTIMATION",
        "step__tag--cyan",
    ),
    (
        "04",
        "Understand",
        "Transparent movement rules check joint angles, alignment and symmetry - "
        "and identify what went wrong.",
        "RULES ENGINE",
        "step__tag--cyan",
    ),
    (
        "05",
        "Improve",
        "You get clear, human-readable feedback you can apply on your very next set.",
        "FEEDBACK OUT",
        "step__tag--lime",
    ),
)


def render() -> None:
    steps = "".join(f"""
        <article class="step" data-animate="step">
          <span class="step__num" aria-hidden="true">{number}</span>
          <div class="step__body">
            <h3 class="step__title">{title}</h3>
            <p class="step__text">{text}</p>
          </div>
          <span class="step__tag {modifier}">{tag}</span>
        </article>
        """ for number, title, text, tag, modifier in STEPS)

    html(strip(f"""
            <div class="ff-page">
            <section class="how section" id="ff-how">
              <div class="container">
                <p class="eyebrow" data-animate="fade-up">How FormFix works</p>
                <h2 class="display display--md" data-animate="fade-up">
                  From your camera roll<br />to a clear correction.
                </h2>
                <div class="how__flow">
                  <div class="how__line" aria-hidden="true"><span class="how__line-progress"></span></div>
                  {steps}
                </div>
              </div>
            </section>
            </div>
            """))
