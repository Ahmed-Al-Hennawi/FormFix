"""
Section 1 - Hero.

A masked three-line headline, the floating athlete with the FormFix landmark
overlay, three HUD chips at different parallax depths, and the scroll hint.
"""

from __future__ import annotations

from utils.assets import asset_url
from utils.exercise_data import SQUAT
from utils.helpers import masked_lines, strip
from utils.pose import render_pose
from utils.styling import html

HEADLINE = (
    "See your form.",
    "Understand <em>the mistake.</em>",
    "Fix the movement.",
)

SUBTITLE = (
    "FormFix AI analyses a short video of your exercise, tracks your body's "
    "movement, and explains - in plain language - where your technique can improve."
)


def render() -> None:
    pose = render_pose(SQUAT.pose, key="hero", extra_class="pose--hero")

    html(
        strip(
            f"""
            <div class="ff-page">
            <section class="hero" id="ff-top">
              <div class="hero__inner">
                <div class="hero__copy">
                  <p class="eyebrow reveal">Explainable computer vision &middot; Exercise technique</p>
                  <h1 class="hero__title">{masked_lines(HEADLINE, animate="")}</h1>
                  <p class="hero__sub reveal">{SUBTITLE}</p>
                  <div class="hero__cta reveal">
                    <a href="#ff-analyse" class="btn btn--primary" data-scroll-to="ff-analyse">
                      Analyse Your Form <span class="btn__arrow">&rarr;</span>
                    </a>
                    <a href="#ff-how" class="btn btn--ghost" data-scroll-to="ff-how">See How It Works</a>
                  </div>
                </div>
                <div class="hero__stage">
                  <div class="hero__glow"></div>
                  <figure class="hero__figure" data-depth="1">
                    <img src="{asset_url(SQUAT.image)}"
                         alt="Athlete performing a barbell squat, analysed by FormFix AI" />
                    {pose}
                    <div class="scanline"></div>
                  </figure>
                  <div class="hero__shadow"></div>
                  <div class="hud hud--tl" data-depth="2"><span class="hud__dot"></span>POSE DETECTED</div>
                  <div class="hud hud--br" data-depth="3">TRACKING 33 LANDMARKS</div>
                  <div class="hud hud--mr" data-depth="2">KNEE ANGLE <strong>92&deg;</strong></div>
                </div>
              </div>
              <a class="hero__scroll-hint" href="#ff-problem" data-scroll-to="ff-problem"
                 aria-label="Scroll to next section">
                <span class="hero__scroll-line"></span>
                <span class="hero__scroll-label">SCROLL</span>
              </a>
            </section>
            </div>
            """
        )
    )
