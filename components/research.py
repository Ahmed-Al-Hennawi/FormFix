"""Section 7b - Academic & technical foundation."""

from __future__ import annotations

from utils.assets import RESEARCH_PAPER, asset_url
from utils.helpers import strip
from utils.styling import html

ITEMS: tuple[tuple[str, str], ...] = (
    (
        "1. Research Question",
        "How can explainable computer vision support beginner gym users in "
        "identifying and correcting common exercise technique mistakes from "
        "uploaded workout videos?",
    ),
    (
        "2. The Problem",
        "Beginners often learn exercises from online videos but struggle to "
        "self-evaluate subtle technical errors. Personal training is expensive, "
        "and asking for help in the gym can cause anxiety.",
    ),
    (
        "3. The Solution &amp; Method",
        "<strong>Vision Pipeline:</strong> Extracts 2D joint coordinates from "
        "standard smartphone video using MediaPipe Pose.<br />"
        "<strong>Kinematic Evaluation:</strong> Measures joint angles against "
        "biomechanical rules.<br />"
        "<strong>Explainable AI (XAI):</strong> Converts raw kinematic data into "
        "plain-English feedback.",
    ),
    (
        "4. Evaluation &amp; Results",
        "Evaluated across three core dimensions: Technical Accuracy (mistake "
        "detection reliability), User Understanding (clarity of generated "
        "feedback), and System Usability (System Usability Scale score).",
    ),
)


def render() -> None:
    items = "".join(
        f"""
        <div class="research__item" data-animate="fade-up">
          <h3 class="research__item-title">{title}</h3>
          <p>{body}</p>
        </div>
        """
        for title, body in ITEMS
    )

    html(
        strip(
            f"""
            <div class="ff-page">
            <section class="research section" id="ff-research">
              <div class="container">
                <div class="research__card" data-animate="fade-up">
                  <h2 class="research__title">
                    Academic &amp; Technical <span class="research__title-accent">Foundation</span>
                  </h2>
                  <p class="research__subtitle">
                    Grounded in Explainable Computer Vision and Human Pose Estimation Research
                  </p>
                  <p class="research__lede">
                    FormFix AI was developed as a university research project investigating how
                    computer vision can provide accessible, explainable movement feedback for
                    beginner gym users. Read the full paper to explore the underlying algorithms,
                    rule-based biomechanical checks, and user evaluation results.
                  </p>
                  <a class="btn btn--paper" href="{asset_url(RESEARCH_PAPER)}"
                     target="_blank" rel="noopener">
                    Read Full Research Paper (PDF)
                  </a>
                  <div class="research__grid">{items}</div>
                  <p class="research__citation" data-animate="fade-up">
                    FormFix AI (2026). An Explainable Computer Vision System for Exercise Form
                    Assessment and Corrective Feedback. Computer Science Undergraduate Thesis /
                    Final Year Project.
                  </p>
                </div>
              </div>
            </section>
            </div>
            """
        )
    )
