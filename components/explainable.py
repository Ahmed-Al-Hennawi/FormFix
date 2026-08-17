"""
Section 5 - Explainable AI.

The black-box verdict set against the FormFix explanation, followed by the
traceability pipeline: what the system saw, the rule it triggered, the feedback
you read.
"""

from __future__ import annotations

from utils.helpers import strip
from utils.styling import html

#: (step label, body, modifier class)
PIPELINE: tuple[tuple[str, str, str], ...] = (
    (
        "DETECTED MOVEMENT",
        "Knee x-position crosses inside the toe line during frames 41-63.",
        "",
    ),
    (
        "RULE TRIGGERED",
        "<code>knee_tracking</code> - medial knee drift exceeded threshold.",
        "",
    ),
    (
        "FEEDBACK",
        '"Keep your knees tracking over your toes as you descend."',
        "out",
    ),
)


def _pipeline() -> str:
    parts = []
    for index, (step, body, modifier) in enumerate(PIPELINE):
        if index:
            parts.append('<div class="pipeline__connector" aria-hidden="true"></div>')
        node_class = "pipeline__node pipeline__node--out" if modifier == "out" else "pipeline__node"
        step_class = "pipeline__step pipeline__step--lime" if modifier == "out" else "pipeline__step"
        parts.append(
            f'<div class="{node_class}">'
            f'<span class="{step_class}">{step}</span>'
            f"<p>{body}</p>"
            "</div>"
        )
    return "".join(parts)


def render() -> None:
    html(
        strip(
            f"""
            <div class="ff-page">
            <section class="explain section" id="ff-explainable">
              <div class="container">
                <p class="eyebrow" data-animate="fade-up">Explainable AI</p>
                <h2 class="display display--md" data-animate="fade-up">
                  Not a score.<br />An explanation.
                </h2>
                <p class="explain__lede" data-animate="fade-up">
                  FormFix AI doesn't simply grade your exercise. Every result can be traced from
                  what the system saw, to the rule it triggered, to the feedback you read.
                </p>
                <div class="explain__contrast">
                  <div class="verdict verdict--blackbox" data-animate="fade-up">
                    <p class="verdict__caption">What a black box tells you</p>
                    <p class="verdict__blackbox-msg"><span aria-hidden="true">&#10005;</span> Incorrect exercise.</p>
                  </div>
                  <div class="verdict verdict--formfix" data-animate="fade-up">
                    <p class="verdict__caption">What FormFix AI tells you</p>
                    <p class="verdict__formfix-msg">"Your knees moved inward during the descent.
                      Try keeping them aligned with your toes."</p>
                  </div>
                </div>
                <div class="pipeline" data-animate="pipeline">{_pipeline()}</div>
              </div>
            </section>
            </div>
            """
        )
    )
