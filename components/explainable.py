"""
Section 5 - explainable feedback: the explanation card, then the trace behind
it. The worked example is quoted from the squat's real depth rule rather than
invented - the three card lines are its own wording, and 115 degrees is the
actual DEPTH_KNEE_ANGLE_WARN.
"""

from __future__ import annotations

from utils.helpers import strip
from utils.styling import html

# (step label, body, modifier class)
TRACE: tuple[tuple[str, str, str], ...] = (
    (
        "WHAT IT SAW",
        "On 2 of your 6 reps, your knees stopped bending at 118&deg;.",
        "",
    ),
    (
        "WHAT IT CHECKED",
        "The <code>depth</code> check looks for 115&deg; or lower. 118&deg; is short of it.",
        "",
    ),
    (
        "WHAT IT SAID",
        '"You stop a little high on 2 of your 6 reps."',
        "out",
    ),
)


def _trace() -> str:
    parts = []
    for index, (step, body, modifier) in enumerate(TRACE):
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
    html(strip(f"""
            <div class="ff-page">
            <section class="explain section" id="ff-explainable">
              <div class="container container--narrow">
                <p class="eyebrow" data-animate="fade-up">Explainable Feedback</p>
                <h2 class="display display--md" data-animate="fade-up">
                  More than a score.<br />An explanation.
                </h2>
                <p class="explain__lede" data-animate="fade-up">
                  FormFix doesn't just hand you a grade. Every correction says what happened,
                  why it matters and what to try instead. Here is one from the session reported
                  further down this page.
                </p>
                <div class="explain__contrast">
                  <div class="verdict verdict--formfix" data-animate="fade-up">
                    <p class="verdict__caption">What FormFix tells you</p>
                    <p class="verdict__title">Squat depth</p>
                    <p class="verdict__formfix-msg">You stop a little high on 2 of your 6 reps.</p>
                    <p class="verdict__why">Squatting lower works your legs through their full range.</p>
                    <p class="verdict__fix"><span class="verdict__fix-tag">Try this</span>Lower until your
                      hips reach about knee height, keeping your whole foot planted.</p>
                  </div>
                </div>
                <div class="explain__trace-head" data-animate="fade-up">
                  <h3 class="explain__trace-title">The rule behind that message</h3>
                  <p class="explain__trace-lede">
                    None of it is guesswork. You can follow any correction back through the three
                    steps that produced it.
                  </p>
                </div>
                <div class="pipeline" data-animate="pipeline">{_trace()}</div>
                <p class="explain__footnote" data-animate="fade-up">
                  That 115&deg; is a starting value, not a fixed truth. FormFix prints the number
                  behind every correction, so you can always see what it was measured against.
                </p>
              </div>
            </section>
            </div>
            """))
