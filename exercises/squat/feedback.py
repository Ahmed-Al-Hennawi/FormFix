"""
The squat's wording. The aggregation is shared (exercises/common/feedback.py);
the only squat-specific parts are the two tables below - which wording belongs
to which rule, and which measured values are worth printing.
"""

from __future__ import annotations

from analysis.models import (
    NotAssessedItem,
    RepSummary,
    RuleResult,
    SessionSummary,
    SquatRep,
)
from exercises.common.feedback import (
    SCORE_FORMULA,
    FeedbackFinding,
    FeedbackTemplate,
    transparent_score,
)
from exercises.common.feedback import (
    build_findings as _build_findings,
)
from exercises.common.feedback import (
    build_not_assessed as _build_not_assessed,
)
from exercises.common.feedback import (
    build_overview as _build_overview,
)
from exercises.common.feedback import (
    build_positives as _build_positives,
)
from exercises.common.feedback import (
    build_rep_summaries as _build_rep_summaries,
)
from exercises.common.feedback import (
    build_summary as _build_summary,
)
from exercises.common.feedback import (
    template_for as _template_for,
)

__all__ = [
    "EVIDENCE_FORMATS",
    "FEEDBACK_TEMPLATES",
    "SCORE_FORMULA",
    "FeedbackFinding",
    "FeedbackTemplate",
    "build_findings",
    "build_not_assessed",
    "build_overview",
    "build_positives",
    "build_rep_summaries",
    "build_summary",
    "template_for",
    "transparent_score",
]

# The line that opens "What you did well". {n} / {s} are filled in.
POSITIVES_OPENING = "{n} full rep{s} completed"

# feedback_key (declared on each rule in config.py) -> wording. Each entry has
# the beginner's three lines plus the plain-language description shown in the
# "how did we detect this" panel.
# has the beginner's three lines plus the plain-language description of the
FEEDBACK_TEMPLATES: dict[str, FeedbackTemplate] = {
    "depth": FeedbackTemplate(
        title="Squat depth",
        rep_pass="Depth",
        rep_issue="Depth needs improvement",
        overview="Depth",
        positive="Good squat depth",
        simple_issue="You stop a little high on {reps}.",
        why="Squatting lower works your legs through their full range.",
        simple_fix="Lower until your hips reach about knee height, keeping your whole foot planted.",
        measures="how far your knees bend at the lowest point of each rep",
        measured_from="your hip, knee and ankle landmarks",
    ),
    "torso_lean": FeedbackTemplate(
        title="Chest position",
        rep_pass="Torso control",
        rep_issue="Torso leaned further forward than the target",
        overview="Torso control",
        positive="Chest stayed upright",
        simple_issue="Your chest tips forward as you lower on {reps}.",
        why="An upright chest keeps the work in your legs.",
        simple_fix="Brace your core and keep your chest lifted on the way down.",
        measures="how far your trunk tilts away from upright during the descent",
        measured_from="your shoulder and hip landmarks",
    ),
    "heel_lift": FeedbackTemplate(
        title="Heel contact",
        rep_pass="Heels planted",
        rep_issue="Heel appears to lift",
        overview="Heel stability",
        positive="Heels stayed planted",
        simple_issue="Your heels look like they come off the floor on {reps}.",
        why="Flat feet give you a more stable base to push from.",
        simple_fix="Keep your whole foot planted and push through your mid-foot.",
        measures="how high your heel sits relative to your toes during the squat",
        measured_from="your heel, toe and ankle landmarks",
    ),
    "extension": FeedbackTemplate(
        title="Standing back up",
        rep_pass="Returned to standing",
        rep_issue="Did not fully return to standing",
        overview="Return to standing",
        positive="Finished each rep standing tall",
        simple_issue="You start the next rep before standing all the way up on {reps}.",
        why="Finishing each rep tall completes the movement.",
        simple_fix="Stand up fully - hips and knees straight - before the next rep.",
        measures="how straight your knees are at the end of each rep, against your own standing posture",
        measured_from="your hip, knee and ankle landmarks",
    ),
    "descent_control": FeedbackTemplate(
        title="Speed on the way down",
        rep_pass="Controlled descent",
        rep_issue="Dropped down quickly",
        overview="Descent control",
        positive="Controlled, steady tempo",
        simple_issue="You drop down quickly on {reps}.",
        why="A controlled descent gives you more stability at the bottom.",
        simple_fix="Take about a second to lower yourself, then stand back up.",
        measures="how long the lowering phase of each rep takes",
        measured_from="the knee angle over time",
    ),
}

# Which measured values a finding prints, and in what units.
EVIDENCE_FORMATS = {
    "min_knee_angle": lambda v: f"minimum knee angle {float(v):.0f} deg",
    "max_torso_lean": lambda v: f"peak torso lean {float(v):.0f} deg",
    "heel_lift_ratio": lambda v: f"heel rise {float(v) * 100:.0f}% of lower-leg length",
    "descent_seconds": lambda v: f"descent {float(v):.1f} s",
    "end_knee_angle": lambda v: f"finished at {float(v):.0f} deg",
}


def template_for(rule: RuleResult) -> FeedbackTemplate:
    """Look up a squat rule's wording by its declared feedback key."""
    return _template_for(rule, FEEDBACK_TEMPLATES)


def build_findings(rule_results: list[RuleResult]) -> list[FeedbackFinding]:
    """The 'what to improve' list, biggest repeated problem first."""
    return _build_findings(rule_results, FEEDBACK_TEMPLATES, EVIDENCE_FORMATS)


def build_positives(rule_results: list[RuleResult], reps: list[SquatRep]) -> list[str]:
    """What went well, limited to what the measurements support."""
    return _build_positives(rule_results, reps, FEEDBACK_TEMPLATES, POSITIVES_OPENING)


def build_rep_summaries(reps: list[SquatRep], rule_results: list[RuleResult]) -> list[RepSummary]:
    """One verdict per rep."""
    return _build_rep_summaries(reps, rule_results, FEEDBACK_TEMPLATES)


def build_overview(rule_results: list[RuleResult]) -> list[str]:
    """One line per evaluated check: "Depth: 3 of 4 repetitions acceptable"."""
    return _build_overview(rule_results, FEEDBACK_TEMPLATES)


def build_not_assessed(rule_results: list[RuleResult]) -> list[NotAssessedItem]:
    """The checks we did not judge, with the reason for each."""
    return _build_not_assessed(rule_results, FEEDBACK_TEMPLATES)


def build_summary(
    reps: list[SquatRep],
    partial_movements: int,
    rule_results: list[RuleResult],
) -> SessionSummary:
    """Everything the results page needs for the set as a whole."""
    return _build_summary(reps, partial_movements, rule_results, FEEDBACK_TEMPLATES, POSITIVES_OPENING)
