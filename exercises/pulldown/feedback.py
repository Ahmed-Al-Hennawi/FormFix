"""
The pulldown's wording. The aggregation is shared
(exercises/common/feedback.py); the only pulldown-specific parts are the two
tables below.
"""

from __future__ import annotations

from analysis.models import (
    NotAssessedItem,
    RepSummary,
    RuleResult,
    SessionSummary,
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
    "build_findings",
    "build_not_assessed",
    "build_overview",
    "build_rep_summaries",
    "build_summary",
    "template_for",
    "transparent_score",
]

POSITIVES_OPENING = "{n} full rep{s} completed"

# feedback_key -> wording. Each entry has the beginner's three lines plus the
# plain-language description shown in the "how did we detect this" panel.
FEEDBACK_TEMPLATES: dict[str, FeedbackTemplate] = {
    "pulldown_rom": FeedbackTemplate(
        title="Movement range",
        rep_pass="Full movement range",
        rep_issue="Movement range was limited",
        overview="Range of motion",
        positive="Full range on every rep",
        simple_issue="Your arms do not travel through the full range on {reps}.",
        why="A fuller range works your back through more of the movement.",
        simple_fix="Pull until your elbows are down by your sides, then let your arms "
        "straighten again before the next rep.",
        measures="how far your elbow opens and closes between the start and the pull",
        measured_from="your shoulder, elbow and wrist landmarks",
    ),
    "pulldown_torso": FeedbackTemplate(
        title="Torso movement",
        rep_pass="Stable torso",
        rep_issue="Torso moved more than the configured range",
        overview="Torso movement",
        positive="Steady torso through the pull",
        simple_issue="Your upper body swings as you pull on {reps}.",
        why="Keeping your torso still keeps the work in your back rather than your momentum.",
        simple_fix="Sit tall and pull with your arms and back, keeping your chest still.",
        measures="how far your trunk moves away from the position it started the rep in",
        measured_from="your shoulder and hip landmarks",
    ),
}

# Which measured values a finding prints, and in what units - the numbers the
# rule compared against its threshold.
# numbers the rule compared against its threshold, so the evidence a user
EVIDENCE_FORMATS = {
    "bottom_elbow_angle": lambda v: f"elbow closed to {float(v):.0f} deg",
    "top_elbow_angle": lambda v: f"elbow extended to {float(v):.0f} deg",
    "rom_degrees": lambda v: f"range {float(v):.0f} deg",
    "max_torso_excursion": lambda v: f"torso moved {float(v):.0f} deg from its top position",
}


def template_for(rule: RuleResult) -> FeedbackTemplate:
    """Look up a pulldown rule's wording by its declared feedback key."""
    return _template_for(rule, FEEDBACK_TEMPLATES)


def build_findings(rule_results: list[RuleResult]) -> list[FeedbackFinding]:
    """The 'what to improve' list, biggest repeated problem first."""
    return _build_findings(rule_results, FEEDBACK_TEMPLATES, EVIDENCE_FORMATS)


def build_rep_summaries(reps: list, rule_results: list[RuleResult]) -> list[RepSummary]:
    """One verdict per rep."""
    return _build_rep_summaries(reps, rule_results, FEEDBACK_TEMPLATES)


def build_overview(rule_results: list[RuleResult]) -> list[str]:
    """One line per evaluated check, stating how often it was acceptable."""
    return _build_overview(rule_results, FEEDBACK_TEMPLATES)


def build_not_assessed(rule_results: list[RuleResult]) -> list[NotAssessedItem]:
    """The checks we did not judge, with the reason for each."""
    return _build_not_assessed(rule_results, FEEDBACK_TEMPLATES)


def build_summary(reps: list, partial_movements: int, rule_results: list[RuleResult]) -> SessionSummary:
    """Everything the results page needs for the set as a whole."""
    return _build_summary(reps, partial_movements, rule_results, FEEDBACK_TEMPLATES, POSITIVES_OPENING)
