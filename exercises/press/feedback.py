"""
Press feedback wording. The logic is shared (exercises/common/feedback.py),
this file just has the press templates and which values to print.
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

POSITIVES_OPENING = "{n} full press rep{s} completed"

# feedback_key -> wording
FEEDBACK_TEMPLATES: dict[str, FeedbackTemplate] = {
    "press_symmetry": FeedbackTemplate(
        title="Uneven arms",
        rep_pass="Even arms",
        rep_issue="One arm led the other",
        overview="Arm symmetry",
        positive="Arms stayed symmetrical",
        simple_issue="One arm leads the other on {reps}.",
        why="Pressing evenly shares the work between both shoulders.",
        simple_fix="Press both weights at the same speed and aim to finish together.",
        measures="the difference between your left and right elbow angle and wrist height",
        measured_from="both shoulder, elbow and wrist chains",
    ),
    "press_alignment": FeedbackTemplate(
        title="Wrist position",
        rep_pass="Wrists stacked",
        rep_issue="Wrist drifted away from the elbow",
        overview="Elbow and wrist alignment",
        positive="Wrists stayed stacked over your elbows",
        simple_issue="Your wrist drifts away from being stacked over your elbow on {reps}.",
        why="A stacked wrist keeps the weight travelling straight up.",
        simple_fix="Keep each wrist directly above its elbow as you press.",
        measures="how far each wrist sits sideways from its elbow, as a share of your shoulder width",
        measured_from="your elbow and wrist landmarks",
    ),
    "press_rom": FeedbackTemplate(
        title="Movement range",
        rep_pass="Full movement range",
        rep_issue="Movement range was limited",
        overview="Range of motion",
        positive="Full range on every press",
        simple_issue="Your presses stop short of the full range on {reps}.",
        why="Pressing through the full range builds strength across the whole movement.",
        simple_fix="Press until your arms are straight overhead, then lower back to shoulder height.",
        measures="how far your elbows open at the top and close at the bottom of each press",
        measured_from="your shoulder, elbow and wrist landmarks",
    ),
}

# which measured values a finding prints, and their units
EVIDENCE_FORMATS = {
    "max_elbow_angle_difference": lambda v: f"left/right elbow difference {float(v):.0f} deg",
    "max_wrist_height_difference": (
        lambda v: f"wrist height difference {float(v) * 100:.0f}% of shoulder width"
    ),
    "rom_difference": lambda v: f"range difference {float(v):.0f} deg",
    "max_alignment_offset": (
        lambda v: f"wrist-to-elbow offset {float(v) * 100:.0f}% of shoulder width"
    ),
    "top_elbow_angle": lambda v: f"elbow extended to {float(v):.0f} deg",
    "bottom_elbow_angle": lambda v: f"elbow returned to {float(v):.0f} deg",
}


def template_for(rule: RuleResult) -> FeedbackTemplate:
    return _template_for(rule, FEEDBACK_TEMPLATES)


def build_findings(rule_results: list[RuleResult]) -> list[FeedbackFinding]:
    """The 'what to improve' list, biggest repeated problem first."""
    return _build_findings(rule_results, FEEDBACK_TEMPLATES, EVIDENCE_FORMATS)


def build_rep_summaries(reps: list, rule_results: list[RuleResult]) -> list[RepSummary]:
    return _build_rep_summaries(reps, rule_results, FEEDBACK_TEMPLATES)


def build_overview(rule_results: list[RuleResult]) -> list[str]:
    """One line per evaluated check, stating how often it was acceptable."""
    return _build_overview(rule_results, FEEDBACK_TEMPLATES)


def build_not_assessed(rule_results: list[RuleResult]) -> list[NotAssessedItem]:
    return _build_not_assessed(rule_results, FEEDBACK_TEMPLATES)


def build_summary(reps: list, partial_movements: int, rule_results: list[RuleResult]) -> SessionSummary:
    return _build_summary(reps, partial_movements, rule_results, FEEDBACK_TEMPLATES, POSITIVES_OPENING)
