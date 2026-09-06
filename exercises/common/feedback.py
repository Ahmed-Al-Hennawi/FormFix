"""
Turning rule results into sentences. Last stage of every pipeline and the only
one allowed to write prose - it measures nothing and decides nothing.

The audience is beginners in a gym: no internal vocabulary, no medical or
injury claims, and always something that went well alongside the corrections.

The score is a plain formula, not a model output:

    score = 100 * (pass = 1, warning = 0.5, fail = 0, summed) / evaluable checks

with one check being one rule on one rep. Checks that couldn't be evaluated
come out of both halves, so a measurement the recording couldn't support is
never scored as a failure.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from analysis.models import (
    NotAssessedItem,
    Reliability,
    RepSummary,
    RuleResult,
    RuleStatus,
    SessionSummary,
)

from .confidence import overall_reliability

SCORE_FORMULA = (
    "score = 100 x (passed checks + 0.5 x warnings) / evaluable checks, "
    "where a check is one rule evaluated on one repetition"
)


def format_timestamp(seconds: float) -> str:
    """00:04.2 style timestamps for evidence lines."""
    if not math.isfinite(seconds):
        return "-"
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:04.1f}"


@dataclass(frozen=True)
class FeedbackTemplate:
    """
    Fixed wording for one kind of finding. The numbers come from the measurements
    but the phrasing never varies, so every sentence a user can see was reviewed
    once instead of generated fresh.

    simple_issue / why / simple_fix are the three short lines a beginner sees;
    measures / measured_from feed the "how did we detect this" panel.
    """

    title: str
    rep_pass: str
    rep_issue: str
    overview: str
    # one line about what went well, used when the whole check passed
    positive: str = ""

    # --- the beginner-facing correction ---
    # what happened, one plain sentence. {flagged}, {total} and {reps} get
    # filled in from the per-rep outcomes.
    simple_issue: str = ""
    why: str = ""
    simple_fix: str = ""

    # --- the explainability panel ---
    measures: str = ""
    # which landmarks it came from ("your hip, knee and ankle")
    measured_from: str = ""


FALLBACK_TEMPLATE = FeedbackTemplate(
    title="Technique check", rep_pass="Check", rep_issue="Needs improvement", overview="Check"
)


def template_for(rule: RuleResult, templates: dict[str, FeedbackTemplate]) -> FeedbackTemplate:
    """Look up a rule's wording by its declared feedback key."""
    return templates.get(rule.feedback_key, FALLBACK_TEMPLATE)


@dataclass
class FeedbackFinding:
    """
    One correction at both levels of detail, with no Streamlit in it. simple_* is
    what the results page shows straight away; the rest is the traceable record
    behind it, which the interface tucks into an expander. One object rather than
    two, so there is no second source of truth.
    """

    rule_id: str
    title: str
    issue: str
    correction: str
    severity: str
    # evidence lines ("Rep 2 - minimum elbow angle 116 deg - 00:04.2")
    evidence_lines: list[str] = field(default_factory=list)
    # which template the wording came from - links a sentence back to its rule
    feedback_key: str = ""
    reliability: Reliability = Reliability.CANNOT_ASSESS

    # --- the beginner-facing three lines ---
    simple_issue: str = ""
    why: str = ""
    simple_fix: str = ""

    # --- the explainability record ---
    measures: str = ""
    measured_from: str = ""
    metric: str = ""
    phase: str = ""
    flagged_reps: int = 0
    evaluable_reps: int = 0
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def priority(self) -> int:
        """Ordering key for the corrections list. A fault that repeats beats a one-off,
        and a finding from clear landmarks beats one from borderline landmarks."""
        severity_rank = {"important": 3, "moderate": 2, "minor": 1}.get(self.severity, 1)
        share = self.flagged_reps / self.evaluable_reps if self.evaluable_reps else 0.0
        return int(severity_rank * 100 + share * 40 + self.reliability.rank * 3)


# --- Score ---


def transparent_score(rule_results: list[RuleResult]) -> tuple[int, int, int]:
    """
    (score 0-100, credits x2 as an int, evaluable check count). Pure function
    so the tests can pin the formula down.
    """
    credits = 0.0
    evaluable = 0
    for rule in rule_results:
        for outcome in rule.per_rep:
            if outcome.status is RuleStatus.NOT_EVALUABLE:
                continue
            evaluable += 1
            if outcome.status is RuleStatus.PASS:
                credits += 1.0
            elif outcome.status is RuleStatus.WARNING:
                credits += 0.5
    score = int(round(100.0 * credits / evaluable)) if evaluable else 0
    return score, int(credits * 2), evaluable


def severity_for(rule: RuleResult) -> str:
    """minor / moderate / important, from how much of the set failed."""
    flagged = [o for o in rule.per_rep if o.status in (RuleStatus.FAIL, RuleStatus.WARNING)]
    evaluable = [o for o in rule.per_rep if o.status is not RuleStatus.NOT_EVALUABLE]
    fails = [o for o in flagged if o.status is RuleStatus.FAIL]
    if fails and evaluable and len(fails) * 2 >= len(evaluable):
        return "important"
    if fails:
        return "moderate"
    return "minor"


# --- Findings ---

# evidence key -> renderer, so a rule's per-rep evidence becomes short lines
EvidenceFormats = dict[str, Callable[[Any], str]]


def evidence_lines(rule: RuleResult, formats: EvidenceFormats) -> list[str]:
    """Flagged per-rep outcomes -> short evidence strings."""
    lines: list[str] = []
    for outcome in rule.per_rep:
        if outcome.status not in (RuleStatus.FAIL, RuleStatus.WARNING):
            continue
        parts = [f"Rep {outcome.rep_number}"]
        for key, render in formats.items():
            value = outcome.evidence.get(key)
            if value is None:
                continue
            try:
                parts.append(render(value))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                parts.append(str(value))
        when = (
            outcome.evidence.get("extreme_time")
            or outcome.evidence.get("bottom_time")
            or outcome.evidence.get("end_time")
            or format_timestamp(outcome.evidence_time)
        )
        if when and when != "-":
            parts.append(str(when))
        lines.append(" - ".join(parts))
    return lines


def count_phrase(flagged: int, total: int) -> str:
    """
    The countable fragment of a plain sentence: "2 of your 6 reps", "your rep".
    Shared so all three exercises count the same way.
    """
    if total <= 1:
        return "your rep"
    if flagged >= total:
        return f"all {total} of your reps"
    return f"{flagged} of your {total} reps"


def build_findings(
    rule_results: list[RuleResult],
    templates: dict[str, FeedbackTemplate],
    formats: EvidenceFormats | None = None,
) -> list[FeedbackFinding]:
    """The corrections, most important first, so the interface can take the top two
    or three and know it is showing what matters."""
    findings: list[FeedbackFinding] = []
    for rule in rule_results:
        if rule.status not in (RuleStatus.FAIL, RuleStatus.WARNING):
            continue
        template = template_for(rule, templates)
        evaluable = [o for o in rule.per_rep if o.status is not RuleStatus.NOT_EVALUABLE]
        flagged = [o for o in evaluable if o.status in (RuleStatus.FAIL, RuleStatus.WARNING)]
        counts = {
            "flagged": len(flagged),
            "total": len(evaluable),
            "reps": count_phrase(len(flagged), len(evaluable)),
        }
        findings.append(
            FeedbackFinding(
                rule_id=rule.rule_id,
                title=template.title,
                issue=rule.explanation,
                correction=rule.correction,
                severity=severity_for(rule),
                evidence_lines=evidence_lines(rule, formats or {}),
                feedback_key=rule.feedback_key,
                reliability=rule.reliability,
                simple_issue=_fill(template.simple_issue, counts),
                why=template.why,
                simple_fix=template.simple_fix or rule.correction,
                measures=template.measures,
                measured_from=template.measured_from,
                metric=rule.metric,
                phase=rule.phase,
                flagged_reps=len(flagged),
                evaluable_reps=len(evaluable),
                settings=dict(rule.evidence),
            )
        )
    findings.sort(key=lambda finding: -finding.priority)
    return findings


def _fill(sentence: str, counts: dict[str, Any]) -> str:
    """Fill a template's {flagged} / {total} / {reps} placeholders. An unknown one
    returns the sentence unchanged - a typo shouldn't kill a working analysis."""
    if not sentence:
        return ""
    try:
        return sentence.format(**counts)
    except (KeyError, IndexError, ValueError):  # pragma: no cover - defensive
        return sentence


def build_positives(
    rule_results: list[RuleResult],
    reps: list,
    templates: dict[str, FeedbackTemplate],
    opening: str = "",
) -> list[str]:
    """What went well, limited to what the measurements support. Never "perfect
    form" - a 2D pose estimate can't prove that."""
    positives: list[str] = []
    if reps and opening:
        positives.append(opening.format(n=len(reps), s="" if len(reps) == 1 else "s"))
    for rule in rule_results:
        if rule.status is RuleStatus.PASS:
            template = template_for(rule, templates)
            if template.positive:
                positives.append(template.positive)
    return positives


# --- Per-repetition and session-level aggregation ---

_REP_STATUS_ORDER = {
    RuleStatus.FAIL: 3,
    RuleStatus.WARNING: 2,
    RuleStatus.PASS: 1,
    RuleStatus.NOT_EVALUABLE: 0,
}


def build_rep_summaries(
    reps: list,
    rule_results: list[RuleResult],
    templates: dict[str, FeedbackTemplate],
) -> list[RepSummary]:
    """One verdict per rep. A single grade for the whole set hides what a beginner
    most wants to know, which is which rep went wrong."""
    summaries: list[RepSummary] = []
    for rep in reps:
        passed: list[str] = []
        issues: list[str] = []
        not_assessed: list[str] = []
        worst = RuleStatus.NOT_EVALUABLE
        reliabilities: list[Reliability] = []

        for rule in rule_results:
            outcome = next((o for o in rule.per_rep if o.rep_number == rep.number), None)
            if outcome is None:
                continue
            template = template_for(rule, templates)
            if outcome.status is RuleStatus.NOT_EVALUABLE:
                not_assessed.append(template.overview)
            elif outcome.status is RuleStatus.PASS:
                passed.append(template.rep_pass)
                reliabilities.append(rule.reliability)
            else:
                issues.append(template.rep_issue)
                reliabilities.append(rule.reliability)
            if _REP_STATUS_ORDER[outcome.status] > _REP_STATUS_ORDER[worst]:
                worst = outcome.status

        if worst is RuleStatus.NOT_EVALUABLE:
            headline = "Not assessed"
        elif issues:
            headline = issues[0]
        else:
            headline = "Good repetition"

        summaries.append(
            RepSummary(
                rep_number=rep.number,
                status=worst,
                headline=headline,
                passed=passed,
                issues=issues,
                not_assessed=not_assessed,
                reliability=overall_reliability(reliabilities),
                start_time=getattr(rep, "start_time", float("nan")),
                bottom_time=getattr(rep, "bottom_time", float("nan")),
                end_time=getattr(rep, "end_time", float("nan")),
            )
        )
    return summaries


def build_overview(rule_results: list[RuleResult], templates: dict[str, FeedbackTemplate]) -> list[str]:
    """
    One line per check: "Range of motion: 3 of 4 repetitions acceptable". How often
    an issue occurred is what tells someone whether they have a habit or a one-off.
    """
    lines: list[str] = []
    for rule in rule_results:
        evaluable = [o for o in rule.per_rep if o.status is not RuleStatus.NOT_EVALUABLE]
        if not evaluable:
            continue
        acceptable = sum(1 for o in evaluable if o.status is RuleStatus.PASS)
        name = template_for(rule, templates).overview
        lines.append(f"{name}: {acceptable} of {len(evaluable)} repetitions acceptable")
    return lines


def build_not_assessed(
    rule_results: list[RuleResult], templates: dict[str, FeedbackTemplate]
) -> list[NotAssessedItem]:
    """The checks we did not judge, with the reason for each."""
    return [
        NotAssessedItem(
            metric=rule.metric or rule.rule_id,
            title=template_for(rule, templates).overview,
            reason=rule.limitation or rule.explanation,
        )
        for rule in rule_results
        if rule.status is RuleStatus.NOT_EVALUABLE
    ]


def build_summary(
    reps: list,
    partial_movements: int,
    rule_results: list[RuleResult],
    templates: dict[str, FeedbackTemplate],
    positives_opening: str = "",
) -> SessionSummary:
    """Everything the results page needs for the set as a whole."""
    score, _, _ = transparent_score(rule_results)
    reliabilities = [
        rule.reliability for rule in rule_results if rule.status is not RuleStatus.NOT_EVALUABLE
    ]
    return SessionSummary(
        complete_reps=len(reps),
        partial_movements=partial_movements,
        positives=build_positives(rule_results, reps, templates, positives_opening),
        score=score,
        score_formula=SCORE_FORMULA,
        rep_summaries=build_rep_summaries(reps, rule_results, templates),
        overview=build_overview(rule_results, templates),
        not_assessed=build_not_assessed(rule_results, templates),
        reliability=overall_reliability(reliabilities),
    )
