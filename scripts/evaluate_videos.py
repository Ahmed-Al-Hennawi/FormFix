"""
Runs FormFix over labelled recordings and writes the evaluation tables.

    python scripts/evaluate_videos.py evaluation/labels.csv

The manifest is a CSV of video,exercise,expected - the video path, one of
squat / pulldown / press, and either "correct" or the rule ids of the faults it
was performed with, separated by ";".

Output goes beside the manifest: results.csv (one row per video), 
error_matrix.csv (TP/FP/FN/TN per rule) and summary.md.

It counts what happened rather than computing an accuracy figure, and a
recording that couldn't be analysed gets its own outcome rather than being
dropped or scored as a miss.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from analysis.models import AnalysisFailure, RuleStatus  # noqa: E402
from exercises import ANALYSERS  # noqa: E402
from exercises.press.config import DEFAULT_CONFIG as PRESS_CONFIG  # noqa: E402
from exercises.press.config import rule_specs as press_rules  # noqa: E402
from exercises.pulldown.config import DEFAULT_CONFIG as PULLDOWN_CONFIG  # noqa: E402
from exercises.pulldown.config import rule_specs as pulldown_rules  # noqa: E402
from exercises.squat.config import DEFAULT_CONFIG as SQUAT_CONFIG  # noqa: E402
from exercises.squat.config import rule_specs as squat_rules  # noqa: E402
from exercises.squat.literature_config import LITERATURE_CONFIG  # noqa: E402
from exercises.squat.literature_config import PROVENANCE as LITERATURE_PROVENANCE  # noqa: E402

# Every rule each exercise can report, so a row can record a true negative
# (rule ran, correctly stayed quiet) and not just the firings.
RULES: dict[str, tuple[str, ...]] = {
    "squat": tuple(spec.rule_id for spec in squat_rules(SQUAT_CONFIG)),
    "pulldown": tuple(spec.rule_id for spec in pulldown_rules(PULLDOWN_CONFIG)),
    "press": tuple(spec.rule_id for spec in press_rules(PRESS_CONFIG)),
}

# A rule counts as having fired if it warned or failed; NOT_EVALUABLE is
# tallied on its own.
FIRED = (RuleStatus.WARNING, RuleStatus.FAIL)


@dataclass
class Row:
    """One analysed recording."""

    video: str
    exercise: str
    expected: list[str]
    analysed: bool = True
    failure: str = ""
    reps: int = 0
    partial: int = 0
    score: int | None = None
    reliability: str = ""
    recording_quality: str = ""
    camera_orientation: str = ""
    statuses: dict[str, str] = field(default_factory=dict)
    reliabilities: dict[str, str] = field(default_factory=dict)
    measurements: dict[str, object] = field(default_factory=dict)

    @property
    def detected(self) -> list[str]:
        return sorted(rule for rule, status in self.statuses.items() if status in ("warning", "fail"))

    @property
    def not_assessed(self) -> list[str]:
        return sorted(rule for rule, status in self.statuses.items() if status == "not_evaluable")


def load_manifest(path: Path) -> list[tuple[Path, str, list[str]]]:
    """Read the labelled-video manifest."""
    entries: list[tuple[Path, str, list[str]]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            video = Path(record["video"].strip())
            if not video.is_absolute():
                video = path.parent / video
            exercise = record["exercise"].strip()
            if exercise not in ANALYSERS:
                raise SystemExit(f"{path}: unknown exercise {exercise!r} for {video.name}")
            raw = record.get("expected", "").strip().lower()
            expected = (
                []
                if raw in ("", "correct", "none")
                else [part.strip() for part in raw.split(";") if part.strip()]
            )
            unknown = set(expected) - set(RULES[exercise])
            if unknown:
                raise SystemExit(f"{path}: unknown expected rule(s) {sorted(unknown)} for {video.name}")
            entries.append((video, exercise, expected))
    return entries


def analyse(
    video: Path,
    exercise: str,
    expected: list[str],
    export_root: Path | None,
    config=None,
) -> Row:
    """Run one recording through its analyser and record what came out. config
    overrides the defaults, which is how the same manifest runs under the
    operational thresholds and the literature preset."""
    row = Row(video=video.name, exercise=exercise, expected=expected)
    kwargs = {"export_root": export_root}
    if config is not None:
        kwargs["config"] = config
    try:
        result = ANALYSERS[exercise](video, **kwargs)
    except AnalysisFailure as failure:
        row.analysed = False
        row.failure = f"{failure.code.value}: {failure.message}"
        return row
    except Exception as exc:  # pragma: no cover - one bad clip shouldn't kill the run
        row.analysed = False
        row.failure = f"error: {exc}"
        return row

    row.reps = result.summary.complete_reps
    row.partial = result.summary.partial_movements
    row.score = result.summary.score
    row.reliability = result.summary.reliability.value
    row.recording_quality = result.validation.quality.value
    row.camera_orientation = result.validation.orientation.value
    row.statuses = {rule.rule_id: rule.status.value for rule in result.rule_results}
    row.reliabilities = {rule.rule_id: rule.reliability.value for rule in result.rule_results}
    row.measurements = {
        "rep_boundaries": result.debug.get("rep_boundaries", []),
        "thresholds": result.debug.get("rep_detection_thresholds", {}),
    }
    return row


def outcome_for(row: Row, rule: str) -> str:
    """The confusion-matrix cell for one rule on one recording. NOT_ASSESSED is its
    own outcome, not a miss - counting it as one would penalise the system
    for declining to judge what the recording couldn't support."""
    if not row.analysed:
        return "NOT_ANALYSED"
    status = row.statuses.get(rule)
    if status is None:
        return "-"
    if status == "not_evaluable":
        return "NOT_ASSESSED"
    fired = status in ("warning", "fail")
    wanted = rule in row.expected
    if fired and wanted:
        return "TP"
    if fired and not wanted:
        return "FP"
    if not fired and wanted:
        return "FN"
    return "TN"


def write_results(rows: list[Row], out_dir: Path) -> Path:
    path = out_dir / "results.csv"
    fields = [
        "video",
        "exercise",
        "expected",
        "analysed",
        "failure",
        "reps",
        "partial_movements",
        "score",
        "analysis_reliability",
        "recording_quality",
        "camera_orientation",
        "detected",
        "not_assessed",
        "rule_statuses",
        "rule_reliabilities",
        "measurements",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video": row.video,
                    "exercise": row.exercise,
                    "expected": ";".join(row.expected) or "correct",
                    "analysed": row.analysed,
                    "failure": row.failure,
                    "reps": row.reps,
                    "partial_movements": row.partial,
                    "score": "" if row.score is None else row.score,
                    "analysis_reliability": row.reliability,
                    "recording_quality": row.recording_quality,
                    "camera_orientation": row.camera_orientation,
                    "detected": ";".join(row.detected) or "correct",
                    "not_assessed": ";".join(row.not_assessed),
                    "rule_statuses": json.dumps(row.statuses),
                    "rule_reliabilities": json.dumps(row.reliabilities),
                    "measurements": json.dumps(row.measurements),
                }
            )
    return path


def write_error_matrix(rows: list[Row], out_dir: Path) -> tuple[Path, dict[str, dict[str, int]]]:
    path = out_dir / "error_matrix.csv"
    tally: dict[str, dict[str, int]] = {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "exercise", "rule", "expected", "detected", "result"])
        for row in rows:
            for rule in RULES[row.exercise]:
                result = outcome_for(row, rule)
                writer.writerow(
                    [
                        row.video,
                        row.exercise,
                        rule,
                        "yes" if rule in row.expected else "no",
                        row.statuses.get(rule, "-"),
                        result,
                    ]
                )
                counts = tally.setdefault(rule, {})
                counts[result] = counts.get(result, 0) + 1
    return path, tally


def write_summary(rows: list[Row], tally: dict[str, dict[str, int]], out_dir: Path) -> Path:
    path = out_dir / "summary.md"
    lines = [
        "# FormFix technical evaluation",
        "",
        f"Recordings in the manifest: **{len(rows)}**  ",
        f"Analysed successfully: **{sum(1 for r in rows if r.analysed)}**  ",
        f"Stopped before analysis: **{sum(1 for r in rows if not r.analysed)}**",
        "",
        "No accuracy percentage is computed here. The counts below are what the",
        "system did; whether a sample of this size supports a rate is a question",
        "for the evaluation chapter, not for a script.",
        "",
        "## Per recording",
        "",
        "| Video | Exercise | Expected | Detected | Reps | Score | Reliability | Recording |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if not row.analysed:
            lines.append(
                f"| {row.video} | {row.exercise} | {';'.join(row.expected) or 'correct'} | "
                f"not analysed ({row.failure.split(':')[0]}) | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {row.video} | {row.exercise} | {';'.join(row.expected) or 'correct'} | "
            f"{';'.join(row.detected) or 'correct'} | {row.reps} | {row.score} | "
            f"{row.reliability} | {row.recording_quality} |"
        )

    lines += [
        "",
        "## Per rule",
        "",
        "TP - the rule fired on a recording labelled with that fault.  ",
        "FP - it fired on a recording not labelled with it.  ",
        "FN - it stayed silent on a recording labelled with it.  ",
        "TN - it correctly stayed silent.  ",
        "NOT ASSESSED - the recording could not support the measurement, so the",
        "system declined to judge it. That is not a miss and is counted apart.",
        "",
        "| Rule | TP | FP | FN | TN | Not assessed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rule, counts in tally.items():
        lines.append(
            f"| {rule} | {counts.get('TP', 0)} | {counts.get('FP', 0)} | "
            f"{counts.get('FN', 0)} | {counts.get('TN', 0)} | {counts.get('NOT_ASSESSED', 0)} |"
        )

    failures = [row for row in rows if not row.analysed]
    if failures:
        lines += ["", "## Recordings that were not analysed", ""]
        lines += [f"- **{row.video}** - {row.failure}" for row in failures]

    lines += [
        "",
        "## Not-assessed measurements",
        "",
    ]
    any_skipped = False
    for row in rows:
        if row.not_assessed:
            any_skipped = True
            lines.append(f"- **{row.video}** ({row.camera_orientation}): {', '.join(row.not_assessed)}")
    if not any_skipped:
        lines.append("None - every rule had usable evidence on every analysed recording.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _configs(args) -> dict:
    """Which config each exercise runs under. Only the squat has a literature
    preset, since it is the only movement the reviewed papers publish joint-angle
    ranges for."""
    from dataclasses import replace

    base = {
        "squat": LITERATURE_CONFIG if args.preset == "literature" else SQUAT_CONFIG,
        "pulldown": PULLDOWN_CONFIG,
        "press": PRESS_CONFIG,
    }
    changes = {}
    if args.strict_uncertainty:
        changes["UNCERTAINTY_STRICT"] = True
    if args.angle_filter:
        changes["ANGLE_FILTER"] = args.angle_filter
    if not changes and args.preset == "default":
        return {}
    return {name: replace(config, **changes) for name, config in base.items()}


def _announce(args) -> None:
    """Print which settings the run used, before any of its results."""
    print(f"Threshold preset : {args.preset}")
    print(
        f"Uncertainty policy: {'strict (downgrades)' if args.strict_uncertainty else 'annotate only'}"
    )
    print(f"Smoothing filter : {args.angle_filter or 'per-config default'}")
    if args.preset == "literature":
        print("\nLiterature preset provenance:")
        for key, why in LITERATURE_PROVENANCE.items():
            print(f"  {key}\n    {why}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("manifest", type=Path, help="CSV of labelled recordings")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: beside the manifest)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="also write the full per-run JSON/CSV bundle for each recording",
    )
    parser.add_argument(
        "--preset",
        choices=("default", "literature"),
        default="default",
        help=(
            "which squat threshold set to run: 'default' is FormFix's own operational "
            "values, 'literature' is the preset derived from Kotiuk et al. (2022), Dill "
            "et al. (2024) and Simoes et al. (2024). Run both over the same manifest and "
            "report the difference; neither is validated, and the comparison is the point"
        ),
    )
    parser.add_argument(
        "--strict-uncertainty",
        action="store_true",
        help=(
            "downgrade any finding whose margin falls inside the published measurement "
            "error of the quantity it rests on, instead of only annotating it"
        ),
    )
    parser.add_argument(
        "--filter",
        dest="angle_filter",
        choices=("ema", "butterworth", "savgol", "moving_average"),
        default=None,
        help="override the movement-signal smoothing filter (default: each config's own)",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")

    out_dir = args.out or args.manifest.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    export_root = out_dir if args.export else None

    overrides = _configs(args)
    _announce(args)

    entries = load_manifest(args.manifest)
    rows: list[Row] = []
    for index, (video, exercise, expected) in enumerate(entries, start=1):
        print(f"[{index}/{len(entries)}] {video.name} ({exercise})", flush=True)
        if not video.is_file():
            row = Row(video=video.name, exercise=exercise, expected=expected, analysed=False)
            row.failure = "file not found"
            rows.append(row)
            continue
        rows.append(analyse(video, exercise, expected, export_root, overrides.get(exercise)))

    results = write_results(rows, out_dir)
    matrix, tally = write_error_matrix(rows, out_dir)
    summary = write_summary(rows, tally, out_dir)
    print(f"\nWrote:\n  {results}\n  {matrix}\n  {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
