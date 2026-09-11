"""
Developer calibration tool - any exercise, one command.

Runs the pipeline on a video and prints each measurement next to the threshold
it was compared against.

    python scripts/calibrate.py path/to/video.mp4 pulldown
    python scripts/calibrate.py path/to/video.mp4 press --frames
    python scripts/calibrate.py path/to/video.mp4 squat --json

Output goes to analysis_results/<run id>/. Not used by the app itself.
(The first line is used as the argparse description.)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from analysis.models import AnalysisFailure, RuleStatus  # noqa: E402
from exercises import ANALYSERS, DISPLAY_NAMES  # noqa: E402

# per-rep fields to print for each exercise (label, format) - everything else
# is still in reps.csv
REP_COLUMNS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "squat": (
        ("min_knee_angle", "min knee", "{:.0f}"),
        ("hip_above_knee_at_bottom", "hip/knee", "{:.2f}"),
        ("max_torso_lean", "torso", "{:.0f}"),
        ("max_heel_lift", "heel", "{:.3f}"),
        ("end_knee_angle", "finish", "{:.0f}"),
        ("max_knee_asymmetry", "L/R", "{:.0f}"),
        ("descent_duration", "descent s", "{:.2f}"),
    ),
    "pulldown": (
        ("top_elbow_angle", "top elbow", "{:.0f}"),
        ("bottom_elbow_angle", "bot elbow", "{:.0f}"),
        ("rom_degrees", "ROM", "{:.0f}"),
        ("max_torso_excursion", "torso exc", "{:.1f}"),
        ("torso_at_top", "torso top", "{:.0f}"),
        ("wrist_travel", "wrist trv", "{:.2f}"),
        ("peak_torso_velocity", "torso d/s", "{:.0f}"),
    ),
    "press": (
        ("top_elbow_angle", "top elbow", "{:.0f}"),
        ("bottom_elbow_angle", "bot elbow", "{:.0f}"),
        ("rom_degrees", "ROM", "{:.0f}"),
        ("max_elbow_angle_difference", "L/R angle", "{:.1f}"),
        ("max_wrist_height_difference", "L/R height", "{:.3f}"),
        ("max_left_alignment_offset", "L align", "{:.2f}"),
        ("max_right_alignment_offset", "R align", "{:.2f}"),
    ),
}

# threshold keys to print next to a rule's verdict
THRESHOLD_KEYS = (
    "threshold_pass_deg",
    "threshold_warn_deg",
    "warn_deg",
    "fail_deg",
    "absolute_fail_deg",
    "top_extension_pass_deg",
    "top_extension_warn_deg",
    "bottom_flexion_pass_deg",
    "bottom_flexion_warn_deg",
    "min_excursion_deg",
    "angle_warn_deg",
    "angle_fail_deg",
    "height_warn_shoulder_widths",
    "height_fail_shoulder_widths",
    "rom_warn_deg",
    "warn_shoulder_widths",
    "fail_shoulder_widths",
    "threshold_ratio",
    "tolerance_deg",
    "min_seconds",
)

STATUS_MARK = {
    RuleStatus.PASS: "PASS",
    RuleStatus.WARNING: "WARN",
    RuleStatus.FAIL: "FAIL",
    RuleStatus.NOT_EVALUABLE: " -- ",
}


def _fmt(value, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def _print_recording(result) -> None:
    checks = result.validation
    print("\n=== Recording =================================================")
    print(f"  exercise            {result.exercise}")
    print(
        f"  file                {result.video.width}x{result.video.height} @ "
        f"{result.video.fps:.1f} fps, {result.video.duration:.1f}s"
    )
    print(f"  quality             {checks.quality.value}")
    print(
        f"  camera orientation  {checks.orientation.value} "
        f"(side-view confidence {checks.side_view_confidence:.2f})"
    )
    print(f"  pose coverage       {checks.metrics.get('pose_frame_ratio')}")
    print(f"  usable frames       {checks.metrics.get('usable_frame_ratio')}")
    print(f"  analysis side       {result.analysis_side}")
    for warning in checks.warnings:
        print(f"  ! {warning}")


def _print_reps(result, exercise_id: str) -> None:
    columns = REP_COLUMNS.get(exercise_id, ())
    print("\n=== Repetitions ===============================================")
    if not result.reps:
        print("  none")
        return
    header = f"  {'rep':>3} {'start':>7} {'turn':>7} {'end':>7}"
    header += "".join(f" {label:>10}" for _, label, _ in columns)
    header += f" {'valid':>6}"
    print(header)
    for rep in result.reps:
        turn = getattr(rep, "bottom_time", getattr(rep, "extreme_time", float("nan")))
        row = f"  {rep.number:>3} {rep.start_time:>7.2f} {turn:>7.2f} {rep.end_time:>7.2f}"
        row += "".join(f" {_fmt(getattr(rep, field, None), spec):>10}" for field, _, spec in columns)
        row += f" {getattr(rep, 'valid_frame_ratio', 0.0):>6.2f}"
        print(row)
    if result.summary.partial_movements:
        print(f"\n  {result.summary.partial_movements} partial movement(s) not counted:")
        for reason in result.debug.get("partial_reasons", []):
            print(f"    - {reason}")


def _print_rules(result) -> None:
    print("\n=== Rules =====================================================")
    for rule in result.rule_results:
        thresholds = {k: v for k, v in rule.evidence.items() if k in THRESHOLD_KEYS}
        per_rep = " ".join(STATUS_MARK[o.status] for o in rule.per_rep)
        print(
            f"\n  {rule.rule_id:<20} {rule.status.value.upper():<14} "
            f"reliability={rule.reliability.value}"
        )
        print(f"    phase={rule.phase}  views={','.join(rule.supported_views)}")
        if thresholds:
            print("    thresholds: " + "  ".join(f"{k}={v}" for k, v in thresholds.items()))
        if per_rep:
            print(f"    per rep:    {per_rep}")
        persistence = result.debug.get("rule_persistence", {}).get(rule.rule_id, [])
        for entry in persistence:
            if entry["phase_frames"]:
                print(
                    f"      rep {entry['rep']}: {entry['violating_frames']}/"
                    f"{entry['phase_frames']} frames violating "
                    f"({entry['violation_ratio']:.0%})"
                )
        print(f"    {rule.explanation}")
        if rule.correction:
            print(f"    -> {rule.correction}")
        if rule.limitation:
            print(f"    (!) {rule.limitation}")


def _save_key_frames(result, exercise_id: str) -> None:
    """Save each rep's start, turning point and end as annotated stills."""
    import cv2

    clip = result.annotated_video_path
    if clip is None or not clip.is_file():
        print("\n  (no annotated clip to sample key frames from)")
        return
    out_dir = clip.parent / "key_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted: list[tuple[str, int]] = []
    for rep in result.reps:
        turn = getattr(rep, "extreme_frame", getattr(rep, "bottom_frame", rep.start_frame))
        wanted += [
            (f"rep{rep.number}_start", rep.start_frame),
            (f"rep{rep.number}_turn", turn),
            (f"rep{rep.number}_end", rep.end_frame),
        ]
    capture = cv2.VideoCapture(str(clip))
    try:
        for name, frame_index in wanted:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
            ok, frame = capture.read()
            if ok:
                cv2.imwrite(str(out_dir / f"{name}.png"), frame)
    finally:
        capture.release()
    print(f"\n  key frames -> {out_dir}")
    del exercise_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("video", type=Path)
    parser.add_argument("exercise", choices=sorted(DISPLAY_NAMES))
    parser.add_argument("--frames", action="store_true", help="save annotated key frames")
    parser.add_argument("--json", action="store_true", help="print the full debug block")
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # I pipe this into a file for the write-up, so only show progress in a terminal
    interactive = sys.stdout.isatty()

    def progress(stage: str, fraction: float, message: str) -> None:
        if interactive:
            print(f"\r[{stage:>9}] {fraction:5.0%}  {message:<44}", end="", flush=True)

    try:
        result = ANALYSERS[args.exercise](args.video, progress=progress, export_root=APP_ROOT)
    except AnalysisFailure as failure:
        print(f"\n\nNot analysed: [{failure.code.value}] {failure.message}")
        for tip in failure.suggestions:
            print(f"  - {tip}")
        if failure.validation:
            print("\n  diagnostics:")
            for key, value in failure.validation.diagnostics().items():
                print(f"    {key}: {value}")
        return 1

    if interactive:
        print("\r" + " " * 78 + "\r", end="")
    _print_recording(result)
    _print_reps(result, args.exercise)
    _print_rules(result)

    print("\n=== Summary ===================================================")
    print(f"  score        {result.summary.score}  ({result.summary.score_formula})")
    print(f"  reliability  {result.summary.reliability.value}")
    for line in result.summary.overview:
        print(f"  {line}")
    for item in result.summary.not_assessed:
        print(f"  not assessed - {item.title}: {item.reason}")

    if result.annotated_video_path:
        print(f"\n  annotated clip -> {result.annotated_video_path}")
    if result.debug.get("export_dir"):
        print(f"  export bundle  -> {result.debug['export_dir']}")
    if args.frames:
        _save_key_frames(result, args.exercise)
    if args.json:
        print("\n=== Debug =====================================================")
        print(json.dumps(_jsonable(result.debug), indent=2)[:20000])
    return 0


def _jsonable(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 4)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
