"""
Calibration tool for the squat. Runs the pipeline on one video and prints the
metrics behind each threshold decision, and can save each rep's start / bottom
/ end as stills.

    python scripts/calibrate_squat.py path/to/video.mp4 [--frames] [--csv]

The export bundle goes to analysis_results/<run id>/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from analysis.models import AnalysisFailure, RuleStatus
from exercises.squat.analyser import analyse_squat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", action="store_true", help="save annotated key frames")
    parser.add_argument("--csv", action="store_true", help="print export paths")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    def progress(stage: str, fraction: float, message: str) -> None:
        print(f"\r[{stage:>9}] {fraction:5.0%}  {message:<40}", end="", flush=True)

    print(f"Analysing {args.video}\n")
    try:
        result = analyse_squat(args.video, progress=progress, export_root=APP_ROOT)
    except AnalysisFailure as failure:
        print(f"\n\nANALYSIS STOPPED: {failure.code.value}\n{failure.message}")
        for tip in failure.suggestions:
            print(f"  - {tip}")
        if failure.validation:
            print("\nRecording diagnostics:")
            for key, value in failure.validation.diagnostics().items():
                print(f"  {key}: {value}")
        return 1

    print("\n")
    print("=" * 64)
    print(
        f"Video: {result.video.width}x{result.video.height} @ {result.video.fps:.1f} fps, "
        f"{result.video.frame_count} frames, {result.video.duration:.1f}s"
    )
    print(
        f"Pose coverage: {result.validation.metrics.get('pose_frame_ratio', 0):.0%}   "
        f"Usable frames: {result.validation.metrics.get('usable_frame_ratio', 0):.0%}   "
        f"Analysis side: {result.analysis_side}   "
        f"Interpolated cells: {result.debug.get('interpolated_cells')}"
    )
    print(
        f"Recording quality: {result.validation.quality.value}   "
        f"Camera: {result.validation.orientation.value} "
        f"(side-view confidence {result.validation.side_view_confidence:.2f})"
    )
    if result.validation.limited_metrics:
        print(f"Not assessed (recording): {', '.join(result.validation.limited_metrics)}")
    for warning in result.validation.warnings:
        print(f"Warning: {warning}")
    print(f"Standing baseline: {result.debug.get('standing_baseline')}")
    print(
        f"Reps: {result.summary.complete_reps} complete, "
        f"{result.summary.partial_movements} partial "
        f"{result.debug.get('partial_reasons', [])}"
    )
    print("-" * 64)
    for rep in result.reps:
        print(
            f"Rep {rep.number}: {rep.start_time:6.2f}s -> bottom {rep.bottom_time:6.2f}s -> "
            f"{rep.end_time:6.2f}s ({rep.duration:.2f}s)  "
            f"min knee {rep.min_knee_angle:6.1f}  torso max {rep.max_torso_lean:5.1f}  "
            f"heel {rep.max_heel_lift if rep.heel_reliable else float('nan'):.3f}  "
            f"end knee {rep.end_knee_angle:6.1f}"
        )
        print(
            f"          phases  descent {rep.descent_duration:.2f}s  "
            f"bottom {rep.bottom_duration:.2f}s  ascent {rep.ascent_duration:.2f}s   "
            f"shin@bottom {rep.shin_inclination_at_bottom:5.1f}  "
            f"L/R gap {rep.max_knee_asymmetry:5.1f} "
            f"({'usable' if rep.symmetry_reliable else 'not usable'})  "
            f"valid frames {rep.valid_frame_ratio:.0%}"
        )
    print("-" * 64)
    for rule in result.rule_results:
        marks = " ".join(
            f"r{o.rep_number}:{o.status.value[0].upper() if o.status is not RuleStatus.NOT_EVALUABLE else '?'}"
            for o in rule.per_rep
        )
        print(f"{rule.rule_id:20s} {rule.status.value:14s} " f"[{rule.reliability.value:13s}] {marks}")
        print(f"    phase: {rule.phase}   views: {', '.join(rule.supported_views)}")
        print(f"    {rule.explanation}")
        print(f"    thresholds: {rule.evidence}")
        persistence = " ".join(
            f"r{o.rep_number}:{o.violating_frames}/{o.phase_frames}" for o in rule.per_rep
        )
        if persistence.strip():
            print(f"    violating frames: {persistence}")
    print("-" * 64)
    for line in result.summary.overview:
        print(f"  {line}")
    for item in result.summary.not_assessed:
        print(f"  NOT ASSESSED - {item.title}: {item.reason}")
    print("-" * 64)
    print(
        f"Score: {result.summary.score}  "
        f"(reliability: {result.summary.reliability.value})  "
        f"({result.summary.score_formula})"
    )
    print(f"Annotated video: {result.annotated_video_path}")
    if "export_dir" in result.debug:
        print(f"Export bundle:   {result.debug['export_dir']}")

    if args.frames and result.annotated_video_path:
        _save_key_frames(result)
    return 0


def _save_key_frames(result) -> None:
    """Save each rep's start, bottom and end frame from the annotated clip."""
    import cv2

    export_dir = Path(result.debug.get("export_dir", result.annotated_video_path.parent))
    capture = cv2.VideoCapture(str(result.annotated_video_path))
    try:
        for rep in result.reps:
            for tag, frame_idx in (
                ("start", rep.start_frame),
                ("bottom", rep.bottom_frame),
                ("end", rep.end_frame),
            ):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = capture.read()
                if ok:
                    out = export_dir / f"rep{rep.number}_{tag}_f{frame_idx}.jpg"
                    cv2.imwrite(str(out), frame)
                    print(f"  saved {out}")
    finally:
        capture.release()


if __name__ == "__main__":
    sys.exit(main())
