"""
JSON/CSV dump of one run, for the evaluation. Writes summary.json,
frame_metrics.csv and reps.csv into analysis_results/<run id>/.
Measurements and config only, nothing identifying.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import math
from enum import Enum
from pathlib import Path
from typing import Any

from .models import SquatAnalysisResult

logger = logging.getLogger(__name__)

EXPORT_ROOT_NAME = "analysis_results"


def _jsonable(value: Any) -> Any:
    """Make dataclasses / enums / paths / NaN JSON-safe."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 4)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalars
        return _jsonable(value.item())
    return value


def export_result(result: SquatAnalysisResult, root: Path, run_id: str) -> Path | None:
    """Write the JSON/CSV bundle; returns the run directory (or None on error)."""
    try:
        run_dir = root / EXPORT_ROOT_NAME / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "exercise": result.exercise,
            "success": result.success,
            "analysis_side": result.analysis_side,
            "video": _jsonable(result.video),
            "validation": _jsonable(result.validation),
            "summary": _jsonable(result.summary),
            "reps": _jsonable(result.reps),
            "rule_results": _jsonable(result.rule_results),
            "debug": _jsonable(result.debug),
        }
        # the temp path is not useful and is mildly identifying
        summary["video"].pop("path", None)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if result.frame_metrics:
            fields = [f.name for f in dataclasses.fields(result.frame_metrics[0])]
            with (run_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for fm in result.frame_metrics:
                    writer.writerow(dataclasses.asdict(fm))

        if result.reps:
            fields = [f.name for f in dataclasses.fields(result.reps[0])]
            with (run_dir / "reps.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for rep in result.reps:
                    writer.writerow(dataclasses.asdict(rep))

        logger.info("Exported analysis bundle to %s", run_dir)
        return run_dir
    except OSError as exc:  # pragma: no cover - export must never break a run
        logger.warning("Analysis export failed: %s", exc)
        return None
