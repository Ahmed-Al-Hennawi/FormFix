"""
Exercise analysers - squat, lat pulldown and dumbbell shoulder press, each built
on the shared code in exercises/common/.

The UI looks up analysers and feedback here, so adding an exercise means a new
package plus one entry. They load lazily because importing an analyser pulls in
OpenCV, and some modules only need the config.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

# exercise id -> display name, needed for failure messages before any result exists
DISPLAY_NAMES: dict[str, str] = {
    "squat": "Squat",
    "pulldown": "Lat Pulldown",
    "press": "Shoulder Press",
}

_CACHE: dict[str, Any] = {}


def _analysers() -> dict[str, Callable]:
    """exercise id -> analyser function (all three have the same signature)."""
    from .press.analyser import analyse_shoulder_press
    from .pulldown.analyser import analyse_lat_pulldown
    from .squat.analyser import analyse_squat

    return {
        "squat": analyse_squat,
        "pulldown": analyse_lat_pulldown,
        "press": analyse_shoulder_press,
    }


def _feedback() -> dict[str, ModuleType]:
    """exercise id -> its feedback module (build_findings and FEEDBACK_TEMPLATES)."""
    from .press import feedback as press_feedback
    from .pulldown import feedback as pulldown_feedback
    from .squat import feedback as squat_feedback

    return {
        "squat": squat_feedback,
        "pulldown": pulldown_feedback,
        "press": press_feedback,
    }


_BUILDERS: dict[str, Callable[[], Any]] = {
    "ANALYSERS": _analysers,
    "FEEDBACK": _feedback,
}


def __getattr__(name: str) -> Any:
    """Build ANALYSERS / FEEDBACK on first access, then cache it."""
    if name in _BUILDERS:
        if name not in _CACHE:
            _CACHE[name] = _BUILDERS[name]()
        return _CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_BUILDERS])


def has_analyser(exercise_id: str) -> bool:
    return exercise_id in DISPLAY_NAMES
