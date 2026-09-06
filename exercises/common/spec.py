"""
The rule specification every exercise declares its checks with. A rule states
what it measures, which phase that means anything in, which camera views can
support it, how much evidence it needs and where its wording comes from. The
evaluation code just compares the measured value against the declared range.

Keeping it declarative means every threshold lives in one config file and a
finding can be traced from metric to sentence. threshold_source records where a
number came from, since these are operational values, not constants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# --- camera views, matching analysis.models.CameraOrientation ---

VIEW_SIDE = "side"
VIEW_DIAGONAL = "diagonal_side"
VIEW_FRONTAL = "frontal"
VIEW_UNKNOWN = "unknown"

# Side-on measurements: joint flexion in the plane of movement, trunk
# inclination, depth, lockout. Unknown is in the list deliberately - we still
# analyse it, at lower reliability, rather than discarding it.
SAGITTAL_VIEWS: tuple[str, ...] = (VIEW_SIDE, VIEW_DIAGONAL, VIEW_UNKNOWN)

# Frontal-plane measurements: left/right comparisons and horizontal alignment
# both need the camera looking across the body.
FRONTAL_VIEWS: tuple[str, ...] = (VIEW_FRONTAL, VIEW_DIAGONAL, VIEW_UNKNOWN)

# For measurements that only track one landmark moving, so any angle works.
ANY_VIEW: tuple[str, ...] = (VIEW_SIDE, VIEW_DIAGONAL, VIEW_FRONTAL, VIEW_UNKNOWN)

# Default provenance for a technique threshold: my value, not the field's.
PROTOTYPE_THRESHOLD = (
    "operational prototype threshold - selected from the exercise definition, "
    "the FormFix reference demonstration and iterative testing; not a "
    "biomechanical constant"
)

# Provenance for a value that only tunes the system, not the technique.
ENGINEERING_THRESHOLD = "engineering stability setting - makes no claim about technique"


@dataclass(frozen=True)
class RuleSpec:
    """
    One technique check. acceptable_min / acceptable_max bound the acceptable range
    (either can be None for a one-sided check) and tolerance is how far past that
    still counts as a warning, not a failure. There is no exact target
    anywhere, because no single number is right for every body.
    """

    name: str
    rule_id: str
    title: str
    # The measured quantity, matching a per-rep / per-frame field name.
    metric: str
    # The movement phase this check is evaluated on.
    phase: str

    acceptable_min: float | None
    acceptable_max: float | None
    # How far beyond the acceptable range still counts as a warning.
    tolerance: float

    # A violation must hold for this many consecutive measurable frames...
    minimum_persistence_frames: int
    # ...and cover at least this share of the phase's measurable frames.
    min_violation_ratio: float
    # Landmark visibility floor below which the check reports "cannot assess".
    minimum_visibility: float

    supported_views: tuple[str, ...]
    # Which feedback template produced the wording.
    feedback_key: str
    # Provenance of the threshold values, shown in the technical details.
    threshold_source: str = PROTOTYPE_THRESHOLD

    def as_dict(self) -> dict:
        return asdict(self)
