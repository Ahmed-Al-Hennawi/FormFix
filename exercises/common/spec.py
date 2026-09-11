"""
The rule spec every exercise uses to declare its checks: what it measures, in
which phase, which camera views work, how much evidence it needs and which
feedback wording it uses. This keeps all thresholds in one config file, and
threshold_source records where each number came from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# --- camera views, matching analysis.models.CameraOrientation ---

VIEW_SIDE = "side"
VIEW_DIAGONAL = "diagonal_side"
VIEW_FRONTAL = "frontal"
VIEW_UNKNOWN = "unknown"

# side-on measurements (flexion, trunk lean, depth, lockout). Unknown is
# included on purpose - it's still analysed, just at lower reliability
SAGITTAL_VIEWS: tuple[str, ...] = (VIEW_SIDE, VIEW_DIAGONAL, VIEW_UNKNOWN)

# front-on measurements: left/right comparisons and horizontal alignment
FRONTAL_VIEWS: tuple[str, ...] = (VIEW_FRONTAL, VIEW_DIAGONAL, VIEW_UNKNOWN)

# measurements that track one landmark moving, so any angle works
ANY_VIEW: tuple[str, ...] = (VIEW_SIDE, VIEW_DIAGONAL, VIEW_FRONTAL, VIEW_UNKNOWN)

# default source for a technique threshold: my own value, not from literature
PROTOTYPE_THRESHOLD = (
    "operational prototype threshold - selected from the exercise definition, "
    "the FormFix reference demonstration and iterative testing; not a "
    "biomechanical constant"
)

# source for values that only tune the system, not the technique
ENGINEERING_THRESHOLD = "engineering stability setting - makes no claim about technique"


@dataclass(frozen=True)
class RuleSpec:
    """
    One technique check. acceptable_min / acceptable_max give the range (either
    can be None) and tolerance is how far past it is still a warning. There is
    no exact target, since no single number is right for every body.
    """

    name: str
    rule_id: str
    title: str
    # matches a per-rep / per-frame field name
    metric: str
    phase: str

    acceptable_min: float | None
    acceptable_max: float | None
    tolerance: float

    # a violation has to last this many frames in a row...
    minimum_persistence_frames: int
    # ...and cover at least this share of the phase
    min_violation_ratio: float
    # below this visibility the check says "cannot assess"
    minimum_visibility: float

    supported_views: tuple[str, ...]
    feedback_key: str
    # where the threshold values came from, shown in the technical details
    threshold_source: str = PROTOTYPE_THRESHOLD

    def as_dict(self) -> dict:
        return asdict(self)
