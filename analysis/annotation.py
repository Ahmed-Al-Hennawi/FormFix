"""
Renders the annotated result video. This module decides what goes on each
frame; overlay.py handles how it is drawn, so the styling can change without
touching the analysis.

Per frame: a thin cyan skeleton, the focus chain heavier and lit, the joints an
active finding was measured from ringed in the warning colour, and a small HUD.
Coordinates come from the cleaned pose.xy, never the raw detections.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    SIDE_LANDMARKS,
    FramePoseData,
    VideoMetadata,
)
from .overlay import (
    AMBER,
    CYAN,
    DEEP_SPACE,
    HEAD_LANDMARK,
    LEVEL_COLOURS,
    LIME,
    RED,
    SEGMENTS,
    SIDE_LIMB_LANDMARKS,
    WHITE,
    JointAngle,
    OverlayCanvas,
    OverlayStyle,
    draw_analysis_label,
    draw_formfix_pose,
    draw_hud,
    draw_joint_angle,
    draw_turning_point_marker,
    draw_watermark,
    highlight_issue_region,
)
from .video_processor import convert_to_h264, iter_frames, open_video, open_writer

logger = logging.getLogger(__name__)

__all__ = [
    "AMBER",
    "CYAN",
    "DEEP_SPACE",
    "LEVEL_COLOURS",
    "LIME",
    "RED",
    "WHITE",
    "FrameState",
    "HudLine",
    "JointAngle",
    "OverlayEvent",
    "render_annotated_video",
]

# what the overlay draws: the skeleton joints plus one head marker. Eyes,
# ears and mouth are tracked but no rule reads them.
DRAWN_LANDMARKS: frozenset[int] = frozenset(
    {landmark for segment in SEGMENTS for landmark in segment} | {HEAD_LANDMARK}
)

# below this confidence, draw nothing. Matched to the analysis floor so the
# overlay can't draw points the measurements already threw away.
MIN_DRAW_VISIBILITY = 0.40

# full-strength visibility; below it a landmark fades in. Has to stay above
# MIN_FAR_SIDE_VISIBILITY or the ramp is dead code for far limbs.
FULL_DRAW_VISIBILITY = 0.95

# Opacity floor for a landmark that is drawn but only just trusted.
FAINT_ALPHA = 0.38

# orientations where one side of the body sits behind the other
SAGITTAL_ORIENTATIONS: frozenset[str] = frozenset({"side", "diagonal_side"})

# how confident a camera-far landmark has to be before it is drawn at all.
# MediaPipe still emits positions for limbs behind the torso, usually 0.2-0.6,
# that swing about because they are guesses.
# drawn at all. MediaPipe still emits positions for the limbs behind the torso,
MIN_FAR_SIDE_VISIBILITY = 0.90

# how close a far landmark has to sit to its near twin, in torso lengths,
# before we call it occluded. This is the check that works side-on: MediaPipe's
# visibility is really "is it inside the frame", so a knee behind the other leg
# still scores 0.8+. 0.12 of a torso is a bit under a thigh's width.
FAR_OCCLUSION_TORSO_FRACTION = 0.12

# far landmark -> its near twin, for the occlusion test. Limbs only.
_MIRROR_LANDMARK: dict[int, int] = {
    LEFT_ELBOW: RIGHT_ELBOW,
    RIGHT_ELBOW: LEFT_ELBOW,
    LEFT_WRIST: RIGHT_WRIST,
    RIGHT_WRIST: LEFT_WRIST,
    LEFT_KNEE: RIGHT_KNEE,
    RIGHT_KNEE: LEFT_KNEE,
    LEFT_ANKLE: RIGHT_ANKLE,
    RIGHT_ANKLE: LEFT_ANKLE,
    LEFT_HEEL: RIGHT_HEEL,
    RIGHT_HEEL: LEFT_HEEL,
    LEFT_FOOT_INDEX: RIGHT_FOOT_INDEX,
    RIGHT_FOOT_INDEX: LEFT_FOOT_INDEX,
    LEFT_EAR: RIGHT_EAR,
    RIGHT_EAR: LEFT_EAR,
}

# biggest believable single-frame jump for one landmark, in torso lengths at
# 30 fps. Beyond it the detector has swapped the point. Scaled by fps below.
MAX_DRAW_JUMP_TORSOS = 0.28

# after this many rejections in a row, accept anyway, else a camera cut would
# suppress the landmark for the rest of the clip
MAX_CONSECUTIVE_JUMP_REJECTS = 3

# Severity order when two findings land on the same frame; the worse one wins.
_LEVEL_RANK: dict[str, int] = {"info": 0, "pass": 0, "warning": 1, "fail": 2}


@dataclass(frozen=True)
class OverlayEvent:
    """A finding shown over a window of frames, plus the joints it was measured
    from, so the overlay can point at the right part of the body."""

    start_frame: int
    end_frame: int
    text: str
    level: str = "warning"  # pass | warning | fail | info
    # Landmark indices to emphasise while this event is on screen.
    highlight_landmarks: tuple[int, ...] = ()
    # two or three words for the label by the body; defaults to text before the dash
    short_text: str = ""

    @property
    def label(self) -> str:
        return self.short_text or self.text.split(" - ")[0].strip()


# phases where the athlete is at rest. Matching on the string rather than the
# enum lets one renderer handle all three exercises.
NEUTRAL_PHASE_VALUES: frozenset[str] = frozenset({"standing", "top", "ready"})


@dataclass(frozen=True)
class HudLine:
    """One chip of the heads-up display."""

    text: str
    colour: tuple[int, int, int] = CYAN
    bold: bool = False


@dataclass(frozen=True)
class FrameState:
    """What the HUD says on one frame. phase is any string enum - each exercise
    names its own phases and the renderer only prints .value."""

    phase: Any = None
    rep_number: int = 0
    total_reps: int = 0
    # true inside the turning-point window (squat bottom, press lockout)
    is_bottom_window: bool = False
    # Extra HUD chips from the exercise, drawn under the phase.
    extra: tuple[HudLine, ...] = field(default_factory=tuple)

    @property
    def phase_label(self) -> str:
        value = getattr(self.phase, "value", self.phase)
        return str(value or "unknown")

    @property
    def inside_repetition(self) -> bool:
        """True while the athlete is actually in a rep. Measured joints go lime here
        and stay cyan at rest, so green means "in the movement, nothing flagged"."""
        return self.rep_number > 0 and self.phase_label not in NEUTRAL_PHASE_VALUES


def render_annotated_video(
    video: VideoMetadata,
    pose: FramePoseData,
    metrics: Sequence[Any],
    frame_states: Sequence[FrameState],
    events: Sequence[OverlayEvent],
    analysis_side: str,
    output_path: Path,
    *,
    camera_orientation: str = "",
    on_frame: Callable[[int, int], None] | None = None,
    emphasis_landmarks: tuple[int, ...] | None = None,
    marker_landmarks: tuple[int, ...] | None = None,
    marker_label: str = "BOTTOM",
    angle_joints: Sequence[JointAngle] = (),
    frame_range: tuple[int, int] | None = None,
) -> Path:
    """
    Render the annotated video and return the final path (H.264 where possible).
    Exercise-specific choices come in as parameters rather than branches:

        emphasis_landmarks  the focus chain (leg chain, one arm, or both arms)
        marker_landmarks    the turning-point marker and its caption
        angle_joints        the joint angles this exercise's rules read
        camera_orientation  sagittal views hold the far limbs to a higher bar
        frame_range         inclusive range of ORIGINAL frames to write
    """
    # two styles on purpose: things drawn on the athlete are sized from the
    # athlete, the HUD and watermark stay sized to the frame
    style = OverlayStyle.for_frame(video.width, video.height, _body_pixels(pose, video))
    chrome = OverlayStyle.for_frame(video.width, video.height)
    canvas = OverlayCanvas(video.width, video.height, style)
    intermediate = output_path.with_name(output_path.stem + "_raw.mp4")

    focus = frozenset(
        emphasis_landmarks if emphasis_landmarks is not None else SIDE_LANDMARKS.get(analysis_side, ())
    )
    policy = _draw_policy(pose, video, analysis_side, camera_orientation, focus)

    by_frame_events: dict[int, list[OverlayEvent]] = {}
    for event in events:
        for f in range(event.start_frame, event.end_frame + 1):
            by_frame_events.setdefault(f, []).append(event)

    # render window in original frame numbers. Frames outside it are decoded but
    # not drawn on or written.
    first_frame, last_frame = _resolve_range(frame_range, pose.frame_count)
    expected_frames = max(1, last_frame - first_frame + 1)
    written = 0

    with (
        open_video(video.path) as capture,
        open_writer(intermediate, video.fps, (video.width, video.height)) as writer,
    ):
        for index, frame in iter_frames(capture):
            if index >= pose.frame_count or index > last_frame:
                break
            if index < first_frame:
                continue
            state = frame_states[index] if index < len(frame_states) else FrameState()
            metric = metrics[index] if index < len(metrics) else None
            active = by_frame_events.get(index, ())

            canvas.begin(frame)
            if _has_drawable_pose(pose, index):
                points, alphas = _drawable_points(pose, index, video, policy)
                _draw_analysis_layer(canvas, points, alphas, style, state, active, focus, angle_joints)
                # Turning-point marker stays off; only the rep counter is on the frame now.
            _draw_readout(canvas, chrome, state, metric, active)
            canvas.commit()

            writer.write(frame)
            written += 1
            if on_frame is not None:
                # Progress counts frames actually rendered, so a trimmed run still fills the bar.
                on_frame(written - 1, expected_frames)

    if convert_to_h264(intermediate, output_path):
        intermediate.unlink(missing_ok=True)
        return output_path

    # No FFmpeg, or it failed: keep the mp4v file rather than lose the run.
    logger.warning("Returning mp4v annotated video (H.264 conversion unavailable)")
    return intermediate


def _resolve_range(frame_range: tuple[int, int] | None, frame_count: int) -> tuple[int, int]:
    """Clamp a requested render window to frames that exist. Anything unusable
    falls back to the whole video - an empty output file looks like a broken
    analysis, which is worse than not trimming."""
    last_index = max(0, frame_count - 1)
    if frame_range is None:
        return 0, last_index
    try:
        first, last = int(frame_range[0]), int(frame_range[1])
    except (TypeError, ValueError, IndexError):
        logger.warning("Unusable frame_range %r; rendering the whole video", frame_range)
        return 0, last_index
    if last < first:
        first, last = last, first
    first = max(0, min(first, last_index))
    last = max(0, min(last, last_index))
    if last < first:
        return 0, last_index
    return first, last


# --- Frame composition ---


def _draw_analysis_layer(
    canvas: OverlayCanvas,
    points: dict[int, tuple[int, int]],
    alphas: dict[int, float],
    style: OverlayStyle,
    state: FrameState,
    active: Sequence[OverlayEvent],
    focus: frozenset[int],
    angle_joints: Sequence[JointAngle],
) -> None:
    """The body layer: skeleton, issue highlights, joint angles."""
    flagged: set[int] = set()
    level = "info"
    for event in active:
        marked = [lm for lm in event.highlight_landmarks if lm in points]
        flagged.update(marked)
        if event.level == "fail" or (event.level == "warning" and level != "fail"):
            level = event.level
    colour = LEVEL_COLOURS.get(level, AMBER)

    draw_formfix_pose(
        canvas,
        points,
        alphas,
        style,
        focus=focus,
        flagged=flagged,
        inside_rep=state.inside_repetition,
    )

    for event in active:
        highlight_issue_region(
            canvas,
            points,
            event.highlight_landmarks,
            style,
            colour=LEVEL_COLOURS.get(event.level, AMBER),
        )

    # angle read-outs and the issue label are off the frame now, but angle_joints
    # is still accepted so each exercise keeps declaring what it measures
    _ = (angle_joints, colour)


def _draw_angles(
    canvas: OverlayCanvas,
    points: dict[int, tuple[int, int]],
    style: OverlayStyle,
    angle_joints: Sequence[JointAngle],
    flagged: set[int],
    flag_colour,
    inside_rep: bool,
) -> None:
    """Draw the rules' joint angles on the joints. Colour means what it does
    everywhere else: warning colour if a finding came from this joint, lime
    mid-rep, cyan at rest."""
    for spec in angle_joints:
        proximal = points.get(spec.proximal)
        vertex = points.get(spec.vertex)
        distal = points.get(spec.distal)
        if proximal is None or vertex is None or distal is None:
            continue
        colour = flag_colour if spec.vertex in flagged else (LIME if inside_rep else CYAN)
        draw_joint_angle(canvas, proximal, vertex, distal, style, colour=colour, arc=spec.arc)


def _label_issue_region(
    canvas: OverlayCanvas,
    points: dict[int, tuple[int, int]],
    style: OverlayStyle,
    active: Sequence[OverlayEvent],
    colour,
) -> None:
    """Two or three words beside the highlighted joints. One label at most and
    never over the face; the pill at the bottom has the fuller wording."""
    for event in active:
        marked = [points[lm] for lm in event.highlight_landmarks if lm in points]
        if not marked:
            continue
        offset = int(round(18 * style.scale))
        y = int(sum(point[1] for point in marked) / len(marked))
        # normally sit to the right of the joints, flip left when there isn't room
        right = max(point[0] for point in marked) + offset
        if right > canvas.width * 0.72:
            anchor, align = (min(point[0] for point in marked) - offset, y), "right"
        else:
            anchor, align = (right, y), "left"
        draw_analysis_label(
            canvas,
            event.label,
            anchor,
            style,
            colour=colour,
            size=style.text_small,
            accent=True,
            align=align,
        )
        return


def _draw_marker(
    canvas: OverlayCanvas,
    points: dict[int, tuple[int, int]],
    marker_landmarks: tuple[int, ...],
    label: str,
    style: OverlayStyle,
) -> None:
    """Turning-point marker, at whichever landmarks the exercise names."""
    marked = [points[lm] for lm in marker_landmarks if lm in points]
    if not marked:
        return
    anchor = (
        int(sum(p[0] for p in marked) / len(marked)),
        int(sum(p[1] for p in marked) / len(marked)),
    )
    draw_turning_point_marker(canvas, anchor, label, style)


def _draw_readout(
    canvas: OverlayCanvas,
    style: OverlayStyle,
    state: FrameState,
    metric: Any,
    active: Sequence[OverlayEvent],
) -> None:
    """On-screen chrome: rep counter and watermark. The phase and measurement chips
    only repeated what the results panel says in plain language."""
    lines: list[tuple[str, tuple[int, int, int], bool]] = []
    if state.total_reps:
        lines.append((f"REP {max(state.rep_number, 0)} / {state.total_reps}", WHITE, True))

    if lines:
        draw_hud(canvas, lines, style)
    draw_watermark(canvas, style)


# --- Landmark selection ---


def _body_pixels(pose: FramePoseData, video: VideoMetadata) -> float:
    """Typical torso length in pixels, used only to size the overlay so someone
    further from the camera gets a smaller skeleton."""
    shoulders = (LEFT_SHOULDER, RIGHT_SHOULDER)
    hips = (LEFT_HIP, RIGHT_HIP)
    valid = pose.valid[:, [*shoulders, *hips]].all(axis=1)
    if not valid.any():
        return 0.0
    xy = pose.xy[valid]
    shoulder = xy[:, shoulders, :].mean(axis=1)
    hip = xy[:, hips, :].mean(axis=1)
    dx = (shoulder[:, 0] - hip[:, 0]) * video.width
    dy = (shoulder[:, 1] - hip[:, 1]) * video.height
    lengths = np.hypot(dx, dy)
    lengths = lengths[np.isfinite(lengths)]
    return float(np.median(lengths)) if lengths.size else 0.0


def _visibility_is_informative(pose: FramePoseData) -> bool:
    """Does the visibility track carry anything? Synthetic fixtures leave it all
    zero, and gating on that would hide the whole skeleton."""
    visibility = pose.visibility
    return bool(
        visibility.size and np.isfinite(visibility).any() and float(np.nanmax(visibility)) > 0.0
    )


def _has_drawable_pose(pose: FramePoseData, index: int) -> bool:
    """
    Is there anything worth drawing on this frame?

    Not pose.pose_found[index] - that says whether the detector fired on the raw
    frame, but short gaps are interpolated afterwards, so a frame can hold good
    landmarks with the flag still False. Gating on it made the skeleton vanish on
    exactly those frames.
    """
    return bool(pose.valid[index].any())


@dataclass
class _DrawPolicy:
    """
    What the overlay may draw for this video, worked out once up front:

        trust_visibility  did the detector report per-landmark confidence at all
        far_limbs         limbs on the away-from-camera side, held to a higher bar
        max_jump_px       how far a landmark may move before it looks swapped
    """

    trust_visibility: bool
    # Landmarks held to the far-side floor - limbs only.
    far_limbs: frozenset[int]
    max_jump_px: float
    # horizontal distance under which a far limb counts as hidden. 0 turns it off.
    occlusion_px: float = 0.0
    _last: dict[int, tuple[float, float, int]] = field(default_factory=dict)

    def accept_position(self, landmark: int, frame_idx: int, x: float, y: float) -> bool:
        """Did this joint move, or did the detector swap the point? The allowance grows
        with the frames since it was last drawn, and after a few rejections in a row
        the gate lapses so a camera cut can recover."""
        if self.max_jump_px <= 0:
            return True
        previous = self._last.get(landmark)
        if previous is not None:
            last_x, last_y, last_frame = previous
            elapsed = frame_idx - last_frame
            if 0 < elapsed <= MAX_CONSECUTIVE_JUMP_REJECTS:
                travel = math.hypot(x - last_x, y - last_y)
                if travel > self.max_jump_px * elapsed:
                    return False
        self._last[landmark] = (x, y, frame_idx)
        return True

    def is_occluded(
        self,
        landmark: int,
        xy: np.ndarray,
        valid: np.ndarray,
        width: float,
    ) -> bool:
        """
        Is this far-side landmark hidden behind its near-side twin?

        Filmed from the side, a limb behind the body projects onto the one in front,
        so the two land at nearly the same horizontal position. Only the horizontal
        gap is compared - vertical separation is real movement.

        False whenever the question can't be answered, since the other gates are a
        better failure mode than hiding a limb on a guess.
        """
        if self.occlusion_px <= 0:
            return False
        twin = _MIRROR_LANDMARK.get(landmark)
        if twin is None:
            return False
        if twin >= len(valid) or not valid[twin]:
            return False
        near_x, far_x = float(xy[twin][0]), float(xy[landmark][0])
        if not (math.isfinite(near_x) and math.isfinite(far_x)):
            return False
        return abs(far_x - near_x) * width < self.occlusion_px


def _draw_policy(
    pose: FramePoseData,
    video: VideoMetadata,
    analysis_side: str,
    camera_orientation: str,
    focus: frozenset[int] = frozenset(),
) -> _DrawPolicy:
    """
    Build the drawing policy for one video. The far side comes from the analysed
    side, not a guess. Nothing in the exercise's measured chain is ever
    suppressed - a press is measured from both arms even filmed from the side.
    """
    far_limbs: frozenset[int] = frozenset()
    sagittal = (camera_orientation or "").lower() in SAGITTAL_ORIENTATIONS
    other = {"left": "right", "right": "left"}.get((analysis_side or "").lower())
    if other is not None and sagittal:
        far_limbs = SIDE_LIMB_LANDMARKS[other] - frozenset(focus)

    torso_px = _body_pixels(pose, video)
    fps = video.fps if getattr(video, "fps", 0) else 30.0
    frame_rate_factor = min(4.0, max(0.5, 30.0 / fps)) if fps > 0 else 1.0
    max_jump_px = torso_px * MAX_DRAW_JUMP_TORSOS * frame_rate_factor if torso_px > 0 else 0.0
    # Only meaningful where there is a far side to test, i.e. a sagittal view.
    occlusion_px = torso_px * FAR_OCCLUSION_TORSO_FRACTION if (torso_px > 0 and far_limbs) else 0.0

    logger.info(
        "Overlay policy: view=%s analysed=%s far-limbs=%d max-jump=%.1fpx occlusion=%.1fpx",
        camera_orientation or "unknown",
        analysis_side or "unknown",
        len(far_limbs),
        max_jump_px,
        occlusion_px,
    )
    return _DrawPolicy(
        trust_visibility=_visibility_is_informative(pose),
        far_limbs=far_limbs,
        max_jump_px=max_jump_px,
        occlusion_px=occlusion_px,
    )


def _drawable_points(
    pose: FramePoseData,
    frame_idx: int,
    video: VideoMetadata,
    policy: _DrawPolicy,
) -> tuple[dict[int, tuple[int, int]], dict[int, float]]:
    """
    Landmarks to draw on one frame, in pixels, with their opacity. A landmark is
    dropped when it is invalid, not finite, under the visibility floor, or moved
    further in one frame than a joint can. Between the floor and full visibility
    it fades in.
    """
    points: dict[int, tuple[int, int]] = {}
    alphas: dict[int, float] = {}
    valid = pose.valid[frame_idx]
    xy = pose.xy[frame_idx]
    visibility = pose.visibility[frame_idx]

    for landmark in range(xy.shape[0]):
        if landmark not in DRAWN_LANDMARKS:
            continue
        if not valid[landmark]:
            continue
        x, y = xy[landmark]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        far = landmark in policy.far_limbs
        # geometry first: MediaPipe reports high confidence for a limb it inferred
        # behind the body, so the visibility gate below can't catch occlusion
        if (
            far
            and policy.trust_visibility
            and policy.is_occluded(landmark, xy, valid, float(video.width))
        ):
            continue

        alpha = 1.0
        if policy.trust_visibility:
            confidence = float(visibility[landmark])
            floor = MIN_FAR_SIDE_VISIBILITY if far else MIN_DRAW_VISIBILITY
            if confidence < floor:
                continue
            if confidence < FULL_DRAW_VISIBILITY:
                span = max(FULL_DRAW_VISIBILITY - floor, 1e-6)
                ramp = min(1.0, max(0.0, (confidence - floor) / span))
                alpha = FAINT_ALPHA + (1.0 - FAINT_ALPHA) * ramp

        x_px = float(x) * video.width
        y_px = float(y) * video.height
        if not policy.accept_position(landmark, frame_idx, x_px, y_px):
            continue

        points[landmark] = (int(round(x_px)), int(round(y_px)))
        alphas[landmark] = alpha
    return points, alphas
