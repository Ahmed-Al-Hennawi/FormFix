"""
Renders the annotated result video. This file decides what goes on each frame
and overlay.py decides how it looks, so I can restyle it without touching the
analysis. Coordinates always come from the cleaned pose.xy, not raw detections.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import filters
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

# skeleton joints plus one head marker - eyes, ears and mouth are tracked but
# no rule uses them
DRAWN_LANDMARKS: frozenset[int] = frozenset(
    {landmark for segment in SEGMENTS for landmark in segment} | {HEAD_LANDMARK}
)

# same floor as the analysis, so the overlay never draws a point the
# measurements already threw away
MIN_DRAW_VISIBILITY = 0.40

# top of the confidence range the gates below work in
FULL_DRAW_VISIBILITY = 0.95

# --- Hysteresis (stops the skeleton flickering) ---
# One yes/no threshold on a noisy signal made limbs blink, so each gate needs
# more to change its answer than to keep it, like a thermostat.

# a drawn landmark is only dropped once it falls to this
DRAW_RELEASE_VISIBILITY = 0.30

# once the answer changes it holds for at least this many frames
MIN_DRAW_DWELL_FRAMES = 5

# a landmark that drops out is held at its last position for this many frames.
# A short gap is detector noise, and since a link needs both ends, one missing
# wrist would take the whole forearm with it.
MAX_HOLD_FRAMES = 4

# orientations where one side of the body sits behind the other
SAGITTAL_ORIENTATIONS: frozenset[str] = frozenset({"side", "diagonal_side"})

# how confident a camera-far landmark has to be before it is drawn. MediaPipe
# still gives positions for limbs behind the torso but they are guesses.
MIN_FAR_SIDE_VISIBILITY = 0.90

# how close (in torso lengths) a far landmark has to sit to its near twin to
# count as occluded. MediaPipe's visibility really means "inside the frame", so
# a knee behind the other leg still scores 0.8+. 0.12 is a bit under a thigh.
FAR_OCCLUSION_TORSO_FRACTION = 0.12

# a far limb that is occluded or unsure on this share of tracked frames is left
# out of the whole video. Deciding it per frame made the far knee blink.
FAR_HIDE_CLIP_FRACTION = 0.5

# far landmark -> its near twin, for the occlusion test
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

# biggest believable one-frame jump, in torso lengths at 30 fps (scaled by fps
# below). Anything bigger means the detector swapped the point.
MAX_DRAW_JUMP_TORSOS = 0.28

# after this many rejections in a row, accept anyway, else a camera cut would
# suppress the landmark for the rest of the clip
MAX_CONSECUTIVE_JUMP_REJECTS = 3

# when two findings share a frame the worse one wins
_LEVEL_RANK: dict[str, int] = {"info": 0, "pass": 0, "warning": 1, "fail": 2}

# --- Display-only smoothing ---
# Only a copy of the coordinates is filtered, for drawing. The whole clip is
# known by render time, so I can use a zero-phase filter (no lag at the turning
# points), which the causal EMA in smoothing.py can't do. Settings are from
# filters.BUTTERWORTH_2HZ (Dill et al., 2024).
RENDER_SMOOTHING_CUTOFF_HZ = 2.0
RENDER_SMOOTHING_ORDER = 4

# --- What the skeleton is made of, per exercise ---
# Drawing every landmark means drawing limbs no rule reads, in the parts the
# detector tracks worst (legs under a pulldown seat, feet behind a press bench).
# Jaiswal et al. (2023) also pick landmarks per exercise, and Kotte et al.
# (2024) found people prefer specific body parts highlighted over the whole
# skeleton lighting up.

# pulldown and press: shoulders, elbows, wrists and hips, cut off at the hip
UPPER_BODY_DRAWN: frozenset[int] = frozenset(
    {
        HEAD_LANDMARK,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
    }
)

# squat: torso and legs, no arms - they are only holding the bar
LOWER_BODY_DRAWN: frozenset[int] = frozenset(
    {
        HEAD_LANDMARK,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE,
        LEFT_HEEL,
        RIGHT_HEEL,
        LEFT_FOOT_INDEX,
        RIGHT_FOOT_INDEX,
    }
)


@dataclass(frozen=True)
class OverlayEvent:
    """A finding shown over a range of frames, with the joints it came from."""

    start_frame: int
    end_frame: int
    text: str
    level: str = "warning"  # pass | warning | fail | info
    highlight_landmarks: tuple[int, ...] = ()
    # two or three words for the label by the body; defaults to text before the dash
    short_text: str = ""

    @property
    def label(self) -> str:
        return self.short_text or self.text.split(" - ")[0].strip()


# rest phases. Matched on the string, not the enum, so one renderer works for
# all three exercises
NEUTRAL_PHASE_VALUES: frozenset[str] = frozenset({"standing", "top", "ready"})


@dataclass(frozen=True)
class HudLine:
    """One chip of the heads-up display."""

    text: str
    colour: tuple[int, int, int] = CYAN
    bold: bool = False


@dataclass(frozen=True)
class FrameState:
    """What the HUD shows on one frame. Each exercise has its own phase enum and
    only .value gets printed."""

    phase: Any = None
    rep_number: int = 0
    total_reps: int = 0
    # true inside the turning-point window (squat bottom, press lockout)
    is_bottom_window: bool = False
    # extra chips from the exercise, drawn under the phase
    extra: tuple[HudLine, ...] = field(default_factory=tuple)

    @property
    def phase_label(self) -> str:
        value = getattr(self.phase, "value", self.phase)
        return str(value or "unknown")

    @property
    def inside_repetition(self) -> bool:
        """True mid-rep. Measured joints go lime here and stay cyan at rest."""
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
    drawn_landmarks: frozenset[int] | None = None,
) -> Path:
    """
    Render the annotated video and return its path (H.264 where possible).
    Anything exercise-specific comes in as a parameter instead of an if/else:

        emphasis_landmarks  the focus chain (leg chain, one arm, or both arms)
        marker_landmarks    the turning-point marker and its caption
        angle_joints        the joint angles this exercise's rules read
        camera_orientation  sagittal views hold the far limbs to a higher bar
        frame_range         inclusive range of ORIGINAL frames to write
        drawn_landmarks     which landmarks make up the skeleton at all
    """
    # two styles on purpose: the skeleton is sized to the athlete, the HUD and
    # watermark to the frame
    style = OverlayStyle.for_frame(video.width, video.height, _body_pixels(pose, video))
    chrome = OverlayStyle.for_frame(video.width, video.height)
    canvas = OverlayCanvas(video.width, video.height, style)
    intermediate = output_path.with_name(output_path.stem + "_raw.mp4")

    # which landmarks to draw, and a smoothed copy of the coordinates just for
    # drawing (the analysis pose is left alone)
    drawn = (DRAWN_LANDMARKS if drawn_landmarks is None else frozenset(drawn_landmarks)) | {
        HEAD_LANDMARK
    }
    render_pose = replace(pose, xy=_render_coordinates(pose, video, drawn))

    focus = frozenset(
        emphasis_landmarks if emphasis_landmarks is not None else SIDE_LANDMARKS.get(analysis_side, ())
    )
    policy = _draw_policy(pose, video, analysis_side, camera_orientation, focus)

    by_frame_events: dict[int, list[OverlayEvent]] = {}
    for event in events:
        for f in range(event.start_frame, event.end_frame + 1):
            by_frame_events.setdefault(f, []).append(event)

    # render window in original frame numbers; frames outside it are skipped
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
            if _has_drawable_pose(render_pose, index):
                points, alphas = _drawable_points(render_pose, index, video, policy, drawn)
                _draw_analysis_layer(canvas, points, alphas, style, state, active, focus, angle_joints)
            _draw_readout(canvas, chrome, state, metric, active)
            canvas.commit()

            writer.write(frame)
            written += 1
            if on_frame is not None:
                # count rendered frames so a trimmed run still fills the progress bar
                on_frame(written - 1, expected_frames)

    if convert_to_h264(intermediate, output_path):
        intermediate.unlink(missing_ok=True)
        return output_path

    # no FFmpeg or it failed - keep the mp4v file rather than lose the run
    logger.warning("Returning mp4v annotated video (H.264 conversion unavailable)")
    return intermediate


def _resolve_range(frame_range: tuple[int, int] | None, frame_count: int) -> tuple[int, int]:
    """Clamp the render window to real frames. Anything unusable falls back to the
    whole video, since an empty output looks like a broken analysis."""
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

    # I took the angle read-outs and issue label off the frame, but angle_joints
    # stays so each exercise still declares what it measures
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
    """Draw joint angles on the joints: warning colour if a finding came from that
    joint, lime mid-rep, cyan at rest."""
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
    """Short label beside the highlighted joints - one at most, never over the face."""
    for event in active:
        marked = [points[lm] for lm in event.highlight_landmarks if lm in points]
        if not marked:
            continue
        offset = int(round(18 * style.scale))
        y = int(sum(point[1] for point in marked) / len(marked))
        # right of the joints, or left if there isn't room
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
    """Rep counter and watermark. I dropped the phase and measurement chips because
    they just repeated the results panel."""
    lines: list[tuple[str, tuple[int, int, int], bool]] = []
    if state.total_reps:
        lines.append((f"REP {max(state.rep_number, 0)} / {state.total_reps}", WHITE, True))

    if lines:
        draw_hud(canvas, lines, style)
    draw_watermark(canvas, style)


# --- Landmark selection ---


def _body_pixels(pose: FramePoseData, video: VideoMetadata) -> float:
    """Typical torso length in pixels, used to size the overlay to the person."""
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
    """False when visibility is all zero (synthetic test data), otherwise gating on
    it would hide the whole skeleton."""
    visibility = pose.visibility
    return bool(
        visibility.size and np.isfinite(visibility).any() and float(np.nanmax(visibility)) > 0.0
    )


def _has_drawable_pose(pose: FramePoseData, index: int) -> bool:
    """
    Is there anything worth drawing on this frame? Uses pose.valid, not
    pose_found, because interpolated frames have good landmarks but pose_found
    is still False, and the skeleton vanished on them.
    """
    return bool(pose.valid[index].any())


@dataclass
class _DrawPolicy:
    """
    What the overlay may draw for this video, worked out once up front:

        trust_visibility  did the detector report per-landmark confidence at all
        far_limbs         limbs on the away-from-camera side, held to a higher bar
        max_jump_px       how far a landmark may move before it looks swapped
        hidden            limbs left out of the whole clip
    """

    trust_visibility: bool
    far_limbs: frozenset[int]
    max_jump_px: float
    # horizontal gap under which a far limb counts as hidden (0 = off)
    occlusion_px: float = 0.0
    hidden: frozenset[int] = frozenset()
    hold_frames: int = MAX_HOLD_FRAMES
    _last: dict[int, tuple[float, float, int]] = field(default_factory=dict)
    # landmark -> (currently drawn, frame the answer last changed)
    _state: dict[int, tuple[bool, int]] = field(default_factory=dict)
    # landmark -> (x, y, frame) of the last position actually drawn
    _held: dict[int, tuple[float, float, int]] = field(default_factory=dict)
    # how many landmarks the previous frame found, for the cut check
    _previous_drawn: int = 0

    def should_draw(self, landmark: int, frame_idx: int, confidence: float | None) -> bool:
        """
        Hysteresis check: MIN_DRAW_VISIBILITY to start drawing, down to
        DRAW_RELEASE_VISIBILITY to stop, and each answer holds for
        MIN_DRAW_DWELL_FRAMES. confidence=None means there is nothing to gate on.
        """
        if confidence is None:
            return True
        known = self._state.get(landmark)
        if known is None:
            drawn = confidence >= MIN_DRAW_VISIBILITY
            self._state[landmark] = (drawn, frame_idx)
            return drawn
        drawn, changed_at = known
        if frame_idx - changed_at < MIN_DRAW_DWELL_FRAMES:
            return drawn
        if drawn and confidence < DRAW_RELEASE_VISIBILITY:
            self._state[landmark] = (False, frame_idx)
            return False
        if not drawn and confidence >= MIN_DRAW_VISIBILITY:
            self._state[landmark] = (True, frame_idx)
            return True
        return drawn

    def continues_from_last_frame(self, drawn_now: int) -> bool:
        """Is this the same shot as the last frame? Losing a landmark or two is
        normal, losing half the figure at once means a cut."""
        before = self._previous_drawn
        self._previous_drawn = drawn_now
        if before <= 0:
            return True
        return drawn_now * 2 >= before

    def remember(self, landmark: int, frame_idx: int, x: float, y: float) -> None:
        self._held[landmark] = (x, y, frame_idx)

    def forget_holds(self) -> None:
        """Clear the held positions after a cut."""
        self._held.clear()

    def hold(self, landmark: int, frame_idx: int) -> tuple[float, float] | None:
        """Last drawn position of a landmark that just went missing, or None once
        it has been gone too long."""
        held = self._held.get(landmark)
        if held is None:
            return None
        x, y, seen = held
        if frame_idx - seen > self.hold_frames:
            return None
        return x, y

    def accept_position(self, landmark: int, frame_idx: int, x: float, y: float) -> bool:
        """Did the joint really move, or did the detector swap the point? The limit
        grows with frames since it was last drawn, and gives up after a few
        rejections in a row so a camera cut can recover."""
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
        Is this far-side landmark hidden behind its near-side twin? From the side
        the two land at almost the same x, so only the horizontal gap is checked.
        Returns False when it can't tell, so a limb isn't hidden on a guess.
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
    side, and the measured chain is never hidden (a press uses both arms even
    from the side).
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
    # only matters in a side view, where there is a far side
    occlusion_px = torso_px * FAR_OCCLUSION_TORSO_FRACTION if (torso_px > 0 and far_limbs) else 0.0

    logger.info(
        "Overlay policy: view=%s analysed=%s far-limbs=%d max-jump=%.1fpx occlusion=%.1fpx",
        camera_orientation or "unknown",
        analysis_side or "unknown",
        len(far_limbs),
        max_jump_px,
        occlusion_px,
    )
    policy = _DrawPolicy(
        trust_visibility=_visibility_is_informative(pose),
        far_limbs=far_limbs,
        max_jump_px=max_jump_px,
        occlusion_px=occlusion_px,
    )
    policy.hidden = _clip_hidden_far_limbs(pose, video, policy)
    if policy.hidden:
        logger.info("Overlay hides %d far landmark(s) for the whole clip", len(policy.hidden))
    return policy


def _clip_hidden_far_limbs(
    pose: FramePoseData, video: VideoMetadata, policy: _DrawPolicy
) -> frozenset[int]:
    """
    Which far-side limbs to leave out of the whole video. Deciding per frame made
    limbs blink, so a limb that is behind the body for most of the clip is never
    drawn and the rest are drawn normally.

    I err towards hiding because of Dill et al. (2024): with the far arm behind
    the torso, their reconstruction attached it to a jacket on a chair.
    """
    if not policy.far_limbs or not policy.trust_visibility:
        return frozenset()

    suspect: dict[int, int] = dict.fromkeys(policy.far_limbs, 0)
    tracked = 0
    width = float(video.width)
    for frame in range(pose.frame_count):
        valid = pose.valid[frame]
        if not valid.any():
            continue
        tracked += 1
        xy = pose.xy[frame]
        visibility = pose.visibility[frame]
        for landmark in policy.far_limbs:
            # untracked, hidden behind its twin, or too unsure - all mean don't draw
            if (
                landmark >= len(valid)
                or not valid[landmark]
                or policy.is_occluded(landmark, xy, valid, width)
                or float(visibility[landmark]) < MIN_FAR_SIDE_VISIBILITY
            ):
                suspect[landmark] += 1

    if tracked == 0:
        return frozenset()
    limit = tracked * FAR_HIDE_CLIP_FRACTION
    return frozenset(landmark for landmark, count in suspect.items() if count >= limit)


def _render_coordinates(
    pose: FramePoseData, video: VideoMetadata, drawn_landmarks: frozenset[int]
) -> np.ndarray:
    """
    Smoothed copy of the coordinates, for drawing only - pose.xy is untouched.
    A zero-phase Butterworth over each drawn landmark. The analysis EMA alone left
    the figure looking like it was floating.
    """
    xy = pose.xy.copy()
    fps = float(getattr(video, "fps", 0.0) or 30.0)
    for landmark in sorted(drawn_landmarks):
        if landmark >= xy.shape[1]:
            continue
        for axis in (0, 1):
            xy[:, landmark, axis] = filters.butterworth_lowpass(
                xy[:, landmark, axis],
                fps,
                cutoff_hz=RENDER_SMOOTHING_CUTOFF_HZ,
                order=RENDER_SMOOTHING_ORDER,
            )
    return xy


def _drawable_points(
    pose: FramePoseData,
    frame_idx: int,
    video: VideoMetadata,
    policy: _DrawPolicy,
    drawn_landmarks: frozenset[int] | None = None,
) -> tuple[dict[int, tuple[int, int]], dict[int, float]]:
    """
    Landmarks to draw on one frame, in pixels, with their opacity. Opacity is
    fixed - tying it to confidence made the figure look like it was breathing.
    """
    points: dict[int, tuple[int, int]] = {}
    alphas: dict[int, float] = {}
    wanted = DRAWN_LANDMARKS if drawn_landmarks is None else drawn_landmarks
    valid = pose.valid[frame_idx]
    xy = pose.xy[frame_idx]
    visibility = pose.visibility[frame_idx]

    fresh: dict[int, tuple[float, float]] = {}
    for landmark in range(xy.shape[0]):
        if landmark not in wanted or landmark in policy.hidden or not valid[landmark]:
            continue
        x, y = xy[landmark]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        confidence = float(visibility[landmark]) if policy.trust_visibility else None
        if not policy.should_draw(landmark, frame_idx, confidence):
            continue
        x_px = float(x) * video.width
        y_px = float(y) * video.height
        if not policy.accept_position(landmark, frame_idx, x_px, y_px):
            continue
        fresh[landmark] = (x_px, y_px)
        policy.remember(landmark, frame_idx, x_px, y_px)

    # holding only makes sense if this frame follows the last one. After a cut
    # the held positions would smear the old pose over the new shot
    if not policy.continues_from_last_frame(len(fresh)):
        policy.forget_holds()

    for landmark in sorted(set(wanted)):
        if landmark >= xy.shape[0] or landmark in policy.hidden:
            continue
        # a frame or two missing is the detector stuttering, so hold the point
        position = fresh.get(landmark) or policy.hold(landmark, frame_idx)
        if position is None:
            continue
        points[landmark] = (int(round(position[0])), int(round(position[1])))
        alphas[landmark] = 1.0
    return points, alphas
