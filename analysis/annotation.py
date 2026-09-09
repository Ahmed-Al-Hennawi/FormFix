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

# what the overlay draws: the skeleton joints plus one head marker. Eyes,
# ears and mouth are tracked but no rule reads them.
DRAWN_LANDMARKS: frozenset[int] = frozenset(
    {landmark for segment in SEGMENTS for landmark in segment} | {HEAD_LANDMARK}
)

# below this confidence, draw nothing. Matched to the analysis floor so the
# overlay can't draw points the measurements already threw away.
MIN_DRAW_VISIBILITY = 0.40

# full-strength visibility. Nothing fades any more - a landmark is drawn or it
# is not - but the constant still marks the top of the confidence range the
# gates below are written in.
FULL_DRAW_VISIBILITY = 0.95

# --- Hysteresis: what stopped the skeleton flickering ---
#
# Every gate below used to be a single per-frame yes/no test on a noisy signal,
# so a landmark sitting anywhere near a threshold switched on and off several
# times a second and the limb blinked. Each gate now takes more to change its
# answer than to keep it, the way a thermostat does.

# A landmark already being drawn is not dropped the moment it dips under
# MIN_DRAW_VISIBILITY; it has to fall this far below it.
DRAW_RELEASE_VISIBILITY = 0.30

# ...and once the answer has changed, it stands for at least this many frames.
MIN_DRAW_DWELL_FRAMES = 5

# A drawn landmark that goes missing is held at its last position for this many
# frames before it is dropped. A two-frame gap is detector noise, not the limb
# leaving the shot - and because a link needs both ends, one missing wrist used
# to take the whole forearm with it.
MAX_HOLD_FRAMES = 4

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

# A far limb is judged once for the whole clip rather than frame by frame: if it
# is occluded or under-confident on this fraction of the tracked frames, it stays
# out of the drawing for the entire video. Asked per frame, the question is
# answered by two noisy signals either side of a threshold, which is exactly what
# made the far knee blink through a squat descent.
FAR_HIDE_CLIP_FRACTION = 0.5

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

# --- Display-only smoothing ---
#
# The coordinates the rules read are left exactly as the analysis left them.
# These filter a copy, used for drawing only. Rendering happens once the whole
# clip is known, so the overlay can afford a zero-phase filter that the causal
# EMA in smoothing.py cannot be - and zero phase means no lag at the turning
# points, which is where a lagging skeleton looks worst. Parameters are the ones
# filters.BUTTERWORTH_2HZ documents (Dill et al., 2024).
RENDER_SMOOTHING_CUTOFF_HZ = 2.0
RENDER_SMOOTHING_ORDER = 4

# --- What the skeleton is made of, per exercise ---
#
# Drawing every landmark on every video means drawing limbs no rule reads, in
# the part of the frame the detector tracks worst: the legs under a pulldown
# seat, the feet behind a press bench. Jaiswal et al. (2023) select landmarks
# per exercise the same way ("the choice of landmarks depended on our
# understanding of the biomechanics of each exercise"), and Kotte et al. (2024)
# report that people find specific body parts highlighted more useful than the
# whole skeleton lighting up.

# Shoulders, elbows, wrists and hips: everything a pulldown or a press measures,
# cut off at the hip.
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

# Torso and both legs, no arms: the squat is measured from the hip, knee and
# ankle, and the arms are holding a bar or held out in front either way.
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
    drawn_landmarks: frozenset[int] | None = None,
) -> Path:
    """
    Render the annotated video and return the final path (H.264 where possible).
    Exercise-specific choices come in as parameters rather than branches:

        emphasis_landmarks  the focus chain (leg chain, one arm, or both arms)
        marker_landmarks    the turning-point marker and its caption
        angle_joints        the joint angles this exercise's rules read
        camera_orientation  sagittal views hold the far limbs to a higher bar
        frame_range         inclusive range of ORIGINAL frames to write
        drawn_landmarks     which landmarks make up the skeleton at all
    """
    # two styles on purpose: things drawn on the athlete are sized from the
    # athlete, the HUD and watermark stay sized to the frame
    style = OverlayStyle.for_frame(video.width, video.height, _body_pixels(pose, video))
    chrome = OverlayStyle.for_frame(video.width, video.height)
    canvas = OverlayCanvas(video.width, video.height, style)
    intermediate = output_path.with_name(output_path.stem + "_raw.mp4")

    # the skeleton this exercise is drawn with, and a display-smoothed copy of
    # the coordinates to draw it from. The analysis keeps its own pose untouched.
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
            if _has_drawable_pose(render_pose, index):
                points, alphas = _drawable_points(render_pose, index, video, policy, drawn)
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
        hidden            limbs left out of the whole clip, decided once up front
    """

    trust_visibility: bool
    # Landmarks held to the far-side floor - limbs only.
    far_limbs: frozenset[int]
    max_jump_px: float
    # horizontal distance under which a far limb counts as hidden. 0 turns it off.
    occlusion_px: float = 0.0
    # Landmarks suppressed for the entire video - see _clip_hidden_far_limbs.
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
        Is this landmark confident enough to draw? Two thresholds rather than
        one: it takes MIN_DRAW_VISIBILITY to start drawing and a fall all the way
        to DRAW_RELEASE_VISIBILITY to stop, and either answer stands for
        MIN_DRAW_DWELL_FRAMES frames. A single threshold on a signal that hovers
        around it is what made limbs blink.

        confidence None means the source reports none, so nothing is gated.
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
        """Is this frame the same shot as the last one? Losing a landmark or two is
        normal; losing half the figure at once is not, and whatever was held is
        then a picture of somewhere the athlete no longer is."""
        before = self._previous_drawn
        self._previous_drawn = drawn_now
        if before <= 0:
            return True
        return drawn_now * 2 >= before

    def remember(self, landmark: int, frame_idx: int, x: float, y: float) -> None:
        """Keep the last position this landmark was actually drawn at."""
        self._held[landmark] = (x, y, frame_idx)

    def forget_holds(self) -> None:
        """Throw the hold buffer away. Called when the frame no longer follows on
        from the last one - see the cut check in _drawable_points."""
        self._held.clear()

    def hold(self, landmark: int, frame_idx: int) -> tuple[float, float] | None:
        """
        Where to draw a landmark that has just gone missing, or None once it has
        been gone too long to pretend. Holding a couple of frames is what stops a
        one-frame detector hiccup taking a limb off the figure.
        """
        held = self._held.get(landmark)
        if held is None:
            return None
        x, y, seen = held
        if frame_idx - seen > self.hold_frames:
            return None
        return x, y

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
    Which far-side limbs to leave out of the entire video.

    The occlusion and far-side confidence tests are both right and both unusable
    frame by frame: they sit on noisy signals, so a limb near either boundary
    switches on and off several times a second and the viewer sees a blink, not a
    judgement. Asked once over the whole clip the answer is stable - a limb that
    spends most of the video behind the body is never drawn, and one that does not
    is drawn like any other, with no far-side gate left to trip on.

    Dill et al. (2024) are the reason this errs towards hiding: with the far arm
    behind the torso their reconstruction attached it to a jacket on a chair. An
    occluded limb does not merely get noisy, it gets attached to the wrong thing.
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
            # untracked, hidden behind its twin, or too unsure to believe - all
            # three are the same answer here: not something to draw
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
    A smoothed copy of the coordinates, for drawing only.

    pose.xy - what every rule reads - is untouched. This is the display layer: it
    runs a zero-phase Butterworth over each drawn landmark, which the analysis
    itself cannot have for free but rendering can, because by then the whole clip
    is known and there is no lag to pay at the turning points. The causal EMA the
    analysis uses barely dents MediaPipe's per-frame noise, which is why the
    figure looked like it was floating while the numbers beside it were calm.
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
    Landmarks to draw on one frame, in pixels, with their opacity.

    Three things changed here after the overlay was judged on how it looked
    rather than on whether it was correct:

    - the far-side and occlusion tests moved out to _clip_hidden_far_limbs,
      where they are answered once for the video instead of once per frame;
    - what is left is hysteretic, so a landmark near a threshold stays put
      instead of chattering, and a landmark that drops out for a frame or two is
      held at its last position rather than deleted;
    - opacity is no longer driven by confidence. It was modulating the whole
      skeleton frame by frame off a noisy signal, which read as the figure
      breathing. A landmark is drawn or it is not.
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

    # Holding a landmark only makes sense if this frame follows on from the last
    # one. When most of the figure goes at once - a cut, a second person, the
    # athlete leaving the shot - the held positions belong to a picture that is
    # no longer on screen, and drawing them smears the old pose over the new one.
    if not policy.continues_from_last_frame(len(fresh)):
        policy.forget_holds()

    for landmark in sorted(set(wanted)):
        if landmark >= xy.shape[0] or landmark in policy.hidden:
            continue
        # a gap of a frame or two is the detector stuttering, not the limb
        # leaving; a link needs both ends, so dropping one costs two links
        position = fresh.get(landmark) or policy.hold(landmark, frame_idx)
        if position is None:
            continue
        points[landmark] = (int(round(position[0])), int(round(position[1])))
        alphas[landmark] = 1.0
    return points, alphas
