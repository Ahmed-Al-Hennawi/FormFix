"""
Drawing code for the annotated video. annotation.py decides what goes on each
frame, this file just paints it through OverlayCanvas.

Watch out: colours are BGR, not the CSS hex (cyan #00F0FF is (255, 240, 0)
here), and all sizes scale with the frame so 480p and 4K look the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import (
    BODY_CONNECTIONS,
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

# --- "Cyber Pulse" palette from the site, in BGR order ---

CYAN = (255, 240, 0)  # #00F0FF - tracking, measurements, the skeleton
LIME = (20, 255, 57)  # #39FF14 - inside the rule
AMBER = (60, 168, 240)  # #F0A83C - warning (site's --warn token)
RED = (118, 90, 255)  # #FF5A76 - failed check (site's --poor token)
WHITE = (255, 255, 255)
DEEP_SPACE = (43, 19, 11)  # #0B132B - label and HUD backgrounds

# status level -> colour, same as the results page
LEVEL_COLOURS: dict[str, tuple[int, int, int]] = {
    "pass": LIME,
    "warning": AMBER,
    "fail": RED,
    "info": CYAN,
}

# limb landmarks per side that disappear behind the torso in a side view.
# Shoulder and hip aren't included, they give the torso its width.
SIDE_LIMB_LANDMARKS: dict[str, frozenset[int]] = {
    "left": frozenset(
        {LEFT_ELBOW, LEFT_WRIST, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX, LEFT_EAR}
    ),
    "right": frozenset(
        {RIGHT_ELBOW, RIGHT_WRIST, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX, RIGHT_EAR}
    ),
}

# the joints a beginner thinks about - only these get rings
KEY_JOINTS: frozenset[int] = frozenset(
    {
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE,
    }
)

# no face landmark feeds any exercise, so just one dim head marker
HEAD_LANDMARK = NOSE

SEGMENTS: tuple[tuple[int, int], ...] = BODY_CONNECTIONS


@dataclass(frozen=True)
class JointAngle:
    """A joint angle to draw, recomputed from the same smoothed coordinates as the
    analysis so the two can't disagree."""

    label: str
    proximal: int
    vertex: int
    distal: int
    arc: bool = True


@dataclass(frozen=True)
class OverlayStyle:
    """
    All drawing sizes for one video. Tuned at a 720px short edge and clamped so
    low-res clips stay readable and 4K doesn't get a 12px skeleton.

    The skeleton is deliberately thin: the person is the subject, the overlay is
    there to show how FormFix followed them.
    """

    width: int
    height: int
    scale: float
    link: int
    link_focus: int
    node_r: int
    key_node_r: int
    core_r: int
    ring_w: int
    text: float
    text_small: float
    text_weight: int
    pad: int
    gap: int
    radius: int
    blur: int

    @classmethod
    def for_frame(cls, width: int, height: int, body_pixels: float | None = None) -> OverlayStyle:
        """
        body_pixels is the athlete's torso length if known. Frame size alone isn't
        enough, the person might fill the frame or stand six feet back.
        """
        short_edge = max(min(int(width), int(height)), 1)
        scale = max(0.55, min(2.0, short_edge / 720.0))
        if body_pixels and body_pixels > 0:
            reference = 0.22 * short_edge  # a full-body figure's torso
            scale *= max(0.62, min(1.35, float(body_pixels) / reference))
            scale = max(0.45, min(2.2, scale))
        px = lambda value, floor=1: max(floor, int(round(value * scale)))  # noqa: E731
        return cls(
            width=int(width),
            height=int(height),
            scale=scale,
            # one weight and one radius everywhere: a skeleton that changes
            # thickness as it moves reads as a bug, not as emphasis
            link=px(2.6, 2),
            link_focus=px(2.6, 2),
            node_r=px(3.9, 3),
            key_node_r=px(3.9, 3),
            core_r=px(1.9, 1),
            ring_w=px(1.0),
            text=0.46 * scale,
            text_small=0.38 * scale,
            text_weight=max(1, int(round(1.15 * scale))),
            pad=px(14, 8),
            gap=px(8, 4),
            radius=px(7, 3),
            blur=max(3, (int(round(7 * scale)) | 1)),
        )


# --- The canvas ---


class OverlayCanvas:
    """
    Drawing surface with per-stroke transparency and glow. OpenCV draws opaque
    shapes, so I paint into a colour buffer plus an alpha mask and blend once per
    frame, with a third buffer blurred in for the glow. Only the touched area is
    cleared and blended, which keeps it fast at 1080p.
    """

    def __init__(self, width: int, height: int, style: OverlayStyle) -> None:
        self.width = int(width)
        self.height = int(height)
        self.style = style
        self._paint = np.zeros((self.height, self.width, 3), np.uint8)
        self._alpha = np.zeros((self.height, self.width), np.uint8)
        self._glow = np.zeros((self.height, self.width, 3), np.uint8)
        self._bounds: tuple[int, int, int, int] | None = None
        self._glow_used = False
        self.frame: np.ndarray | None = None

    def begin(self, frame: np.ndarray) -> None:
        """Start a new frame, clearing only what the previous one touched."""
        if self._bounds is not None:
            x0, y0, x1, y1 = self._bounds
            self._paint[y0:y1, x0:x1] = 0
            self._alpha[y0:y1, x0:x1] = 0
            if self._glow_used:
                self._glow[y0:y1, x0:x1] = 0
        self._bounds = None
        self._glow_used = False
        self.frame = frame

    def commit(self, glow_strength: float = 0.62) -> None:
        """Composite this frame's strokes and their bloom onto the video."""
        if self.frame is None or self._bounds is None:
            return
        x0, y0, x1, y1 = self._bounds
        roi = self.frame[y0:y1, x0:x1]

        if self._glow_used:
            # blur the glow at quarter res - cheaper and softer
            glow = np.ascontiguousarray(self._glow[y0:y1, x0:x1])
            height, width = glow.shape[:2]
            small = cv2.resize(
                glow, (max(width // 4, 1), max(height // 4, 1)), interpolation=cv2.INTER_LINEAR
            )
            k = max(3, self.style.blur // 2 | 1)
            small = cv2.blur(small, (k, k))
            glow = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
            roi[:] = cv2.addWeighted(np.ascontiguousarray(roi), 1.0, glow, glow_strength, 0.0)

        mask = np.ascontiguousarray(self._alpha[y0:y1, x0:x1])
        if not mask.any():
            return
        # integer alpha compositing in OpenCV, not a float32 round-trip
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        painted = np.ascontiguousarray(self._paint[y0:y1, x0:x1])
        foreground = cv2.multiply(painted, mask3, scale=1.0 / 255.0)
        background = cv2.multiply(np.ascontiguousarray(roi), cv2.bitwise_not(mask3), scale=1.0 / 255.0)
        roi[:] = cv2.add(foreground, background)

    def _touch(self, points, margin: int) -> None:
        """Grow the dirty rect so commit knows what to composite."""
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        x0 = max(min(xs) - margin, 0)
        y0 = max(min(ys) - margin, 0)
        x1 = min(max(xs) + margin + 1, self.width)
        y1 = min(max(ys) + margin + 1, self.height)
        if x0 >= x1 or y0 >= y1:
            return
        if self._bounds is None:
            self._bounds = (x0, y0, x1, y1)
        else:
            bx0, by0, bx1, by1 = self._bounds
            self._bounds = (min(bx0, x0), min(by0, y0), max(bx1, x1), max(by1, y1))

    @staticmethod
    def _alpha_value(alpha: float) -> int:
        return int(round(max(0.0, min(1.0, alpha)) * 255))

    def _stroke(self, draw, colour, alpha: float) -> None:
        """Paint one primitive into the colour and coverage buffers."""
        draw(self._paint, colour)
        draw(self._alpha, self._alpha_value(alpha))

    def _bloom(self, draw, colour, strength: float) -> None:
        if strength <= 0:
            return
        dim = tuple(int(round(c * max(0.0, min(1.0, strength)))) for c in colour)
        draw(self._glow, dim)
        self._glow_used = True

    # --- primitives ---

    def line(
        self,
        p1,
        p2,
        colour,
        thickness: int,
        alpha: float = 1.0,
        glow: float = 0.0,
    ) -> None:
        a = (int(p1[0]), int(p1[1]))
        b = (int(p2[0]), int(p2[1]))
        self._touch((a, b), thickness * 2 + self.style.blur if glow else thickness + 2)
        if glow:
            width = thickness + max(2, int(round(3 * self.style.scale)))
            self._bloom(lambda t, c: cv2.line(t, a, b, c, width, cv2.LINE_AA), colour, glow)
        self._stroke(lambda t, c: cv2.line(t, a, b, c, thickness, cv2.LINE_AA), colour, alpha)

    def circle(
        self,
        centre,
        radius: int,
        colour,
        thickness: int = -1,
        alpha: float = 1.0,
        glow: float = 0.0,
    ) -> None:
        c = (int(centre[0]), int(centre[1]))
        margin = radius + max(thickness, 0) + (self.style.blur if glow else 2)
        self._touch((c,), margin)
        if glow:
            self._bloom(
                lambda t, col: cv2.circle(t, c, radius + self.style.ring_w, col, -1, cv2.LINE_AA),
                colour,
                glow,
            )
        self._stroke(
            lambda t, col: cv2.circle(t, c, radius, col, thickness, cv2.LINE_AA), colour, alpha
        )

    def arc(
        self,
        centre,
        radius: int,
        start_deg: float,
        end_deg: float,
        colour,
        thickness: int,
        alpha: float = 1.0,
    ) -> None:
        c = (int(centre[0]), int(centre[1]))
        self._touch((c,), radius + thickness + 2)
        self._stroke(
            lambda t, col: cv2.ellipse(
                t, c, (radius, radius), 0.0, start_deg, end_deg, col, thickness, cv2.LINE_AA
            ),
            colour,
            alpha,
        )

    def rounded_rect(
        self,
        top_left,
        size,
        colour,
        radius: int,
        alpha: float = 1.0,
        thickness: int = -1,
    ) -> None:
        x, y = int(top_left[0]), int(top_left[1])
        w, h = int(size[0]), int(size[1])
        r = max(0, min(int(radius), w // 2, h // 2))
        self._touch(((x, y), (x + w, y + h)), max(thickness, 1) + 1)

        def draw(target, col):
            if thickness < 0:
                cv2.rectangle(target, (x + r, y), (x + w - r, y + h), col, -1, cv2.LINE_AA)
                cv2.rectangle(target, (x, y + r), (x + w, y + h - r), col, -1, cv2.LINE_AA)
                for cx, cy in (
                    (x + r, y + r),
                    (x + w - r, y + r),
                    (x + r, y + h - r),
                    (x + w - r, y + h - r),
                ):
                    cv2.circle(target, (cx, cy), r, col, -1, cv2.LINE_AA)
            else:
                cv2.line(target, (x + r, y), (x + w - r, y), col, thickness, cv2.LINE_AA)
                cv2.line(target, (x + r, y + h), (x + w - r, y + h), col, thickness, cv2.LINE_AA)
                cv2.line(target, (x, y + r), (x, y + h - r), col, thickness, cv2.LINE_AA)
                cv2.line(target, (x + w, y + r), (x + w, y + h - r), col, thickness, cv2.LINE_AA)
                for (cx, cy), angle in (
                    ((x + r, y + r), 180),
                    ((x + w - r, y + r), 270),
                    ((x + w - r, y + h - r), 0),
                    ((x + r, y + h - r), 90),
                ):
                    cv2.ellipse(
                        target, (cx, cy), (r, r), 0, angle, angle + 90, col, thickness, cv2.LINE_AA
                    )

        self._stroke(draw, colour, alpha)

    def text(
        self,
        value: str,
        origin,
        colour,
        size: float,
        weight: int = 1,
        alpha: float = 1.0,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
    ) -> None:
        org = (int(origin[0]), int(origin[1]))
        (tw, th), _ = cv2.getTextSize(value, font, size, weight)
        self._touch(((org[0], org[1] - th), (org[0] + tw, org[1] + th)), 2)
        self._stroke(
            lambda t, c: cv2.putText(t, value, org, font, size, c, weight, cv2.LINE_AA),
            colour,
            alpha,
        )


# --- Composed elements ---


def _text_size(value: str, style: OverlayStyle, size: float | None = None) -> tuple[int, int]:
    scale = style.text if size is None else size
    (w, h), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, style.text_weight)
    return w, h


def draw_connection(
    canvas: OverlayCanvas,
    start,
    end,
    style: OverlayStyle,
    *,
    colour=CYAN,
    focus: bool = False,
    alpha: float = 0.9,
    glow: float = 0.0,
    shadow: bool = True,
) -> None:
    """One skeleton link: a thin line over a thinner dark edge, because cyan on a
    pale wall or a white t-shirt is almost invisible. No glow - it spread the
    skeleton over the body and looked like a screensaver."""
    del glow
    thickness = style.link_focus if focus else style.link
    if shadow:
        canvas.line(start, end, DEEP_SPACE, thickness + 2, alpha=alpha * 0.30)
    canvas.line(start, end, colour, thickness, alpha=alpha)


def draw_joint_node(
    canvas: OverlayCanvas,
    point,
    style: OverlayStyle,
    *,
    colour=WHITE,
    key: bool = False,
    alpha: float = 1.0,
    glow: float = 0.0,
) -> None:
    """A small filled dot with a thin dark edge, the same size at every joint."""
    del key, glow
    canvas.circle(point, style.node_r + 1, DEEP_SPACE, -1, alpha=alpha * 0.45)
    canvas.circle(point, style.node_r, colour, -1, alpha=alpha)


def draw_analysis_label(
    canvas: OverlayCanvas,
    value: str,
    anchor,
    style: OverlayStyle,
    *,
    colour=CYAN,
    align: str = "left",
    size: float | None = None,
    accent: bool = False,
    alpha: float = 0.92,
) -> tuple[int, int]:
    """Small translucent chip. Returns (width, height) so chips can be stacked."""
    scale = style.text if size is None else size
    tw, th = _text_size(value, style, scale)
    pad_x = max(4, int(round(7 * style.scale)))
    pad_y = max(3, int(round(5 * style.scale)))
    bar = max(2, int(round(2.5 * style.scale))) if accent else 0
    width = tw + pad_x * 2 + bar
    height = th + pad_y * 2

    x, y = int(anchor[0]), int(anchor[1])
    if align == "right":
        x -= width
    elif align == "centre":
        x -= width // 2
    x = max(0, min(x, canvas.width - width))
    y = max(0, min(y, canvas.height - height))

    canvas.rounded_rect((x, y), (width, height), DEEP_SPACE, style.radius, alpha=0.66)
    canvas.rounded_rect(
        (x, y), (width, height), colour, style.radius, alpha=0.30, thickness=max(1, style.ring_w - 1)
    )
    if bar:
        canvas.rounded_rect(
            (x + pad_x // 2, y + pad_y), (bar, height - pad_y * 2), colour, bar // 2, alpha=0.95
        )
    canvas.text(
        value,
        (x + pad_x + bar, y + pad_y + th),
        colour,
        scale,
        style.text_weight,
        alpha=alpha,
    )
    return width, height


def _draw_head(
    canvas: OverlayCanvas,
    points: dict[int, tuple[int, int]],
    alphas: dict[int, float],
    style: OverlayStyle,
) -> None:
    """
    One faint dot for the head and a neck line to the shoulders. No face
    landmarks: MediaPipe tracks eyes, ears and mouth, but drawing them puts a
    mask over the person's face, which is the part of the overlay everybody
    finds unpleasant.
    """
    head = points.get(HEAD_LANDMARK)
    if head is None:
        return
    alpha = 0.55 * alphas.get(HEAD_LANDMARK, 1.0)
    left, right = points.get(LEFT_SHOULDER), points.get(RIGHT_SHOULDER)
    if left is not None and right is not None:
        neck = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
        draw_connection(canvas, head, neck, style, colour=CYAN, alpha=alpha, shadow=False)
    draw_joint_node(canvas, head, style, colour=WHITE, alpha=alpha)


def draw_formfix_pose(
    canvas: OverlayCanvas,
    points: dict[int, tuple[int, int]],
    alphas: dict[int, float],
    style: OverlayStyle,
    *,
    focus: frozenset[int] | set[int] = frozenset(),
    flagged: frozenset[int] | set[int] = frozenset(),
    inside_rep: bool = False,
) -> None:
    """
    The skeleton. One style throughout: cyan links, white joints, the same weight
    everywhere. The only thing that varies is opacity, so the side away from the
    camera is quieter than the side the measurements come from.

    Nothing here is coloured by whether the technique passed - a skeleton that
    turns red mid-rep reads as an error in the software, and the verdict belongs
    on the results page.
    """
    del focus, flagged, inside_rep
    for a, b in SEGMENTS:
        pa, pb = points.get(a), points.get(b)
        if pa is None or pb is None:
            continue
        alpha = min(alphas.get(a, 1.0), alphas.get(b, 1.0))
        draw_connection(canvas, pa, pb, style, colour=CYAN, alpha=0.88 * alpha)

    _draw_head(canvas, points, alphas, style)

    for landmark, point in points.items():
        if landmark == HEAD_LANDMARK:
            continue
        draw_joint_node(canvas, point, style, colour=WHITE, alpha=alphas.get(landmark, 1.0))


# --- Heads-up display ---


def draw_hud(canvas: OverlayCanvas, lines, style: OverlayStyle) -> None:
    """Rep counter, phase and at most two live values as small chips."""
    x = style.pad
    y = style.pad
    for text, colour, accent in lines[:4]:
        _, height = draw_analysis_label(canvas, text, (x, y), style, colour=colour, accent=accent)
        y += height + style.gap


def draw_watermark(canvas: OverlayCanvas, style: OverlayStyle, text: str = "FORMFIX") -> None:
    """Small logo text in the top-right corner."""
    canvas.text(
        text,
        (
            canvas.width - style.pad - _text_size(text, style, style.text_small)[0],
            style.pad + int(9 * style.scale),
        ),
        CYAN,
        style.text_small,
        max(1, style.text_weight),
        alpha=0.42,
    )
