"""
Draws the landmark overlay on the exercise figures. Back to front: a faint
dashed spine, cyan hairline links, the focus limb the rules read, a light
travelling along it, the measured angle arc, then the ringed nodes.

All three exercises come through here, which is the only reason the node sizes
and stroke weights stay identical between them.
"""

from __future__ import annotations

from .exercise_data import POSE_VIEWBOX, Pose

# Node geometry in viewBox units. Key and ordinary landmarks share a radius on
# purpose: the key one is picked out by colour and its rings, not by size.
NODE_RING_R = 13
NODE_CORE_R = 4.6
KEY_RING_R = NODE_RING_R
KEY_CORE_R = NODE_CORE_R
KEY_PING_R = 18
KEY_ORBIT_R = 25


def _fmt(value: float) -> str:
    """Trim trailing zeros so the markup stays readable."""
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _line(segment) -> str:
    (x1, y1), (x2, y2) = segment
    return f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" />'


def _node(x: float, y: float, key: bool = False) -> str:
    cx, cy = _fmt(x), _fmt(y)
    if key:
        return (
            '<g class="pn pn--key">'
            f'<circle class="pn__ping" cx="{cx}" cy="{cy}" r="{KEY_PING_R}" />'
            f'<circle class="pn__orbit" cx="{cx}" cy="{cy}" r="{KEY_ORBIT_R}" />'
            f'<circle class="pn__ring" cx="{cx}" cy="{cy}" r="{KEY_RING_R}" />'
            f'<circle class="pn__core" cx="{cx}" cy="{cy}" r="{KEY_CORE_R}" />'
            "</g>"
        )
    return (
        '<g class="pn">'
        f'<circle class="pn__ring" cx="{cx}" cy="{cy}" r="{NODE_RING_R}" />'
        f'<circle class="pn__core" cx="{cx}" cy="{cy}" r="{NODE_CORE_R}" />'
        "</g>"
    )


def render_pose(pose: Pose, key: str, extra_class: str = "") -> str:
    """Render the landmark overlay for one figure. key must be unique on the page -
    it namespaces the SVG gradient id."""
    gradient_id = f"ffFocus-{key}"
    x1, y1, x2, y2 = pose.gradient

    spine_points = " ".join(f"{_fmt(px)},{_fmt(py)}" for px, py in pose.spine)
    links = "".join(_line(segment) for segment in pose.links)
    focus = "".join(_line(segment) for segment in pose.focus)
    nodes = "".join(_node(px, py) for px, py in pose.nodes)
    key_nodes = "".join(_node(px, py, key=True) for px, py in pose.key_nodes)

    classes = f"pose {extra_class}".strip()

    return (
        f'<svg class="{classes}" viewBox="{POSE_VIEWBOX}" fill="none" aria-hidden="true">'
        "<defs>"
        f'<linearGradient id="{gradient_id}" gradientUnits="userSpaceOnUse" '
        f'x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}">'
        '<stop offset="0" stop-color="#00F0FF" />'
        '<stop offset="1" stop-color="#39FF14" />'
        "</linearGradient>"
        "</defs>"
        f'<g class="pose__spine"><polyline points="{spine_points}" /></g>'
        f'<g class="pose__links">{links}</g>'
        f'<g class="pose__links pose__links--focus" stroke="url(#{gradient_id})">{focus}</g>'
        f'<g class="pose__flow">{focus}</g>'
        f'<g class="pose__arc"><path d="{pose.arc}" /></g>'
        f'<g class="pose__nodes">{nodes}{key_nodes}</g>'
        "</svg>"
    )
