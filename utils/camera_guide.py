"""
The small overhead diagram above the uploader showing where to put the phone.

Testers read the written instructions and still filmed from the wrong side, so
I draw the camera position instead. It's a bird's-eye view: the figure faces up
the page, the dashed ring is every possible phone position, and the solid band
is where this exercise can actually be measured from.

Positions are angles from the person's front, not pixels (angle=90 = side-on).
The numbers per exercise are in exercise_data.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Canvas ---
# viewBox units. The figure is a bit above centre to leave room for the bench
# or seat behind it
CENTRE_X = 150.0
CENTRE_Y = 110.0

ORBIT_R = 78.0

# square and centred on the ring, wide enough that the phone never goes over
# the edge
VIEWBOX = "52 12 196 196"

# half-width of the view cone - not a real field of view, just has to look right
CONE_HALF_WIDTH = 44.0


@dataclass(frozen=True)
class CameraSetup:
    """Where to put the camera for one exercise. Angles are degrees clockwise from
    the person's front: 0 face-on, 90 their left side, 180 behind."""

    angle: float
    # usable band on the ring, (from, to)
    arc: tuple[float, float]
    # how high and how far back, in as few words as possible
    height: str
    distance: str
    # outline of the equipment so it's recognisable at a glance
    equipment: str = ""
    # true when either side works
    either_side: bool = False


def _point(angle: float, radius: float) -> tuple[float, float]:
    """Polar to plan coordinates (0 = in front of the figure = up the page, clockwise)."""
    radians = math.radians(angle)
    return (
        CENTRE_X + radius * math.sin(radians),
        CENTRE_Y - radius * math.cos(radians),
    )


def _fmt(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _arc_path(start: float, end: float, radius: float) -> str:
    x1, y1 = _point(start, radius)
    x2, y2 = _point(end, radius)
    large = 1 if abs(end - start) > 180 else 0
    sweep = 1 if end > start else 0
    return (
        f"M {_fmt(x1)} {_fmt(y1)} "
        f"A {_fmt(radius)} {_fmt(radius)} 0 {large} {sweep} {_fmt(x2)} {_fmt(y2)}"
    )


def _cone(angle: float) -> str:
    """Wedge from the lens across the body."""
    cam_x, cam_y = _point(angle, ORBIT_R)
    # perpendicular to the line of sight, so the wedge opens across the figure
    radians = math.radians(angle)
    px, py = math.cos(radians), math.sin(radians)
    ax = CENTRE_X + px * CONE_HALF_WIDTH
    ay = CENTRE_Y + py * CONE_HALF_WIDTH
    bx = CENTRE_X - px * CONE_HALF_WIDTH
    by = CENTRE_Y - py * CONE_HALF_WIDTH
    points = f"{_fmt(cam_x)},{_fmt(cam_y)} {_fmt(ax)},{_fmt(ay)} {_fmt(bx)},{_fmt(by)}"
    return f'<polygon class="cg-cone" points="{points}" />'


def _camera(angle: float) -> str:
    """A phone with its lens towards the figure (rotate(angle + 180) points it at
    the centre)."""
    x, y = _point(angle, ORBIT_R)
    return (
        f'<g class="cg-cam" transform="translate({_fmt(x)} {_fmt(y)}) '
        f'rotate({_fmt(angle + 180)})">'
        '<rect class="cg-cam__body" x="-9.5" y="-16" width="19" height="32" rx="4.5" />'
        '<circle class="cg-cam__lens" cx="0" cy="-9" r="3.6" />'
        '<rect class="cg-cam__screen" x="-5.5" y="-2" width="11" height="13" rx="2" />'
        "</g>"
    )


def _equipment(kind: str) -> str:
    """Outline of the equipment: bench behind, machine in front, bar across the shoulders."""
    if kind == "barbell":
        return (
            '<g class="cg-kit">'
            '<line class="cg-kit__bar" x1="98" y1="106" x2="202" y2="106" />'
            '<circle class="cg-kit__plate" cx="104" cy="106" r="7" />'
            '<circle class="cg-kit__plate" cx="196" cy="106" r="7" />'
            "</g>"
        )
    if kind == "bench":
        return (
            '<g class="cg-kit">'
            '<rect class="cg-kit__pad" x="137" y="104" width="26" height="52" rx="10" />'
            "</g>"
        )
    if kind == "machine":
        return (
            '<g class="cg-kit">'
            '<rect class="cg-kit__pad" x="139" y="112" width="22" height="30" rx="8" />'
            '<rect class="cg-kit__frame" x="118" y="52" width="64" height="20" rx="6" />'
            '<line class="cg-kit__cable" x1="150" y1="72" x2="150" y2="92" />'
            "</g>"
        )
    return ""


def _figure() -> str:
    """The person from above, facing up the page."""
    return (
        '<g class="cg-figure">'
        '<ellipse class="cg-figure__ground" cx="150" cy="112" rx="34" ry="26" />'
        '<rect class="cg-figure__shoulders" x="124" y="101" width="52" height="18" rx="9" />'
        '<circle class="cg-figure__head" cx="150" cy="107" r="11.5" />'
        '<path class="cg-figure__facing" d="M 150 92 L 150 80" />'
        '<path class="cg-figure__nose" d="M 144.5 82.5 L 150 76 L 155.5 82.5" />'
        "</g>"
    )


def render_camera_plan(setup: CameraSetup, key: str, label: str) -> str:
    """The overhead plan for one exercise. `key` must be unique on the page (used
    for the gradient) and `label` is for screen readers."""
    gradient_id = f"cgCone-{key}"
    start, end = setup.arc

    bands = [f'<path class="cg-band" d="{_arc_path(start, end, ORBIT_R)}" />']
    if setup.either_side:
        # the other side works too, drawn faintly without a phone
        bands.append(f'<path class="cg-band cg-band--alt" d="{_arc_path(-end, -start, ORBIT_R)}" />')

    return (
        f'<svg class="cg-plan" viewBox="{VIEWBOX}" role="img" '
        f'aria-label="{label}" preserveAspectRatio="xMidYMid meet">'
        "<defs>"
        f'<linearGradient id="{gradient_id}" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#00f0ff" stop-opacity="0.26" />'
        '<stop offset="100%" stop-color="#00f0ff" stop-opacity="0.02" />'
        "</linearGradient>"
        "</defs>"
        f'<circle class="cg-orbit" cx="{_fmt(CENTRE_X)}" cy="{_fmt(CENTRE_Y)}" '
        f'r="{_fmt(ORBIT_R)}" />'
        f"{''.join(bands)}"
        f'<g style="--cg-cone-fill: url(#{gradient_id})">{_cone(setup.angle)}</g>'
        f"{_equipment(setup.equipment)}"
        f"{_figure()}"
        f"{_camera(setup.angle)}"
        "</svg>"
    )
