"""
The little overhead map shown above the uploader: where to stand the phone for
one exercise.

Testers read the written instructions and then filmed from the wrong side
anyway, so the camera position is drawn rather than described - the words beside
it are cut down to three lines. Everything is a bird's-eye view: the figure in
the middle faces up the page, the dashed ring is every place the phone could
stand, and the solid band on it is the part of the ring this exercise's rules
can actually be measured from.

Geometry is angles-from-facing, not pixels: `angle=90` means "a quarter turn
round from straight in front of you", which is what side-on means, and the
renderer turns that into coordinates. Nothing here knows about a specific
exercise - exercise_data.py owns the numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Canvas -----------------------------------------------------------------
# Bird's-eye plan in viewBox units. The figure sits a little above centre so the
# equipment drawn behind it (a bench, a seat) still has room.
CENTRE_X = 150.0
CENTRE_Y = 110.0

# The dashed ring of possible camera positions.
ORBIT_R = 78.0

# Square, and centred on the ring rather than on the drawing: the ring is the
# thing that has to look deliberately placed inside its tile. Wide enough to
# hold the camera wherever it sits on the ring, so nothing spills over the edge.
VIEWBOX = "52 12 196 196"

# Half-width of the slice of the body the view cone is drawn around. Not a real
# field of view - it just has to read as "this is what the camera sees".
CONE_HALF_WIDTH = 44.0


@dataclass(frozen=True)
class CameraSetup:
    """Where to put the camera for one exercise, in plain geometry.

    Angles are degrees clockwise from straight in front of the person, so 0 is
    face-on, 90 is their left-hand side, 180 is behind them.
    """

    # where the phone is drawn
    angle: float
    # the usable band on the ring, as (from, to) in the same degrees
    arc: tuple[float, float]
    # how high to hold it, and how far back, in the fewest words possible
    height: str
    distance: str
    # a hint of what they are on, so the plan is recognisable at a glance
    equipment: str = ""
    # true when the mirror image of the arc works just as well (either side)
    either_side: bool = False


def _point(angle: float, radius: float) -> tuple[float, float]:
    """Polar to plan coordinates. 0 degrees is straight in front of the figure,
    which is up the page, and the angle increases clockwise."""
    radians = math.radians(angle)
    return (
        CENTRE_X + radius * math.sin(radians),
        CENTRE_Y - radius * math.cos(radians),
    )


def _fmt(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _arc_path(start: float, end: float, radius: float) -> str:
    """The band on the ring between two angles."""
    x1, y1 = _point(start, radius)
    x2, y2 = _point(end, radius)
    large = 1 if abs(end - start) > 180 else 0
    sweep = 1 if end > start else 0
    return (
        f"M {_fmt(x1)} {_fmt(y1)} "
        f"A {_fmt(radius)} {_fmt(radius)} 0 {large} {sweep} {_fmt(x2)} {_fmt(y2)}"
    )


def _cone(angle: float) -> str:
    """What the camera sees: a wedge from the lens across the body."""
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
    """A phone, lens towards the figure. Drawn upright in its own coordinates and
    turned to face the middle: rotate(angle + 180) points its lens back at the
    centre, since the phone stands at `angle` on the ring."""
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
    """A trace of the kit, so the plan reads as the right exercise. Behind the
    figure for a bench, in front for a machine, across the shoulders for a bar."""
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
    """The person from above: shoulders across, head on top, facing up the page."""
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
    """The overhead plan for one exercise. `key` only has to be unique on the
    page - it names the gradient - and `label` is what a screen reader hears."""
    gradient_id = f"cgCone-{key}"
    start, end = setup.arc

    bands = [f'<path class="cg-band" d="{_arc_path(start, end, ORBIT_R)}" />']
    if setup.either_side:
        # the mirror image is just as valid; drawn faintly, without a phone on it
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
