#!/usr/bin/env python3
"""
FormFix - wireframe generator.

Writes the mid-fidelity wireframes as plain SVG, with the real interface copy
in them, so every screen and state is documented without a design tool.

    python3 wireframes/generate_wireframes.py

The SVGs land next to this file and re-running regenerates all of them.

Conventions in the output: a grey block is a container, stacked grey bars are
body copy, a box with a cross is media, a dark button is a primary action, an
outlined one secondary, a dashed outline is conditional on interface state, and
italic grey is an annotation rather than part of the interface.

Standard library only.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

OUT = Path(__file__).resolve().parent

# --- ink ---
SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "SFMono-Regular, Menlo, Consolas, monospace"

CANVAS = "#e9e9ea"  # around the device frame
PAPER = "#ffffff"  # the page itself
INK = "#2c2c2e"  # headings
BODY = "#5f5f63"  # readable body text
MUTED = "#8a8a8f"  # secondary / meta
FAINT = "#b4b4b8"  # annotation text
LINE = "#c2c2c6"  # box borders
HAIR = "#dededf"  # separators
BOX = "#f4f4f5"  # panel fill
BOX2 = "#ebebec"  # nested panel fill
BAR = "#d8d8da"  # placeholder text bar
DARK = "#3a3a3c"  # primary button
MEDIA = "#eeeeef"  # image placeholder fill


def esc(value: object) -> str:
    return _html.escape(str(value), quote=True)


def wrap(text: str, width_px: float, size: float) -> list[str]:
    """Greedy word wrap, using an average glyph width for the font size."""
    per_line = max(8, int(width_px / (size * 0.5)))
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= per_line:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class Wire:
    """One wireframe sheet: a titled canvas containing a device frame."""

    PAD = 44
    HEAD = 96
    FOOT = 46

    def __init__(
        self,
        slug: str,
        title: str,
        subtitle: str,
        dev_w: int,
        dev_h: int,
        chrome: str = "browser",
        url: str = "formfix.app/",
    ) -> None:
        self.slug = slug
        self.title = title
        self.subtitle = subtitle
        self.dw = dev_w
        self.dh = dev_h
        self.chrome = chrome
        self.url = url
        self.ch = {"browser": 40, "phone": 34, "none": 0}[chrome]
        self.W = max(dev_w + self.PAD * 2, 760)
        self.H = self.HEAD + self.ch + dev_h + self.FOOT
        self.fx = (self.W - dev_w) / 2
        self.ox = self.fx
        self.oy = self.HEAD + self.ch
        self.parts: list[str] = []

    # --- primitives ---
    def raw(self, markup: str) -> None:
        self.parts.append(markup)

    def r(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = BOX,
        stroke: str = LINE,
        rx: float = 5,
        sw: float = 1,
        dash: str | None = None,
        opacity: float | None = None,
    ) -> None:
        attrs = [
            f'x="{self.ox + x:.1f}" y="{self.oy + y:.1f}"',
            f'width="{w:.1f}" height="{h:.1f}" rx="{rx}"',
            f'fill="{fill}"',
        ]
        if stroke:
            attrs.append(f'stroke="{stroke}" stroke-width="{sw}"')
        else:
            attrs.append('stroke="none"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        self.parts.append(f"<rect {' '.join(attrs)} />")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = HAIR,
        sw: float = 1,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{self.ox + x1:.1f}" y1="{self.oy + y1:.1f}" '
            f'x2="{self.ox + x2:.1f}" y2="{self.oy + y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{dash_attr} stroke-linecap="round" />'
        )

    def circle(
        self, cx: float, cy: float, r: float, fill: str = "none", stroke: str = LINE, sw: float = 1
    ) -> None:
        self.parts.append(
            f'<circle cx="{self.ox + cx:.1f}" cy="{self.oy + cy:.1f}" r="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'
        )

    def path(
        self, d: str, stroke: str = LINE, sw: float = 1, fill: str = "none", dash: str | None = None
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<path d="{d}" transform="translate({self.ox},{self.oy})" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{dash_attr} stroke-linecap="round" />'
        )

    def t(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 13,
        fill: str = INK,
        weight: str = "400",
        anchor: str = "start",
        family: str = SANS,
        italic: bool = False,
        spacing: float | None = None,
    ) -> None:
        extra = ' font-style="italic"' if italic else ""
        if spacing is not None:
            extra += f' letter-spacing="{spacing}"'
        self.parts.append(
            f'<text x="{self.ox + x:.1f}" y="{self.oy + y:.1f}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{extra}>{esc(text)}</text>'
        )

    # --- composites ---
    def para(
        self,
        x: float,
        y: float,
        text: str,
        w: float,
        size: float = 13,
        fill: str = BODY,
        lh: float | None = None,
        weight: str = "400",
        max_lines: int | None = None,
    ) -> float:
        """Real wrapped copy. Returns the y below the last line."""
        lh = lh or size * 1.55
        lines = wrap(text, w, size)
        if max_lines:
            lines = lines[:max_lines]
        for index, line in enumerate(lines):
            self.t(x, y + index * lh, line, size=size, fill=fill, weight=weight)
        return y + (len(lines) - 1) * lh

    def bars(
        self,
        x: float,
        y: float,
        w: float,
        n: int = 3,
        gap: float = 13,
        h: float = 7,
        last: float = 0.62,
        fill: str = BAR,
    ) -> float:
        """Placeholder body copy. Returns the y below the last bar."""
        for index in range(n):
            width = w * last if index == n - 1 and n > 1 else w
            self.r(x, y + index * gap, width, h, fill=fill, stroke="", rx=3.5)
        return y + (n - 1) * gap + h

    def media(self, x: float, y: float, w: float, h: float, label: str = "") -> None:
        self.r(x, y, w, h, fill=MEDIA, stroke=LINE)
        self.line(x, y, x + w, y + h, stroke="#d2d2d5")
        self.line(x + w, y, x, y + h, stroke="#d2d2d5")
        if label:
            self.r(x + 10, y + h - 30, len(label) * 5.6 + 20, 20, fill=PAPER, stroke=LINE, rx=10)
            self.t(x + 20, y + h - 16, label, size=9.5, fill=MUTED, spacing=0.5)

    def btn(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        primary: bool = True,
        size: float = 12.5,
        dash: str | None = None,
        disabled: bool = False,
    ) -> None:
        if primary:
            self.r(x, y, w, h, fill="#bfbfc3" if disabled else DARK, stroke="", rx=h / 2, dash=dash)
            self.t(
                x + w / 2,
                y + h / 2 + size * 0.35,
                label,
                size=size,
                fill="#ffffff",
                weight="600",
                anchor="middle",
            )
        else:
            self.r(x, y, w, h, fill=BOX2 if disabled else PAPER, stroke=LINE, rx=h / 2, dash=dash)
            self.t(
                x + w / 2,
                y + h / 2 + size * 0.35,
                label,
                size=size,
                fill=MUTED if disabled else INK,
                weight="500",
                anchor="middle",
            )

    @staticmethod
    def tag_width(label: str, size: float = 9.5, pad: float = 9) -> float:
        """How wide tag will draw a pill, for right-aligning it first."""
        # 0.67em is semibold uppercase. The tracking tag() applies counts
        # towards the width too, or long labels spill out of their own pill.
        return len(label) * (size * 0.67 + 0.7) + pad * 2

    def tag(
        self,
        x: float,
        y: float,
        label: str,
        size: float = 9.5,
        pad: float = 9,
        h: float = 19,
        fill: str = BOX2,
        stroke: str = LINE,
    ) -> float:
        spacing = 0.7
        w = self.tag_width(label, size, pad)
        self.r(x, y, w, h, fill=fill, stroke=stroke, rx=h / 2)
        self.t(
            x + pad,
            y + h / 2 + size * 0.36,
            label,
            size=size,
            fill=MUTED,
            weight="600",
            spacing=spacing,
        )
        return w

    def eyebrow(self, x: float, y: float, text: str) -> None:
        self.t(x, y, text.upper(), size=10, fill=MUTED, weight="700", spacing=1.6)

    def heading(
        self, x: float, y: float, text: str, size: float = 30, fill: str = INK, weight: str = "700"
    ) -> None:
        self.t(x, y, text, size=size, fill=fill, weight=weight)

    def note(self, x: float, y: float, text: str, w: float = 300, align: str = "left") -> None:
        """An annotation about the design, not part of the interface itself."""
        lines = wrap(text, w, 10.5)
        self.line(x, y - 9, x, y - 9 + len(lines) * 14 + 2, stroke=FAINT, sw=1, dash="3 3")
        for index, line in enumerate(lines):
            self.t(x + 7, y + index * 14, line, size=10.5, fill=FAINT, italic=True)

    def divider(self, x: float, y: float, w: float) -> None:
        self.line(x, y, x + w, y, stroke=HAIR)

    def pose(self, x: float, y: float, w: float, h: float) -> None:
        """The landmark overlay: a stick figure of joints and links."""
        cx = x + w / 2
        pts = {
            "head": (cx, y + h * 0.13),
            "sh_l": (cx - w * 0.14, y + h * 0.28),
            "sh_r": (cx + w * 0.14, y + h * 0.28),
            "el_l": (cx - w * 0.21, y + h * 0.44),
            "el_r": (cx + w * 0.21, y + h * 0.44),
            "wr_l": (cx - w * 0.17, y + h * 0.58),
            "wr_r": (cx + w * 0.17, y + h * 0.58),
            "hip_l": (cx - w * 0.11, y + h * 0.55),
            "hip_r": (cx + w * 0.11, y + h * 0.55),
            "kn_l": (cx - w * 0.14, y + h * 0.74),
            "kn_r": (cx + w * 0.14, y + h * 0.74),
            "an_l": (cx - w * 0.12, y + h * 0.92),
            "an_r": (cx + w * 0.12, y + h * 0.92),
        }
        links = [
            ("sh_l", "sh_r"),
            ("hip_l", "hip_r"),
            ("sh_l", "hip_l"),
            ("sh_r", "hip_r"),
            ("sh_l", "el_l"),
            ("el_l", "wr_l"),
            ("sh_r", "el_r"),
            ("el_r", "wr_r"),
            ("hip_l", "kn_l"),
            ("kn_l", "an_l"),
            ("hip_r", "kn_r"),
            ("kn_r", "an_r"),
        ]
        for a, b in links:
            ax, ay = pts[a]
            bx, by = pts[b]
            self.line(ax, ay, bx, by, stroke="#9a9aa0", sw=1.4)
        hx, hy = pts["head"]
        sx = (pts["sh_l"][0] + pts["sh_r"][0]) / 2
        sy = (pts["sh_l"][1] + pts["sh_r"][1]) / 2
        self.line(hx, hy + 9, sx, sy, stroke="#9a9aa0", sw=1.4)
        self.circle(hx, hy, 9, fill=PAPER, stroke="#8f8f95", sw=1.4)
        for key, (px, py) in pts.items():
            if key == "head":
                continue
            emphasis = key in ("kn_l", "kn_r")
            self.circle(
                px, py, 4 if emphasis else 3, fill=DARK if emphasis else PAPER, stroke="#8f8f95", sw=1.3
            )

    def fit(self, y: float, pad: float = 48) -> None:
        """Trim the sheet to the content actually drawn on it."""
        self.dh = max(140, y + pad)
        self.H = self.HEAD + self.ch + self.dh + self.FOOT

    # --- output ---
    def _chrome(self) -> str:
        out: list[str] = []
        fx, fy = self.fx, self.HEAD
        if self.chrome == "browser":
            out.append(
                f'<path d="M{fx} {fy + 10} a10 10 0 0 1 10 -10 h{self.dw - 20} '
                f'a10 10 0 0 1 10 10 v30 h-{self.dw} z" fill="#f0f0f1" stroke="{LINE}" />'
            )
            for index in range(3):
                out.append(
                    f'<circle cx="{fx + 20 + index * 15}" cy="{fy + 20}" r="4.5" ' f'fill="#d5d5d8" />'
                )
            out.append(
                f'<rect x="{fx + 76}" y="{fy + 10}" width="{min(360, self.dw - 120)}" '
                f'height="20" rx="10" fill="{PAPER}" stroke="{LINE}" />'
            )
            out.append(
                f'<text x="{fx + 88}" y="{fy + 24}" font-family="{MONO}" font-size="10.5" '
                f'fill="{MUTED}">{esc(self.url)}</text>'
            )
        elif self.chrome == "phone":
            out.append(
                f'<path d="M{fx} {fy + 22} a22 22 0 0 1 22 -22 h{self.dw - 44} '
                f'a22 22 0 0 1 22 22 v12 h-{self.dw} z" fill="#f0f0f1" stroke="{LINE}" />'
            )
            out.append(
                f'<rect x="{fx + self.dw / 2 - 34}" y="{fy + 7}" width="68" height="13" '
                f'rx="6.5" fill="#dcdcdf" />'
            )
            out.append(
                f'<text x="{fx + 16}" y="{fy + 22}" font-family="{SANS}" font-size="9.5" '
                f'font-weight="600" fill="{MUTED}">9:41</text>'
            )
        return "".join(out)

    def emit(self) -> Path:
        frame_y = self.HEAD
        frame_h = self.ch + self.dh
        radius = 22 if self.chrome == "phone" else (10 if self.chrome == "browser" else 4)
        head = [
            f'<rect width="{self.W}" height="{self.H}" fill="{CANVAS}" />',
            f'<text x="{self.PAD}" y="{40}" font-family="{SANS}" font-size="19" '
            f'font-weight="700" fill="{INK}">{esc(self.title)}</text>',
            f'<text x="{self.PAD}" y="{62}" font-family="{SANS}" font-size="11.5" '
            f'fill="{MUTED}">{esc(self.subtitle)}</text>',
            f'<text x="{self.W - self.PAD}" y="{40}" font-family="{SANS}" font-size="10.5" '
            f'font-weight="700" fill="{FAINT}" text-anchor="end" letter-spacing="1.8">'
            f"FORMFIX &#183; WIREFRAME</text>",
            f'<text x="{self.W - self.PAD}" y="{62}" font-family="{MONO}" font-size="10" '
            f'fill="{FAINT}" text-anchor="end">{esc(self.slug)}.svg</text>',
            f'<rect x="{self.fx}" y="{frame_y}" width="{self.dw}" height="{frame_h}" '
            f'rx="{radius}" fill="{PAPER}" stroke="{LINE}" stroke-width="1.4" />',
            self._chrome(),
            (
                f'<line x1="{self.fx}" y1="{self.oy}" x2="{self.fx + self.dw}" y2="{self.oy}" '
                f'stroke="{LINE}" />'
                if self.ch
                else ""
            ),
        ]
        foot = (
            f'<text x="{self.PAD}" y="{self.H - 18}" font-family="{SANS}" font-size="10" '
            f'fill="{FAINT}">Mid-fidelity wireframe &#183; real interface copy &#183; '
            f"grey bars represent body text &#183; not to scale</text>"
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
            f'viewBox="0 0 {self.W} {self.H}" role="img" '
            f'aria-label="{esc(self.title)} wireframe">'
            f"<title>{esc(self.title)}</title>"
            f"{''.join(head)}"
            f'<g clip-path="url(#clip)">{"".join(self.parts)}</g>'
            f"{foot}"
            '<defs><clipPath id="clip">'
            f'<rect x="{self.fx}" y="{self.oy}" width="{self.dw}" height="{self.dh}" />'
            "</clipPath></defs>"
            "</svg>"
        )
        target = OUT / f"{self.slug}.svg"
        target.write_text(svg, encoding="utf-8")
        return target


# ===========================================================================
# Real interface copy, taken from the application modules
# ===========================================================================

NAV_ITEMS = ("How It Works", "Exercises", "Explainable Feedback", "Technology", "Research")

HERO_HEADLINE = ("See your form.", "Understand your movement.", "Improve your technique.")
HERO_SUB = (
    "FormFix analyses a short video of your squat, dumbbell shoulder press or lat "
    "pulldown, tracks how your body moves, and explains - in plain language - what to "
    "change and why."
)

ABOUT_TITLE = "A second pair of eyes on the lifts you're learning."
ABOUT_LEDE = (
    "Upload a short video of your set. FormFix follows how you move, then tells you "
    "which part of your technique to work on - and why it matters."
)
ABOUT_CHIPS_LABEL = "The three movements"
ABOUT_CHIPS = ("Barbell squat", "Shoulder press", "Lat pulldown")
ABOUT_ASIDE = "No account, no setup. Your video is deleted the moment the analysis finishes."
ABOUT_PANELS = (
    (
        "Built for you if",
        "✓",
        (
            "You are learning these three lifts",
            "You train without a coach watching",
            "You would rather know what to fix than get a score",
        ),
    ),
    (
        "What it does not do",
        "—",
        (
            "Write your training programme",
            "Tell you how much to lift",
            "Make any claim about injury or health",
        ),
    ),
)

PROBLEM_HEADLINE = ("Watching the movement is easy.", "Understanding your own movement is harder.")
PROBLEM_A = (
    "Most beginners learn exercises from short videos and copy what they see. Without "
    "feedback, small technique errors - a knee drifting inward, an elbow slipping out of "
    "position - go unnoticed and quietly become habits."
)
PROBLEM_B = (
    "A coach would catch them. Not everyone has one - and asking a stranger mid-workout is "
    "harder than it sounds. FormFix bridges that gap: it watches your movement the way a "
    "coach would, and tells you what it sees."
)

STEPS = (
    ("01", "Record", "Film a short clip of your set. Any phone camera works.", "VIDEO IN"),
    ("02", "Upload", "Drop the video into FormFix. No account, no setup.", "UPLOAD.MP4"),
    (
        "03",
        "Analyse",
        "Computer vision tracks 33 body landmarks through every frame of the " "movement.",
        "POSE ESTIMATION",
    ),
    (
        "04",
        "Understand",
        "Transparent movement rules check joint angles, alignment and "
        "symmetry - and identify what went wrong.",
        "RULES ENGINE",
    ),
    (
        "05",
        "Improve",
        "You get clear, human-readable feedback you can apply on your very " "next set.",
        "FEEDBACK OUT",
    ),
)

EXERCISES = (
    (
        "Squat",
        "Side-on",
        "Analysed from the side, FormFix tracks how deep your hips descend, how far your "
        "chest leans forward, whether your heels stay planted, and whether you stand tall "
        "between repetitions.",
        ("Hip depth", "Torso angle", "Heel contact", "Return to standing"),
        "KNEE ANGLE 92°",
    ),
    (
        "Shoulder Press",
        "Front-on",
        "Filmed from the front, FormFix compares your two arms through the press, checks "
        "that each wrist stays stacked over its elbow, and measures how far each repetition "
        "travels at the shoulders and overhead.",
        ("Arm symmetry", "Elbow and wrist alignment", "Range of motion"),
        "ELBOW ANGLE 84°",
    ),
    (
        "Lat Pulldown",
        "Side or three-quarter",
        "Filmed from the side, FormFix measures how much your torso moves away from its "
        "starting position while you pull, and how far your arms travel at both ends of each "
        "repetition.",
        ("Torso movement", "Range of motion"),
        "TORSO MOVEMENT ±15°",
    ),
)

PIPELINE = (
    ("WHAT IT SAW", "On 2 of your 6 reps, your knees stopped bending at 118°."),
    ("WHAT IT CHECKED", "The depth check looks for 115° or lower. 118° is short of it."),
    ("WHAT IT SAID", "“You stop a little high on 2 of your 6 reps.”"),
)

TECH_ITEMS = (
    (
        "01",
        "Computer Vision",
        "Your uploaded video is processed frame by frame. A phone camera is all it takes.",
    ),
    (
        "02",
        "Pose Estimation",
        "A 33-landmark human pose model maps your joints through the entire movement.",
    ),
    (
        "03",
        "Movement Rules",
        "Transparent, testable rules evaluate angles, alignment and symmetry. No black box.",
    ),
    (
        "04",
        "Explainable Feedback",
        "Every message traces back to the exact rule - and the exact moment - that triggered it.",
    ),
)

RESEARCH_ITEMS = (
    (
        "1. Research Question",
        "How can explainable computer vision support beginner gym users in identifying and "
        "correcting common exercise technique mistakes from uploaded workout videos?",
    ),
    (
        "2. The Problem",
        "Beginners often learn exercises from online videos but struggle to self-evaluate "
        "subtle technical errors. Personal training is expensive, and asking for help in the "
        "gym can cause anxiety.",
    ),
    (
        "3. The Solution & Method",
        "Vision Pipeline: 2D joint coordinates from smartphone video using MediaPipe Pose. "
        "Kinematic Evaluation: joint angles against biomechanical rules. Explainable Feedback: raw "
        "kinematics converted into plain-English feedback.",
    ),
    (
        "4. Evaluation & Results",
        "Evaluated across three dimensions: Technical Accuracy, User Understanding, and "
        "System Usability (System Usability Scale score).",
    ),
)

REPORT_ROWS = (
    ("Pose detection", "", 100, "Successful"),
    ("Squat depth", "4 of 6 reps", 67, "Needs Attention"),
    ("Forward torso lean", "6 of 6 reps", 92, "Good"),
    ("Heel stability", "6 of 6 reps", 100, "Good"),
    ("Return to standing", "6 of 6 reps", 100, "Good"),
)
REPORT_FEEDBACK = (
    "Two of your repetitions stopped above the configured depth target. Lowering a little "
    "further under control will even the set out."
)

SQUAT_TIPS = (
    "Film from the side, roughly level with your hips.",
    "Keep your hips, knees and ankles in frame for the whole set.",
    "Keep the camera still, and record several controlled repetitions.",
    "Start standing, perform several full squats, and finish standing.",
)

ANALYSIS_STAGES = (
    "Preparing video",
    "Checking recording",
    "Detecting body landmarks",
    "Measuring movement",
    "Detecting repetitions",
    "Assessing technique",
    "Building feedback",
    "Rendering analysed video",
)

TRACKER = ("Upload Video", "Analyse Form", "Feedback")

CORRECTIONS = (
    (
        "01",
        "Squat depth",
        "Two of your six repetitions stopped above parallel.",
        "Depth is what makes the set count - a shallow rep trains a shorter range.",
        "Lower until your hip crease passes your knee, under control.",
    ),
    (
        "02",
        "Forward torso lean",
        "Your chest dropped forward on the third repetition.",
        "A forward chest shifts load off the legs and onto the lower back.",
        "Think about keeping your chest up as you come out of the bottom.",
    ),
)

POSITIVES = (
    "Your heels stayed planted on every repetition.",
    "You stood fully tall between reps.",
    "Descent speed stayed controlled throughout the set.",
)

CHECK_ROWS = (
    ("Squat depth", "4 of 6 reps · High reliability", 67),
    ("Forward torso lean", "6 of 6 reps · High reliability", 92),
    ("Heel stability", "6 of 6 reps · High reliability", 100),
    ("Return to standing", "6 of 6 reps · Medium reliability", 100),
    ("Descent control", "5 of 6 reps · Medium reliability", 83),
)

PREVIEW_ROWS = (
    ("Form score", "An overall score out of 100"),
    ("What you did well", "The parts of the movement that looked right"),
    ("What to improve", "Each issue, with a clear correction"),
)


# ===========================================================================
# Homepage sections (desktop) - each takes a y and returns the y below itself
# ===========================================================================

M = 120  # page margin
CW = 1200  # content width


def sec_nav(w: Wire, y: float = 22) -> float:
    """The floating pill nav. Fixed, so it sits over the page."""
    pill_w, pill_h = 1060, 62
    x = (w.dw - pill_w) / 2
    w.r(x, y, pill_w, pill_h, fill=PAPER, stroke=LINE, rx=pill_h / 2)
    w.r(x + 22, y + 17, 30, 28, fill=BOX2, stroke=LINE, rx=4)
    w.t(x + 60, y + 36, "FORMFIX", size=14, weight="800", spacing=0.6)
    lx = x + 240
    for label in NAV_ITEMS:
        w.t(lx, y + 36, label, size=12, fill=BODY, weight="500")
        lx += len(label) * 6.9 + 34
    w.btn(x + pill_w - 168, y + 13, 146, 36, "Analyse Form")
    return y + pill_h


def sec_hero(w: Wire, y: float) -> float:
    top = y + 96
    w.eyebrow(M, top, "Explainable computer vision · For people learning to lift")
    hy = top + 54
    for index, line in enumerate(HERO_HEADLINE):
        w.heading(
            M, hy + index * 58, line, size=46, weight="800", fill=INK if index != 1 else "#4a4a4e"
        )
    sy = hy + 3 * 58 + 8
    w.para(M, sy, HERO_SUB, 500, size=14, lh=24)
    by = sy + 90
    w.btn(M, by, 224, 50, "Analyse Your Form  →", size=14)
    w.btn(M + 240, by, 190, 50, "See How It Works", primary=False, size=14)
    w.note(
        M,
        by + 92,
        "Fixed pill navigation. The landmark overlay and the three HUD "
        "chips sit over the photograph and parallax on scroll.",
        w=430,
    )

    stage_x, stage_y = 790, top + 6
    w.media(stage_x, stage_y, 520, 540, "athlete photograph")
    w.pose(stage_x + 90, stage_y + 40, 340, 460)
    w.tag(stage_x - 44, stage_y + 54, "● POSE DETECTED", fill=PAPER)
    w.tag(stage_x + 340, stage_y + 486, "TRACKING 33 LANDMARKS", fill=PAPER)
    w.tag(stage_x + 404, stage_y + 250, "KNEE ANGLE 92°", fill=PAPER)

    hint_y = by + 118
    w.line(w.dw / 2, hint_y, w.dw / 2, hint_y + 34, stroke=LINE)
    w.t(
        w.dw / 2,
        hint_y + 50,
        "SCROLL",
        size=9.5,
        fill=MUTED,
        weight="700",
        anchor="middle",
        spacing=1.8,
    )
    return hint_y + 78


def sec_about(w: Wire, y: float) -> float:
    """The "what this is" block: what it does, who for, what it does not do."""
    w.eyebrow(M, y + 60, "What this is")
    w.heading(M, y + 104, "A second pair of eyes on", size=26)
    w.heading(M, y + 138, "the lifts you're learning.", size=26)
    w.line(M, y + 168, M + 56, y + 168, stroke=DARK, sw=2)
    w.para(M, y + 194, ABOUT_LEDE, 520, size=13.5, lh=23)
    w.t(M, y + 268, ABOUT_CHIPS_LABEL.upper(), size=9, fill=MUTED, weight="700", spacing=1.5)
    cx = M
    for chip in ABOUT_CHIPS:
        cx += w.tag(cx, y + 282, chip.upper(), size=9) + 10
    w.circle(M + 4, y + 334, 3, fill=DARK, stroke="")
    w.para(M + 18, y + 338, ABOUT_ASIDE, 420, size=11.5, lh=18, fill=MUTED)

    # Both halves are lists, since someone deciding whether this is for them
    # scans rather than reads. The first panel gets the emphasis.
    px, py = M + 640, y + 96
    for index, (title, marker, lines) in enumerate(ABOUT_PANELS):
        lead = index == 0
        w.r(
            px,
            py,
            560,
            146,
            fill=PAPER if lead else BOX,
            stroke=DARK if lead else LINE,
            sw=1.4 if lead else 1,
        )
        w.t(
            px + 24,
            py + 30,
            title.upper(),
            size=9.5,
            fill=INK if lead else MUTED,
            weight="700",
            spacing=1.3,
        )
        w.line(px + 24, py + 44, px + 536, py + 44, stroke=LINE)
        ly = py + 72
        for line in lines:
            w.circle(px + 32, ly - 4, 8, fill=BOX2 if lead else PAPER, stroke=LINE)
            w.t(px + 32, ly, marker, size=9, fill=INK if lead else MUTED, anchor="middle")
            w.t(px + 52, ly, line, size=12, fill=BODY)
            ly += 28
        py += 166
    w.note(
        M + 640,
        y + 440,
        "Added after supervisor feedback: the hero says what the "
        "system does but not who it is for. A statement beside two scannable panels, "
        "so the exclusions are as visible as the promise.",
        w=520,
    )
    return y + 500


def sec_problem(w: Wire, y: float) -> float:
    w.eyebrow(M, y + 70, "The problem")
    w.heading(M, y + 118, PROBLEM_HEADLINE[0], size=32)
    w.heading(M, y + 160, PROBLEM_HEADLINE[1], size=32, fill="#9b9ba0")
    col = 520
    w.para(M, y + 216, PROBLEM_A, col, size=13.5, lh=23)
    w.para(M + 600, y + 216, PROBLEM_B, col, size=13.5, lh=23)
    return y + 320


def sec_how(w: Wire, y: float) -> float:
    w.eyebrow(M, y + 70, "How FormFix works")
    w.heading(M, y + 116, "From your camera roll", size=32)
    w.heading(M, y + 158, "to a clear correction.", size=32)
    rail_x = M + 26
    top = y + 210
    row_h = 108
    w.line(rail_x, top, rail_x, top + row_h * len(STEPS) - 30, stroke=HAIR, sw=2)
    w.line(rail_x, top, rail_x, top + row_h * 2, stroke=DARK, sw=2)
    for index, (number, title, text, tagname) in enumerate(STEPS):
        ry = top + index * row_h
        w.circle(rail_x, ry + 14, 15, fill=PAPER, stroke=LINE)
        w.t(rail_x, ry + 18.5, number, size=11, weight="700", fill=INK, anchor="middle")
        w.t(rail_x + 44, ry + 14, title, size=17, weight="700")
        w.para(rail_x + 44, ry + 40, text, 620, size=12.5, lh=20)
        w.tag(M + CW - 180, ry + 4, tagname)
    w.note(
        M + 700,
        top + 6,
        "Rail progress line is scrubbed by scroll position; the active step's number "
        "lights up while it owns the viewport.",
        w=250,
    )
    return top + row_h * len(STEPS) + 40


def sec_explorer(w: Wire, y: float, active: int = 0) -> float:
    w.eyebrow(M, y + 70, "Exercise explorer")
    w.heading(M, y + 116, "Three movements.", size=32)
    w.heading(M, y + 158, "One analysis engine.", size=32)

    tabs_y = y + 202
    tx = M
    for index, (name, *_rest) in enumerate(EXERCISES):
        tw = len(name) * 8 + 44
        current = index == active
        w.r(tx, tabs_y, tw, 40, fill=DARK if current else PAPER, stroke="" if current else LINE, rx=20)
        w.t(
            tx + tw / 2,
            tabs_y + 25,
            name,
            size=12.5,
            weight="600",
            fill=PAPER if current else BODY,
            anchor="middle",
        )
        tx += tw + 12

    stage_y = tabs_y + 70
    name, view, desc, areas, metric = EXERCISES[active]
    w.t(M, stage_y + 8, f"0{active + 1}", size=12, weight="700", fill=INK)
    w.t(M + 24, stage_y + 8, f"/ 0{len(EXERCISES)}", size=12, fill=MUTED)
    w.heading(M, stage_y + 52, name, size=26)
    w.para(M, stage_y + 88, desc, 320, size=12.5, lh=21)
    ly = stage_y + 190
    for area in areas:
        w.circle(M + 4, ly - 4, 3, fill=DARK, stroke="")
        w.t(M + 18, ly, area, size=12, fill=BODY)
        ly += 24

    w.media(M + 400, stage_y - 10, 400, 430, "exercise photograph")
    w.pose(M + 460, stage_y + 20, 280, 370)
    chip = f"● {metric}"
    w.tag(M + 800 - 14 - w.tag_width(chip), stage_y + 392, chip, fill=PAPER)

    side_x = M + 880
    w.r(side_x, stage_y + 40, 44, 44, fill=PAPER, stroke=LINE, rx=22)
    w.t(side_x + 22, stage_y + 68, "←", size=15, fill=BODY, anchor="middle")
    w.r(side_x + 56, stage_y + 40, 44, 44, fill=PAPER, stroke=LINE, rx=22)
    w.t(side_x + 78, stage_y + 68, "→", size=15, fill=BODY, anchor="middle")
    w.t(side_x, stage_y + 152, "NEXT", size=9.5, fill=MUTED, weight="700", spacing=1.6)
    w.r(side_x, stage_y + 168, 200, 92, fill=BOX, stroke=LINE)
    w.media(side_x + 12, stage_y + 180, 68, 68)
    nxt = EXERCISES[(active + 1) % len(EXERCISES)][0]
    w.t(side_x + 92, stage_y + 220, nxt, size=13, weight="600")
    w.note(
        side_x,
        stage_y + 300,
        "Tabs, arrows and the NEXT card all drive the same client-side transition - "
        "no Streamlit rerun, so scroll position survives.",
        w=210,
    )
    return stage_y + 470


def sec_explainable(w: Wire, y: float) -> float:
    w.eyebrow(M, y + 70, "Explainable Feedback")
    w.heading(M, y + 116, "Not a score.", size=32)
    w.heading(M, y + 158, "An explanation.", size=32)
    w.para(
        M,
        y + 200,
        "FormFix doesn't just hand you a grade. Every correction says what "
        "happened, why it matters and what to try instead. Here is one from the session "
        "reported further down this page.",
        640,
        size=13.5,
        lh=22,
    )

    cy = y + 270
    w.r(M, cy, 580, 220, fill=BOX, stroke=LINE)
    w.t(M + 26, cy + 34, "What a black box tells you", size=11, fill=MUTED, weight="600")
    w.t(M + 26, cy + 82, "✕  Incorrect exercise.", size=20, weight="700", fill="#7a7a7f")
    # Same shape as the real correction card: title, what happened, why it
    # matters, what to try.
    w.r(M + 620, cy, 580, 220, fill=PAPER, stroke=DARK, sw=1.4)
    w.t(M + 646, cy + 34, "What FormFix tells you", size=11, fill=MUTED, weight="600")
    w.t(M + 646, cy + 62, "SQUAT DEPTH", size=10, fill=MUTED, weight="700", spacing=1.3)
    w.para(
        M + 646,
        cy + 92,
        "You stop a little high on 2 of your 6 reps.",
        500,
        size=15,
        lh=24,
        fill=INK,
        weight="600",
    )
    w.para(
        M + 646,
        cy + 126,
        "Squatting lower works your legs through their full range.",
        500,
        size=12.5,
        lh=21,
    )
    w.tag(M + 646, cy + 158, "TRY THIS")
    w.para(
        M + 646,
        cy + 194,
        "Lower until your hips reach about knee height, keeping your " "whole foot planted.",
        500,
        size=12.5,
        lh=21,
        fill=INK,
    )

    # The three nodes get an intro line - they are the most technical
    # thing on the page.
    w.t(M, cy + 288, "The rule behind that message", size=19, weight="700", fill=INK)
    w.para(
        M,
        cy + 314,
        "None of it is guesswork. You can follow any correction back "
        "through the three steps that produced it.",
        560,
        size=12.5,
        lh=21,
    )
    py = cy + 366
    node_w = 356
    for index, (step, body) in enumerate(PIPELINE):
        nx = M + index * (node_w + 66)
        last = index == len(PIPELINE) - 1
        w.r(
            nx,
            py,
            node_w,
            132,
            fill=PAPER if last else BOX,
            stroke=DARK if last else LINE,
            sw=1.4 if last else 1,
        )
        w.t(nx + 20, py + 30, step, size=9.5, weight="700", fill=MUTED, spacing=1.3)
        w.para(nx + 20, py + 60, body, node_w - 40, size=12, lh=19)
        if index:
            w.line(nx - 56, py + 66, nx - 10, py + 66, stroke=LINE)
            w.t(nx - 16, py + 70, "▶", size=8, fill=MUTED)
    w.para(
        M,
        py + 176,
        "That 115° is a starting value, not a fixed truth. FormFix prints "
        "the number behind every correction, so you can always see what it was measured "
        "against.",
        760,
        size=12,
        lh=20,
        fill=MUTED,
    )
    return py + 250


def sec_example(w: Wire, y: float) -> float:
    w.eyebrow(M + 160, y + 70, "Example analysis")
    w.heading(M + 160, y + 118, "What a session looks like.", size=32)
    card_x, card_y, card_w = M + 160, y + 170, 880
    rows_h = len(REPORT_ROWS) * 46
    card_h = 118 + rows_h + 118
    w.r(card_x, card_y, card_w, card_h, fill=PAPER, stroke=LINE, rx=10)
    w.t(card_x + 30, card_y + 44, "Squat Analysis", size=17, weight="700")
    w.t(
        card_x + 30,
        card_y + 68,
        "squat_session_004.mp4 · 00:12 · reps detected: 6",
        size=11.5,
        fill=MUTED,
    )
    w.tag(card_x + card_w - 210, card_y + 32, "● ANALYSIS COMPLETE", fill=BOX2)
    w.divider(card_x + 30, card_y + 96, card_w - 60)
    ry = card_y + 122
    for label, detail, score, badge in REPORT_ROWS:
        w.t(card_x + 30, ry + 16, label, size=13, weight="600")
        if detail:
            w.t(card_x + 30, ry + 33, detail, size=10.5, fill=MUTED)
        meter_x = card_x + 300
        w.r(meter_x, ry + 12, 320, 8, fill=BOX2, stroke="", rx=4)
        w.r(meter_x, ry + 12, 320 * score / 100, 8, fill=DARK, stroke="", rx=4)
        w.tag(card_x + card_w - 180, ry + 6, badge.upper(), fill=BOX)
        ry += 46
    w.divider(card_x + 30, ry + 8, card_w - 60)
    w.t(card_x + 30, ry + 44, "FEEDBACK", size=9.5, weight="700", fill=MUTED, spacing=1.4)
    w.para(card_x + 30, ry + 70, REPORT_FEEDBACK, card_w - 60, size=13, lh=21)
    w.note(
        card_x + card_w + 30,
        card_y + 130,
        "Same markup as the live results card, so the promise on the marketing page "
        "and the real output cannot drift apart.",
        w=180,
    )
    return card_y + card_h + 90


def sec_tech(w: Wire, y: float) -> float:
    w.eyebrow(M, y + 70, "Under the hood")
    w.heading(M, y + 116, "Transparent by design.", size=32)
    grid_y = y + 176
    col_w = (CW - 3 * 30) / 4
    for index, (number, title, body) in enumerate(TECH_ITEMS):
        gx = M + index * (col_w + 30)
        w.r(gx, grid_y, col_w, 200, fill=BOX, stroke=LINE)
        w.t(gx + 22, grid_y + 42, number, size=13, weight="700", fill=MUTED)
        w.t(gx + 22, grid_y + 78, title, size=15, weight="700")
        w.para(gx + 22, grid_y + 108, body, col_w - 44, size=12, lh=19)
    return grid_y + 260


def sec_research(w: Wire, y: float) -> float:
    card_y = y + 60
    card_h = 590
    w.r(M, card_y, CW, card_h, fill=BOX, stroke=LINE, rx=12)
    w.t(
        w.dw / 2, card_y + 60, "Academic & Technical Foundation", size=28, weight="700", anchor="middle"
    )
    w.t(
        w.dw / 2,
        card_y + 90,
        "Grounded in Explainable Computer Vision and Human Pose " "Estimation Research",
        size=12,
        fill=MUTED,
        anchor="middle",
    )
    lede = (
        "FormFix was developed as a university research project investigating how "
        "computer vision can provide accessible, explainable movement feedback for "
        "beginner gym users."
    )
    lines = wrap(lede, 720, 13)
    for index, line in enumerate(lines):
        w.t(w.dw / 2, card_y + 128 + index * 22, line, size=13, fill=BODY, anchor="middle")
    w.btn(w.dw / 2 - 155, card_y + 190, 310, 46, "Read Full Research Paper (PDF)", size=13)

    gy = card_y + 270
    col_w = (CW - 120 - 30) / 2
    for index, (title, body) in enumerate(RESEARCH_ITEMS):
        gx = M + 60 + (index % 2) * (col_w + 30)
        yy = gy + (index // 2) * 130
        w.r(gx, yy, col_w, 112, fill=PAPER, stroke=LINE)
        w.t(gx + 20, yy + 30, title, size=13, weight="700")
        w.para(gx + 20, yy + 54, body, col_w - 40, size=11.5, lh=17, max_lines=3)
    w.t(
        w.dw / 2,
        card_y + card_h - 34,
        "FormFix (2026). An Explainable Computer Vision "
        "System for Exercise Form Assessment and Corrective Feedback.",
        size=10.5,
        fill=MUTED,
        anchor="middle",
    )
    return card_y + card_h + 70


def sec_outro(w: Wire, y: float) -> float:
    w.media(w.dw / 2 - 110, y + 60, 220, 76, "logo")
    w.t(w.dw / 2, y + 200, "Better movement starts", size=36, weight="800", anchor="middle")
    w.t(w.dw / 2, y + 246, "with understanding it.", size=36, weight="800", anchor="middle")
    w.btn(w.dw / 2 - 140, y + 292, 280, 56, "Analyse Your Form  →", size=15)
    fy = y + 400
    w.divider(M, fy, CW)
    w.t(
        w.dw / 2,
        fy + 36,
        "FormFix is an educational exercise-technique tool and is not "
        "a replacement for professional coaching or medical advice.",
        size=11,
        fill=MUTED,
        anchor="middle",
    )
    w.t(
        w.dw / 2,
        fy + 66,
        "FORMFIX     ·     MSc Computer Science · Thesis " "Project     ·     2026",
        size=10,
        fill=FAINT,
        anchor="middle",
        spacing=0.8,
    )
    return fy + 100


# ===========================================================================
# /analyse - shared building blocks
# ===========================================================================


def ring(w: Wire, cx: float, cy: float, r: float, pct: int | None, label: str) -> None:
    """The score ring: a track, an arc for the value, and the number inside."""
    circumference = 2 * 3.14159 * r
    w.raw(
        f'<g transform="translate({w.ox},{w.oy})">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{BOX2}" stroke-width="9" />'
        + (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{DARK}" '
            f'stroke-width="9" stroke-linecap="round" '
            f'stroke-dasharray="{circumference * pct / 100:.1f} {circumference:.1f}" '
            f'transform="rotate(-90 {cx} {cy})" />'
            if pct is not None
            else ""
        )
        + "</g>"
    )
    w.t(cx, cy + 8, str(pct) if pct is not None else "—", size=30, weight="800", anchor="middle")
    w.t(cx, cy + 28, label, size=10, fill=MUTED, anchor="middle")


def ax_topbar(w: Wire, y: float = 26) -> float:
    w.t(M, y + 26, "←   Back Home", size=13, weight="600", fill=BODY)
    w.r(w.dw - M - 150, y + 6, 30, 26, fill=BOX2, stroke=LINE, rx=4)
    w.t(w.dw - M - 112, y + 26, "FORMFIX", size=13, weight="800")
    return y + 52


def ax_header(w: Wire, y: float) -> float:
    w.eyebrow(M, y + 50, "Analyse your form")
    w.heading(M, y + 100, "Upload a set.", size=38, weight="800")
    w.heading(M, y + 148, "Get an explanation.", size=38, weight="800")
    w.para(
        M,
        y + 190,
        "Record a single set, drop the clip in, and FormFix will track your body "
        "through the movement and explain what it sees. Each exercise has its own camera "
        "view - the guidance below the exercise picker tells you which.",
        760,
        size=13.5,
        lh=22,
    )
    return y + 250


def ax_tracker(w: Wire, y: float, stage: str) -> float:
    index = {"idle": 0, "ready": 0, "analysing": 1, "complete": 3}[stage]
    done_first = stage in ("ready", "analysing", "complete")
    total_w = 760
    x0 = (w.dw - total_w) / 2
    gap = total_w / (len(TRACKER) - 1)
    for position, label in enumerate(TRACKER):
        cx = x0 + position * gap
        if position < index or (position == 0 and done_first):
            state = "done"
        elif position == index:
            state = "active"
        else:
            state = "wait"
        if position:
            px = x0 + (position - 1) * gap
            link_done = position <= index
            w.line(
                px + 26,
                y + 20,
                cx - 26,
                y + 20,
                stroke=DARK if link_done else HAIR,
                sw=2,
                dash="6 5" if (position == index and stage == "analysing") else None,
            )
        w.circle(
            cx,
            y + 20,
            19,
            fill=DARK if state == "done" else PAPER,
            stroke=DARK if state != "wait" else LINE,
            sw=1.6,
        )
        if state == "done":
            w.t(cx, y + 25, "✓", size=13, fill=PAPER, weight="700", anchor="middle")
        elif state == "active":
            w.circle(cx, y + 20, 6, fill=DARK, stroke="")
        w.t(
            cx,
            y + 58,
            label,
            size=11.5,
            weight="600" if state != "wait" else "400",
            fill=INK if state != "wait" else MUTED,
            anchor="middle",
        )
    return y + 84


def panel_head(
    w: Wire, x: float, y: float, number: str, title: str, note: str = "", live: bool = False
) -> float:
    w.r(x, y, 34, 26, fill=DARK if live else BOX2, stroke="" if live else LINE, rx=5)
    w.t(x + 17, y + 18, number, size=12, weight="700", fill=PAPER if live else INK, anchor="middle")
    w.t(x + 48, y + 19, title, size=17, weight="700")
    if note:
        w.t(x + 48 + len(title) * 9.4 + 16, y + 19, note, size=10.5, fill=MUTED)
    return y + 42


def ax_upload(w: Wire, x: float, y: float, cw: float, state: str) -> float:
    """The left column. state: idle | ready | analysing | complete."""
    y = panel_head(w, x, y, "01", "Your video", "MP4 · MOV · AVI · 3-120 seconds · one set per clip")

    w.t(x, y + 26, "Which exercise?", size=12, weight="600", fill=BODY)
    ry = y + 42
    for index, (name, *_r) in enumerate(EXERCISES):
        cx = x + index * 168
        w.circle(cx + 8, ry + 10, 7, fill=PAPER, stroke=LINE)
        if index == 0:
            w.circle(cx + 8, ry + 10, 3.5, fill=DARK, stroke="")
        w.t(cx + 24, ry + 14, name, size=12.5, fill=INK)
    y = ry + 40

    # "How to record" guidance
    guide_h = 34 + len(SQUAT_TIPS) * 22
    w.r(x, y, cw, guide_h, fill=BOX, stroke=LINE)
    w.t(x + 18, y + 24, "HOW TO RECORD", size=9.5, weight="700", fill=MUTED, spacing=1.3)
    w.tag(x + cw - 130, y + 10, "SIDE-ON VIEW", fill=PAPER)
    for index, tip in enumerate(SQUAT_TIPS):
        w.circle(x + 22, y + 44 + index * 22, 2.5, fill=MUTED, stroke="")
        w.t(x + 34, y + 48 + index * 22, tip, size=11.5, fill=BODY)
    y += guide_h + 26

    # Drop zone
    w.r(x, y, cw, 104, fill=BOX, stroke=LINE, dash="6 5")
    w.r(x + 24, y + 34, 34, 34, fill=PAPER, stroke=LINE, rx=6)
    w.t(x + 41, y + 56, "↥", size=15, fill=MUTED, anchor="middle")
    w.t(x + 74, y + 46, "Drag and drop file here", size=13, weight="600")
    w.t(x + 74, y + 66, "Limit 200MB per file · MP4, MOV, AVI", size=11, fill=MUTED)
    w.btn(x + cw - 132, y + 36, 108, 32, "Browse files", primary=False, size=12)
    y += 128

    if state == "idle":
        w.t(x, y + 8, "Drag & drop or browse · .mp4 · .mov · .avi", size=11, fill=MUTED)
        w.t(
            x,
            y + 26,
            "Your video is deleted as soon as it has been analysed, the result is "
            "erased a few hours later, and no account is needed.",
            size=10.5,
            fill=MUTED,
        )
        w.note(
            x,
            y + 46,
            "Idle state: no video selected. The right-hand column shows what the result "
            "will contain, and nothing is analysed until the user presses Analyse Form.",
            w=cw - 20,
        )
        return y + 100

    # Confirmation + preview + actions (states: ready / analysing / complete)
    w.r(x, y, cw, 56, fill=BOX, stroke=LINE)
    w.circle(x + 30, y + 28, 13, fill=DARK, stroke="")
    w.t(x + 30, y + 33, "✓", size=12, fill=PAPER, weight="700", anchor="middle")
    w.t(x + 56, y + 25, "Video uploaded successfully", size=13, weight="700")
    w.t(x + 56, y + 43, "squat_session_004.mp4 · 8.4 MB", size=11, fill=MUTED)
    y += 76

    w.media(x, y, cw, 300, "uploaded video preview")
    w.r(x + 12, y + 300 - 40, cw - 24, 28, fill=PAPER, stroke=LINE, rx=14)
    w.t(x + 30, y + 300 - 21, "▶", size=10, fill=BODY)
    w.line(x + 48, y + 300 - 26, x + cw - 60, y + 300 - 26, stroke=HAIR, sw=3)
    w.t(x + cw - 44, y + 300 - 21, "0:12", size=9.5, fill=MUTED)
    y += 324

    primary = "Analyse again" if state == "complete" else "Analyse Form"
    disabled = state == "analysing"
    w.btn(x, y, cw * 0.58, 48, primary, size=13.5, disabled=disabled)
    w.btn(x + cw * 0.6, y, cw * 0.4, 48, "Remove video", primary=False, size=13.5, disabled=disabled)
    if disabled:
        w.note(x, y + 74, "Both controls are greyed out while the analysis is running.", w=cw - 20)
    return y + 60


def ax_waiting(w: Wire, x: float, y: float, cw: float, ready: bool) -> float:
    height = 300
    w.r(x, y, cw, height, fill=BOX, stroke=LINE, dash=None if ready else "6 5")
    w.eyebrow(x + 28, y + 40, "Ready to analyse" if ready else "Waiting for a video")
    w.t(
        x + 28,
        y + 76,
        "Press Analyse Form" if ready else "Your results will appear here",
        size=20,
        weight="700",
    )
    body = (
        "Your clip is loaded. Start the analysis and FormFix will walk through the "
        "movement frame by frame."
        if ready
        else "Choose your exercise and upload a short clip of one set. Nothing is analysed "
        "until you press Analyse Form."
    )
    w.para(x + 28, y + 106, body, cw - 56, size=12.5, lh=20)
    ly = y + 168
    for label, note in PREVIEW_ROWS:
        w.circle(x + 32, ly - 4, 3.5, fill=MUTED, stroke="")
        w.t(x + 48, ly, label, size=12.5, weight="700")
        w.t(x + 48 + len(label) * 7.4 + 10, ly, note, size=11.5, fill=MUTED)
        ly += 30
    return y + height


def ax_running(w: Wire, x: float, y: float, cw: float) -> float:
    y = panel_head(w, x, y, "02", "Analysing your form", "This takes a few seconds", live=True)
    stage_h = 330
    w.media(x, y + 16, cw, stage_h, "frame being analysed")
    w.pose(x + cw / 2 - 130, y + 50, 260, 260)
    w.line(x + 10, y + 190, x + cw - 10, y + 190, stroke=DARK, sw=2, dash="10 6")
    w.t(x + cw - 150, y + 205, "scan line", size=9.5, fill=MUTED, italic=True)
    w.tag(x + cw - 216, y + stage_h - 22, "● TRACKING 33 LANDMARKS", fill=PAPER)
    yy = y + stage_h + 40
    for index, stage in enumerate(ANALYSIS_STAGES):
        done = index < 3
        active = index == 3
        mark = "✓" if done else ("●" if active else "—")
        w.t(x + 6, yy + index * 26, mark, size=11, fill=DARK if done or active else FAINT, weight="700")
        label = f"{stage} · 46%" if active else stage
        w.t(
            x + 30,
            yy + index * 26,
            label,
            size=12.5,
            weight="700" if active else "400",
            fill=INK if done or active else MUTED,
        )
    end = yy + len(ANALYSIS_STAGES) * 26
    w.note(
        x,
        end + 20,
        "Each stage lights up when the pipeline genuinely reaches it; the frame-by-frame "
        "stages report a live percentage.",
        w=cw - 20,
    )
    return end + 60


def ax_results(
    w: Wire, x: float, y: float, cw: float, expanded: bool = False, quality_note: bool = False
) -> float:
    if quality_note:
        w.r(x, y, cw, 168, fill=BOX, stroke=DARK, sw=1.4)
        w.eyebrow(x + 24, y + 30, "Recording quality")
        w.t(x + 24, y + 60, "Analysis completed with a partial camera angle", size=15, weight="700")
        w.para(
            x + 24,
            y + 86,
            "FormFix could analyse this recording. Squat is analysed from a side-on view. "
            "Keeping the whole movement in frame, with the camera still, gives the most "
            "accurate measurements.",
            cw - 48,
            size=11.5,
            lh=18,
        )
        w.circle(x + 30, y + 146, 2.5, fill=MUTED, stroke="")
        w.t(x + 42, y + 150, "Camera was closer to three-quarter than side-on.", size=11.5, fill=BODY)
        y += 196

    # Analysed video
    w.t(x, y + 18, "Your analysed movement", size=17, weight="700")
    w.para(
        x,
        y + 42,
        "FormFix tracked your body through the movement. The highlighted "
        "joints are the ones each finding was measured from.",
        cw,
        size=11.5,
        lh=18,
    )
    w.media(x, y + 84, cw, 300, "annotated video with landmark overlay")
    w.pose(x + cw / 2 - 110, y + 110, 220, 250)
    y += 412

    # Verdict
    w.r(x, y, cw, 150, fill=BOX, stroke=LINE)
    ring(w, x + 92, y + 75, 46, 78, "/ 100")
    w.eyebrow(x + 174, y + 44, "Overall form")
    w.t(x + 174, y + 76, "Good Form", size=24, weight="800")
    w.para(
        x + 174, y + 100, "Depth was the one thing holding this set back.", cw - 200, size=12.5, lh=18
    )
    w.t(x + 174, y + 126, "Squat · 6 reps · 00:12", size=11, fill=MUTED)
    y += 176

    # What to fix
    w.t(x, y + 16, "What to fix", size=17, weight="700")
    w.tag(x + cw - 92, y + 2, "2 THINGS")
    yy = y + 36
    for number, title, issue, why, fix in CORRECTIONS:
        card_h = 128
        w.r(x, yy, cw, card_h, fill=PAPER, stroke=LINE)
        w.t(x + 22, yy + 30, number, size=11, weight="700", fill=MUTED)
        w.t(x + 52, yy + 30, title, size=14.5, weight="700")
        w.para(x + 22, yy + 56, issue, cw - 44, size=12, lh=18, fill=INK)
        w.para(x + 22, yy + 78, why, cw - 44, size=11.5, lh=17, fill=MUTED)
        w.tag(x + 22, yy + 94, "TRY THIS", fill=BOX2)
        w.t(x + 98, yy + 107, fix, size=11.5, fill=BODY)
        yy += card_h + 14
    y = yy + 14

    # What you did well
    w.r(x, y, cw, 40 + len(POSITIVES) * 26, fill=BOX, stroke=LINE)
    w.t(x + 22, y + 28, "What you did well", size=15, weight="700")
    for index, item in enumerate(POSITIVES):
        w.t(x + 22, y + 56 + index * 26, "✓", size=11, weight="700", fill=DARK)
        w.t(x + 42, y + 56 + index * 26, item, size=12, fill=BODY)
    y += 40 + len(POSITIVES) * 26 + 30

    # Measured checks
    w.t(x, y + 16, "Measured checks", size=17, weight="700")
    w.tag(x + cw - 96, y + 2, "5 CHECKS")
    yy = y + 40
    for label, detail, score in CHECK_ROWS:
        w.t(x, yy + 12, label, size=12.5, weight="600")
        w.t(x, yy + 29, detail, size=10.5, fill=MUTED)
        w.r(x + cw - 250, yy + 8, 190, 8, fill=BOX2, stroke="", rx=4)
        w.r(x + cw - 250, yy + 8, 190 * score / 100, 8, fill=DARK, stroke="", rx=4)
        w.t(x + cw - 24, yy + 15, str(score), size=12, weight="700", anchor="end")
        yy += 42
    y = yy + 16

    # Reference card
    w.r(x, y, cw, 130, fill=BOX, stroke=LINE)
    w.t(x + 22, y + 30, "See correct squat form", size=15, weight="700")
    w.media(x + 22, y + 46, 130, 68)
    w.circle(x + 87, y + 80, 14, fill=PAPER, stroke=LINE)
    w.t(x + 87, y + 85, "▶", size=10, fill=DARK, anchor="middle")
    w.eyebrow(x + 172, y + 62, "Reference movement")
    w.t(x + 172, y + 84, "Squat — Correct Technique", size=13, weight="700")
    w.para(
        x + 172,
        y + 102,
        "Watch how the movement should look, then compare it with " "your own set above.",
        cw - 200,
        size=11,
        lh=15,
        fill=MUTED,
    )
    y += 156

    # Progressive-disclosure expander
    w.r(x, y, cw, 46, fill=PAPER, stroke=LINE)
    w.t(x + 20, y + 29, "▾" if expanded else "▸", size=11, fill=BODY)
    w.t(x + 42, y + 29, "See the full detail — how FormFix measured every rep", size=13, weight="600")
    y += 46
    if not expanded:
        w.note(
            x,
            y + 30,
            "Collapsed by default: levels 1 and 2 stay readable in five "
            "seconds, the traceability record is one click away.",
            w=cw - 20,
        )
        return y + 70
    return y


DETECTIONS = (
    (
        "Squat depth",
        "Two of your six repetitions stopped above parallel.",
        "Lower until your hip crease passes your knee, under control.",
        False,
    ),
    (
        "Forward torso lean",
        "Your chest dropped forward on the third repetition.",
        "Think about keeping your chest up as you come out of the bottom.",
        False,
    ),
    (
        "Descent control",
        "One repetition was lowered noticeably faster than the rest.",
        "Take about two seconds to lower, then drive back up.",
        True,
    ),
)

REPS = (
    ("Rep 01", "Good repetition", "Passed: depth, torso, heels", "", "0:02"),
    ("Rep 02", "Stopped above depth", "Squat depth", "Passed: torso, heels", "0:04"),
    ("Rep 03", "Chest dropped forward", "Forward torso lean", "Passed: depth, heels", "0:06"),
    ("Rep 04", "Good repetition", "Passed: depth, torso, heels", "", "0:08"),
)

NOT_ASSESSED = (
    ("Knee tracking", "Needs a front-on camera; this clip was filmed from the side."),
    ("Bar path", "No barbell was detected in the frame."),
)


def ax_details(w: Wire, x: float, y: float, cw: float) -> float:
    """The contents of the expanded 'See the full detail' panel."""
    w.r(x, y, cw, 8, fill=BOX, stroke="", rx=0)
    y += 22
    w.para(
        x + 20,
        y + 14,
        "Everything behind the summary above: how each finding was "
        "detected, how every repetition scored, and which checks could not be assessed "
        "from this recording.",
        cw - 40,
        size=11.5,
        lh=18,
    )
    yy = y + 66
    for title, issue, fix, minor in DETECTIONS:
        w.r(x + 20, yy, cw - 40, 92, fill=PAPER, stroke=LINE)
        w.t(x + 40, yy + 28, title, size=13.5, weight="700")
        if minor:
            w.tag(x + 40 + len(title) * 8 + 12, yy + 14, "ALSO NOTICED", fill=BOX2)
        w.para(x + 40, yy + 52, issue, cw - 80, size=11.5, lh=17)
        w.tag(x + 40, yy + 64, "TRY THIS", fill=BOX2)
        w.t(x + 112, yy + 77, fix, size=11.5, fill=BODY)
        yy += 104
    yy += 10

    w.t(x + 20, yy + 16, "Repetition by repetition", size=14, weight="700")
    w.tag(x + cw - 110, yy + 2, "6 REPS")
    yy += 40
    for number, headline, meta_a, meta_b, stamp in REPS:
        w.r(x + 20, yy, cw - 40, 62, fill=BOX, stroke=LINE)
        w.t(x + 40, yy + 26, number, size=11.5, weight="700")
        w.t(x + 110, yy + 26, headline, size=12.5, weight="600")
        w.t(
            x + 110,
            yy + 46,
            " · ".join(part for part in (meta_a, meta_b) if part),
            size=10.5,
            fill=MUTED,
        )
        w.t(x + cw - 44, yy + 26, stamp, size=10.5, fill=MUTED, anchor="end")
        yy += 70
    w.t(x + 40, yy + 6, "…", size=14, fill=MUTED)
    yy += 34

    w.t(x + 20, yy + 16, "Not assessed", size=14, weight="700")
    w.para(
        x + 20,
        yy + 38,
        "These checks need something this recording could not show, so "
        "FormFix left them unscored rather than guessing.",
        cw - 40,
        size=11.5,
        lh=17,
    )
    yy += 62
    for title, reason in NOT_ASSESSED:
        w.r(x + 20, yy, cw - 40, 46, fill=BOX, stroke=LINE, dash="5 4")
        w.t(x + 40, yy + 28, title, size=12.5, weight="600")
        w.t(x + 200, yy + 28, reason, size=11, fill=MUTED)
        yy += 54
    w.note(
        x + 20,
        yy + 24,
        "Under FORMFIX_DEBUG each finding also prints its measurement, landmarks, "
        "movement phase, rule settings and threshold provenance.",
        w=cw - 60,
    )
    return yy + 80


def ax_failure(w: Wire, x: float, y: float, cw: float) -> float:
    w.r(x, y, cw, 330, fill=BOX, stroke=DARK, sw=1.4)
    w.eyebrow(x + 28, y + 40, "Analysis not possible")
    w.t(x + 28, y + 78, "We couldn't analyse this recording", size=20, weight="700")
    w.para(
        x + 28,
        y + 108,
        "FormFix could not see your hips, knees and ankles clearly enough to measure the "
        "movement. Nothing was scored.",
        cw - 56,
        size=12.5,
        lh=20,
    )
    w.r(x + 28, y + 156, cw - 56, 118, fill=PAPER, stroke=LINE)
    w.t(x + 48, y + 182, "FOR THE BEST RESULT", size=9.5, weight="700", fill=MUTED, spacing=1.3)
    for index, tip in enumerate(SQUAT_TIPS[:3]):
        w.circle(x + 52, y + 204 + index * 22, 2.5, fill=MUTED, stroke="")
        w.t(x + 64, y + 208 + index * 22, tip, size=11.5, fill=BODY)
    w.para(
        x + 28,
        y + 298,
        "Nothing was scored - FormFix never guesses when it cannot see "
        "the movement clearly. Adjust the recording and upload again.",
        cw - 56,
        size=11,
        lh=16,
        fill=MUTED,
    )
    w.note(
        x,
        y + 348,
        "This screen carries whichever reason applies. As well as an unclear recording, "
        "a run stops when the movement contradicts the exercise selected (a bench press "
        "uploaded as a shoulder press) or when tracking repeatedly loses which person to "
        "follow. Other people merely being in shot only adds a note to the result.",
        w=cw - 20,
    )
    return y + 430


# ===========================================================================
# Desktop screens
# ===========================================================================

DESKTOP_W = 1440


def home_full() -> None:
    w = Wire(
        "01-home-full-desktop",
        "Homepage — full page",
        "Desktop 1440px · views/home.py · every section in order",
        DESKTOP_W,
        5760,
        url="formfix.app/",
    )
    y = 0
    y = sec_hero(w, y)
    w.divider(M, y, CW)
    y = sec_about(w, y)
    w.divider(M, y, CW)
    y = sec_problem(w, y)
    w.divider(M, y, CW)
    y = sec_how(w, y)
    w.divider(M, y, CW)
    y = sec_explorer(w, y)
    w.divider(M, y, CW)
    y = sec_explainable(w, y)
    w.divider(M, y, CW)
    y = sec_example(w, y)
    w.divider(M, y, CW)
    y = sec_tech(w, y)
    y = sec_research(w, y)
    y = sec_outro(w, y)
    sec_nav(w)  # fixed pill, drawn last so it sits on top
    w.fit(y)
    w.emit()


def home_hero() -> None:
    w = Wire(
        "02-home-hero-desktop",
        "Homepage — hero and navigation",
        "Desktop 1440px · components/navbar.py, components/hero.py",
        DESKTOP_W,
        860,
    )
    y = sec_hero(w, 0)
    sec_nav(w)
    w.fit(y)
    w.emit()


def home_about() -> None:
    w = Wire(
        "02b-home-about-desktop",
        "Homepage \u2014 what this is, and who it is for",
        "Desktop 1440px \u00b7 components/about.py \u00b7 section 1b, between hero and problem",
        DESKTOP_W,
        560,
    )
    y = sec_about(w, -20)
    w.fit(y - 40)
    w.emit()


def home_explorer() -> None:
    w = Wire(
        "03-home-exercise-explorer-desktop",
        "Homepage — exercise explorer",
        "Desktop 1440px · components/exercise_section.py · squat tab active",
        DESKTOP_W,
        820,
    )
    y = sec_explorer(w, -50)
    w.fit(y)
    w.emit()


def home_explainable() -> None:
    w = Wire(
        "04-home-explainable-and-example-desktop",
        "Homepage — explainable feedback and the example analysis",
        "Desktop 1440px · components/explainable.py, components/results_section.py",
        DESKTOP_W,
        1560,
    )
    y = sec_explainable(w, -40)
    w.divider(M, y, CW)
    y = sec_example(w, y)
    w.fit(y)
    w.emit()


def home_research() -> None:
    w = Wire(
        "05-home-technology-research-outro-desktop",
        "Homepage — technology, research and closing call to action",
        "Desktop 1440px · components/technology.py, research.py, outro.py, footer.py",
        DESKTOP_W,
        1500,
    )
    y = sec_tech(w, -60)
    y = sec_research(w, y)
    y = sec_outro(w, y)
    w.fit(y)
    w.emit()


def analyse_idle() -> None:
    w = Wire(
        "06-analyse-idle-desktop",
        "Analyse — idle (no video selected)",
        "Desktop 1440px · components/analyse_page.py · session stage: idle",
        DESKTOP_W,
        1180,
        url="formfix.app/analyse",
    )
    y = ax_topbar(w)
    y = ax_header(w, y)
    y = ax_tracker(w, y + 20, "idle")
    cw = 560
    left_y = ax_upload(w, M, y + 40, cw, "idle")
    right_y = ax_waiting(w, M + 640, y + 40, cw, ready=False)
    w.note(
        M + 640,
        right_y + 46,
        "Two-column layout on desktop; Streamlit stacks the columns below 640px, so the "
        "upload panel comes first on mobile.",
        w=cw - 20,
    )
    w.fit(max(left_y, right_y + 90))
    w.emit()


def analyse_ready() -> None:
    w = Wire(
        "07-analyse-ready-desktop",
        "Analyse — video uploaded, ready to analyse",
        "Desktop 1440px · session stage: ready",
        DESKTOP_W,
        1560,
        url="formfix.app/analyse",
    )
    y = ax_topbar(w)
    y = ax_header(w, y)
    y = ax_tracker(w, y + 20, "ready")
    cw = 560
    left_y = ax_upload(w, M, y + 40, cw, "ready")
    right_y = ax_waiting(w, M + 640, y + 40, cw, ready=True)
    w.fit(max(left_y, right_y))
    w.emit()


def analyse_running() -> None:
    w = Wire(
        "08-analyse-analysing-desktop",
        "Analyse — pipeline running",
        "Desktop 1440px · session stage: analysing · live stage reporting",
        DESKTOP_W,
        1620,
        url="formfix.app/analyse",
    )
    y = ax_topbar(w)
    y = ax_header(w, y)
    y = ax_tracker(w, y + 20, "analysing")
    cw = 560
    left_y = ax_upload(w, M, y + 40, cw, "analysing")
    right_y = ax_running(w, M + 640, y + 40, cw)
    w.fit(max(left_y, right_y))
    w.emit()


def analyse_results() -> None:
    w = Wire(
        "09-analyse-results-desktop",
        "Analyse — results (progressive disclosure)",
        "Desktop 1440px · session stage: complete · levels 1 and 2 visible",
        DESKTOP_W,
        2340,
        url="formfix.app/analyse",
    )
    y = ax_topbar(w)
    y = ax_header(w, y)
    y = ax_tracker(w, y + 20, "complete")
    cw = 560
    left_y = ax_upload(w, M, y + 40, cw, "complete")
    right_y = ax_results(w, M + 640, y + 40, cw, expanded=False)
    w.fit(max(left_y, right_y))
    w.emit()


def analyse_details() -> None:
    w = Wire(
        "10-analyse-results-expanded-desktop",
        "Analyse — results with the detail panel expanded",
        "Desktop 1440px · level 3 of the progressive disclosure",
        DESKTOP_W,
        1500,
        url="formfix.app/analyse",
    )
    x, cw = M + 640, 560
    w.t(M, 40, "Results column, scrolled to the bottom", size=15, weight="700")
    w.para(
        M,
        68,
        "The first two levels (verdict, corrections, positives, measured checks, "
        "annotated video and reference clip) are shown in the previous wireframe. This "
        "sheet documents what opens underneath them.",
        480,
        size=12,
        lh=19,
    )
    w.note(
        M,
        170,
        "Level 3 stays in plain language for the athlete. The full traceability "
        "chain — measurement, landmarks, movement phase, rule id, threshold provenance — "
        "is level 4 and appears only under FORMFIX_DEBUG.",
        w=460,
    )
    w.r(x, 30, cw, 46, fill=PAPER, stroke=LINE)
    w.t(x + 20, 59, "▾", size=11, fill=BODY)
    w.t(x + 42, 59, "See the full detail — how FormFix measured every rep", size=13, weight="600")
    end = ax_details(w, x, 76, cw)
    w.fit(end)
    w.emit()


def analyse_failure() -> None:
    w = Wire(
        "11-analyse-failure-desktop",
        "Analyse — recording could not be analysed",
        "Desktop 1440px · validation failure · nothing is scored",
        DESKTOP_W,
        1240,
        url="formfix.app/analyse",
    )
    y = ax_topbar(w)
    y = ax_header(w, y)
    y = ax_tracker(w, y + 20, "complete")
    cw = 560
    left_y = ax_upload(w, M, y + 40, cw, "complete")
    right_y = ax_failure(w, M + 640, y + 40, cw)
    w.note(
        M + 640,
        right_y + 30,
        "Never a bare 'analysis failed': the panel states what "
        "could not be measured, why, and what to change before uploading again.",
        w=cw - 20,
    )
    w.fit(max(left_y, right_y + 80))
    w.emit()


def analyse_quality() -> None:
    w = Wire(
        "12-analyse-limited-quality-desktop",
        "Analyse — results with a recording-quality note",
        "Desktop 1440px · analysis ran, but the camera angle limited some checks",
        DESKTOP_W,
        1180,
        url="formfix.app/analyse",
    )
    y = ax_topbar(w)
    y = ax_header(w, y)
    y = ax_tracker(w, y + 20, "complete")
    cw = 560
    left_y = ax_upload(w, M, y + 40, cw, "complete")
    x = M + 640
    yy = y + 40
    w.r(x, yy, cw, 168, fill=BOX, stroke=DARK, sw=1.4)
    w.eyebrow(x + 24, yy + 30, "Recording quality")
    w.t(x + 24, yy + 60, "Analysis completed with a partial camera angle", size=15, weight="700")
    w.para(
        x + 24,
        yy + 86,
        "FormFix could analyse this recording. Squat is analysed from a "
        "side-on view. Keeping the whole movement in frame, with the camera still, gives "
        "the most accurate measurements.",
        cw - 48,
        size=11.5,
        lh=18,
    )
    w.circle(x + 30, yy + 146, 2.5, fill=MUTED, stroke="")
    w.t(x + 42, yy + 150, "Camera was closer to three-quarter than side-on.", size=11.5, fill=BODY)
    yy += 196
    w.r(x, yy, cw, 150, fill=BOX, stroke=LINE)
    ring(w, x + 92, yy + 75, 46, None, "not scored")
    w.eyebrow(x + 174, yy + 44, "Overall form")
    w.t(x + 174, yy + 76, "Not scored", size=24, weight="800")
    w.para(
        x + 174,
        yy + 100,
        "No check had reliable evidence from this camera angle.",
        cw - 200,
        size=12.5,
        lh=18,
    )
    w.t(x + 174, yy + 126, "Squat · 6 reps · 00:12", size=11, fill=MUTED)
    w.note(
        x,
        yy + 190,
        "A missing score is reported as '—  not scored', never as a 0 — a "
        "zero would read as poor technique rather than as absent evidence.",
        w=cw - 20,
    )
    w.fit(max(left_y, yy + 250))
    w.emit()


def reference_modal() -> None:
    w = Wire(
        "13-reference-modal-desktop",
        "Analyse — reference technique lightbox",
        "Desktop 1440px · components/analysis_results.reference_modal",
        DESKTOP_W,
        900,
        url="formfix.app/analyse",
    )
    w.r(0, 0, w.dw, 900, fill="#dcdcde", stroke="", rx=0)
    w.t(M, 40, "page content behind the overlay", size=11, fill=MUTED, italic=True)
    mx, my, mw, mh = 300, 90, 840, 700
    w.r(mx, my, mw, mh, fill=PAPER, stroke=LINE, rx=12)
    w.r(mx + mw - 54, my + 18, 34, 34, fill=PAPER, stroke=LINE, rx=17)
    w.t(mx + mw - 37, my + 40, "✕", size=12, fill=BODY, anchor="middle")
    w.media(mx + 30, my + 66, mw - 60, 470, "reference clip · video player")
    w.pose(mx + mw / 2 - 130, my + 110, 260, 380)
    w.r(mx + 54, my + 494, mw - 108, 28, fill=PAPER, stroke=LINE, rx=14)
    w.t(mx + 72, my + 513, "▶", size=10, fill=BODY)
    w.line(mx + 92, my + 508, mx + mw - 96, my + 508, stroke=HAIR, sw=3)
    w.t(mx + 30, my + 578, "Squat — Correct Technique", size=17, weight="700")
    w.t(
        mx + 30,
        my + 604,
        "Reference clip · Hip depth · Torso angle · Heel contact · " "Return to standing",
        size=11.5,
        fill=MUTED,
    )
    w.note(
        mx + 30,
        my + 640,
        "Closes on the ✕, on a backdrop click and on Escape. "
        "scripts/analyse.js moves the dialog to <body> so its fixed positioning "
        "resolves against the viewport.",
        w=mw - 80,
    )
    w.fit(my + mh, pad=30)
    w.emit()


# ===========================================================================
# Mobile screens (390px)
# ===========================================================================

MW = 390  # device width
MM = 20  # page margin
MCW = MW - MM * 2


def m_nav(w: Wire, y: float = 12, open_menu: bool = False) -> float:
    w.r(MM, y, MCW, 54, fill=PAPER, stroke=LINE, rx=27)
    w.r(MM + 16, y + 15, 26, 24, fill=BOX2, stroke=LINE, rx=4)
    w.t(MM + 50, y + 33, "FORMFIX", size=12, weight="800")
    w.r(MM + MCW - 46, y + 16, 26, 22, fill=BOX2, stroke=LINE, rx=4)
    w.line(MM + MCW - 40, y + 23, MM + MCW - 26, y + 23, stroke=INK, sw=1.6)
    w.line(MM + MCW - 40, y + 31, MM + MCW - 26, y + 31, stroke=INK, sw=1.6)
    return y + 54


def mobile_home_hero() -> None:
    w = Wire(
        "m01-home-hero-mobile",
        'Homepage — hero and "what this is"',
        "Mobile 390px · single column, navigation collapses to a menu button",
        MW,
        1180,
        chrome="phone",
        url="formfix.app/",
    )
    y = m_nav(w) + 34
    w.eyebrow(MM, y, "Explainable computer vision")
    w.eyebrow(MM, y + 16, "For people learning to lift")
    y += 50
    for line in HERO_HEADLINE:
        # 25px, not the desktop's 27: "Understand the mistake." overran the column.
        w.t(MM, y, line, size=25, weight="800")
        y += 34
    y += 8
    y = w.para(MM, y, HERO_SUB, MCW, size=12.5, lh=20) + 30
    w.btn(MM, y, MCW, 48, "Analyse Your Form  →", size=13.5)
    w.btn(MM, y + 58, MCW, 48, "See How It Works", primary=False, size=13.5)
    y += 132
    w.media(MM, y, MCW, 330, "athlete photograph")
    w.pose(MM + 60, y + 26, 230, 280)
    w.tag(MM + 12, y + 14, "● POSE DETECTED", fill=PAPER)
    w.tag(
        MM + MCW - 12 - w.tag_width("TRACKING 33 LANDMARKS"),
        y + 298,
        "TRACKING 33 LANDMARKS",
        fill=PAPER,
    )
    y += 360
    w.t(MW / 2, y + 20, "SCROLL", size=9, fill=MUTED, weight="700", anchor="middle", spacing=1.6)

    # The orientation block matters most here: on a phone the hero fills the
    # screen, so this is the first thing a reader meets after scrolling once.
    y += 62
    w.divider(MM, y, MCW)
    w.eyebrow(MM, y + 40, "What this is")
    w.t(MM, y + 76, "A second pair of eyes on", size=17, weight="800")
    w.t(MM, y + 100, "the lifts you're learning.", size=17, weight="800")
    w.line(MM, y + 118, MM + 48, y + 118, stroke=DARK, sw=2)
    y = w.para(MM, y + 140, ABOUT_LEDE, MCW, size=12.5, lh=20) + 22
    w.t(MM, y, ABOUT_CHIPS_LABEL.upper(), size=8.5, fill=MUTED, weight="700", spacing=1.4)
    y += 14
    # Wrap the pills rather than letting them run off a 390px column.
    cx = MM
    for chip in ABOUT_CHIPS:
        label = chip.upper()
        if cx + len(label) * (8.5 * 0.67 + 0.7) + 18 > MM + MCW:
            cx, y = MM, y + 28
        cx += w.tag(cx, y, label, size=8.5) + 8
    y += 40
    w.circle(MM + 3, y - 4, 3, fill=DARK, stroke="")
    y = w.para(MM + 16, y, ABOUT_ASIDE, MCW - 16, size=11, lh=17, fill=MUTED) + 22
    for index, (title, marker, lines) in enumerate(ABOUT_PANELS):
        lead = index == 0
        h = 58 + len(lines) * 34
        w.r(
            MM,
            y,
            MCW,
            h,
            fill=PAPER if lead else BOX,
            stroke=DARK if lead else LINE,
            sw=1.4 if lead else 1,
        )
        w.t(
            MM + 18,
            y + 26,
            title.upper(),
            size=9,
            fill=INK if lead else MUTED,
            weight="700",
            spacing=1.2,
        )
        w.line(MM + 18, y + 40, MM + MCW - 18, y + 40, stroke=LINE)
        ly = y + 64
        for line in lines:
            w.circle(MM + 26, ly - 4, 7, fill=BOX2 if lead else PAPER, stroke=LINE)
            w.t(MM + 26, ly, marker, size=8, fill=INK if lead else MUTED, anchor="middle")
            w.para(MM + 44, ly, line, MCW - 64, size=11.5, lh=16)
            ly += 34
        y += h + 14
    w.fit(y + 8)
    w.emit()


def mobile_menu() -> None:
    w = Wire(
        "m02-home-menu-mobile",
        "Homepage — overlay menu (mobile)",
        "Mobile 390px · the nav toggle opens the full-width menu below 900px",
        MW,
        720,
        chrome="phone",
        url="formfix.app/",
    )
    w.r(0, 0, MW, 760, fill="#e4e4e6", stroke="", rx=0)
    y = m_nav(w)
    w.r(MM, y + 12, MCW, 420, fill=PAPER, stroke=LINE)
    ly = y + 62
    for label in NAV_ITEMS:
        w.t(MM + 26, ly, label, size=16, weight="600")
        w.divider(MM + 26, ly + 20, MCW - 52)
        ly += 58
    w.btn(MM + 26, ly + 6, MCW - 52, 48, "Analyse Form", size=13.5)
    w.t(MM, ly + 84, "page content dimmed behind the menu", size=11, fill=MUTED, italic=True)
    w.note(
        MM,
        ly + 120,
        "Menu links scroll smoothly to their section and close the "
        "overlay; the CTA is the one link that leaves the page.",
        w=MCW,
    )
    w.fit(ly + 190)
    w.emit()


def mobile_home_sections() -> None:
    w = Wire(
        "m03-home-sections-mobile",
        "Homepage — how it works and exercise explorer " "(mobile)",
        "Mobile 390px · rail collapses to a single column, explorer " "stacks copy above figure",
        MW,
        1560,
        chrome="phone",
        url="formfix.app/",
    )
    y = 40
    w.eyebrow(MM, y, "How FormFix works")
    w.t(MM, y + 36, "From your camera roll", size=23, weight="800")
    w.t(MM, y + 66, "to a clear correction.", size=23, weight="800")
    y += 104
    for number, title, text, _tagname in STEPS:
        w.r(MM, y, MCW, 116, fill=BOX, stroke=LINE)
        w.circle(MM + 30, y + 30, 14, fill=PAPER, stroke=LINE)
        w.t(MM + 30, y + 34, number, size=10.5, weight="700", anchor="middle")
        w.t(MM + 58, y + 34, title, size=15, weight="700")
        w.para(MM + 18, y + 60, text, MCW - 36, size=11.5, lh=17, max_lines=3)
        y += 128
    y += 20
    w.eyebrow(MM, y, "Exercise explorer")
    w.t(MM, y + 36, "Three movements.", size=23, weight="800")
    w.t(MM, y + 66, "One analysis engine.", size=23, weight="800")
    y += 92
    tx = MM
    for index, (name, *_r) in enumerate(EXERCISES):
        tw = len(name) * 6.6 + 26
        w.r(tx, y, tw, 34, fill=DARK if index == 0 else PAPER, stroke="" if index == 0 else LINE, rx=17)
        w.t(
            tx + tw / 2,
            y + 22,
            name,
            size=10.5,
            weight="600",
            fill=PAPER if index == 0 else BODY,
            anchor="middle",
        )
        tx += tw + 8
    y += 52
    w.t(MM, y + 14, "Squat", size=20, weight="700")
    w.t(MM + MCW, y + 14, "01 / 03", size=11, fill=MUTED, anchor="end")
    y = w.para(MM, y + 40, EXERCISES[0][2], MCW, size=12, lh=19) + 26
    for area in EXERCISES[0][3]:
        w.circle(MM + 4, y - 4, 3, fill=DARK, stroke="")
        w.t(MM + 16, y, area, size=11.5, fill=BODY)
        y += 22
    y += 12
    w.media(MM, y, MCW, 300, "exercise photograph")
    w.pose(MM + 70, y + 24, 210, 250)
    w.tag(MM + 10, y + 268, "● KNEE ANGLE 92°", fill=PAPER)
    y += 320
    w.r(MM, y, 40, 40, fill=PAPER, stroke=LINE, rx=20)
    w.t(MM + 20, y + 26, "←", size=14, fill=BODY, anchor="middle")
    w.r(MM + 50, y, 40, 40, fill=PAPER, stroke=LINE, rx=20)
    w.t(MM + 70, y + 26, "→", size=14, fill=BODY, anchor="middle")
    w.r(MM + MCW - 150, y, 150, 40, fill=BOX, stroke=LINE, rx=8)
    w.t(MM + MCW - 136, y + 25, "NEXT  Shoulder Press", size=10, weight="600", fill=BODY)
    w.fit(y + 40)
    w.emit()


def mobile_analyse_idle() -> None:
    w = Wire(
        "m04-analyse-idle-mobile",
        "Analyse — idle (mobile)",
        "Mobile 390px · columns stack: upload first, then the results placeholder",
        MW,
        1240,
        chrome="phone",
        url="formfix.app/analyse",
    )
    w.t(MM, 34, "←  Back Home", size=12, weight="600", fill=BODY)
    w.t(MM + MCW, 34, "FORMFIX", size=11, weight="800", anchor="end")
    y = 74
    w.eyebrow(MM, y, "Analyse your form")
    w.t(MM, y + 34, "Upload a set.", size=25, weight="800")
    w.t(MM, y + 66, "Get an explanation.", size=25, weight="800")
    y = (
        w.para(
            MM,
            y + 96,
            "Record a single set, drop the clip in, and FormFix will "
            "track your body through the movement and explain what it sees.",
            MCW,
            size=12,
            lh=19,
        )
        + 40
    )

    # compact tracker
    for index, label in enumerate(TRACKER):
        cx = MM + 28 + index * ((MCW - 56) / 2)
        if index:
            w.line(
                cx - ((MCW - 56) / 2) + 20,
                y + 16,
                cx - 20,
                y + 16,
                stroke=DARK if index == 0 else HAIR,
                sw=2,
            )
        w.circle(cx, y + 16, 15, fill=PAPER, stroke=DARK if index == 0 else LINE, sw=1.5)
        if index == 0:
            w.circle(cx, y + 16, 5, fill=DARK, stroke="")
        w.t(
            cx,
            y + 50,
            label.split()[0],
            size=10.5,
            fill=INK if index == 0 else MUTED,
            weight="600" if index == 0 else "400",
            anchor="middle",
        )
    y += 82

    y = panel_head(w, MM, y, "01", "Your video")
    w.t(MM, y + 18, "MP4 · MOV · AVI · 3-120 seconds · one set per clip", size=10.5, fill=MUTED)
    y += 40
    for index, (name, *_r) in enumerate(EXERCISES):
        w.r(MM, y + index * 40, MCW, 34, fill=BOX if index == 0 else PAPER, stroke=LINE)
        w.circle(MM + 20, y + index * 40 + 17, 7, fill=PAPER, stroke=LINE)
        if index == 0:
            w.circle(MM + 20, y + index * 40 + 17, 3.5, fill=DARK, stroke="")
        w.t(MM + 38, y + index * 40 + 21, name, size=12, fill=INK)
    y += len(EXERCISES) * 40 + 16

    guide_h = 32 + len(SQUAT_TIPS) * 34
    w.r(MM, y, MCW, guide_h, fill=BOX, stroke=LINE)
    w.t(MM + 16, y + 22, "HOW TO RECORD", size=9, weight="700", fill=MUTED, spacing=1.2)
    w.tag(MM + MCW - 108, y + 10, "SIDE-ON VIEW", size=8.5, fill=PAPER)
    for index, tip in enumerate(SQUAT_TIPS):
        w.circle(MM + 20, y + 44 + index * 34, 2.5, fill=MUTED, stroke="")
        w.para(MM + 32, y + 48 + index * 34, tip, MCW - 52, size=10.5, lh=14, max_lines=2)
    y += guide_h + 20

    w.r(MM, y, MCW, 110, fill=BOX, stroke=LINE, dash="6 5")
    w.r(MM + MCW / 2 - 18, y + 22, 36, 32, fill=PAPER, stroke=LINE, rx=6)
    w.t(MM + MCW / 2, y + 44, "↥", size=14, fill=MUTED, anchor="middle")
    w.t(MM + MCW / 2, y + 72, "Drag and drop file here", size=12, weight="600", anchor="middle")
    w.btn(MM + MCW / 2 - 60, y + 82, 120, 26, "Browse files", primary=False, size=10.5)
    w.para(
        MM, y + 108,
        "Your video is deleted as soon as it has been analysed, and no account is needed.",
        MCW, size=10, lh=14,
    )
    y += 150

    w.r(MM, y, MCW, 190, fill=BOX, stroke=LINE, dash="6 5")
    w.eyebrow(MM + 20, y + 32, "Waiting for a video")
    w.t(MM + 20, y + 62, "Your results will appear here", size=15, weight="700")
    w.para(
        MM + 20, y + 88, "Nothing is analysed until you press Analyse Form.", MCW - 40, size=11.5, lh=17
    )
    for index, (label, _note) in enumerate(PREVIEW_ROWS):
        w.circle(MM + 26, y + 122 + index * 22 - 4, 3, fill=MUTED, stroke="")
        w.t(MM + 38, y + 122 + index * 22, label, size=11.5, weight="600")
    w.fit(y + 190)
    w.emit()


def mobile_analyse_running() -> None:
    w = Wire(
        "m05-analyse-analysing-mobile",
        "Analyse — pipeline running (mobile)",
        "Mobile 390px · session stage: analysing",
        MW,
        1060,
        chrome="phone",
        url="formfix.app/analyse",
    )
    w.t(MM, 34, "←  Back Home", size=12, weight="600", fill=BODY)
    y = 66
    for index, label in enumerate(TRACKER):
        cx = MM + 28 + index * ((MCW - 56) / 2)
        if index:
            w.line(
                cx - ((MCW - 56) / 2) + 20,
                y + 16,
                cx - 20,
                y + 16,
                stroke=DARK if index <= 1 else HAIR,
                sw=2,
                dash="6 5" if index == 1 else None,
            )
        done, active = index < 1, index == 1
        w.circle(
            cx,
            y + 16,
            15,
            fill=DARK if done else PAPER,
            stroke=DARK if done or active else LINE,
            sw=1.5,
        )
        if done:
            w.t(cx, y + 21, "✓", size=11, fill=PAPER, weight="700", anchor="middle")
        elif active:
            w.circle(cx, y + 16, 5, fill=DARK, stroke="")
        w.t(
            cx,
            y + 50,
            label.split()[0],
            size=10.5,
            fill=INK if done or active else MUTED,
            anchor="middle",
        )
    y += 84
    y = panel_head(w, MM, y, "02", "Analysing", "a few seconds", live=True)
    w.media(MM, y + 16, MCW, 300, "frame being analysed")
    w.pose(MM + 80, y + 44, 190, 240)
    w.line(MM + 8, y + 170, MM + MCW - 8, y + 170, stroke=DARK, sw=2, dash="10 6")
    w.tag(MM + MCW - 168, y + 286, "● TRACKING 33 LANDMARKS", size=8.5, fill=PAPER)
    y += 340
    for index, stage in enumerate(ANALYSIS_STAGES):
        done, active = index < 3, index == 3
        w.t(
            MM + 4,
            y + index * 26,
            "✓" if done else ("●" if active else "—"),
            size=10.5,
            fill=DARK if done or active else FAINT,
            weight="700",
        )
        w.t(
            MM + 26,
            y + index * 26,
            f"{stage} · 46%" if active else stage,
            size=11.5,
            weight="700" if active else "400",
            fill=INK if done or active else MUTED,
        )
    y += len(ANALYSIS_STAGES) * 26 + 20
    w.note(
        MM,
        y + 20,
        "The upload panel is above this on mobile; both of its buttons are "
        "disabled while the pipeline runs.",
        w=MCW,
    )
    w.fit(y + 70)
    w.emit()


def mobile_analyse_results() -> None:
    w = Wire(
        "m06-analyse-results-mobile",
        "Analyse — results (mobile)",
        "Mobile 390px · session stage: complete · same progressive disclosure",
        MW,
        1720,
        chrome="phone",
        url="formfix.app/analyse",
    )
    w.t(MM, 34, "←  Back Home", size=12, weight="600", fill=BODY)
    y = 62
    w.t(MM, y + 18, "Your analysed movement", size=15, weight="700")
    w.media(MM, y + 34, MCW, 250, "annotated video")
    w.pose(MM + 90, y + 56, 170, 200)
    y += 306

    w.r(MM, y, MCW, 190, fill=BOX, stroke=LINE)
    ring(w, MM + MCW / 2, y + 66, 44, 78, "/ 100")
    w.t(MM + MCW / 2, y + 132, "Good Form", size=20, weight="800", anchor="middle")
    w.t(MM + MCW / 2, y + 154, "Overall form", size=10, fill=MUTED, anchor="middle", weight="700")
    w.t(MM + MCW / 2, y + 174, "Squat · 6 reps · 00:12", size=10.5, fill=MUTED, anchor="middle")
    y += 214

    w.t(MM, y + 14, "What to fix", size=15, weight="700")
    w.tag(MM + MCW - 80, y, "2 THINGS", size=8.5)
    y += 32
    for number, title, issue, _why, fix in CORRECTIONS:
        w.r(MM, y, MCW, 140, fill=PAPER, stroke=LINE)
        w.t(MM + 16, y + 26, number, size=10.5, weight="700", fill=MUTED)
        w.t(MM + 42, y + 26, title, size=13, weight="700")
        w.para(MM + 16, y + 50, issue, MCW - 32, size=11, lh=16, max_lines=2)
        w.tag(MM + 16, y + 82, "TRY THIS", size=8.5, fill=BOX2)
        w.para(MM + 16, y + 120, fix, MCW - 32, size=10.5, lh=14, max_lines=2)
        y += 152
    y += 6

    w.r(MM, y, MCW, 36 + len(POSITIVES) * 24, fill=BOX, stroke=LINE)
    w.t(MM + 16, y + 24, "What you did well", size=13.5, weight="700")
    for index, item in enumerate(POSITIVES):
        w.t(MM + 16, y + 50 + index * 24, "✓", size=10.5, weight="700", fill=DARK)
        w.para(MM + 32, y + 50 + index * 24, item, MCW - 52, size=10.5, lh=14, max_lines=1)
    y += 36 + len(POSITIVES) * 24 + 26

    w.t(MM, y + 14, "Measured checks", size=15, weight="700")
    y += 34
    for label, detail, score in CHECK_ROWS:
        w.t(MM, y + 12, label, size=11.5, weight="600")
        w.t(MM, y + 28, detail, size=9.5, fill=MUTED)
        w.r(MM + MCW - 120, y + 8, 92, 7, fill=BOX2, stroke="", rx=3.5)
        w.r(MM + MCW - 120, y + 8, 92 * score / 100, 7, fill=DARK, stroke="", rx=3.5)
        w.t(MM + MCW, y + 14, str(score), size=11, weight="700", anchor="end")
        y += 40
    y += 8
    w.r(MM, y, MCW, 110, fill=BOX, stroke=LINE)
    w.t(MM + 16, y + 26, "See correct squat form", size=13, weight="700")
    w.media(MM + 16, y + 40, 104, 54)
    w.t(MM + 136, y + 62, "Squat — Correct", size=11.5, weight="600")
    w.t(MM + 136, y + 80, "Technique", size=11.5, weight="600")
    y += 128
    w.r(MM, y, MCW, 44, fill=PAPER, stroke=LINE)
    w.t(MM + 16, y + 28, "▸  See the full detail", size=12, weight="600")
    w.fit(y + 44)
    w.emit()


# ===========================================================================
# Screen flow / navigation map
# ===========================================================================


def node(
    w: Wire,
    x: float,
    y: float,
    bw: float,
    bh: float,
    title: str,
    lines: tuple[str, ...] = (),
    kind: str = "screen",
) -> None:
    fills = {"screen": PAPER, "state": BOX, "variant": PAPER, "overlay": BOX2}
    dash = "6 5" if kind == "overlay" else None
    w.r(
        x,
        y,
        bw,
        bh,
        fill=fills[kind],
        stroke=DARK if kind == "screen" else LINE,
        sw=1.5 if kind == "screen" else 1,
        dash=dash,
    )
    w.t(x + 16, y + 26, title, size=13.5, weight="700")
    for index, line in enumerate(lines):
        w.t(x + 16, y + 48 + index * 17, line, size=10.5, fill=MUTED)


def arrow(
    w: Wire,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str = "",
    bend: bool = False,
    label_dx: float = 0,
    label_dy: float = -12,
) -> None:
    if bend:
        mid = (y1 + y2) / 2
        w.path(f"M {x1} {y1} L {x1} {mid} L {x2} {mid} L {x2} {y2}", stroke=MUTED, sw=1.4)
    else:
        w.line(x1, y1, x2, y2, stroke=MUTED, sw=1.4)
    angle = 0 if abs(y2 - y1) < 1 else (90 if y2 > y1 else -90)
    if abs(x2 - x1) < 1:
        head = (
            f"M {x2 - 5} {y2 - 8} L {x2} {y2} L {x2 + 5} {y2 - 8}"
            if y2 > y1
            else f"M {x2 - 5} {y2 + 8} L {x2} {y2} L {x2 + 5} {y2 + 8}"
        )
    else:
        head = (
            f"M {x2 - 8} {y2 - 5} L {x2} {y2} L {x2 - 8} {y2 + 5}"
            if x2 > x1
            else f"M {x2 + 8} {y2 - 5} L {x2} {y2} L {x2 + 8} {y2 + 5}"
        )
    w.path(head, stroke=MUTED, sw=1.4)
    _ = angle
    if label:
        cx = (x1 + x2) / 2 + label_dx
        cy = (y1 + y2) / 2 + label_dy
        w.t(cx, cy, label, size=10, fill=BODY, anchor="middle", italic=True)


def flow_map() -> None:
    w = Wire(
        "00-screen-flow",
        "Screen flow and interface states",
        "How the two pages connect, and the state machine behind /analyse",
        1740,
        880,
        chrome="none",
    )
    w.t(40, 44, "PAGES", size=10, weight="700", fill=MUTED, spacing=1.4)
    node(
        w,
        40,
        62,
        300,
        226,
        "Homepage  /",
        (
            "1   Hero",
            "1b  What this is / who it is for",
            "2   The problem",
            "3   How it works (5 steps)",
            "4   Exercise explorer (3 tabs)",
            "5   Explainable Feedback",
            "6   Example analysis",
            "7   Technology + Research",
            "8   Closing CTA + footer",
        ),
    )
    node(
        w,
        40,
        326,
        300,
        96,
        "Overlay menu",
        (
            "Below 900px only",
            "Same links + Analyse Form CTA",
        ),
        kind="overlay",
    )
    arrow(w, 190, 288, 190, 326, "nav toggle", label_dx=54, label_dy=4)

    node(
        w,
        430,
        130,
        250,
        112,
        "Analyse  /analyse",
        (
            "Upload column (left)",
            "State column (right)",
            "Progress tracker on top",
        ),
    )
    arrow(w, 340, 150, 430, 150, '"Analyse Form"')
    arrow(w, 430, 205, 340, 205, '"Back Home"', label_dy=18)

    w.t(
        430,
        300,
        'SESSION STATE  ·  st.session_state["ff_ax_stage"]',
        size=10,
        weight="700",
        fill=MUTED,
        spacing=1.2,
    )
    states = (
        ("IDLE", ("nothing uploaded", "results placeholder")),
        ("READY", ("clip confirmed", '"Press Analyse Form"')),
        ("ANALYSING", ("8 pipeline stages", "controls disabled")),
        ("COMPLETE", ("an AnalysisResult", "is on screen")),
    )
    sx, sw_, gap = 430, 240, 60
    for index, (name, lines) in enumerate(states):
        node(w, sx + index * (sw_ + gap), 322, sw_, 92, name, lines, kind="state")
        if index:
            arrow(
                w,
                sx + index * (sw_ + gap) - gap,
                368,
                sx + index * (sw_ + gap),
                368,
                ("upload a clip", "press Analyse Form", "pipeline finishes")[index - 1],
            )

    # loops back
    w.path(
        f"M {sx + 3 * (sw_ + gap) + sw_ / 2} 414 L {sx + 3 * (sw_ + gap) + sw_ / 2} 452 "
        f"L {sx + sw_ / 2} 452 L {sx + sw_ / 2} 414",
        stroke=MUTED,
        sw=1.4,
        dash="5 4",
    )
    w.path(
        f"M {sx + sw_ / 2 - 5} 422 L {sx + sw_ / 2} 414 L {sx + sw_ / 2 + 5} 422", stroke=MUTED, sw=1.4
    )
    w.t(
        sx + sw_ * 1.9,
        468,
        '"Remove video"  →  back to IDLE',
        size=10,
        fill=BODY,
        anchor="middle",
        italic=True,
    )
    w.path(
        f"M {sx + 3 * (sw_ + gap) + 40} 322 L {sx + 3 * (sw_ + gap) + 40} 296 "
        f"L {sx + 2 * (sw_ + gap) + sw_ / 2} 296 L {sx + 2 * (sw_ + gap) + sw_ / 2} 322",
        stroke=MUTED,
        sw=1.4,
        dash="5 4",
    )
    w.t(
        sx + 2 * (sw_ + gap) + sw_ / 2 + 90,
        290,
        '"Analyse again"',
        size=10,
        fill=BODY,
        anchor="middle",
        italic=True,
    )

    w.t(430, 528, "WHAT COMPLETE CAN SHOW", size=10, weight="700", fill=MUTED, spacing=1.2)
    variants = (
        (
            "Results",
            (
                "verdict + score ring",
                "what to fix / what went well",
                "measured checks, annotated video",
            ),
        ),
        (
            "Results + quality note",
            ("analysis ran, camera angle limited it", 'score may read "— not scored"'),
        ),
        (
            "Analysis not possible",
            ("validation failed", "nothing is scored", "what to change, then upload again"),
        ),
    )
    complete_cx = sx + 3 * (sw_ + gap) + sw_ / 2
    w.path(f"M {complete_cx} 414 L {complete_cx} 506", stroke=MUTED, sw=1.4)
    w.line(590, 506, complete_cx, 506, stroke=MUTED, sw=1.4)
    for index, (name, lines) in enumerate(variants):
        vx = 430 + index * 350
        node(w, vx, 550, 320, 108, name, lines, kind="variant")
        arrow(w, vx + 160, 506, vx + 160, 550)
    w.t(600, 496, "the same COMPLETE stage, three possible results", size=10, fill=BODY, italic=True)

    node(
        w,
        430,
        700,
        320,
        96,
        "Reference lightbox",
        (
            "opens over the results",
            "✕ / backdrop / Escape closes",
        ),
        kind="overlay",
    )
    node(
        w,
        780,
        700,
        320,
        96,
        "Full detail panel",
        (
            "level 3: findings, rep by rep,",
            "not assessed",
        ),
        kind="overlay",
    )
    node(
        w,
        1130,
        700,
        320,
        96,
        "FORMFIX_DEBUG",
        (
            "level 4: measurement, landmarks,",
            "phase, rule id, thresholds",
        ),
        kind="overlay",
    )
    arrow(w, 590, 658, 590, 700, '"See correct form"', label_dx=90, label_dy=4)
    arrow(w, 700, 658, 940, 700, '"See the full detail"', label_dx=40, label_dy=-4)
    arrow(w, 1100, 748, 1130, 748)
    w.t(1115, 692, "env flag", size=9.5, fill=BODY, anchor="middle", italic=True)

    w.t(40, 700, "PROGRESSIVE DISCLOSURE", size=10, weight="700", fill=MUTED, spacing=1.2)
    for index, line in enumerate(
        (
            "L1  verdict, corrections, positives, checks",
            "L2  annotated video, reference clip",
            "L3  every finding, rep by rep, not assessed",
            "L4  full traceability (developer only)",
        )
    ):
        w.t(40, 728 + index * 22, line, size=11, fill=BODY)
    w.fit(810)
    w.emit()


# ===========================================================================

BUILDERS = (
    flow_map,
    home_full,
    home_hero,
    home_about,
    home_explorer,
    home_explainable,
    home_research,
    analyse_idle,
    analyse_ready,
    analyse_running,
    analyse_results,
    analyse_details,
    analyse_failure,
    analyse_quality,
    reference_modal,
    mobile_home_hero,
    mobile_menu,
    mobile_home_sections,
    mobile_analyse_idle,
    mobile_analyse_running,
    mobile_analyse_results,
)


def main() -> None:
    for builder in BUILDERS:
        builder()
    files = sorted(p.name for p in OUT.glob("*.svg"))
    print(f"{len(files)} wireframes written to {OUT}")
    for name in files:
        print("  ", name)


if __name__ == "__main__":
    main()
