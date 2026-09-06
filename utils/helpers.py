"""Small string helpers used by the markup components."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape


def esc(text: str) -> str:
    """Escape a string for inclusion in markup."""
    return escape(str(text), quote=True)


def masked_lines(lines: Iterable[str], animate: str = "line", dim_from: int | None = None) -> str:
    """
    Masked line-reveal markup: each line sits in an overflow:hidden mask so it can
    slide up into view. The data-animate hook has to go on the mask, not the line -
    the line starts translated fully below it, so it is clipped to zero area and an
    IntersectionObserver watching it never fires.
    """
    out = []
    for index, line in enumerate(lines):
        dim = " line--dim" if dim_from is not None and index >= dim_from else ""
        attr = f' data-animate="{animate}"' if animate else ""
        out.append(f'<span class="line-mask"{attr}><span class="line{dim}">{line}</span></span>')
    return "".join(out)


def eyebrow(text: str, animate: bool = True) -> str:
    """The small uppercase label that opens each section."""
    attr = ' data-animate="fade-up"' if animate else ""
    return f'<p class="eyebrow"{attr}>{esc(text)}</p>'


def section_open(name: str, section_id: str, extra_class: str = "") -> str:
    classes = " ".join(part for part in (name, "section", extra_class) if part)
    return f'<section class="{classes}" id="{section_id}">'


SECTION_CLOSE = "</section>"


def container(inner: str, narrow: bool = False, extra_class: str = "") -> str:
    classes = "container"
    if narrow:
        classes += " container--narrow"
    if extra_class:
        classes += f" {extra_class}"
    return f'<div class="{classes}">{inner}</div>'


def strip(markup: str) -> str:
    """
    Flatten a triple-quoted markup block onto unindented, blank-line-free lines.
    Streamlit's markdown renderer reads four-space indentation as a code block, and
    a blank line ends an HTML block, so both have to go.
    """
    lines = (line.strip() for line in markup.strip().splitlines())
    return "\n".join(line for line in lines if line)
