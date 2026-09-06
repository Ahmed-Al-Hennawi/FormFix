"""
The page footer. It sits inside the outro section, so the useful entry point
is markup(); render() is there for using it standalone.
"""

from __future__ import annotations

from utils.helpers import strip
from utils.styling import html

META = ("FORMFIX", "MSc Computer Science &middot; Thesis Project", "2026")


def markup() -> str:
    meta = "".join(f"<span>{item}</span>" for item in META)
    return strip(f"""
        <footer class="footer">
          <p class="footer__disclaimer">
            FormFix is an educational exercise-technique tool and is not a replacement
            for professional coaching or medical advice.
          </p>
          <div class="footer__meta">{meta}</div>
        </footer>
        """)


def render() -> None:
    html(f'<div class="ff-page">{markup()}</div>')
