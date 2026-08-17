"""
The page footer.

It lives inside the outro section (as in the original prototype), so it is
exposed as :func:`markup` rather than rendering itself - but it can also be
rendered standalone with :func:`render`.
"""

from __future__ import annotations

from utils.helpers import strip
from utils.styling import html

META = ("FORMFIX AI", "MSc Computer Science &middot; Thesis Prototype", "2026")


def markup() -> str:
    meta = "".join(f"<span>{item}</span>" for item in META)
    return strip(
        f"""
        <footer class="footer">
          <p class="footer__disclaimer">
            FormFix AI is an educational exercise-technique tool and is not a replacement
            for professional coaching or medical advice.
          </p>
          <div class="footer__meta">{meta}</div>
        </footer>
        """
    )


def render() -> None:
    html(f'<div class="ff-page">{markup()}</div>')
