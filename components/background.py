"""
Two fixed layers: the scroll-progress hairline at the top of the viewport and
the atmosphere behind everything. main.js moves both to document.body after
render - inside a Streamlit block container, position: fixed resolves against
the container, not the viewport.
"""

from __future__ import annotations

from utils.helpers import strip
from utils.styling import html


def render() -> None:
    html(strip("""
            <div class="scroll-progress" aria-hidden="true">
              <span class="scroll-progress__bar"></span>
            </div>
            <div class="atmosphere" aria-hidden="true">
              <div class="atmosphere__pool atmosphere__pool--a"></div>
              <div class="atmosphere__pool atmosphere__pool--b"></div>
              <div class="atmosphere__grain"></div>
            </div>
            """))
