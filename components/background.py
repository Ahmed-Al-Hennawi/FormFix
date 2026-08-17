"""
The two fixed layers that sit behind and above everything else:

* the **scroll progress line** - a 2px hairline pinned to the very top of the
  viewport, scaled on the X axis from ``main.js`` (0% at the top of the page,
  ~50% halfway down, 100% at the bottom)
* the **atmosphere** - two blurred radial light pools plus a fine grain layer,
  re-tinted whenever the active exercise changes

Both are emitted here and then relocated to ``document.body`` by ``main.js``,
so their ``position: fixed`` is always resolved against the viewport rather
than against a Streamlit block container.
"""

from __future__ import annotations

from utils.helpers import strip
from utils.styling import html


def render() -> None:
    html(
        strip(
            """
            <div class="scroll-progress" aria-hidden="true">
              <span class="scroll-progress__bar"></span>
            </div>
            <div class="atmosphere" aria-hidden="true">
              <div class="atmosphere__pool atmosphere__pool--a"></div>
              <div class="atmosphere__pool atmosphere__pool--b"></div>
              <div class="atmosphere__grain"></div>
            </div>
            """
        )
    )
