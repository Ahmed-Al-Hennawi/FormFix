"""
The upload / analysis page.

Thin wrapper: the page-specific stylesheet, the shared atmosphere layers, the
experience itself (``components/analyse_page.py``) and finally the two
behaviour layers - the shared one from the homepage, then the page-specific
one (reference lightbox, score count-up).
"""

from __future__ import annotations

import components
from utils.styling import (
    inject_analyse_behaviour,
    inject_analyse_styles,
    inject_behaviour,
)


def render() -> None:
    inject_analyse_styles()

    # Same atmosphere and scroll-progress line as the homepage, no navbar -
    # the page stays focused on the analysis.
    components.background.render()

    components.analyse_page.render()

    inject_behaviour()
    inject_analyse_behaviour()
