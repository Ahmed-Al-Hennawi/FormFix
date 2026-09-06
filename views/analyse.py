"""The /analyse page. A wrapper: page stylesheet, background layers, then
components/analyse_page.py, then the two behaviour scripts."""

from __future__ import annotations

import components
from utils.styling import (
    inject_analyse_behaviour,
    inject_analyse_styles,
    inject_behaviour,
)


def render() -> None:
    inject_analyse_styles()

    # Same background as the homepage but no navbar, to keep the page focused.
    components.background.render()

    components.analyse_page.render()

    inject_behaviour()
    inject_analyse_behaviour()
