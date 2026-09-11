"""The /analyse page: stylesheet, background, components/analyse_page.py, then the JS."""

from __future__ import annotations

import components
from utils.styling import (
    inject_analyse_behaviour,
    inject_analyse_styles,
    inject_behaviour,
)


def render() -> None:
    inject_analyse_styles()

    # same background as the homepage but no navbar, to keep it focused
    components.background.render()

    components.analyse_page.render()

    inject_behaviour()
    inject_analyse_behaviour()
