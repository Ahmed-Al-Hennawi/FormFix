"""
The floating pill navigation.

Behaviour ported from the original prototype and implemented in ``main.js``:

* permanently fixed - the pill never leaves the viewport
* translucent at rest, dark glass once the page starts moving (``is-scrolled``)
* smooth scrolling to sections, with an offset so the pill never covers a heading
* active section indication via IntersectionObserver
* a responsive overlay menu below 900px

The "Analyse Form" call to action is the one link that leaves the page: it
opens ``/analyse``, the dedicated upload and analysis experience.
"""

from __future__ import annotations

from utils.assets import LOGO_MARK, asset_url
from utils.helpers import strip
from utils.routing import analyse_url
from utils.styling import html

#: (label, section id) - identical to the original prototype's nav.
NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("How It Works", "how"),
    ("Exercises", "exercises"),
    ("Explainable AI", "explainable"),
    ("Technology", "technology"),
    ("Research", "research"),
)


def render() -> None:
    links = "".join(
        f'<li><a href="#ff-{target}" class="nav__link" data-scroll-to="ff-{target}">{label}</a></li>'
        for label, target in NAV_ITEMS
    )
    mobile_links = "".join(
        f'<a href="#ff-{target}" data-scroll-to="ff-{target}">{label}</a>'
        for label, target in NAV_ITEMS
    )

    html(
        strip(
            f"""
            <header class="nav-wrap">
              <nav class="nav" aria-label="Primary">
                <a class="nav__brand" href="#ff-top" data-scroll-to="ff-top" aria-label="FormFix AI home">
                  <img class="nav__mark" src="{asset_url(LOGO_MARK)}" alt="" width="44" height="33" />
                  <span class="nav__wordmark">FORMFIX<span class="nav__ai">AI</span></span>
                </a>
                <ul class="nav__links">{links}</ul>
                <div class="nav__actions">
                  <a href="{analyse_url()}" class="btn btn--nav" target="_self">Analyse Form</a>
                  <button class="nav__toggle" aria-expanded="false" aria-controls="mobile-menu" aria-label="Open menu">
                    <span></span><span></span>
                  </button>
                </div>
              </nav>
              <div class="mobile-menu" id="mobile-menu" hidden>
                {mobile_links}
                <a href="{analyse_url()}" class="mobile-menu__cta" target="_self">Analyse Form</a>
              </div>
            </header>
            """
        )
    )
