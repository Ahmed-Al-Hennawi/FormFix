"""
Floating pill navbar. Just the markup - main.js handles the glass effect,
smooth scrolling, active section and the mobile menu below 900px.
"""

from __future__ import annotations

from utils.assets import LOGO_MARK, asset_url
from utils.helpers import strip
from utils.routing import analyse_url
from utils.styling import html

# (label, section id)
NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("How It Works", "how"),
    ("Exercises", "exercises"),
    ("Explainable Feedback", "explainable"),
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

    html(strip(f"""
            <header class="nav-wrap">
              <nav class="nav" aria-label="Primary">
                <a class="nav__brand" href="#ff-top" data-scroll-to="ff-top" aria-label="FormFix home">
                  <img class="nav__mark" src="{asset_url(LOGO_MARK)}" alt="" width="44" height="33" />
                  <span class="nav__wordmark">FORMFIX</span>
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
            """))
