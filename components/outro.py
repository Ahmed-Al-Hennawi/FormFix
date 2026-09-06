"""Section 8 - Final call to action."""

from __future__ import annotations

from utils.assets import LOGO_FULL, asset_url
from utils.helpers import masked_lines, strip
from utils.routing import analyse_url
from utils.styling import html

from . import footer

HEADLINE = ("Better movement starts", "with understanding it.")


def render() -> None:
    html(strip(f"""
            <div class="ff-page">
            <section class="outro section" id="ff-outro">
              <div class="container container--narrow outro__inner">
                <img class="outro__logo" src="{asset_url(LOGO_FULL)}"
                     alt="FormFix - Analyse, Correct, Improve" data-animate="fade-up" />
                <h2 class="display outro__title">{masked_lines(HEADLINE)}</h2>
                <a href="{analyse_url()}" class="btn btn--primary btn--lg" target="_self"
                   data-animate="fade-up">
                  Analyse Your Form <span class="btn__arrow">&rarr;</span>
                </a>
              </div>
              {footer.markup()}
            </section>
            </div>
            """))
