"""
Getting the prototype's CSS and JS into the Streamlit document.

CSS is easy - st.markdown puts a style element in the same document. JavaScript
isn't, because Streamlit strips <script> out of markdown, so it goes in through
a zero-height components.html iframe whose script reaches into
window.parent.document and installs the behaviour there.
"""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from .paths import ANALYSE_CSS, ANALYSE_JS, MAIN_CSS, MAIN_JS, read_text

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2"
    "?family=Space+Grotesk:wght@400;500;600;700"
    "&family=Inter:wght@400;500;600"
    "&display=swap"
)


def inject_styles() -> None:
    """Load styles/main.css into the page."""
    css = read_text(MAIN_CSS)
    st.markdown(
        f'<style>@import url("{GOOGLE_FONTS}");\n{css}</style>',
        unsafe_allow_html=True,
    )


def inject_page_styles(path) -> None:
    """Load an extra stylesheet on top of main.css, so /analyse styles stay off the homepage."""
    css = read_text(path)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def inject_behaviour() -> None:
    """Install scripts/main.js into the parent document. Call it at the end of a
    page, once the markup it enhances exists."""
    inject_script(MAIN_JS, element_id="formfix-behaviour", teardown="FormFixTeardown")


def inject_script(path, element_id: str, teardown: str | None = None) -> None:
    """
    Install a JavaScript file into the parent (Streamlit) document. element_id
    names the injected script so a rerun replaces it rather than stacking
    listeners; teardown is a window function the previous copy exposed.

    The bootstrap carries a per-run nonce: without it the iframe content is
    identical every rerun, Streamlit reuses the frame and the script never runs
    again - which showed up as new feedback cards rendering invisible.
    """
    js = read_text(path)
    if not js:
        return

    nonce = st.session_state.get("_ff_script_nonce", 0) + 1
    st.session_state["_ff_script_nonce"] = nonce

    # json.dumps gives a JS-safe string literal; the replace stops a literal
    # "</script>" in the source from closing this tag early.
    payload = json.dumps(js).replace("</", "<\\/")
    teardown_js = (
        f"""
  if (window.parent.{teardown}) {{
    try {{ window.parent.{teardown}(); }} catch (e) {{}}
  }}"""
        if teardown
        else ""
    )

    bootstrap = f"""
<script>
/* run {nonce} */
(function () {{
  const doc = window.parent && window.parent.document;
  if (!doc) return;

  // Streamlit reruns re-execute this component. Replace the previous copy
  // rather than stacking duplicate listeners on top of each other.
  const existing = doc.getElementById({json.dumps(element_id)});
  if (existing) existing.remove();{teardown_js}

  const script = doc.createElement("script");
  script.id = {json.dumps(element_id)};
  script.type = "text/javascript";
  script.textContent = {payload};
  doc.head.appendChild(script);
}})();
</script>
"""
    components.html(bootstrap, height=0, width=0)


def inject_analyse_styles() -> None:
    """The /analyse stylesheet, loaded on that page only."""
    inject_page_styles(ANALYSE_CSS)


def inject_analyse_behaviour() -> None:
    """The /analyse behaviour layer. Goes in after main.js so the shared
    scroll-reveal observers are already running."""
    inject_script(ANALYSE_JS, element_id="formfix-analyse", teardown="FormFixAnalyseTeardown")


def html(markup: str) -> None:
    """Shorthand for writing a block of custom markup."""
    st.markdown(markup, unsafe_allow_html=True)
