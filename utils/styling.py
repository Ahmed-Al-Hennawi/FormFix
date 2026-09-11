"""
Gets the CSS and JS into the Streamlit page. CSS just goes in with st.markdown.
Streamlit strips <script> from markdown, so the JS goes through a zero-height
components.html iframe that installs it into window.parent.document.
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
    css = read_text(MAIN_CSS)
    st.markdown(
        f'<style>@import url("{GOOGLE_FONTS}");\n{css}</style>',
        unsafe_allow_html=True,
    )


def inject_page_styles(path) -> None:
    """Extra stylesheet on top of main.css, so /analyse styles stay off the homepage."""
    css = read_text(path)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def inject_behaviour() -> None:
    """Install scripts/main.js. Call it at the end of the page, after the markup."""
    inject_script(MAIN_JS, element_id="formfix-behaviour", teardown="FormFixTeardown")


def inject_script(path, element_id: str, teardown: str | None = None) -> None:
    """
    Install a JS file into the parent Streamlit page. element_id lets a rerun
    replace the script instead of stacking listeners, and teardown is a function
    the old copy exposed. A per-run nonce is needed, otherwise Streamlit reuses the
    iframe, the script doesn't run again and new feedback cards stay invisible.
    """
    js = read_text(path)
    if not js:
        return

    nonce = st.session_state.get("_ff_script_nonce", 0) + 1
    st.session_state["_ff_script_nonce"] = nonce

    # json.dumps gives a safe JS string, and the replace stops a "</script>" in
    # the source closing the tag early
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
    inject_page_styles(ANALYSE_CSS)


def inject_analyse_behaviour() -> None:
    """The /analyse JS. Goes in after main.js so the scroll-reveal is already running."""
    inject_script(ANALYSE_JS, element_id="formfix-analyse", teardown="FormFixAnalyseTeardown")


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)
