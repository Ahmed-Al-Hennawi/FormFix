"""
Pages.

    home      the marketing site, unchanged
    analyse   the upload / analysis studio

Both are registered in ``app.py`` with ``st.navigation``:

    /            -> views.home.render
    /analyse     -> views.analyse.render
"""

from . import analyse, home

__all__ = ["analyse", "home"]
