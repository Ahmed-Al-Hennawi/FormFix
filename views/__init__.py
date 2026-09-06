"""
The two pages: home (the marketing site) and analyse (the analysis studio).
app.py registers both with st.navigation.
"""

from . import analyse, home

__all__ = ["analyse", "home"]
