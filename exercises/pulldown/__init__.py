"""
Lat-pulldown analyser. Written for the movement in the FormFix reference clip:
seated, bilateral, pronated grip, bar in front of the head to the upper chest.
Behind-the-neck, close-grip, single-arm and standing variations are not
described correctly by these rules.

Two checks: excessive torso movement during the pull, and incomplete range of
motion at both ends. Trunk angular velocity is computed and exported for a
possible "swinging" check but no rule reads it.
"""
