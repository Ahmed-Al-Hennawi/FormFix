"""
The exercise-agnostic half of the analysers. What changes between exercises is
the movement itself - which landmarks matter, which signal marks a rep, which
thresholds apply. The machinery around that doesn't, so it lives here: the rule
spec, persistence filtering, the rep state machine, phase scoping, reliability,
feedback aggregation and the failure mapping.

Nothing in here knows what a squat is.
"""
