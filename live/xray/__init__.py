"""X-Ray: read-only episode introspection for live mode (π0.5 backend).

Computation happens ONLY on replan ticks and only for enabled overlays;
when no overlay is enabled the episode loop takes the untouched act() path
(equivalence gate: setup/xray_equivalence_check.py).
"""
