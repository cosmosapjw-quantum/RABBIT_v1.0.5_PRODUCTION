"""
rabbit.config — Conventions, grids, fidelity levels, solver configuration.

Modules (planned)
-----------------
conventions.py    Metric signature, time variable N = ln a, species enum,
                  ℓ_max = 2 exact for Type I (not a truncation parameter).
fidelity.py       FidelityLevel enum: REFERENCE_EXACT → STUB.
grids.py          MomentumGrid (Gauss–Laguerre, default N_q = 80),
                  MultipoleSpec (ℓ_max = 2 for Type I).
solver_config.py  SciPy Radau/BDF defaults, rtol/atol.
"""
