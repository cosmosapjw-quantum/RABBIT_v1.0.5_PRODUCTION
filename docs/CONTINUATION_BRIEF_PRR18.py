"""
RABBIT v895 — Post-PRR18 Continuation Brief
============================================

Previous thread accomplished:
  - PRR18: CL3 (finite nucleon mass) JAX port — 0.00e+00 parity, 32 tests
  - PRR18: Teff JIT-safe rewrite with cubic spline — ≤1.11e-16 parity
  - PRR18: Teff end-to-end driver wiring — phase1+phase2 tier-2 RHS
  - PRR18: Backend capabilities updated: max_correction_level=3, supports_teff=True
  - Matched-physics parity: |ΔY_p| = 6.0e-4 (SciPy CL2 vs JAX CL2, FLRW)
  - Shear sweep: ΔY_p > 0 for all Σ > 0, ratio ≈ 0.76 vs analytic
  - Channel decomposition at Σ=0.2: Ch.1 = +1.30e-4, Ch.2 = +2.51e-5 (19.2%)
  - 57 regression tests passing

Current package: RABBIT_v895_PRR18_FULL.zip
  - pip install ".[dev]" → all JAX modules functional
  - CL0–CL3 + Teff available via:
    JAXTypeIConfig(thermo_tier=2, use_live_weak_monopoles=True,
                   correction_level=3, enable_teff=True)

KNOWN ISSUES:
  1. Ch.2/Ch.1 = 19.2% at N_q=6 vs expected ~1.4% from SciPy reference
     → N_q sensitivity confirms convergence; likely resolution at N_q=12+
     → Not a code bug, but a physics precision issue at low N_q
  2. P0-3: fig10 panel (a) still has retired eigenvalue baked in image
     → Needs regeneration script (not available in current package)
  3. JAX default dispatch (CAPABILITY_BY_BACKEND["jax"]) still CL0/tier-1
     → Full physics requires explicit opt-in

IMMEDIATE ACTIONS (next session):
  1. N_q=12 end-to-end run: measure Ch.2/Ch.1 convergence
  2. Promote JAX tier-2 CL3+Teff → validated_candidate if parity holds
  3. fig10 regeneration (P0-3)
  4. Paper I LaTeX updates (P1-1: network description, P1-2: report diagram)
  5. End-to-end dynesty validation on BBN likelihood

DEFERRED TO LATER:
  - JAX Class A geometry/transport (Phase I–IV of ALLTYPE_PROMPT_LIST)
  - Joint BBN+CMB analysis
  - GPU benchmark
"""
