"""
rabbit.weak — Live Born (+corrections) weak-rate functionals.

Modules (planned)
-----------------
channels.py      Six channels (a)–(f) with explicit 1D integrands.
                 Born calibration: K = 1/(τ_n λ₀ m_e⁵), λ₀ ≈ 1.63609.
quadrature.py    Gauss–Laguerre in E_e (N_E = 32 default).
                 Gauss–Legendre on bounded [m_e, Δ] for channels (c), (f).
live_rates.py    Distribution functional: λ[f₀(q)].
                 Born level: 1D integral of monopole f₀(E) only.
                 No table fallback. No perturbative correction factor.
reference_quadrature.py
                 Dense fixed-grid trapezoid reference evaluator for
                 GL32/GL64/trapz1000 validation; not a production solver.
augmented_bridge.py
                 Extracts augmented-distribution monopoles for live
                 weak-rate kernels.

Key physics (R03)
-----------------
At Born level, angular orthogonality of the V−A matrix element ensures
that only the monopole component f₀(E) of the neutrino distribution
contributes to weak n ↔ p rates.  Higher multipoles enter at O(E/m_N)
~ 10⁻³ through recoil and weak magnetism corrections.
"""

# Phase 3 deflation: the augmented_angular_rates / augmented_bridge re-exports were
# removed with the AP65 audit track (zero live consumers; only augmented tests used them).
# Import the live submodules directly, e.g. `from rabbit.weak.live_rates import ...`.
__all__: list[str] = []
