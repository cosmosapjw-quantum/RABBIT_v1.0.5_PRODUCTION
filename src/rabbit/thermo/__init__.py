"""
rabbit.thermo — QED EOS, incomplete decoupling, entropy tracking.

Modules (planned)
-----------------
eos_photon_electron.py          ρ(T), p(T), s(T), dρ/dT with finite m_e.
                                O(e³) QED correction mandatory (ΔN_eff ~ −0.001).
incomplete_decoupling.py        Live ν–e energy exchange dQ_ν/dt.
                                Four tiers: parametric → nudec_BSM_v2 → ν–ν → oscillation.
plasma_perfect_fluid_justification.py
                                Documentation: Kn ~ 10⁻¹⁸, δY_p ~ 10⁻²¹.
                                e±/γ anisotropic stress is identically negligible.

Key physics (R05)
-----------------
The electromagnetic plasma (e±, γ) is a perfect fluid to extraordinary
precision during BBN.  The Knudsen number Kn ~ 10⁻¹⁸ at T = 1 MeV, and
the resulting δY_p ~ 10⁻²¹ is negligible by 17 orders of magnitude.
Angular orthogonality further ensures that the ℓ = 2 quadrupole of e±
vanishes identically in the angle-integrated weak rates.  No existing
BBN code includes this correction.
"""
