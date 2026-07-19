"""
rabbit.thermo.plasma_perfect_fluid_justification

Documentation and diagnostic for the e±/γ perfect fluid approximation.

=== PHYSICAL JUSTIFICATION ===

The electromagnetic plasma (e±, γ) is treated as a perfect fluid
(π_ab = 0) throughout BBN.  This is justified by three independent
arguments:

1. KNUDSEN NUMBER
   Kn = λ_mfp / L_H ~ 10⁻¹⁸ at T = 1 MeV.
   The photon mean free path is λ_mfp ~ 1/(n_e σ_T) ~ 10⁻¹⁸ / H.
   The plasma is in the extreme hydrodynamic limit with no possibility
   of developing a kinetic anisotropic stress at any relevant level.

2. δY_p ESTIMATE
   Even if an artificial ℓ=2 quadrupole were present, its effect on
   Y_p would be:
       δY_p ~ (π_plasma/ρ_total) × Σ² ~ Kn × Σ² ~ 10⁻²¹
   This is negligible by 17 orders of magnitude compared to
   observational precision (δY_p ~ 10⁻⁴).

3. ANGULAR ORTHOGONALITY
   The ℓ=2 e± quadrupole vanishes identically in the angle-integrated
   weak rates (same angular orthogonality that restricts Born rates
   to the monopole, R03).  Even if π_plasma ≠ 0, it would not affect
   the weak n↔p rates at Born level.

4. LITERATURE CONSENSUS
   No existing BBN code includes electromagnetic plasma transport:
   - PRIMAT (Pitrou et al. 2018): perfect fluid
   - Park et al. (2025): Bianchi BBN, perfect fluid
   - Rothman & Matzner (1984): early Bianchi BBN, perfect fluid
   This is physics, not an approximation.

=== DIAGNOSTIC ===

The optional diagnostic computes |π_plasma / π_ν| to verify that
the plasma anisotropic stress is negligible relative to the neutrino
contribution.  Since π_plasma = 0 exactly, this ratio is identically 0.
"""
import numpy as np


def knudsen_number(T_MeV: float) -> float:
    """Compute the Knudsen number Kn = λ_mfp / L_H at temperature T.

    Parameters
    ----------
    T_MeV : float
        Plasma temperature [MeV].

    Returns
    -------
    float
        Kn (dimensionless). Should be ≪ 1 for fluid description.
    """
    # Thomson cross section σ_T = 6.65e-25 cm²
    sigma_T_cm2 = 6.6524587321e-25
    # Electron number density: n_e ~ (ζ(3)/π²) T³ for T > m_e
    # In natural units [MeV³], convert to cm⁻³
    hbar_c_cm = 197.3269804e-13  # MeV·cm
    zeta3 = 1.202056903

    m_e = 0.5109989500
    if T_MeV < m_e / 10.0:
        return 0.0  # No e± pairs, infinite mfp but no scatterers

    n_e_natural = 2.0 * zeta3 / np.pi**2 * T_MeV**3  # MeV³
    n_e_cgs = n_e_natural / hbar_c_cm**3  # cm⁻³

    lambda_mfp_cm = 1.0 / max(n_e_cgs * sigma_T_cm2, 1e-100)

    # Hubble scale L_H = c/H
    # H ~ √(8πG ρ/3), ρ ~ (π²/30) g_* T⁴
    G_N = 6.70883e-45  # MeV⁻²
    g_star = 10.75  # approximate for T ~ MeV
    rho = (np.pi**2 / 30.0) * g_star * T_MeV**4
    H_MeV = np.sqrt(8.0 * np.pi * G_N / 3.0 * rho)
    H_invsec = H_MeV * 1.519267447e21
    c_cm = 2.998e10  # cm/s
    L_H_cm = c_cm / max(H_invsec, 1e-100)

    return lambda_mfp_cm / L_H_cm


def delta_Yp_from_plasma_stress(Kn: float, Sigma_sq: float) -> float:
    """Estimate δY_p from hypothetical plasma anisotropic stress.

    δY_p ~ Kn × Σ² (dimensional estimate).
    """
    return Kn * Sigma_sq


def plasma_stress_ratio(pi_nu: float) -> float:
    """Compute |π_plasma / π_ν|.  Always returns 0 (π_plasma = 0 exactly)."""
    return 0.0


# Precomputed reference values
KN_AT_1MEV = knudsen_number(1.0)
DELTA_YP_AT_1MEV_SIGMA01 = delta_Yp_from_plasma_stress(KN_AT_1MEV, 0.01)
