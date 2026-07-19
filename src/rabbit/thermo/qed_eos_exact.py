"""
rabbit.thermo.qed_eos_exact — Exact QED corrections to the e±γ plasma EOS.

Implements the O(e²) + O(e³) finite-temperature QED corrections to
the pressure and energy density of the photon-electron-positron plasma,
following Bennett et al. (2020, JCAP 03, 003; arXiv:1911.04504) and
the compact formulation of Akita & Yamaguchi (2022, arXiv:2210.10307).

The corrections enter the plasma equation of state as:
    P = P_ideal + P^(2) + P^(3)
    ρ = ρ_ideal + ρ^(2) + ρ^(3)

where the superscript denotes the order in the QED coupling e = √(4πα).

O(e²) = O(α) corrections have three terms:
    1. Photon thermal self-energy from the electron loop (dominant)
    2. e⁺e⁻ Fock exchange energy
    3. Finite-mass corrections to exchange

O(e³) = O(α^{3/2}) correction:
    Ring-diagram resummation (Debye screening / plasmon contribution)

Physical effects:
    - δN_eff^{(e²)} ≈ +0.010 (from modified e±γ ↔ ν energy transfer)
    - δN_eff^{(e³)} ≈ −0.001 (Bennett et al. 2020 — previously overlooked)
    - Combined δY_p ~ 10⁻⁴ (marginal for Y_p, essential for N_eff)

This is a PLASMA-FRAME SCALAR EOS (isotropic plasma assumption).
The e±γ plasma is isotropic to Kn ~ 10⁻¹⁸ during BBN.
Anisotropic QED EOS (tensor response) requires separate offline
QFT computation and is NOT implemented here (Gate G_QED).

References:
    Bennett et al. 2020, JCAP 03, 003 [arXiv:1911.04504]
    Akita & Yamaguchi 2022, arXiv:2210.10307, Eq. 2.51-2.54
    Mangano et al. 2002, Nucl. Phys. B 729, 221
    Heckler 1994, Phys. Rev. D 49, 611
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.laguerre import laggauss


# ═══════════════════════════════════════════════════════════════
# §1. Constants
# ═══════════════════════════════════════════════════════════════

_M_E: float = 0.5109989500          # electron mass [MeV]
_ALPHA: float = 1.0 / 137.035999084  # fine structure constant
_E2: float = 4.0 * np.pi * _ALPHA    # e² = 4πα
_E: float = np.sqrt(_E2)             # e = √(4πα)
_PI: float = np.pi
_PI2: float = np.pi**2
_PI4: float = np.pi**4


# ═══════════════════════════════════════════════════════════════
# §2. Core Fermi-Dirac integrals for QED corrections
# ═══════════════════════════════════════════════════════════════

# Quadrature setup: Gauss-Laguerre for semi-infinite integrals
_N_QUAD: int = 80
_GL_NODES, _GL_WEIGHTS = laggauss(_N_QUAD)


def _validate_mu(chemical_potential_MeV: float) -> float:
    mu = float(chemical_potential_MeV)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential_MeV must be finite.")
    return mu


def _N_F(E: np.ndarray, T: float, chemical_potential_MeV: float = 0.0) -> np.ndarray:
    """Fermi-Dirac occupation summed over particle + antiparticle.

    N_F(p) = 2/(exp(E_p/T) + 1)

    At nonzero signed electron chemical potential, this becomes
    f_e(E, mu_e) + f_pos(E, mu_e).  The resulting scalar occupation extension
    is still an isotropic plasma-frame EOS, not a tensor QED response.
    """
    if T < 1e-10:
        return np.zeros_like(E)
    mu = _validate_mu(chemical_potential_MeV)
    electron_arg = np.clip((E - mu) / T, -500, 500)
    positron_arg = np.clip((E + mu) / T, -500, 500)
    return 1.0 / (np.exp(electron_arg) + 1.0) + 1.0 / (np.exp(positron_arg) + 1.0)


def compute_I1(T: float, chemical_potential_MeV: float = 0.0) -> float:
    """Integral I₁(T) = ∫₀^∞ dp (p²/E_p) N_F(p).

    Uses Gauss-Laguerre quadrature with substitution p = T×t.
    """
    if T < 1e-10:
        return 0.0

    # Substitution: p = T × t, dp = T dt
    # Integrand: (T²t²/E_p) × N_F(p) × T × e^t (undo Laguerre weight)
    t = _GL_NODES
    w = _GL_WEIGHTS
    p = T * t
    E_p = np.sqrt(p**2 + _M_E**2)
    nf = _N_F(E_p, T, chemical_potential_MeV)

    integrand = (p**2 / E_p) * nf * np.exp(t) * T
    return float(np.sum(w * integrand))


def compute_I2(T: float, n_quad: int = 40, chemical_potential_MeV: float = 0.0) -> float:
    """Integral I₂(T) = ∫∫ dp dp' (pp')/(E_p E_{p'}) ln|(p+p')/(p-p')| N_F(p) N_F(p').

    2D integral computed as double Gauss-Laguerre sum.
    Uses reduced quadrature (n_quad) for speed — 2D cost is O(n²).
    """
    if T < 1e-10:
        return 0.0

    nodes, weights = laggauss(n_quad)
    p = T * nodes
    E_p = np.sqrt(p**2 + _M_E**2)
    nf = _N_F(E_p, T, chemical_potential_MeV)

    # Precompute 1D factors
    factor = (p / E_p) * nf * np.exp(nodes) * T  # includes dp = T dt

    result = 0.0
    for i in range(n_quad):
        for j in range(n_quad):
            if abs(p[i] - p[j]) < 1e-30:
                # L'Hôpital: ln|(p+p')/(p-p')| → 2p'/p when p→p'
                log_term = 2.0 * p[j] / max(p[i], 1e-30)
            else:
                log_term = np.log(abs((p[i] + p[j]) / (p[i] - p[j])))
            result += weights[i] * weights[j] * factor[i] * factor[j] * log_term

    return result


def compute_I_D(T: float, chemical_potential_MeV: float = 0.0) -> float:
    """Integral I_D(T) = ∫₀^∞ dp [(p² + E²_p)/E_p] N_F(p).

    Related to the Debye mass: m²_D = e² I_D / (2π²).
    """
    if T < 1e-10:
        return 0.0

    t = _GL_NODES
    w = _GL_WEIGHTS
    p = T * t
    E_p = np.sqrt(p**2 + _M_E**2)
    nf = _N_F(E_p, T, chemical_potential_MeV)

    integrand = ((p**2 + E_p**2) / E_p) * nf * np.exp(t) * T
    return float(np.sum(w * integrand))


# ═══════════════════════════════════════════════════════════════
# §3. QED pressure corrections
# ═══════════════════════════════════════════════════════════════

def P_qed_e2(T: float, chemical_potential_MeV: float = 0.0) -> float:
    """O(e²) QED correction to pressure [MeV⁴].

    P^(2) = -(e²T²)/(12π²) I₁ - (e²)/(8π⁴) I₁² + (e²m_e²)/(16π⁴) I₂

    Three contributions:
    1. Photon self-energy from electron loop (dominant, negative)
    2. Fock exchange energy (negative)
    3. Finite-mass exchange correction (positive, partial cancellation)
    """
    i1 = compute_I1(T, chemical_potential_MeV=chemical_potential_MeV)
    i2 = compute_I2(T, chemical_potential_MeV=chemical_potential_MeV)

    term1 = -(_E2 * T**2) / (12.0 * _PI2) * i1
    term2 = -_E2 / (8.0 * _PI4) * i1**2
    term3 = _E2 * _M_E**2 / (16.0 * _PI4) * i2

    return term1 + term2 + term3


def P_qed_e3(T: float, chemical_potential_MeV: float = 0.0) -> float:
    """O(e³) QED correction to pressure (ring diagram) [MeV⁴].

    P^(3) = (e³ T)/(12π⁴) × I_D^{3/2}

    This is the Debye screening / plasmon contribution.
    Negative contribution to pressure (partially cancels O(e²)).
    """
    i_D = compute_I_D(T, chemical_potential_MeV=chemical_potential_MeV)
    if i_D <= 0:
        return 0.0
    return (_E**3 * T) / (12.0 * _PI4) * i_D**1.5


# ═══════════════════════════════════════════════════════════════
# §4. QED energy density from thermodynamic relation
# ═══════════════════════════════════════════════════════════════

def _rho_from_P(P_func, T: float, dT_frac: float = 1e-4) -> float:
    """Energy density from pressure via ρ = -P + T(∂P/∂T).

    Uses central finite difference for ∂P/∂T.
    """
    h = max(T * dT_frac, 1e-8)
    P_plus = P_func(T + h)
    P_minus = P_func(T - h)
    dPdT = (P_plus - P_minus) / (2.0 * h)
    return -P_func(T) + T * dPdT


def rho_qed_e2(T: float, chemical_potential_MeV: float = 0.0) -> float:
    """O(e²) QED correction to energy density [MeV⁴]."""
    mu = _validate_mu(chemical_potential_MeV)
    return _rho_from_P(lambda temp: P_qed_e2(temp, chemical_potential_MeV=mu), T)


def rho_qed_e3(T: float, chemical_potential_MeV: float = 0.0) -> float:
    """O(e³) QED correction to energy density [MeV⁴]."""
    mu = _validate_mu(chemical_potential_MeV)
    return _rho_from_P(lambda temp: P_qed_e3(temp, chemical_potential_MeV=mu), T)


# ═══════════════════════════════════════════════════════════════
# §5. Combined exact QED EOS
# ═══════════════════════════════════════════════════════════════

def qed_correction_exact(T: float, chemical_potential_MeV: float = 0.0) -> dict:
    """Compute exact QED corrections to pressure and energy density.

    Parameters
    ----------
    T : float
        Temperature [MeV].

    Returns
    -------
    dict with keys:
        P_e2, P_e3, P_total : O(e²), O(e³), combined pressure correction
        rho_e2, rho_e3, rho_total : corresponding energy density corrections
        delta_rho_rel : δρ/ρ_ideal relative correction
        I1, I2, ID : raw integral values for diagnostics
    """
    mu = _validate_mu(chemical_potential_MeV)
    i1 = compute_I1(T, chemical_potential_MeV=mu)
    i2 = compute_I2(T, chemical_potential_MeV=mu)
    i_D = compute_I_D(T, chemical_potential_MeV=mu)

    # Pressure
    P2_t1 = -(_E2 * T**2) / (12.0 * _PI2) * i1
    P2_t2 = -_E2 / (8.0 * _PI4) * i1**2
    P2_t3 = _E2 * _M_E**2 / (16.0 * _PI4) * i2
    P2 = P2_t1 + P2_t2 + P2_t3

    P3 = (_E**3 * T) / (12.0 * _PI4) * max(i_D, 0.0)**1.5 if i_D > 0 else 0.0

    # Energy density via thermodynamic relation
    r2 = rho_qed_e2(T, chemical_potential_MeV=mu)
    r3 = rho_qed_e3(T, chemical_potential_MeV=mu)

    # Ideal energy density for relative comparison
    rho_gamma = (_PI2 / 15.0) * T**4
    rho_e_rel = (7.0 / 4.0) * rho_gamma  # relativistic e± limit
    rho_ideal_approx = rho_gamma + rho_e_rel  # rough, for δρ/ρ estimate

    return {
        'P_e2': P2, 'P_e3': P3, 'P_total': P2 + P3,
        'rho_e2': r2, 'rho_e3': r3, 'rho_total': r2 + r3,
        'delta_rho_rel': (r2 + r3) / max(rho_ideal_approx, 1e-100),
        'I1': i1, 'I2': i2, 'ID': i_D,
        'chemical_potential_MeV': mu,
        'qed_contract': 'exact_finite_mu_scalar_occupation_v1',
    }


def delta_rho_qed_exact(T: float) -> float:
    """Total QED correction to energy density δρ [MeV⁴].

    Drop-in replacement for the approximate qed_delta_rho().
    """
    return rho_qed_e2(T) + rho_qed_e3(T)


def delta_P_qed_exact(T: float) -> float:
    """Total QED correction to pressure δP [MeV⁴]."""
    return P_qed_e2(T) + P_qed_e3(T)


def delta_rho_qed_exact_with_electron_mu(
    T: float,
    chemical_potential_MeV: float = 0.0,
) -> float:
    """Scalar finite-mu extension of the exact O(e²)+O(e³) QED energy correction."""

    mu = _validate_mu(chemical_potential_MeV)
    return rho_qed_e2(T, chemical_potential_MeV=mu) + rho_qed_e3(
        T,
        chemical_potential_MeV=mu,
    )


def delta_P_qed_exact_with_electron_mu(
    T: float,
    chemical_potential_MeV: float = 0.0,
) -> float:
    """Scalar finite-mu extension of the exact O(e²)+O(e³) QED pressure correction."""

    mu = _validate_mu(chemical_potential_MeV)
    return P_qed_e2(T, chemical_potential_MeV=mu) + P_qed_e3(
        T,
        chemical_potential_MeV=mu,
    )
